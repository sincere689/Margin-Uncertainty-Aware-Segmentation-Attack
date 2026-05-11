import os
import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, FCN_ResNet50_Weights
from tqdm import tqdm

from dataset import get_dataloader
from attacks import get_attack, normalize_image
from metrics import ConfusionMatrix
from logger import ExperimentLogger

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    print("❌ 缺少库！请运行: pip install transformers timm")


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='对抗攻击实验控制台 - 多架构支持版')
    parser.add_argument('--model_arch', type=str, default='deeplabv3', choices=['deeplabv3', 'fcn', 'segformer'],
                        help='模型架构')
    parser.add_argument('--dataset', type=str, default=None, choices=['voc', 'cityscapes'], help='手动指定数据集')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--norm_type', type=str, default='inf', choices=['inf', 'l2', 'two'])

    parser.add_argument('--no_weights', action='store_true')

    parser.add_argument('--use_margin_weight', action='store_true',
                        help='显式开启 margin 权重')
    parser.add_argument('--use_uncertainty_blind_weight', action='store_true',
                        help='显式开启 uncertainty blind-spot 权重')

    parser.add_argument('--high_margin_ratio', type=float, default=0.2,
                        help='高 margin 主攻区比例')
    parser.add_argument('--uncertainty_gamma', type=float, default=0.5,
                        help='盲区不确定性权重强度')
    parser.add_argument('--uncertainty_max_weight', type=float, default=1.5,
                        help='盲区不确定性权重上限')
    parser.add_argument('--uncertainty_blind_lambda', type=float, default=0.3,
                        help='盲区不确定性分支损失系数')
    parser.add_argument('--data_root', type=str, default="./data")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--attack', type=str, default='mr-cospgd',
                        choices=['clean', 'bim', 'pgd', 'segpgd', 'cospgd', 'mr-cospgd'])
    parser.add_argument('--epsilon', type=float, default=8 / 255)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--steps', type=str, default='1,5,10,20,40')
    parser.add_argument('--eval_steps', type=str, default=None,
                        help='在单次最长迭代攻击过程中记录中间步结果，例如 3,5,10。若不设置，则保持原来的独立多步运行方式')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_model(device, model_arch='deeplabv3'):
    print("=" * 60)
    if model_arch == 'segformer':
        print("📦 加载模型：SegFormer-B0 (Cityscapes 19类权重)")
        base_model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        ).to(device).eval()

        class SegFormerWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m

            def forward(self, x):
                outputs = self.model(pixel_values=x, return_dict=True)
                return {
                    'out': torch.nn.functional.interpolate(
                        outputs.logits, size=x.shape[-2:], mode="bilinear", align_corners=False
                    )
                }

        model = SegFormerWrapper(base_model)
    elif model_arch == 'fcn':
        print("📦 加载模型：FCN + ResNet50 (VOC 21类)")
        model = segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).to(device).eval()
    else:
        print("📦 加载模型：DeepLabV3 + ResNet50 (VOC 21类)")
        model = segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device).eval()

    print(f"✅ 模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print("=" * 60)
    return model


def build_high_margin_mask(outputs, labels, top_ratio=0.2, ignore_index=255):
    probs = torch.softmax(outputs, dim=1)
    valid_mask = (labels != ignore_index)
    pred = outputs.argmax(dim=1)
    correct_mask = (pred == labels) & valid_mask

    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0

    p_true = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)

    tmp_probs = probs.clone()
    tmp_probs.scatter_(1, safe_labels.unsqueeze(1), -1e9)
    p_other_max = tmp_probs.max(dim=1)[0]

    margin = p_true - p_other_max
    margin[~correct_mask] = -1e9

    B, H, W = margin.shape
    high_margin_mask = torch.zeros_like(margin, dtype=torch.float)

    for b in range(B):
        correct_margin = margin[b][correct_mask[b]]
        if correct_margin.numel() == 0:
            continue

        k = max(1, int(top_ratio * correct_margin.numel()))
        threshold = torch.topk(correct_margin, k=k).values.min()
        high_margin_mask[b] = ((margin[b] >= threshold) & correct_mask[b]).float()

    return high_margin_mask


def build_uncertainty_blind_mask(outputs, labels, top_ratio=0.2, ignore_index=255):
    high_margin_mask = build_high_margin_mask(
        outputs=outputs,
        labels=labels,
        top_ratio=top_ratio,
        ignore_index=ignore_index
    )
    pred = outputs.argmax(dim=1)
    valid_mask = (labels != ignore_index)
    correct_mask = ((pred == labels) & valid_mask).float()
    blind_mask = correct_mask * (1.0 - high_margin_mask)
    return blind_mask


def compute_region_energy_ratio(delta, region_mask):
    delta_energy = delta.abs().mean(dim=1)
    region_energy = (delta_energy * region_mask).sum(dim=(1, 2))
    total_energy = delta_energy.sum(dim=(1, 2)) + 1e-12
    return region_energy / total_energy


def evaluate_single_attack(model, loader, device, args, attack_name, iterations):
    metric = ConfusionMatrix(num_classes=args.num_classes)
    attack_logs = []

    linf_values = []
    margin_region_energy_values = []
    blind_region_energy_values = []

    print(f"\n>>> [启动] 算法：{attack_name} | 迭代：{iterations} | Eps: {args.epsilon:.4f}")

    for i, (images, labels) in enumerate(tqdm(loader, desc=f"{attack_name}@{iterations}it")):
        images, labels = images.to(device), labels.to(device)

        if attack_name == 'Clean':
            with torch.no_grad():
                outputs = model(normalize_image(images))['out']
                preds = torch.argmax(outputs, dim=1)
        else:
            adv_images, batch_attack_logs = get_attack(
                model=model,
                images=images,
                labels=labels,
                device=device,
                method_name=attack_name,
                epsilon=args.epsilon,
                alpha=args.alpha,
                iterations=iterations,
                num_classes=args.num_classes,
                beta=args.beta,
                norm_type=args.norm_type,
                model_arch=args.model_arch,
                use_margin_weight=args.effective_use_margin_weight,
                use_uncertainty_blind_weight=args.effective_use_uncertainty_blind_weight,
                high_margin_ratio=args.high_margin_ratio,
                uncertainty_gamma=args.uncertainty_gamma,
                uncertainty_max_weight=args.uncertainty_max_weight,
                uncertainty_blind_lambda=args.uncertainty_blind_lambda
            )

            delta = adv_images - images
            linf_batch = delta.abs().view(delta.shape[0], -1).max(dim=1)[0]

            linf_values.extend(linf_batch.detach().cpu().tolist())

            for row in batch_attack_logs:
                row['batch_idx'] = i
            attack_logs.extend(batch_attack_logs)

            with torch.no_grad():
                outputs = model(normalize_image(adv_images))['out']
                preds = torch.argmax(outputs, dim=1)

                high_margin_mask = build_high_margin_mask(
                    outputs=outputs,
                    labels=labels,
                    top_ratio=args.high_margin_ratio,
                    ignore_index=255
                )

                blind_mask = build_uncertainty_blind_mask(
                    outputs=outputs,
                    labels=labels,
                    top_ratio=args.high_margin_ratio,
                    ignore_index=255
                )

                margin_region_energy = compute_region_energy_ratio(delta, high_margin_mask)
                blind_region_energy = compute_region_energy_ratio(delta, blind_mask)

                margin_region_energy_values.extend(margin_region_energy.detach().cpu().tolist())
                blind_region_energy_values.extend(blind_region_energy.detach().cpu().tolist())

        metric.update(labels.cpu().numpy(), preds.cpu().numpy())
        if args.debug and i >= 19:
            break

    results = metric.get_results()
    linf_mean = float(np.mean(linf_values)) if len(linf_values) > 0 else 0.0
    margin_region_energy_mean = float(np.mean(margin_region_energy_values)) if len(
        margin_region_energy_values) > 0 else 0.0
    blind_region_energy_mean = float(np.mean(blind_region_energy_values)) if len(
        blind_region_energy_values) > 0 else 0.0

    print(
        f"✅ [{attack_name}@{iterations}it] "
        f"mIoU: {results['Mean IoU'] * 100:.2f}% | "
        f"mAcc: {results['Mean Acc'] * 100:.2f}% | "
        f"Linf: {linf_mean:.6f}"
    )

    return {
        'mIoU': results['Mean IoU'],
        'mAcc': results['Mean Acc'],
        'linf': linf_mean,
        'margin_region_energy': margin_region_energy_mean,
        'blind_region_energy': blind_region_energy_mean,
        'attack_logs': attack_logs
    }


def evaluate_iterations(model, loader, device, args, attack_name):
    iterations = [int(x) for x in args.steps.split(',')]
    results, debug_results = [], []

    for T in iterations:
        score = evaluate_single_attack(model, loader, device, args, attack_name, T)
        results.append({
            'attack': attack_name,
            'iterations': T,
            'mIoU': score['mIoU'] * 100,
            'mAcc': score['mAcc'] * 100,
            'linf': score['linf'],
            'margin_region_energy': score['margin_region_energy'],
            'blind_region_energy': score['blind_region_energy'],
            'epsilon': args.epsilon
        })

        if attack_name in ['CosPGD', 'MR-CoSPGD']:
            for row in score['attack_logs']:
                row['attack'], row['iterations'] = attack_name, T
                debug_results.append(row)

    return results, debug_results



def evaluate_checkpoint_iterations(model, loader, device, args, attack_name):
    """
    Run one attack trajectory up to max(args.steps), and evaluate checkpoints
    specified by args.eval_steps. This avoids independently re-running attacks
    for 3/5/10/etc. Use only when args.eval_steps is not None.
    """
    max_iterations = max(int(x) for x in args.steps.split(','))
    eval_steps = sorted(set(int(x) for x in args.eval_steps.split(',')))

    if any(t <= 0 for t in eval_steps):
        raise ValueError("--eval_steps 中的迭代次数必须为正整数")
    if max(eval_steps) > max_iterations:
        raise ValueError(
            f"--eval_steps 的最大值 {max(eval_steps)} 不能大于 --steps 中的最大迭代次数 {max_iterations}"
        )

    metrics = {t: ConfusionMatrix(num_classes=args.num_classes) for t in eval_steps}
    linf_values = {t: [] for t in eval_steps}
    margin_region_energy_values = {t: [] for t in eval_steps}
    blind_region_energy_values = {t: [] for t in eval_steps}
    debug_results = []

    print(
        f"\n>>> [启动] 算法：{attack_name} | 最长迭代：{max_iterations} | "
        f"记录步数：{eval_steps} | Eps: {args.epsilon:.4f}"
    )

    for i, (images, labels) in enumerate(tqdm(loader, desc=f"{attack_name}@checkpoint-{max_iterations}it")):
        images, labels = images.to(device), labels.to(device)

        adv_images, batch_attack_logs, adv_checkpoints = get_attack(
            model=model,
            images=images,
            labels=labels,
            device=device,
            method_name=attack_name,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iterations=max_iterations,
            num_classes=args.num_classes,
            beta=args.beta,
            norm_type=args.norm_type,
            model_arch=args.model_arch,
            use_margin_weight=args.effective_use_margin_weight,
            use_uncertainty_blind_weight=args.effective_use_uncertainty_blind_weight,
            high_margin_ratio=args.high_margin_ratio,
            uncertainty_gamma=args.uncertainty_gamma,
            uncertainty_max_weight=args.uncertainty_max_weight,
            uncertainty_blind_lambda=args.uncertainty_blind_lambda,
            save_steps=eval_steps
        )

        for row in batch_attack_logs:
            row['batch_idx'] = i
            row['attack'] = attack_name
            row['iterations'] = max_iterations
        debug_results.extend(batch_attack_logs)

        with torch.no_grad():
            for t in eval_steps:
                if t not in adv_checkpoints:
                    raise RuntimeError(f"未找到第 {t} 轮的 checkpoint，请检查 attacks.py 中 save_steps 逻辑")

                adv_t = adv_checkpoints[t]
                outputs = model(normalize_image(adv_t))['out']
                preds = torch.argmax(outputs, dim=1)

                metrics[t].update(labels.cpu().numpy(), preds.cpu().numpy())

                delta = adv_t - images
                linf_batch = delta.abs().view(delta.shape[0], -1).max(dim=1)[0]
                linf_values[t].extend(linf_batch.detach().cpu().tolist())

                high_margin_mask = build_high_margin_mask(
                    outputs=outputs,
                    labels=labels,
                    top_ratio=args.high_margin_ratio,
                    ignore_index=255
                )
                blind_mask = build_uncertainty_blind_mask(
                    outputs=outputs,
                    labels=labels,
                    top_ratio=args.high_margin_ratio,
                    ignore_index=255
                )

                margin_region_energy = compute_region_energy_ratio(delta, high_margin_mask)
                blind_region_energy = compute_region_energy_ratio(delta, blind_mask)

                margin_region_energy_values[t].extend(margin_region_energy.detach().cpu().tolist())
                blind_region_energy_values[t].extend(blind_region_energy.detach().cpu().tolist())

        if args.debug and i >= 19:
            break

    results = []
    for t in eval_steps:
        score = metrics[t].get_results()
        linf_mean = float(np.mean(linf_values[t])) if len(linf_values[t]) > 0 else 0.0
        margin_energy_mean = float(np.mean(margin_region_energy_values[t])) if len(margin_region_energy_values[t]) > 0 else 0.0
        blind_energy_mean = float(np.mean(blind_region_energy_values[t])) if len(blind_region_energy_values[t]) > 0 else 0.0

        print(
            f"✅ [{attack_name}@{t}it checkpoint] "
            f"mIoU: {score['Mean IoU'] * 100:.2f}% | "
            f"mAcc: {score['Mean Acc'] * 100:.2f}% | "
            f"Linf: {linf_mean:.6f}"
        )

        results.append({
            'attack': attack_name,
            'iterations': t,
            'mIoU': score['Mean IoU'] * 100,
            'mAcc': score['Mean Acc'] * 100,
            'linf': linf_mean,
            'margin_region_energy': margin_energy_mean,
            'blind_region_energy': blind_energy_mean,
            'epsilon': args.epsilon,
            'eval_mode': f'checkpoint@{max_iterations}'
        })

    return results, debug_results


def print_synergy_terminal_summary(debug_df):
    """Print compact GM-BRC synergy statistics at the end of the run."""
    synergy_cols = [
        'correct_ratio_before_gm',
        'correct_ratio_after_gm',
        'residual_blind_ratio'
    ]

    if debug_df is None or len(debug_df) == 0:
        return
    if not all(col in debug_df.columns for col in synergy_cols):
        return

    valid_df = debug_df.dropna(subset=synergy_cols).copy()
    if len(valid_df) == 0:
        return

    group_cols = [col for col in ['attack', 'iterations', 'iter'] if col in valid_df.columns]
    summary = valid_df.groupby(group_cols, as_index=False)[synergy_cols].mean()

    display_df = summary.copy()
    for col in synergy_cols:
        display_df[col] = display_df[col] * 100.0

    print("\n📌 GM-BRC 协同指标汇总（单位：%，按有效像素统计）:")
    print(display_df.to_string(index=False, formatters={
        'correct_ratio_before_gm': lambda x: f'{x:.2f}',
        'correct_ratio_after_gm': lambda x: f'{x:.2f}',
        'residual_blind_ratio': lambda x: f'{x:.2f}',
    }))

    if 'iter' in summary.columns:
        base_group_cols = [c for c in group_cols if c != 'iter']
        if len(base_group_cols) > 0:
            idx = summary.groupby(base_group_cols)['iter'].idxmax()
            final_df = summary.loc[idx].copy()
        else:
            final_df = summary.loc[[summary['iter'].idxmax()]].copy()
        for col in synergy_cols:
            final_df[col] = final_df[col] * 100.0
        print("\n✅ 最终轮次协同指标:")
        print(final_df.to_string(index=False, formatters={
            'correct_ratio_before_gm': lambda x: f'{x:.2f}',
            'correct_ratio_after_gm': lambda x: f'{x:.2f}',
            'residual_blind_ratio': lambda x: f'{x:.2f}',
        }))
def main():
    args = get_args()
    # Final paper setting: Blind always uses residual mode.
    # The old pure mode has been removed from the CLI and attack call path.
    args.blind_region_mode = 'residual'
    set_seed(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    if args.no_weights:
        args.effective_use_margin_weight = False
        args.effective_use_uncertainty_blind_weight = False
    else:
        args.effective_use_margin_weight = args.use_margin_weight
        args.effective_use_uncertainty_blind_weight = args.use_uncertainty_blind_weight

    if args.model_arch == 'segformer':
        if args.dataset is None:
            args.dataset = 'cityscapes'
        args.num_classes = 19
        if args.data_root == "./data":
            args.data_root = "./data/cityscapes"
    else:
        if args.dataset is None:
            args.dataset = 'voc'
        args.num_classes = 21

    logger = ExperimentLogger(args)
    print("=" * 60)
    print(f"对抗攻击实验 - {args.model_arch.upper()} on {args.dataset.upper()}")
    print(f"📁 保存目录：{logger.exp_dir}")
    print(
        f"⚙️ Margin: {'ON' if args.effective_use_margin_weight else 'OFF'} | "
        f"UncertaintyBlind: {'ON' if args.effective_use_uncertainty_blind_weight else 'OFF'} | "
        f"BlindMode: {args.blind_region_mode}"
    )

    model = load_model(device, args.model_arch)

    try:
        val_loader = get_dataloader(
            root=args.data_root,
            batch_size=args.batch_size,
            split='val',
            dataset_type=args.dataset
        )
        print(f"✅ 数据加载成功！共 {len(val_loader)} 个批次")
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        return

    name_map = {
        'clean': 'Clean',
        'bim': 'BIM',
        'pgd': 'PGD',
        'segpgd': 'SegPGD',
        'cospgd': 'CosPGD',
        'mr-cospgd': 'MR-CoSPGD'
    }

    target_attacks = [name_map[args.attack]]

    all_results, all_debug_results = [], []
    for attack_name in target_attacks:
        if attack_name == 'Clean':
            score = evaluate_single_attack(model, val_loader, device, args, 'Clean', 0)
            all_results.append({
                'attack': 'Clean',
                'iterations': 0,
                'mIoU': score['mIoU'] * 100,
                'mAcc': score['mAcc'] * 100,
                'linf': score['linf'],
                'margin_region_energy': score['margin_region_energy'],
                'blind_region_energy': score['blind_region_energy'],
                'epsilon': args.epsilon
            })
        else:
            if args.eval_steps is not None:
                res, dbg = evaluate_checkpoint_iterations(model, val_loader, device, args, attack_name)
            else:
                res, dbg = evaluate_iterations(model, val_loader, device, args, attack_name)
            all_results.extend(res)
            all_debug_results.extend(dbg)

    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(logger.exp_dir, 'results.csv')
    results_df.to_csv(results_csv, index=False)

    if len(all_debug_results) > 0:
        debug_df = pd.DataFrame(all_debug_results)
        debug_csv = os.path.join(logger.exp_dir, 'attack_logs.csv')
        debug_df.to_csv(debug_csv, index=False)
        print(f"💾 逐步攻击日志已保存：{debug_csv}")
        print_synergy_terminal_summary(debug_df)

    print(f"\n📊 最终结果汇总:\n{results_df.to_string(index=False)}")


if __name__ == '__main__':
    main()
