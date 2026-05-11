# 后面实验图 找到凸显我算法的图

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloader
from attacks import get_attack, normalize_image
from metrics import ConfusionMatrix
from main import get_args, load_model, set_seed


TITLE_MAP = {
    'cospgd': 'CosPGD',
    'cospgd-mc': 'CosPGD-MC',
    'margin-cospgd': 'Margin-CosPGD',
}


def calc_single_miou(preds, labels, num_classes=21):
    """计算单张图片的 mIoU。"""
    metric = ConfusionMatrix(num_classes=num_classes)
    metric.update(labels.cpu().numpy(), preds.cpu().numpy())
    return metric.get_results()['Mean IoU']


def forward_out(model, images):
    outputs = model(normalize_image(images))
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    elif isinstance(outputs, dict):
        outputs = outputs.get('out', list(outputs.values())[0])
    return outputs


def plot_comparison(
    img,
    label,
    clean_pred,
    base_pred,
    weighted_pred,
    batch_idx,
    save_dir,
    base_name='CosPGD',
    weighted_name='CosPGD-MC'
):
    """画 1x5 对比图：原图 / GT / CleanPred / 基线攻击 / 两权重攻击。"""
    img_show = img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    label_show = label.detach().cpu().squeeze().numpy()
    clean_show = clean_pred.detach().cpu().squeeze().numpy()
    base_show = base_pred.detach().cpu().squeeze().numpy()
    weighted_show = weighted_pred.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Clean Image', 'Ground Truth', 'Clean Pred', base_name, weighted_name]
    images_to_show = [img_show, label_show, clean_show, base_show, weighted_show]

    vmax = 20
    for ax, title, show_img in zip(axes, titles, images_to_show):
        if title == 'Clean Image':
            ax.imshow(np.clip(show_img, 0, 1))
        else:
            ax.imshow(show_img, cmap='nipy_spectral', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'best_case_batch_{batch_idx}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"📸 成功保存对比图至: {save_path}")


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 与 main.py 保持一致的默认逻辑
    if getattr(args, 'no_weights', False):
        args.effective_use_margin_weight = False
        args.effective_use_class_weight = False
    else:
        if getattr(args, 'use_margin_weight', False) or getattr(args, 'use_class_weight', False):
            args.effective_use_margin_weight = getattr(args, 'use_margin_weight', False)
            args.effective_use_class_weight = getattr(args, 'use_class_weight', False)
        else:
            args.effective_use_margin_weight = True
            args.effective_use_class_weight = True

    if args.model_arch == 'segformer':
        if args.dataset is None:
            args.dataset = 'cityscapes'
        args.num_classes = 19
        if args.data_root == './data':
            args.data_root = './data/cityscapes'
    else:
        if args.dataset is None:
            args.dataset = 'voc'
        args.num_classes = 21

    save_dir = f"visualizations_{args.model_arch}_{'cospgd-mc'}"
    os.makedirs(save_dir, exist_ok=True)

    model = load_model(device, args.model_arch)
    loader = get_dataloader(
        root=args.data_root,
        batch_size=1,
        split='val',
        dataset_type=args.dataset
    )

    # 基线：原始 CosPGD
    base_attack = 'cospgd'
    # 你的方法：两个权重都加后的 CosPGD-MC
    weighted_attack = 'cospgd-mc'

    print(f"\n🔍 开始海选：寻找 {TITLE_MAP[weighted_attack]} 相比 {TITLE_MAP[base_attack]} 更强的样本...")
    results = []

    # 先在前100张里筛
    search_limit = 100
    iterations = 10

    for i, (images, labels) in enumerate(tqdm(loader, total=search_limit)):
        if i >= search_limit:
            break

        images, labels = images.to(device), labels.to(device)

        # 1. 干净图像预测
        with torch.no_grad():
            clean_out = forward_out(model, images)
            clean_pred = torch.argmax(clean_out, dim=1)

        # 2. 基线 CosPGD
        base_adv, _ = get_attack(
            model=model,
            images=images,
            labels=labels,
            device=device,
            method_name=base_attack,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iterations=iterations,
            num_classes=args.num_classes,
            beta=args.beta,
            eta_lambda=args.eta_lambda,
            rho=args.rho,
            lambda_max=args.lambda_max,
            norm_type=args.norm_type,
            debug=getattr(args, 'debug', False),
            debug_sp=getattr(args, 'debug_sp', False),
            use_margin_weight=False,
            use_class_weight=False,
        )
        with torch.no_grad():
            base_out = forward_out(model, base_adv)
            base_pred = torch.argmax(base_out, dim=1)
        base_miou = calc_single_miou(base_pred, labels, num_classes=args.num_classes)

        # 3. 两个权重后的 CosPGD-MC
        weighted_adv, _ = get_attack(
            model=model,
            images=images,
            labels=labels,
            device=device,
            method_name=weighted_attack,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iterations=iterations,
            num_classes=args.num_classes,
            beta=args.beta,
            eta_lambda=args.eta_lambda,
            rho=args.rho,
            lambda_max=args.lambda_max,
            norm_type=args.norm_type,
            debug=getattr(args, 'debug', False),
            debug_sp=getattr(args, 'debug_sp', False),
            use_margin_weight=True,
            use_class_weight=True,
        )
        with torch.no_grad():
            weighted_out = forward_out(model, weighted_adv)
            weighted_pred = torch.argmax(weighted_out, dim=1)
        weighted_miou = calc_single_miou(weighted_pred, labels, num_classes=args.num_classes)

        # diff 越大，说明你的方法比基线破坏更强
        diff = base_miou - weighted_miou
        results.append({
            'batch_idx': i,
            'diff': diff,
            'base_miou': base_miou,
            'weighted_miou': weighted_miou,
            'img': images.detach().cpu(),
            'label': labels.detach().cpu(),
            'clean_pred': clean_pred.detach().cpu(),
            'base_pred': base_pred.detach().cpu(),
            'weighted_pred': weighted_pred.detach().cpu(),
        })

    # 排序，找出最能凸显你算法优势的样本
    results.sort(key=lambda x: x['diff'], reverse=True)
    top_3 = results[:3]

    print("\n🏆 海选完成！正在生成 Top-3 对比图...")
    for rank, res in enumerate(top_3):
        print(
            f"Top {rank + 1} (Batch {res['batch_idx']}): "
            f"{TITLE_MAP[base_attack]} mIoU={res['base_miou']:.4f} | "
            f"{TITLE_MAP[weighted_attack]} mIoU={res['weighted_miou']:.4f} "
            f"(差值: {res['diff']:.4f})"
        )
        plot_comparison(
            img=res['img'],
            label=res['label'],
            clean_pred=res['clean_pred'],
            base_pred=res['base_pred'],
            weighted_pred=res['weighted_pred'],
            batch_idx=res['batch_idx'],
            save_dir=save_dir,
            base_name=TITLE_MAP[base_attack],
            weighted_name=TITLE_MAP[weighted_attack],
        )


if __name__ == '__main__':
    main()