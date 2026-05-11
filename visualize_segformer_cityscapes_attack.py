import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from attacks import get_attack, normalize_image

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    raise ImportError("缺少 transformers，请先运行: pip install transformers timm")


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < 19)
    color[valid] = CITYSCAPES_PALETTE[mask[valid]]
    return color


def load_cityscapes_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    img_t = torch.from_numpy(img_np).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    return img_np, img_t


class SegFormerWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, x):
        outputs = self.model(pixel_values=x, return_dict=True)
        return {
            'out': F.interpolate(
                outputs.logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        }


def build_model(device):
    print("=" * 60)
    print("📦 加载模型：SegFormer-B0 (Cityscapes 19类权重)")
    base_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ).to(device).eval()
    model = SegFormerWrapper(base_model).to(device).eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"✅ 模型参数量：{num_params:.2f}M")
    print("=" * 60)
    return model


@torch.no_grad()
def forward_pred(model, x):
    outputs = model(normalize_image(x))
    logits = outputs['out'] if isinstance(outputs, dict) else outputs
    pred = logits.argmax(dim=1)
    return pred


def generate_adv(model, x, y_dummy, args, device):
    print("\n[DEBUG] framework attack config:")
    print("method_name = mr-cospgd")
    print("use_margin_weight = True")
    print("use_class_weight = False")
    print("use_uncertainty_blind_weight = True")
    print(f"high_margin_ratio = {args.high_margin_ratio}")
    print(f"uncertainty_gamma = {args.uncertainty_gamma}")
    print(f"uncertainty_max_weight = {args.uncertainty_max_weight}")
    print(f"uncertainty_blind_lambda = {args.uncertainty_blind_lambda}")

    adv_x, attack_logs = get_attack(
        model=model,
        images=x,
        labels=y_dummy,
        device=device,
        method_name='mr-cospgd',
        epsilon=args.epsilon,
        alpha=args.alpha,
        iterations=args.steps,
        num_classes=19,
        beta=args.beta,
        norm_type='inf',
        model_arch='segformer',
        use_margin_weight=True,
        use_class_weight=False,
        class_weight_mode='dynamic',
        use_uncertainty_blind_weight=True,
        high_margin_ratio=args.high_margin_ratio,
        uncertainty_gamma=args.uncertainty_gamma,
        uncertainty_max_weight=args.uncertainty_max_weight,
        uncertainty_blind_lambda=args.uncertainty_blind_lambda,
    )
    return adv_x.clamp(0.0, 1.0), attack_logs


def main():
    parser = argparse.ArgumentParser(description="论文框架图：原图/正常分割/对抗样本/错误分割")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="segformer_mr_cospgd_framework.png")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--high_margin_ratio", type=float, default=0.6)
    parser.add_argument("--uncertainty_gamma", type=float, default=0.5)
    parser.add_argument("--uncertainty_max_weight", type=float, default=1.5)
    parser.add_argument("--uncertainty_blind_lambda", type=float, default=0.05)
    parser.add_argument("--title_fontsize", type=int, default=20)
    parser.add_argument("--fig_w", type=float, default=18)
    parser.add_argument("--fig_h", type=float, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUNNING FILE: {os.path.abspath(__file__)}")
    print(f"DEVICE: {device}")

    raw_img_np, img_t = load_cityscapes_image(args.image_path)
    img_t = img_t.to(device)

    model = build_model(device)

    pred_clean = forward_pred(model, img_t)
    pred_clean_np = pred_clean.squeeze(0).cpu().numpy().astype(np.uint8)

    # 用 clean prediction 作为攻击所需标签，适合做框架图演示
    y_dummy = pred_clean.detach().clone().long().to(device)

    adv_t, attack_logs = generate_adv(model, img_t, y_dummy, args, device)
    pred_adv = forward_pred(model, adv_t)
    pred_adv_np = pred_adv.squeeze(0).cpu().numpy().astype(np.uint8)

    adv_np = adv_t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    adv_np = np.clip(adv_np * 255.0, 0, 255).astype(np.uint8)

    clean_color = colorize_mask(pred_clean_np)
    adv_color = colorize_mask(pred_adv_np)

    changed_ratio = float((pred_clean_np != pred_adv_np).mean())
    linf = float((adv_t - img_t).detach().abs().max().item())

    print("\n" + "=" * 60)
    print("框架图统计")
    print(f"image_path: {args.image_path}")
    print(f"steps: {args.steps}")
    print(f"epsilon: {args.epsilon:.6f}")
    print(f"alpha: {args.alpha:.6f}")
    print(f"L_inf(|adv-clean|): {linf:.6f}")
    print(f"Prediction changed ratio: {changed_ratio * 100:.2f}%")
    if len(attack_logs) > 0:
        print(f"Last iter final_loss: {attack_logs[-1].get('final_loss', 0.0):.6f}")
        print(f"Last iter main_loss: {attack_logs[-1].get('main_loss', 0.0):.6f}")
        print(f"Last iter blind_loss: {attack_logs[-1].get('blind_loss', 0.0):.6f}")
    print("=" * 60)

    plt.figure(figsize=(args.fig_w, args.fig_h))

    plt.subplot(1, 4, 1)
    plt.imshow(raw_img_np)
    plt.title("Original Image", fontsize=args.title_fontsize)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(clean_color)
    plt.title("Clean Prediction", fontsize=args.title_fontsize)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(adv_np)
    plt.title("Adversarial Image", fontsize=args.title_fontsize)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(adv_color)
    plt.title("Adversarial Prediction", fontsize=args.title_fontsize)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[✓] 框架图已保存到: {args.save_path}")


if __name__ == "__main__":
    main()
