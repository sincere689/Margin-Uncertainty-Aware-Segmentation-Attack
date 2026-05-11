import os
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def build_segformer_model(device: torch.device):
    # 修正：确保传入了正确的预训练模型名称
    model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def get_logits(image: Image.Image, processor, model, device: torch.device):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits

    h, w = image.size[1], image.size[0]
    logits = F.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )
    return logits


def compute_uncertainty_blind_map(logits: torch.Tensor,
                                  entropy_power: float = 1.5,
                                  edge_boost: float = 0.6):
    probs = F.softmax(logits, dim=1).squeeze(0)

    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
    entropy = entropy.pow(entropy_power)

    top1_prob, _ = torch.max(probs, dim=0)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=top1_prob.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=top1_prob.device
    ).view(1, 1, 3, 3)

    top1_prob_4d = top1_prob.unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(top1_prob_4d, sobel_x, padding=1)
    grad_y = F.conv2d(top1_prob_4d, sobel_y, padding=1)
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()

    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

    blind_map = entropy + edge_boost * edge
    blind_map = (blind_map - blind_map.min()) / (blind_map.max() - blind_map.min() + 1e-8)

    return blind_map.cpu().numpy()


def compute_margin_map(logits: torch.Tensor,
                       nms_kernel: int = 91,
                       top_k_points: int = 50,
                       soft_blur_sigma: int = 35,
                       min_margin: float = 0.70):
    """
    寻找“中间态”的 Margin Map：
    1. 动态选点：通过 NMS 平衡稀疏度。
    2. 局部显著性：保证全图有 50 个左右的局部最稳定点，信息量适中。
    3. 斑块渲染：柔和的高斯衰减效果，提升论文插图质感。
    """
    probs = F.softmax(logits, dim=1).squeeze(0)
    sorted_probs, _ = torch.sort(probs, dim=0, descending=True)
    margin = sorted_probs[0] - sorted_probs[1]

    # 注入微小扰动打破平坦区数值相等的情况
    noise = torch.randn_like(margin) * 1e-6
    margin_perturbed = margin + noise

    # NMS 提取局部峰值
    x = margin_perturbed.unsqueeze(0).unsqueeze(0)
    local_max = F.max_pool2d(
        x, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2
    ).squeeze(0).squeeze(0)

    # 峰值筛选
    peak_mask = (margin_perturbed == local_max) & (margin > min_margin)

    # 全局 Top-K 再次筛选，确保点数稳定
    candidate_values = margin[peak_mask]
    if candidate_values.numel() > top_k_points:
        top_k_val = torch.topk(candidate_values.flatten(), top_k_points).values[-1]
        peak_mask = peak_mask & (margin >= top_k_val)

    # 构造点阵
    sparse_points = torch.zeros_like(margin)
    sparse_points[peak_mask] = margin[peak_mask]

    # 渲染斑块
    px = sparse_points.unsqueeze(0).unsqueeze(0)
    blob = F.max_pool2d(px, kernel_size=7, stride=1, padding=3)

    # 通过多次平滑产生自然的斑块淡出效果
    for _ in range(2):
        blob = F.avg_pool2d(blob, kernel_size=soft_blur_sigma, stride=1, padding=soft_blur_sigma // 2)

    final_map = blob.squeeze()

    if final_map.max() > 0:
        final_map = final_map / (final_map.max() + 1e-8)

    return final_map.cpu().numpy()


def build_light_red_cmap():
    # 论文配色：深红到淡红
    colors = ["#fffafa", "#ffcccc", "#ee5555", "#990000"]
    return LinearSegmentedColormap.from_list("paper_red", colors)


def save_margin_overlay(image: Image.Image,
                        arr: np.ndarray,
                        save_path: str,
                        cmap,
                        title: str,
                        alpha: float = 0.82,
                        bg_darkness: float = 0.82,
                        dpi: int = 300):
    image_np = np.array(image).astype(np.float32) / 255.0
    dimmed = np.clip(image_np * bg_darkness + 0.08, 0.0, 1.0)

    heat = cmap(arr)[..., :3]

    alpha_map = (arr ** 1.0) * alpha
    alpha_map = np.clip(alpha_map, 0.0, 1.0)[..., None]

    overlay = dimmed * (1.0 - alpha_map) + heat * alpha_map
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(overlay)
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./paper_weight_maps")
    parser.add_argument("--device", type=str, default="cuda")

    # 中间态默认参数
    parser.add_argument("--nms_kernel", type=int, default=91)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--blur_sigma", type=int, default=35)
    parser.add_argument("--margin_alpha", type=float, default=0.82)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ensure_dir(args.save_dir)

    image = load_image(args.image_path)
    # 建立模型
    processor, model = build_segformer_model(device)
    logits = get_logits(image, processor, model, device)

    # 计算权重图
    margin_map = compute_margin_map(
        logits,
        nms_kernel=args.nms_kernel,
        top_k_points=args.top_k,
        soft_blur_sigma=args.blur_sigma,
        min_margin=0.70
    )

    blind_map = compute_uncertainty_blind_map(logits)

    paper_red_cmap = build_light_red_cmap()
    blind_cmap = plt.get_cmap("viridis")

    # 保存 Margin 图
    save_margin_overlay(
        image=image,
        arr=margin_map,
        save_path=os.path.join(args.save_dir, "sparse_margin_based_focus_map_overlay.png"),
        cmap=paper_red_cmap,
        title="Sparse Margin Weight Map",
        alpha=args.margin_alpha
    )

    # 保存 Blind 图
    image_np = np.array(image)
    plt.figure(figsize=(10, 5))
    plt.imshow(image_np)
    plt.imshow(blind_map, cmap=blind_cmap, alpha=0.6, interpolation="bilinear")
    plt.title("Uncertainty-Driven Blind-Region Focus Map", fontsize=16)
    plt.axis("off")
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(args.save_dir, "uncertainty_driven_blind_region_focus_map_overlay.png"),
                bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()

    print(f"Done. Processed image saved to {args.save_dir}")


if __name__ == "__main__":
    main()