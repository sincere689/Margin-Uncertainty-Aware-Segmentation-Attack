# 找到画sparse margin-based weight map和uncertainty-driven blind-region weight map框架图最合适的的筛选脚本文件
# 发现最好的是D:\GMM\cospgd-main\Adversarial_Segmentation\data\Cityscapes\leftImg8bit\val\frankfurt\frankfurt_000000_012868_leftImg8bit.png

import os
import glob
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def analyze_single_image(logits, margin_keep_ratio=0.08, blur_kernel_size=3):
    probs = F.softmax(logits, dim=1).squeeze(0)  # [C, H, W]

    sorted_probs, _ = torch.sort(probs, dim=0, descending=True)
    top1_prob = sorted_probs[0]
    top2_prob = sorted_probs[1]

    # -------- sparse margin-based weight map --------
    margin = top1_prob - top2_prob
    threshold = torch.quantile(margin, 1.0 - margin_keep_ratio)

    sparse_margin_map = torch.where(
        margin >= threshold,
        margin,
        torch.zeros_like(margin)
    )
    sparse_margin_map = sparse_margin_map / (sparse_margin_map.max() + 1e-8)

    margin_nonzero_ratio = (sparse_margin_map > 0).float().mean().item()
    margin_std = sparse_margin_map.std().item()

    # -------- uncertainty-driven blind-region weight map --------
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=0)
    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    if blur_kernel_size and blur_kernel_size > 1:
        pad = blur_kernel_size // 2
        entropy_4d = entropy.unsqueeze(0).unsqueeze(0)
        entropy = F.avg_pool2d(
            F.pad(entropy_4d, (pad, pad, pad, pad), mode='reflect'),
            kernel_size=blur_kernel_size,
            stride=1
        ).squeeze(0).squeeze(0)
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    uncertainty_map = entropy
    uncertainty_std = uncertainty_map.std().item()

    # -------- prediction complexity --------
    pred = probs.argmax(dim=0)
    unique_classes = torch.unique(pred).numel()

    horizontal_changes = (pred[:, 1:] != pred[:, :-1]).float().mean().item()
    vertical_changes = (pred[1:, :] != pred[:-1, :]).float().mean().item()
    boundary_complexity = 0.5 * (horizontal_changes + vertical_changes)

    diff_score = torch.mean(torch.abs(sparse_margin_map - uncertainty_map)).item()

    # -------- final score --------
    class_score = min(unique_classes / 10.0, 1.0)
    boundary_score = min(boundary_complexity * 8.0, 1.0)
    margin_score = min(margin_std * 8.0, 1.0)
    uncertainty_score = min(uncertainty_std * 6.0, 1.0)
    separation_score = min(diff_score * 5.0, 1.0)

    final_score = (
        0.20 * class_score +
        0.20 * boundary_score +
        0.20 * margin_score +
        0.15 * uncertainty_score +
        0.25 * separation_score
    )

    return {
        "margin_map": sparse_margin_map.detach().cpu().numpy(),
        "uncertainty_map": uncertainty_map.detach().cpu().numpy(),
        "score": final_score,
        "unique_classes": unique_classes,
        "boundary_complexity": boundary_complexity,
        "margin_nonzero_ratio": margin_nonzero_ratio,
        "margin_std": margin_std,
        "uncertainty_std": uncertainty_std,
        "diff_score": diff_score,
    }


def save_visualizations(image_pil, analysis, save_dir, stem):
    os.makedirs(save_dir, exist_ok=True)

    image_pil.save(os.path.join(save_dir, f"{stem}_input.png"))

    plt.figure(figsize=(8, 4))
    plt.imshow(analysis["margin_map"], cmap="hot", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        os.path.join(save_dir, f"{stem}_margin.png"),
        dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(analysis["uncertainty_map"], cmap="viridis", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        os.path.join(save_dir, f"{stem}_uncertainty.png"),
        dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.close()

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image_pil)
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(analysis["margin_map"], cmap="hot", interpolation="nearest")
    ax2.set_title("Sparse Margin Map")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(analysis["uncertainty_map"], cmap="viridis", interpolation="nearest")
    ax3.set_title("Blind Uncertainty Map")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{stem}_preview.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    # ===========================
    # 改这里：你的 Cityscapes 根目录
    # ===========================
    cityscapes_root = r"D:\GMM\cospgd-main\Adversarial_Segmentation\data\Cityscapes"
    output_dir = "selected_cityscapes_results"
    top_k = 10
    margin_keep_ratio = 0.08
    blur_kernel_size = 3
    # ===========================

    val_dir = os.path.join(cityscapes_root, "leftImg8bit", "val")
    image_paths = sorted(glob.glob(os.path.join(val_dir, "*", "*_leftImg8bit.png")))

    if len(image_paths) == 0:
        print(f"[ERROR] 没找到 Cityscapes 验证集图片，请检查路径：{val_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} Cityscapes val images.")

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = deeplabv3_resnet50(weights=weights).to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths):
            try:
                image_pil = Image.open(image_path).convert("RGB")
                input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
                logits = model(input_tensor)["out"]

                analysis = analyze_single_image(
                    logits,
                    margin_keep_ratio=margin_keep_ratio,
                    blur_kernel_size=blur_kernel_size
                )

                results.append({
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    **analysis
                })

                print(
                    f"[{idx+1}/{len(image_paths)}] "
                    f"{os.path.basename(image_path)} | "
                    f"score={analysis['score']:.4f} | "
                    f"classes={analysis['unique_classes']} | "
                    f"boundary={analysis['boundary_complexity']:.4f} | "
                    f"diff={analysis['diff_score']:.4f}"
                )

            except Exception as e:
                print(f"[SKIP] {image_path} -> {e}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    ranking_file = os.path.join(output_dir, "ranking.txt")
    with open(ranking_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(
                f"Rank {i+1}\n"
                f"Image: {r['image_name']}\n"
                f"Score: {r['score']:.6f}\n"
                f"Unique Classes: {r['unique_classes']}\n"
                f"Boundary Complexity: {r['boundary_complexity']:.6f}\n"
                f"Margin Nonzero Ratio: {r['margin_nonzero_ratio']:.6f}\n"
                f"Margin Std: {r['margin_std']:.6f}\n"
                f"Uncertainty Std: {r['uncertainty_std']:.6f}\n"
                f"Map Difference: {r['diff_score']:.6f}\n"
                f"Path: {r['image_path']}\n"
                f"{'-'*60}\n"
            )

    top_dir = os.path.join(output_dir, f"top_{top_k}")
    os.makedirs(top_dir, exist_ok=True)

    for i, r in enumerate(results[:top_k]):
        rank_dir = os.path.join(top_dir, f"rank_{i+1:02d}")
        os.makedirs(rank_dir, exist_ok=True)

        image_pil = Image.open(r["image_path"]).convert("RGB")
        stem = Path(r["image_name"]).stem

        save_visualizations(image_pil, r, rank_dir, stem)

        shutil.copy2(r["image_path"], os.path.join(rank_dir, r["image_name"]))

        with open(os.path.join(rank_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"Rank: {i+1}\n"
                f"Image: {r['image_name']}\n"
                f"Score: {r['score']:.6f}\n"
                f"Unique Classes: {r['unique_classes']}\n"
                f"Boundary Complexity: {r['boundary_complexity']:.6f}\n"
                f"Margin Nonzero Ratio: {r['margin_nonzero_ratio']:.6f}\n"
                f"Margin Std: {r['margin_std']:.6f}\n"
                f"Uncertainty Std: {r['uncertainty_std']:.6f}\n"
                f"Map Difference: {r['diff_score']:.6f}\n"
                f"Original Path: {r['image_path']}\n"
            )

    print("\nDone.")
    print(f"Ranking file: {ranking_file}")
    print(f"Top-{top_k} results saved to: {top_dir}")


if __name__ == "__main__":
    main()