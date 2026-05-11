import os
import re
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def parse_info_txt(info_path):
    """
    从 info.txt 里解析 clean / cospgd / ours 的 mIoU
    兼容常见写法，例如：
    clean_mIoU: 72.31
    cospgd_mIoU: 15.28
    ours_mIoU: 7.54
    delta: 7.74
    """
    metrics = {
        "clean_miou": None,
        "cospgd_miou": None,
        "ours_miou": None,
        "delta": None,
    }

    if not os.path.exists(info_path):
        return metrics

    with open(info_path, "r", encoding="utf-8") as f:
        text = f.read()

    patterns = {
        "clean_miou": [
            r"clean[_\s-]*mIoU\s*[:=]\s*([0-9.]+)",
            r"clean[_\s-]*miou\s*[:=]\s*([0-9.]+)",
        ],
        "cospgd_miou": [
            r"cospgd[_\s-]*mIoU\s*[:=]\s*([0-9.]+)",
            r"cospgd[_\s-]*miou\s*[:=]\s*([0-9.]+)",
        ],
        "ours_miou": [
            r"ours[_\s-]*mIoU\s*[:=]\s*([0-9.]+)",
            r"ours[_\s-]*miou\s*[:=]\s*([0-9.]+)",
        ],
        "delta": [
            r"delta\s*[:=]\s*([0-9.]+)",
        ],
    }

    for key, plist in patterns.items():
        for p in plist:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                metrics[key] = float(m.group(1))
                break

    return metrics


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到图片: {path}")
    return Image.open(path).convert("RGB")


def build_figure(sample_dirs, output_path, dpi=300):
    """
    每个 sample_dir 内应包含：
      clean.png
      clean_pred.png
      cospgd_pred.png
      ours_pred.png
      info.txt
    """
    n_rows = len(sample_dirs)
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.8 * n_rows))

    if n_rows == 1:
        axes = [axes]

    column_titles = ["Clean Image", "Clean Prediction", "CoSPGD", "Ours"]

    for row_idx, sample_dir in enumerate(sample_dirs):
        info_path = os.path.join(sample_dir, "info.txt")
        metrics = parse_info_txt(info_path)

        clean_img = load_image(os.path.join(sample_dir, "clean.png"))
        clean_pred = load_image(os.path.join(sample_dir, "clean_pred.png"))
        cospgd_pred = load_image(os.path.join(sample_dir, "cospgd_pred.png"))
        ours_pred = load_image(os.path.join(sample_dir, "ours_pred.png"))

        images = [clean_img, clean_pred, cospgd_pred, ours_pred]

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.imshow(images[col_idx])
            ax.axis("off")

            # 第一行显示列标题
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=16, pad=12)

        # 第二列开始以下面小标题方式标 mIoU 更稳
        if metrics["clean_miou"] is not None:
            axes[row_idx][1].text(
                0.5, -0.10,
                f"mIoU: {metrics['clean_miou']:.2f}%",
                transform=axes[row_idx][1].transAxes,
                ha="center", va="top", fontsize=12
            )

        if metrics["cospgd_miou"] is not None:
            axes[row_idx][2].text(
                0.5, -0.10,
                f"mIoU: {metrics['cospgd_miou']:.2f}%",
                transform=axes[row_idx][2].transAxes,
                ha="center", va="top", fontsize=12
            )

        if metrics["ours_miou"] is not None:
            axes[row_idx][3].text(
                0.5, -0.10,
                f"mIoU: {metrics['ours_miou']:.2f}%",
                transform=axes[row_idx][3].transAxes,
                ha="center", va="top", fontsize=12
            )

        # 左侧行标记
        sample_name = os.path.basename(sample_dir)
        axes[row_idx][0].text(
            -0.06, 0.5,
            f"Sample {row_idx + 1}",
            transform=axes[row_idx][0].transAxes,
            ha="right", va="center",
            fontsize=14, rotation=90
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.35)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ 已保存最终论文可视化图：{output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="ranked_top_candidates 的根目录"
    )
    parser.add_argument(
        "--samples",
        type=str,
        nargs="+",
        required=True,
        help="选择的样本文件夹名，例如 rank_012_sample_00024_delta_7.74 rank_013_sample_00096_delta_7.41"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/final_qualitative_segformer_cityscapes.png",
        help="输出图路径"
    )
    args = parser.parse_args()

    sample_dirs = [os.path.join(args.root, s) for s in args.samples]
    build_figure(sample_dirs, args.output)


if __name__ == "__main__":
    main()