import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


def generate_clean_academic_zoom(
    img_name,
    save_name,
    x,
    y,
    size=80,
    zoom=3.5,
    vis_dir='visualizations_deeplabv3_cospgd-mc',
    cols=(0, 3, 4),
):
    """
    对 1x5 对比图做同步局部放大。
    默认放大第 0/3/4 列：Clean / CosPGD / CosPGD-MC。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, vis_dir, img_name)
    save_path = os.path.join(current_dir, save_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 错误：找不到图片，请检查路径！\n尝试路径: {img_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    col_w = w // 5

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.imshow(img)
    ax.axis('off')

    for col_idx in cols:
        real_x = col_idx * col_w + x

        axins = zoomed_inset_axes(
            ax,
            zoom,
            loc='lower left',
            bbox_to_anchor=(col_idx / 5 + 0.01, 1.05, 0.18, 0.18),
            bbox_transform=ax.transAxes
        )

        x1, x2 = real_x - size // 2, real_x + size // 2
        y1, y2 = y - size // 2, y + size // 2

        patch = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        axins.imshow(patch)
        axins.set_xticks([])
        axins.set_yticks([])

        for spine in axins.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2.5)

        rect = patches.Rectangle(
            (x1, y1),
            size,
            size,
            linewidth=2.5,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 成功生成放大对比图: {save_path}")
    plt.show()


if __name__ == '__main__':
    # 这里改成你实际筛出来的那张图名字
    generate_clean_academic_zoom(
        img_name='best_case_batch_6.png',
        save_name='Figure_Zoom_MC.png',
        x=180,
        y=460,
        size=100,
        zoom=3.8,
        vis_dir='visualizations_deeplabv3_cospgd-mc',
        cols=(0, 3, 4),
    )