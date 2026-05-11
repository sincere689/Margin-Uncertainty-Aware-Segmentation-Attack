import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

# ==========================================================
# Cityscapes 19 类官方映射表
# ==========================================================
CITY_ID_TO_TRAIN_ID = np.array([
    255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255,
    255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255,
    255, 16, 17, 18
])

# ==========================================================
# Cityscapes 统一输入尺寸
# ==========================================================
CITYSCAPES_TARGET_SIZE = (512, 1024)


def target_transform_voc(mask):
    mask_np = np.array(mask, dtype=np.int64)
    return torch.tensor(mask_np, dtype=torch.long).clone()


def target_transform_city(mask):
    # 标签必须使用最近邻插值，避免产生错误类别 ID
    mask = transforms.functional.resize(
        mask,
        CITYSCAPES_TARGET_SIZE,
        interpolation=InterpolationMode.NEAREST
    )
    mask_np = np.array(mask)

    # 执行 34 -> 19 类映射
    target = CITY_ID_TO_TRAIN_ID[mask_np]
    return torch.tensor(target, dtype=torch.long).clone()


def segmentation_collate_fn(batch):
    """
    兼容 VOC 原图尺寸不一致 + 多进程 DataLoader。
    - image: padding 到当前 batch 内最大 H/W，padding 值为 0
    - label: padding 到当前 batch 内最大 H/W，padding 值为 255(ignore_index)
    - clone(): 避免 non-resizable storage 导致 default_collate 报错
    """
    images, labels = zip(*batch)

    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []
    padded_labels = []

    for img, lab in zip(images, labels):
        img = img.clone().contiguous()
        lab = lab.clone().contiguous()

        h, w = img.shape[-2], img.shape[-1]
        pad_h = max_h - h
        pad_w = max_w - w

        # F.pad 参数顺序: (left, right, top, bottom)
        img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        lab = F.pad(lab, (0, pad_w, 0, pad_h), value=255)

        padded_images.append(img)
        padded_labels.append(lab)

    return torch.stack(padded_images, dim=0), torch.stack(padded_labels, dim=0)


def get_dataloader(root, batch_size=4, split='val', dataset_type='voc'):
    if not os.path.exists(root):
        raise FileNotFoundError(f"❌ 路径 '{root}' 不存在，请检查。")

    print(f"🚀 正在加载 {dataset_type.upper()} 数据集: {root} ...")

    # ----------------------------------------------------------
    # 模式 A: 加载 Pascal VOC
    # ----------------------------------------------------------
    if dataset_type.lower() == 'voc':
        img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = datasets.VOCSegmentation(
            root=root,
            year='2012',
            image_set=split,
            download=False,
            transform=img_transform,
            target_transform=target_transform_voc
        )

    # ----------------------------------------------------------
    # 模式 B: 加载 Cityscapes
    # ----------------------------------------------------------
    elif dataset_type.lower() == 'cityscapes':
        img_transform = transforms.Compose([
            transforms.Resize(
                CITYSCAPES_TARGET_SIZE,
                interpolation=InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
        ])

        dataset = datasets.Cityscapes(
            root=root,
            split=split,
            mode='fine',
            target_type='semantic',
            transform=img_transform,
            target_transform=target_transform_city
        )

    else:
        raise ValueError("❌ 不支持的数据集类型！可选: 'voc', 'cityscapes'")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=segmentation_collate_fn
    )

    return loader
