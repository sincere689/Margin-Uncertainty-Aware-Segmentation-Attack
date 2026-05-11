import os
import csv
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset import get_dataloader
from attacks import get_attack, normalize_image
from main import load_model, set_seed


CITYSCAPES_PALETTE = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

VOC_PALETTE = np.array([
    [  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0], [  0,   0, 128],
    [128,   0, 128], [  0, 128, 128], [128, 128, 128], [ 64,   0,   0], [192,   0,   0],
    [ 64, 128,   0], [192, 128,   0], [ 64,   0, 128], [192,   0, 128], [ 64, 128, 128],
    [192, 128, 128], [  0,  64,   0], [128,  64,   0], [  0, 192,   0], [128, 192,   0],
    [  0,  64, 128]
], dtype=np.uint8)


def get_args():
    parser = argparse.ArgumentParser(
        description="Select qualitative adversarial attack visualization samples."
    )
    parser.add_argument('--model_arch', type=str, default='segformer', choices=['deeplabv3', 'fcn', 'segformer'])
    parser.add_argument('--dataset', type=str, default=None, choices=['voc', 'cityscapes'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='qualitative_attack_candidates')

    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=8 / 255)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--norm_type', type=str, default='inf', choices=['inf', 'l2', 'two'])

    parser.add_argument('--high_margin_ratio', type=float, default=0.3)
    parser.add_argument('--uncertainty_gamma', type=float, default=0.5)
    parser.add_argument('--uncertainty_max_weight', type=float, default=1.5)
    parser.add_argument('--uncertainty_blind_lambda', type=float, default=0.05)

    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--min_clean_miou', type=float, default=20.0,
                        help='Minimum clean single-image mIoU (%) for candidate selection.')
    parser.add_argument('--min_cospgd_miou', type=float, default=5.0,
                        help='Minimum CoSPGD single-image mIoU (%) to avoid fully collapsed baseline samples.')
    parser.add_argument('--min_delta', type=float, default=2.0,
                        help='Minimum mIoU gap: CoSPGD - Ours (%).')
    parser.add_argument('--save_all', action='store_true',
                        help='Save all processed samples, not only samples passing the filter.')
    return parser.parse_args()


def infer_dataset_and_classes(args):
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
    return args


def single_image_miou(pred, label, num_classes, ignore_index=255):
    pred = pred.detach().cpu().numpy().astype(np.int64)
    label = label.detach().cpu().numpy().astype(np.int64)
    valid = label != ignore_index

    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        label_c = (label == cls) & valid
        union = pred_c | label_c
        if union.sum() == 0:
            continue
        inter = pred_c & label_c
        ious.append(inter.sum() / (union.sum() + 1e-12))

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious) * 100.0)


def tensor_to_uint8_image(x):
    # x: [3, H, W], expected in [0, 1]
    x = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def colorize_mask(mask, dataset, ignore_index=255):
    mask = mask.detach().cpu().numpy().astype(np.int64)
    palette = CITYSCAPES_PALETTE if dataset == 'cityscapes' else VOC_PALETTE
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask != ignore_index) & (mask >= 0) & (mask < len(palette))
    color[valid] = palette[mask[valid]]
    color[mask == ignore_index] = np.array([255, 255, 255], dtype=np.uint8)
    return color


def predict_mask(model, images):
    with torch.no_grad():
        outputs = model(normalize_image(images))['out']
        preds = torch.argmax(outputs, dim=1)
    return preds


def run_attack_once(
    model, images, labels, device, method_name, args,
    use_margin_weight=False,
    use_uncertainty_blind_weight=False,
):
    adv_images, _ = get_attack(
        model=model,
        images=images,
        labels=labels,
        device=device,
        method_name=method_name,
        epsilon=args.epsilon,
        alpha=args.alpha,
        iterations=args.steps,
        num_classes=args.num_classes,
        beta=args.beta,
        norm_type=args.norm_type,
        model_arch=args.model_arch,
        use_margin_weight=use_margin_weight,
        use_uncertainty_blind_weight=use_uncertainty_blind_weight,
        high_margin_ratio=args.high_margin_ratio,
        uncertainty_gamma=args.uncertainty_gamma,
        uncertainty_max_weight=args.uncertainty_max_weight,
        uncertainty_blind_lambda=args.uncertainty_blind_lambda,
    )
    preds = predict_mask(model, adv_images)
    return adv_images, preds


def save_sample_folder(sample_dir, image, label, clean_pred, cospgd_pred, ours_pred, info, dataset):
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(image)).save(sample_dir / 'clean.png')
    Image.fromarray(colorize_mask(label, dataset)).save(sample_dir / 'gt.png')
    Image.fromarray(colorize_mask(clean_pred, dataset)).save(sample_dir / 'clean_pred.png')
    Image.fromarray(colorize_mask(cospgd_pred, dataset)).save(sample_dir / 'cospgd_pred.png')
    Image.fromarray(colorize_mask(ours_pred, dataset)).save(sample_dir / 'ours_pred.png')

    with open(sample_dir / 'info.txt', 'w', encoding='utf-8') as f:
        for k, v in info.items():
            f.write(f'{k}: {v}\n')


def write_csv(csv_path, rows):
    fieldnames = [
        'rank', 'sample_index', 'folder',
        'clean_mIoU', 'cospgd_mIoU', 'ours_mIoU', 'delta',
        'pass_filter', 'steps', 'epsilon', 'alpha',
        'high_margin_ratio', 'uncertainty_blind_lambda'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def main():
    args = infer_dataset_and_classes(get_args())
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Model: {args.model_arch} | Dataset: {args.dataset} | Classes: {args.num_classes}')
    print(f'Output: {args.output_dir}')

    output_dir = Path(args.output_dir)
    all_dir = output_dir / 'all_processed'
    ranked_dir = output_dir / 'ranked_top_candidates'
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(device, args.model_arch)

    loader = get_dataloader(
        root=args.data_root,
        batch_size=1,
        split='val',
        dataset_type=args.dataset
    )

    rows = []
    processed = 0

    for sample_index, (images, labels) in enumerate(tqdm(loader, desc='Selecting qualitative samples')):
        if processed >= args.max_samples:
            break
        processed += 1

        images = images.to(device)
        labels = labels.to(device)

        image = images[0]
        label = labels[0]

        clean_pred = predict_mask(model, images)[0]
        clean_miou = single_image_miou(clean_pred, label, args.num_classes)

        # Reset the seed before each attack so CoSPGD and Ours start from the same random initialization.
        torch.manual_seed(args.seed + sample_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + sample_index)
        _, cospgd_pred_batch = run_attack_once(
            model=model,
            images=images,
            labels=labels,
            device=device,
            method_name='cospgd',
            args=args,
            use_margin_weight=False,
            use_uncertainty_blind_weight=False,
        )
        cospgd_pred = cospgd_pred_batch[0]
        cospgd_miou = single_image_miou(cospgd_pred, label, args.num_classes)

        torch.manual_seed(args.seed + sample_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + sample_index)
        _, ours_pred_batch = run_attack_once(
            model=model,
            images=images,
            labels=labels,
            device=device,
            method_name='mr-cospgd',
            args=args,
            use_margin_weight=True,
            use_uncertainty_blind_weight=True,
        )
        ours_pred = ours_pred_batch[0]
        ours_miou = single_image_miou(ours_pred, label, args.num_classes)

        delta = cospgd_miou - ours_miou
        pass_filter = (
            clean_miou >= args.min_clean_miou and
            cospgd_miou >= args.min_cospgd_miou and
            ours_miou < cospgd_miou and
            delta >= args.min_delta
        )

        folder_name = f'sample_{sample_index:05d}_delta_{delta:.2f}_cospgd_{cospgd_miou:.2f}_ours_{ours_miou:.2f}'
        sample_dir = all_dir / folder_name

        row = {
            'rank': '',
            'sample_index': sample_index,
            'folder': str(sample_dir),
            'clean_mIoU': round(clean_miou, 4),
            'cospgd_mIoU': round(cospgd_miou, 4),
            'ours_mIoU': round(ours_miou, 4),
            'delta': round(delta, 4),
            'pass_filter': int(pass_filter),
            'steps': args.steps,
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'high_margin_ratio': args.high_margin_ratio,
            'uncertainty_blind_lambda': args.uncertainty_blind_lambda,
        }

        if pass_filter or args.save_all:
            save_sample_folder(
                sample_dir=sample_dir,
                image=image,
                label=label,
                clean_pred=clean_pred,
                cospgd_pred=cospgd_pred,
                ours_pred=ours_pred,
                info=row,
                dataset=args.dataset,
            )

        rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: (x['pass_filter'], x['delta']), reverse=True)
    for rank, row in enumerate(rows_sorted, start=1):
        row['rank'] = rank

    write_csv(output_dir / 'candidates_all.csv', rows_sorted)

    top_rows = [r for r in rows_sorted if int(r['pass_filter']) == 1][:args.top_k]
    write_csv(output_dir / 'candidates_top.csv', top_rows)

    # Copy top-k candidate folders into ranked_top_candidates/rank_xxx_* for quick manual review.
    if ranked_dir.exists():
        shutil.rmtree(ranked_dir)
    ranked_dir.mkdir(parents=True, exist_ok=True)

    for rank, row in enumerate(top_rows, start=1):
        src = Path(row['folder'])
        if not src.exists():
            continue
        dst = ranked_dir / f'rank_{rank:03d}_sample_{int(row["sample_index"]):05d}_delta_{float(row["delta"]):.2f}'
        shutil.copytree(src, dst)

    print('\nDone.')
    print(f'Processed samples: {processed}')
    print(f'Passed filter: {sum(int(r["pass_filter"]) for r in rows)}')
    print(f'All CSV: {output_dir / "candidates_all.csv"}')
    print(f'Top CSV: {output_dir / "candidates_top.csv"}')
    print(f'Top candidate images: {ranked_dir}')
    print('\nNext step: open ranked_top_candidates and manually choose 2 samples with clear visual differences.')


if __name__ == '__main__':
    main()
