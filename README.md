# Margin-Uncertainty-Aware-Segmentation-Attack

This repository provides the implementation of a margin- and uncertainty-aware adversarial attack for semantic segmentation.

The method is designed for white-box, non-targeted adversarial attacks under the \(L_\infty\) perturbation constraint. It adopts a sequential attack strategy that first emphasizes correctly classified pixels with large classification margins and then applies an uncertainty-aware complementary attack to residual correctly classified pixels after the main attack update.

## Core Files

- `main.py`: Main entry for running adversarial attack experiments.
- `attack_implementations.py`: Implementation of the proposed attack and related attack variants.
- `attacks.py`: Basic attack components and baseline attack interfaces.
- `dataset.py`: Dataset loading and preprocessing.
- `metrics.py`: Evaluation metrics for semantic segmentation.
- `logger.py`: Logging utilities.
- `requirements.txt`: Required Python packages.

Several auxiliary scripts are also provided for qualitative visualization and sample selection.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Notes

Datasets, pretrained model weights, experimental logs, and visualization results are not included in this repository. Please prepare the required datasets and model weights separately.

## Citation

Citation information will be updated after publication.
