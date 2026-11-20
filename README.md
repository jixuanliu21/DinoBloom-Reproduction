# DinoBloom Reproduction: Hematology Image Representation Learning

## Overview

This project reproduces and evaluates the downstream classification performance of **DinoBloom-S**, a self-supervised vision transformer adapted to hematological images.

The primary objective is to investigate whether hematology-specific representation learning improves blood cell classification compared with the general-purpose **DINOv2-S** baseline.

The experiments were implemented and executed locally on an Apple Silicon Mac using PyTorch and MPS acceleration.

This is an independent reproduction study based on the publicly available DinoBloom model and dataset. It is not an original implementation of the DinoBloom training algorithm.

## Models

| Model | Architecture | Feature Dimension |
|---|---|---|
| DINOv2-S | ViT-S/14 | 384 |
| DinoBloom-S | ViT-S/14 | 384 |

Both models were evaluated using frozen feature extractors and the same downstream classifiers.

## Dataset

The experiments use the publicly available Acevedo Peripheral Blood Cell dataset.

- Original image files identified: 17,093
- Initially selected samples: 14,543
- Classification task: 10 blood cell categories
- Samples excluded due to conflicting labels: 2
- Final grouped evaluation samples: 14,541

The preprocessing pipeline resizes images to 224 × 224 pixels and applies ImageNet mean and standard deviation normalization.

The category mapping was constructed from image filename prefixes. Samples without sufficiently specific labels were excluded.

## Methodology

1. Load pretrained DinoBloom-S and DINOv2-S checkpoints.
2. Extract 384-dimensional feature vectors from blood cell images.
3. Audit the dataset for exact duplicate files and conflicting labels.
4. Perform five-fold stratified group cross-validation.
5. Evaluate 1-NN, 20-NN and multinomial logistic regression.
6. Report Weighted F1 and Balanced Accuracy.

The grouped evaluation prevents exact duplicate files identified by SHA-256 from appearing in different training and testing folds.

## Results

### Grouped Five-Fold Cross-Validation

| Classifier | DINOv2-S Weighted F1 | DinoBloom-S Weighted F1 |
|---|---:|---:|
| 1-NN | 70.65% | 85.22% |
| 20-NN | 76.84% | 88.56% |
| Linear Probe | 87.28% | 89.26% |

### Balanced Accuracy

| Classifier | DINOv2-S | DinoBloom-S |
|---|---:|---:|
| 1-NN | 65.49% | 81.61% |
| 20-NN | 71.47% | 85.63% |
| Linear Probe | 84.30% | 86.59% |

DinoBloom-S outperformed DINOv2-S under all three evaluated classification settings.

The largest gains were observed with nearest-neighbor classifiers, suggesting that the hematology-adapted embeddings offer more discriminative representations for this dataset.

## Data Quality Audit

The SHA-256 audit identified:

- 17 groups containing exactly duplicate image files
- 12 duplicate groups spanning different folds under the original evaluation
- One group containing identical images with conflicting class labels

The corrected evaluation excludes the conflicting-label pair and assigns the remaining exact duplicates to the same cross-validation fold.

## Limitations

- The experiment does not reproduce DinoBloom's original pretraining process.
- The selected dataset subset and evaluation protocol have not been established as identical to those in the published paper.
- Patient-level independence could not be verified.
- Near-duplicate images and alternative crops may remain undetected by SHA-256.
- The linear-probe optimization did not converge in every fold under the original 1,000-iteration setting.
- Consequently, the reported linear-probe results should be considered provisional.

The KNN experiments are unaffected by the logistic-regression convergence warning.

## Reproducibility

Python 3.11 and PyTorch with MPS were used for local execution.

The project scripts require the official DinoBloom repository, the publicly available pretrained weights and the Acevedo dataset. These external assets are not redistributed in this repository.

See `docs/experiment_protocol.md` for the exact evaluation configuration and setup instructions.

## Original Work and Attribution

This reproduction builds on:

- [DinoBloom official repository](https://github.com/marrlab/DinoBloom)
- [DinoBloom pretrained checkpoints](https://zenodo.org/records/10908163)
- [Acevedo Peripheral Blood Cell dataset](https://data.mendeley.com/datasets/snkd93bnjr/1)

The original model architecture, pretrained weights and research contributions belong to their respective authors. Please refer to the original paper and repository for the authoritative citation and licensing information.

## Scope

This repository documents independent reproduction experiments, macOS-compatible inference and downstream evaluation. It does not claim to introduce a new foundation model or a novel classification algorithm.