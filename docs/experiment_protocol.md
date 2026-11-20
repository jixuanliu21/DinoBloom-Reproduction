# Experiment Protocol

## Dataset
Acevedo Peripheral Blood Cell dataset.

- 14,543 initially selected images
- 10 target classes
- 2 conflicting-label samples excluded
- 14,541 samples used in grouped evaluation

## Models
- Original DINOv2-S, ViT-S/14
- DinoBloom-S, ViT-S/14

Both use frozen 384-dimensional image embeddings.

## Feature extraction
- Input resolution: 224 x 224
- ImageNet normalization
- Batch size: 32
- No downstream fine-tuning of the backbones

## Evaluation
- Five-fold StratifiedGroupKFold
- shuffle=True
- random_state=42
- Exact duplicate images grouped using SHA-256
- Conflicting-label duplicate pair excluded
- Classifiers: 1-NN, 20-NN, Logistic Regression
- KNN uses scikit-learn default distance metric
- Logistic Regression: lbfgs, C=38.4, max_iter=1000
- Metrics: Weighted F1, Balanced Accuracy

## Important limitations
- No verified patient-level split
- No near-duplicate image analysis
- Linear Probe did not converge in all folds
- Dataset selection may differ from the original paper
- This reproduces downstream evaluation, not pretraining

## Required external assets
- Official marrlab/DinoBloom repository
- Official DinoBloom-S checkpoint
- Acevedo dataset

## Local configuration
Set DINOBLOOM_REPO to the official repository location
if it is not stored alongside this reproduction project.

The provided scripts currently use HemaAgent-local paths under
the user's home directory. These must be made configurable
before the project can be considered fully portable.
