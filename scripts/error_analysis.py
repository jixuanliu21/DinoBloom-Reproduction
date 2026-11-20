
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

base = Path.home() / "HemaAgent-local/results/DinoBloom"

with np.load(base / "features_all.npz") as data:
    X = data["features"]
    y = data["labels"]

classes = np.unique(y)

cv = StratifiedKFold(n_splits=5, shuffle=False)

# Store predictions from all five held-out folds
predictions = np.empty_like(y)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    model = LogisticRegression(
        C=X.shape[1] * len(classes) / 100,
        max_iter=1000,
        solver="lbfgs",
    )

    model.fit(X[train_idx], y[train_idx])
    predictions[test_idx] = model.predict(X[test_idx])

    print(f"Fold {fold} completed")

# Report individual class performance
print("\nClassification Report:")
print(classification_report(y, predictions, digits=4))

# Create row-normalized confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))

ConfusionMatrixDisplay.from_predictions(
    y,
    predictions,
    labels=classes,
    display_labels=classes,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
    xticks_rotation=45,
    ax=ax,
)

ax.set_title("DinoBloom-S | Acevedo | Linear Probe")
fig.tight_layout()

output = base / "confusion_matrix.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output)
