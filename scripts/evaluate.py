
from pathlib import Path
import csv

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

# 1. Load extracted features
base = Path.home() / "HemaAgent-local/results/DinoBloom"
data = np.load(base / "features_all.npz")

X = data["features"]
y = data["labels"]

print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Classes:", len(np.unique(y)))

assert X.shape == (14543, 384)
assert len(y) == len(X)
assert np.isfinite(X).all()

# 2. Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=False,
)

results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y), start=1
):
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Same two KNN settings as the paper
    models = {
        "1-NN": KNeighborsClassifier(n_neighbors=1),
        "20-NN": KNeighborsClassifier(n_neighbors=20),
        "Linear Probe": LogisticRegression(
            C=(X.shape[1] * len(np.unique(y_train))) / 100,
            max_iter=1000,
            solver="lbfgs",
        ),
    }

    for name, classifier in models.items():
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        result = {
            "fold": fold,
            "model": name,
            "weighted_f1": f1_score(
                y_test, predictions, average="weighted"
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "accuracy": accuracy_score(y_test, predictions),
        }

        results.append(result)
        print(
            f"Fold {fold} | {name:13s} | "
            f"wF1={result['weighted_f1']:.4f} | "
            f"bAcc={result['balanced_accuracy']:.4f}"
        )

# 3. Save fold-level results
output = base / "evaluation_folds.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

# 4. Report mean and standard deviation
print("\n===== Five-fold results =====")

for name in ["1-NN", "20-NN", "Linear Probe"]:
    subset = [r for r in results if r["model"] == name]

    for metric in ["weighted_f1", "balanced_accuracy"]:
        values = np.array([r[metric] for r in subset])
        print(
            f"{name:13s} {metric:20s}: "
            f"{values.mean():.4f} +/- {values.std():.4f}"
        )

print("\nSaved:", output)
