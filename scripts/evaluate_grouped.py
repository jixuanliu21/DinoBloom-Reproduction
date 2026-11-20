
from pathlib import Path
from collections import defaultdict
import csv
import json
import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
)
from sklearn.exceptions import ConvergenceWarning

BASE = Path.home() / "HemaAgent-local/results"
AUDIT = BASE / "audit/dataset_audit.json"
OUTPUT = BASE / "grouped_evaluation"

OUTPUT.mkdir(parents=True, exist_ok=True)

# 1. Load audit and both feature matrices
audit = json.loads(AUDIT.read_text())

with np.load(BASE / "DinoBloom/features_all.npz") as a, \
     np.load(BASE / "DINOv2/features_all.npz") as b:

    bloom_X = a["features"].copy()
    dino_X = b["features"].copy()
    paths = a["paths"].copy()
    labels = a["labels"].copy()

    assert np.array_equal(paths, b["paths"])
    assert np.array_equal(labels, b["labels"])

assert bloom_X.shape == dino_X.shape
assert np.isfinite(bloom_X).all()
assert np.isfinite(dino_X).all()

# 2. Group exact duplicates
path_to_idx = {
    str(p): i for i, p in enumerate(paths)
}

groups = np.arange(len(paths))
excluded = set()

for duplicate_group in audit["duplicates"]:
    indices = [
        path_to_idx[item["path"]]
        for item in duplicate_group
        if item["path"] in path_to_idx
    ]

    if not indices:
        continue

    group_labels = {labels[i] for i in indices}

    if len(group_labels) > 1:
        excluded.update(indices)
    else:
        group_id = min(indices)
        for i in indices:
            groups[i] = group_id

# 3. Exclude conflicting-label samples
keep = np.array([
    i not in excluded
    for i in range(len(paths))
])

X_bloom = bloom_X[keep]
X_dino = dino_X[keep]
y = labels[keep]
g = groups[keep]

print("Original samples:", len(paths))
print("Excluded samples:", len(excluded))
print("Remaining samples:", len(y))
print("Classes:", len(np.unique(y)))

assert len(np.unique(y)) == 10

# 4. Group-aware cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

folds = list(cv.split(X_bloom, y, groups=g))

for fold, (train_idx, test_idx) in enumerate(folds, 1):
    train_groups = set(g[train_idx])
    test_groups = set(g[test_idx])

    assert train_groups.isdisjoint(test_groups)
    assert set(y[train_idx]) == set(y[test_idx])

    print(
        f"Fold {fold}: "
        f"train={len(train_idx)}, "
        f"test={len(test_idx)} "
        "group overlap=0"
    )

# 5. Evaluate both feature extractors
results = []

for model_name, X in [
    ("DinoBloom-S", X_bloom),
    ("DINOv2-S", X_dino),
]:
    print(f"\n===== {model_name} =====")

    for fold, (train_idx, test_idx) in enumerate(folds, 1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        n_classes = len(np.unique(y_train))
        C = X.shape[1] * n_classes / 100

        classifiers = {
            "1-NN": KNeighborsClassifier(n_neighbors=1),
            "20-NN": KNeighborsClassifier(n_neighbors=20),
            "Linear Probe": LogisticRegression(
                C=C,
                max_iter=1000,
                solver="lbfgs",
            ),
        }

        for method, clf in classifiers.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter(
                    "always", ConvergenceWarning
                )
                clf.fit(X_train, y_train)

            converged = not any(
                issubclass(w.category, ConvergenceWarning)
                for w in caught
            )

            pred = clf.predict(X_test)

            result = {
                "model": model_name,
                "fold": fold,
                "method": method,
                "weighted_f1": f1_score(
                    y_test, pred, average="weighted"
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, pred
                ),
                "converged": converged,
            }

            results.append(result)

            print(
                f"Fold {fold} | {method:13s} | "
                f"wF1={result['weighted_f1']:.4f} | "
                f"bAcc={result['balanced_accuracy']:.4f} | "
                f"converged={converged}"
            )

# 6. Save fold-level results
output = OUTPUT / "grouped_folds.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

# 7. Summary
print("\n===== GROUPED CV SUMMARY =====")

for model in ["DinoBloom-S", "DINOv2-S"]:
    for method in ["1-NN", "20-NN", "Linear Probe"]:
        subset = [
            r for r in results
            if r["model"] == model
            and r["method"] == method
        ]

        wf1 = np.mean([r["weighted_f1"] for r in subset])
        bacc = np.mean([
            r["balanced_accuracy"] for r in subset
        ])
        converged = sum(r["converged"] for r in subset)

        print(
            f"{model:12s} | {method:13s} | "
            f"wF1={wf1:.4f} | "
            f"bAcc={bacc:.4f} | "
            f"converged={converged}/5"
        )

print("\nSaved:", output)
