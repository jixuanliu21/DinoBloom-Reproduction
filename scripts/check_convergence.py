
from pathlib import Path
import csv
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

# 1. Configuration
base = Path.home() / "HemaAgent-local/results"
models = ["DinoBloom", "DINOv2"]

max_iterations = 3000
results = []

# 2. Evaluate both feature extractors
for model_name in models:
    path = base / model_name / "features_all.npz"

    with np.load(path) as data:
        X = data["features"]
        y = data["labels"]

    assert X.shape == (14543, 384)
    assert np.isfinite(X).all()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=False
    )

    print(f"\n===== {model_name} =====")

    # 3. Train a linear probe on each fold
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        n_classes = len(np.unique(y_train))
        C = X_train.shape[1] * n_classes / 100

        classifier = LogisticRegression(
            C=C,
            max_iter=max_iterations,
            solver="lbfgs",
            tol=1e-4,
        )

        # Capture convergence warnings explicitly
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter(
                "always", ConvergenceWarning
            )

            classifier.fit(X_train, y_train)

        converged = not any(
            issubclass(w.category, ConvergenceWarning)
            for w in caught
        )

        predictions = classifier.predict(X_test)

        result = {
            "model": model_name,
            "fold": fold,
            "weighted_f1": f1_score(
                y_test, predictions, average="weighted"
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "iterations": int(classifier.n_iter_[0]),
            "converged": converged,
        }

        results.append(result)

        print(
            f"Fold {fold} | "
            f"wF1={result['weighted_f1']:.4f} | "
            f"bAcc={result['balanced_accuracy']:.4f} | "
            f"iterations={result['iterations']} | "
            f"converged={converged}"
        )

# 4. Save without overwriting old results
output = base / "convergence_check_3000.csv"

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

# 5. Summary
print("\n===== Summary =====")

for name in models:
    subset = [r for r in results if r["model"] == name]

    wf1 = np.mean([r["weighted_f1"] for r in subset])
    bacc = np.mean([
        r["balanced_accuracy"] for r in subset
    ])
    converged_count = sum(
        r["converged"] for r in subset
    )

    print(
        f"{name}: "
        f"wF1={wf1:.4f}, "
        f"bAcc={bacc:.4f}, "
        f"converged={converged_count}/5"
    )

print("\nSaved:", output)
