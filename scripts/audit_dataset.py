
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json

import numpy as np

base = Path.home() / "HemaAgent-local"
manifest = base / "datasets/Acevedo/dataset_manifest.csv"
results_dir = base / "results"
audit_dir = results_dir / "audit"
audit_dir.mkdir(parents=True, exist_ok=True)

with manifest.open(newline="") as f:
    rows = list(csv.DictReader(f))

print("Total samples:", len(rows))
print("Classes:", len({r["label"] for r in rows}))

# 1. Check missing files and duplicate paths
missing = [
    r["image_path"] for r in rows
    if not Path(r["image_path"]).is_file()
]

paths = [r["image_path"] for r in rows]
duplicate_paths = len(paths) - len(set(paths))

# 2. Check exact duplicate image files using SHA-256
hash_groups = defaultdict(list)

for i, row in enumerate(rows, 1):
    path = Path(row["image_path"])
    if not path.is_file():
        continue

    sha = hashlib.file_digest(
        path.open("rb"), "sha256"
    ).hexdigest()

    hash_groups[sha].append({
        "path": str(path),
        "label": row["label"],
    })

    if i % 3000 == 0:
        print(f"Hashed {i}/{len(rows)}")

duplicates = [
    group for group in hash_groups.values()
    if len(group) > 1
]

conflicting = [
    group for group in duplicates
    if len({item["label"] for item in group}) > 1
]

# 3. Check alignment between model features
with np.load(
    results_dir / "DinoBloom/features_all.npz"
) as a, np.load(
    results_dir / "DINOv2/features_all.npz"
) as b:

    same_paths = bool(np.array_equal(
        a["paths"], b["paths"]
    ))
    same_labels = bool(np.array_equal(
        a["labels"], b["labels"]
    ))
    same_features = bool(np.array_equal(
        a["features"], b["features"]
    ))

    shapes = {
        "DinoBloom": list(a["features"].shape),
        "DINOv2": list(b["features"].shape),
    }

summary = {
    "samples": len(rows),
    "class_counts": dict(Counter(
        r["label"] for r in rows
    )),
    "missing_files": len(missing),
    "duplicate_paths": duplicate_paths,
    "exact_duplicate_groups": len(duplicates),
    "conflicting_label_groups": len(conflicting),
    "same_paths": same_paths,
    "same_labels": same_labels,
    "same_features": same_features,
    "feature_shapes": shapes,
}

report = {
    "summary": summary,
    "missing": missing,
    "duplicates": duplicates,
    "conflicting_labels": conflicting,
}

output = audit_dir / "dataset_audit.json"
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

print("\n===== DATA AUDIT SUMMARY =====")
print(json.dumps(summary, indent=2))
print("\nReport saved:", output)
