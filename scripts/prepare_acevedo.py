
from pathlib import Path
import csv
from collections import Counter

root = Path.home() / "HemaAgent-local/datasets/Acevedo/PBC_dataset_normal_DIB"
output = Path.home() / "HemaAgent-local/datasets/Acevedo/dataset_manifest.csv"

label_map = {
    "BA": "basophil",
    "EO": "eosinophil",
    "ERB": "erythroblast",
    "LY": "lymphocyte",
    "MO": "monocyte",
    "MMY": "metamyelocyte",
    "MY": "myelocyte",
    "PMY": "promyelocyte",
    "BNE": "neutrophil_band",
    "SNE": "neutrophil_segmented",
}

rows = []
excluded = Counter()

for image in sorted(root.rglob("*.jpg")):
    if image.name.startswith("."):
        excluded["hidden"] += 1
        continue

    prefix = image.stem.split("_")[0]

    if prefix not in label_map:
        excluded[prefix] += 1
        continue

    rows.append({
        "image_path": str(image.resolve()),
        "label": label_map[prefix],
    })

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
    writer.writeheader()
    writer.writerows(rows)

counts = Counter(row["label"] for row in rows)

print("Saved:", output)
print("Valid images:", len(rows))
print("Classes:", len(counts))
print("\nClass counts:")
for label, count in sorted(counts.items()):
    print(f"{label:25s} {count}")

print("\nExcluded:")
for reason, count in sorted(excluded.items()):
    print(f"{reason:25s} {count}")

assert len(rows) == 14543
assert len(counts) == 10
