

from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os

OFFICIAL_REPO = Path(
    os.environ.get(
        "DINOBLOOM_REPO",
        str(Path(__file__).resolve().parents[2] / "DinoBloom")
    )
)

assert (OFFICIAL_REPO / "hubconf.py").is_file(), (
    f"Official DinoBloom repository not found: {OFFICIAL_REPO}"
)


# =====================================================
# 1. Paths
# =====================================================

base = Path.home() / "HemaAgent-local"

manifest = base / "datasets/Acevedo/dataset_manifest.csv"

output_dir = base / "results/DinoBloom"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "features_all.npz"


# =====================================================
# 2. Select device
# =====================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# =====================================================
# 3. Load original DINOv2-S
# =====================================================

# Run this script from the DinoBloom repository root.
# Use the local architecture but original DINOv2 weights.


# Load DinoBloom-S architecture and trained weights

model = torch.hub.load(
    str(OFFICIAL_REPO),
    "dinov2_vits14",
    source="local",
    pretrained=False,
)

checkpoint_path = (
    Path.home()
    / "HemaAgent-local/checkpoints/DinoBloom-S.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

state_dict = {
    key.removeprefix("backbone."): value
    for key, value in checkpoint["teacher"].items()
    if key.startswith("backbone.")
}

if "pos_embed" in state_dict:
    model.pos_embed = torch.nn.Parameter(
        torch.empty_like(state_dict["pos_embed"])
    )

model.load_state_dict(state_dict, strict=True)

model = model.to(device).eval()

print("DinoBloom-S weights loaded successfully")


print("Loaded original DINOv2-S successfully")


# =====================================================
# 4. Image preprocessing
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# =====================================================
# 5. Read dataset manifest
# =====================================================

with manifest.open("r", newline="") as f:
    rows = list(csv.DictReader(f))

print("Total images:", len(rows))

assert len(rows) == 14543, "Unexpected dataset size"


# =====================================================
# 6. Batch feature extraction
# =====================================================

batch_size = 32
all_features = []

with torch.inference_mode():

    for start in range(0, len(rows), batch_size):

        batch_rows = rows[start:start + batch_size]

        images = []

        for row in batch_rows:

            image_path = Path(row["image_path"])

            with Image.open(image_path) as img:
                image = img.convert("RGB")
                tensor = transform(image)

            images.append(tensor)

        batch = torch.stack(images).to(device)

        # Extract 384-dimensional embeddings
        features = model(batch)

        all_features.append(
            features.cpu().numpy()
        )

        processed = min(start + batch_size, len(rows))

        if processed % 320 == 0 or processed == len(rows):
            print(f"Processed {processed}/{len(rows)}")


# =====================================================
# 7. Combine extracted features
# =====================================================

features_np = np.concatenate(
    all_features,
    axis=0
)

labels = np.array([
    row["label"] for row in rows
])

paths = np.array([
    row["image_path"] for row in rows
])

assert features_np.shape == (14543, 384)
assert np.isfinite(features_np).all()

print("\nFeature extraction completed!")
print("Feature shape:", features_np.shape)
print("Labels shape:", labels.shape)


# =====================================================
# 8. Save results
# =====================================================

np.savez_compressed(
    output_file,
    features=features_np,
    labels=labels,
    paths=paths,
)

print("Saved to:", output_file)
print("\nDINOv2-S baseline feature extraction SUCCESS")
