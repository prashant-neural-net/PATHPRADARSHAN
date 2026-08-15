import os
import numpy as np
import rasterio
import torch
from model import UNet

IMAGE_DIR = "dataset_v3_original/images"
MASK_DIR = "dataset_v3_original/masks"
MODEL_PATH = "road_unet_v5_best.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".tif")
)

n = len(files)

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(
    n,
    generator=generator
).tolist()

train_size = int(0.8 * n)
val_indices = indices[train_size:]

print("Total samples:", n)
print("Validation samples:", len(val_indices))

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

thresholds = [
    0.20, 0.30, 0.40, 0.50,
    0.55, 0.60, 0.65, 0.70,
    0.75, 0.80
]

results = {
    t: {"tp": 0, "fp": 0, "fn": 0}
    for t in thresholds
}

for idx in val_indices:

    image_name = files[idx]
    mask_name = image_name.replace(
        "image_", "mask_"
    )

    with rasterio.open(
        os.path.join(IMAGE_DIR, image_name)
    ) as src:
        image = src.read().astype(np.float32)

    with rasterio.open(
        os.path.join(MASK_DIR, mask_name)
    ) as src:
        mask = src.read(1).astype(np.float32)

    image /= 10000.0

    x = torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(
            model(x)
        ).squeeze().cpu().numpy()

    target = mask == 1

    for t in thresholds:

        pred = probs >= t

        tp = np.logical_and(pred, target).sum()
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()

        results[t]["tp"] += int(tp)
        results[t]["fp"] += int(fp)
        results[t]["fn"] += int(fn)

print()
print("================================")
print("U-NET V5 THRESHOLD EVALUATION")
print("================================")

best_t = None
best_dice = -1

for t in thresholds:

    tp = results[t]["tp"]
    fp = results[t]["fp"]
    fn = results[t]["fn"]

    dice = 2 * tp / (
        2 * tp + fp + fn + 1e-8
    )

    iou = tp / (
        tp + fp + fn + 1e-8
    )

    precision = tp / (
        tp + fp + 1e-8
    )

    recall = tp / (
        tp + fn + 1e-8
    )

    print(
        f"Threshold {t:.2f} | "
        f"Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
    )

    if dice > best_dice:
        best_dice = dice
        best_t = t

print()
print("Best threshold:", best_t)
print("Best Dice:", round(best_dice, 4))
