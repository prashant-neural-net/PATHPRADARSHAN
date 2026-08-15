import numpy as np
import rasterio
import torch

from model import UNet


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = "sentinel_bengaluru_large_aoi.tif"
MODEL_PATH = "road_unet_v5_best.pth"
OUTPUT_PROB = "bengaluru_v5_probability.tif"
OUTPUT_MASK = "bengaluru_v5_road_mask.tif"

PATCH_SIZE = 256
STRIDE = 128
THRESHOLD = 0.55


DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded:", MODEL_PATH)


# ============================================================
# OPEN SATELLITE IMAGE
# ============================================================

with rasterio.open(INPUT_IMAGE) as src:

    image = src.read([1, 2, 3, 4]).astype(np.float32)

    profile = src.profile.copy()

    height = src.height
    width = src.width

    transform = src.transform


print("Image shape:", image.shape)
print("Image size:", width, height)


# ============================================================
# ACCUMULATION ARRAYS
# ============================================================

probability_sum = np.zeros(
    (height, width),
    dtype=np.float32
)

prediction_count = np.zeros(
    (height, width),
    dtype=np.uint16
)


# ============================================================
# PATCH INFERENCE
# ============================================================

total_rows = (
    (height - 1) // STRIDE
) + 1

processed = 0

for y in range(
    0,
    height,
    STRIDE
):

    for x in range(
        0,
        width,
        STRIDE
    ):

        y_end = min(
            y + PATCH_SIZE,
            height
        )

        x_end = min(
            x + PATCH_SIZE,
            width
        )

        patch_h = y_end - y
        patch_w = x_end - x

        # ----------------------------------------------------
        # Ignore very small edge patches
        # ----------------------------------------------------

        if patch_h < 64 or patch_w < 64:
            continue

        patch = image[
            :,
            y:y_end,
            x:x_end
        ]

        # ----------------------------------------------------
        # Pad edge patches to 256x256
        # ----------------------------------------------------

        padded = np.zeros(
            (
                4,
                PATCH_SIZE,
                PATCH_SIZE
            ),
            dtype=np.float32
        )

        padded[
            :,
            :patch_h,
            :patch_w
        ] = patch

        # Sentinel normalization
        padded /= 10000.0

        padded = np.ascontiguousarray(
            padded
        )

        tensor = torch.from_numpy(
            padded
        ).unsqueeze(0).float().to(
            DEVICE
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor
            )

            probs = torch.sigmoid(
                logits
            )

        probs = (
            probs[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        probs = probs[
            :patch_h,
            :patch_w
        ]

        # ----------------------------------------------------
        # Accumulate overlapping patches
        # ----------------------------------------------------

        probability_sum[
            y:y_end,
            x:x_end
        ] += probs

        prediction_count[
            y:y_end,
            x:x_end
        ] += 1

        processed += 1

        if processed % 25 == 0:

            print(
                "Patches processed:",
                processed
            )


# ============================================================
# AVERAGE OVERLAPPING PREDICTIONS
# ============================================================

valid = prediction_count > 0

probability_map = np.zeros_like(
    probability_sum
)

probability_map[valid] = (
    probability_sum[valid]
    /
    prediction_count[valid]
)

# ============================================================
# SAVE PROBABILITY MAP
# ============================================================

prob_profile = profile.copy()

prob_profile.update(
    count=1,
    dtype="float32",
    nodata=0,
    compress="LZW"
)

with rasterio.open(
    OUTPUT_PROB,
    "w",
    **prob_profile
) as dst:

    dst.write(
        probability_map.astype(np.float32),
        1
    )

print("Saved probability map:", OUTPUT_PROB)


# ============================================================
# THRESHOLD
# ============================================================

road_mask = (
    probability_map >= THRESHOLD
).astype(np.uint8)


# ============================================================
# SAVE MASK
# ============================================================

profile.update(
    count=1,
    dtype="uint8",
    nodata=0,
    compress="LZW"
)

with rasterio.open(
    OUTPUT_MASK,
    "w",
    **profile
) as dst:

    dst.write(
        road_mask,
        1
    )


# ============================================================
# STATS
# ============================================================

road_pixels = int(
    road_mask.sum()
)

total_pixels = (
    height * width
)

road_percent = (
    road_pixels
    /
    total_pixels
) * 100


print()
print("==============================")
print("V5 FULL AOI INFERENCE")
print("==============================")
print("Processed patches:", processed)
print("Threshold:", THRESHOLD)
print("Road pixels:", road_pixels)
print(
    "Road percentage:",
    round(road_percent, 2),
    "%"
)
print("Saved:", OUTPUT_MASK)
