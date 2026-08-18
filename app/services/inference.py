### inference

from pathlib import Path

import numpy as np
import rasterio
import torch

from model.model import UNet

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "road_unet_v5_best.pth"
)

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

#### loading the model

def load_model():
    
    model = UNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model

#### reading raster file

def read_raster(input_path):
    with rasterio.open(input_path) as src:

        image = src.read(
            [1,2,3,4]
        ).astype(np.float32)

        profile = src.profile.copy()

        height = src.height
        width = src.width

    return image, profile, height, width

#### predicting patches

def predict_patches(image, model):
    _, height, width = image.shape
    
    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )
    
    predict_count = np.zeros(
        (height, width),
        dtype=np.uint16
    )

    for y in range(0, height, STRIDE):
        for x in range(0, width, STRIDE):

            #patch boundaries
            y_end = min(
                y + PATCH_SIZE,
                height
            )

            x_end = min(
                x + PATCH_SIZE,
                width
            )
            # calculate patch size 
            patch_h = y_end - y
            patch_w = x_end - x

            #ignoring tiny patches
            if patch_h < 64 or patch_w < 64:
                continue

            patch = image[
                :,
                y:y_end,
                x:x_end
            ]

            # padding missing values
            padded = np.zeros(
               (4,
                PATCH_SIZE,
                PATCH_SIZE),
                dtype=np.float32
            )

            padded[:,
                   :patch_h,
                   :patch_w
            ] = patch
