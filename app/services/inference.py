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
            # normalize
            padded /= 10000.0

            # Numpy to pytorch
            padded = np.ascontiguousarray(padded)

            tensor = torch.from_numpy(
                padded
            ).unsqueeze(0).float().to(DEVICE)

            # model inference
            with torch.no_grad():
                logits = model(
                    tensor
                )

                probs = torch.sigmoid(logits)

            # pytorch to numpy
            # may cause error
            probs = (
                probs[0, 0].detach().to(DEVICE).numpy()
            )

            # remove padding

            probs = probs[:patch_h, :patch_w]

            probability_sum[y:y_end, x:x_end] += probs

            # count predictions
            predict_count[y:y_end, x:x_end] += 1
        # Average overlapping predictions
    valid = predict_count > 0

    probability_map = np.zeros_like(
        probability_sum
    )

    probability_map[valid] = (
        probability_sum[valid]
        /
        predict_count[valid]
    )

    return probability_map


    ### save probability map
def save_probability_map(
            probability_map,
            profile,
            output_path
):
    prob_profile = profile.copy()

    prob_profile.update(
            count=1,
            dtype="float32",
            nodata=0,
            compress="LZN"
    )

    with rasterio.open(
            output_path,
            "w",
            **prob_profile
    ) as dst:
            
        dst.write(
            probability_map.astype(
                np.float32
        ),
        1
)
        
def create_road_mask(
        probability_map,
):
    road_mask = (
        probability_map >= THRESHOLD
    ).astype(np.uint8)

    return road_mask

def save_road_mask(
        road_mask,
        profile,
        output_path
):
    mask_profile = profile.copy()

    mask_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="LZW"
    )

    with rasterio.open(
        output_path,
        "w",
        **mask_profile
    ) as dst:
        
        dst.write(
            road_mask,
            1,
        )

### running inference
def run_inference(
        input_path,
        probability_path,
        mask_path
):
    model = load_model()

    image, profile, height, width = read_raster(input_path)

    probability_map = predict_patches(
        image,
        model
    )

    save_probability_map(
        probability_map,
        profile,
        probability_path
    )

    road_mask = create_road_mask(
        probability_map
    )

    save_road_mask(
        road_mask,
        profile,
        mask_path
    )

    return {
        "probability_path": str(
            probability_path
        ),
        "mask_path": str(
            mask_path
        ),
    }
