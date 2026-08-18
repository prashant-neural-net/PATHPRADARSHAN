from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .raster import (
    inspect_band
)

from .inference import run_inference

UPLOAD_DIR = Path("data/uploads")
RESULT_DIR = Path("data/results")


async def save_analysis_image(
    file: UploadFile
):
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis_id = str(uuid4())

    extension = Path(file.filename or "").suffix.lower()

    if extension not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF files are supported.")

    output_path = UPLOAD_DIR / f"{analysis_id}{extension}"

    with output_path.open("wb") as buffer:

        while chunk := await file.read(1024 * 1024):

            buffer.write(chunk)
    
    raster = inspect_band(output_path)

    RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

    probability_path = (
        RESULT_DIR
        / f"{analysis_id}_probability.tif"
    )

    mask_path = (
        RESULT_DIR
        / f"{analysis_id}_road_mask.tif"
    )

    # run inference model
    inference_result = run_inference(
        output_path,
        probability_path,
        mask_path,
    )

    return {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "status": "uploaded",
        "path": str(output_path),
        "raster": raster
    }

