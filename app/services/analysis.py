from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

UPLOAD_DIR = Path("data/uploads")


async def save_analysis_image(
    file: UploadFile,
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

    return {
        "analysis_id": analysis_id,
        "filename": file.filename,
        "status": "uploaded",
        "path": str(output_path),
    }
