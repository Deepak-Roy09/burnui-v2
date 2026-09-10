from fastapi import APIRouter, File, UploadFile

from app.datasets.validation import validate_csv_upload

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/validate")
async def validate_dataset(file: UploadFile = File(...)) -> dict:
    """Validate a normalized C-MAPSS-compatible CSV without persisting it."""
    return validate_csv_upload(file.filename, await file.read())
