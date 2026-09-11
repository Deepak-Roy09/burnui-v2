from fastapi import APIRouter, Depends, File, UploadFile

from app.auth import User, require_role
from app.datasets.validation import validate_csv_upload

router = APIRouter(prefix="/datasets", tags=["datasets"])
require_dataset_user = require_role("INSPECTOR", "ENGINEER", "ADMIN")


@router.post("/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    _: User = Depends(require_dataset_user),
) -> dict:
    """Validate a normalized C-MAPSS-compatible CSV without persisting it."""
    return validate_csv_upload(file.filename, await file.read())
