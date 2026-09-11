"""Separate, research-only C-MAPSS drift-prediction endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import User, require_role
from app.evaluation.drift_prediction import run_fd001_drift_evaluation


router = APIRouter(prefix="/drift-evaluation", tags=["drift-evaluation"])
require_drift_user = require_role("INSPECTOR", "ENGINEER", "ADMIN")


@router.get("/run")
def run_drift_evaluation(_: User = Depends(require_drift_user)) -> dict[str, object]:
    """Run the isolated FD001 drift evaluation; it does not affect screening."""
    try:
        return run_fd001_drift_evaluation().to_dict()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
