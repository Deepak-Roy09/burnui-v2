"""Authenticated, read-only model-evidence endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth import User, require_role
from app.evaluation.evidence import build_anomaly_evidence


router = APIRouter(prefix="/evaluation", tags=["evaluation"])
require_evaluation_user = require_role("INSPECTOR", "ENGINEER", "ADMIN")


@router.get("/anomaly-evidence")
def anomaly_evidence(_: User = Depends(require_evaluation_user)) -> dict[str, object]:
    try:
        return build_anomaly_evidence()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail={"message": "Anomaly evaluation evidence is unavailable."}) from exc
