"""Run the existing real-only anomaly screen for an uploaded sensor CSV."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.datasets.contract import CYCLE_COLUMN, ENGINE_ID_COLUMN, REQUIRED_NON_SENSOR_COLUMNS, sensor_columns
from app.datasets.validation import validate_csv_upload
from app.ml.production_training import load_production_pipeline, production_artifact_paths
from app.preparation.early_features import extract_early_features

router = APIRouter(prefix="/screening", tags=["screening"])


@router.post("/run")
async def run_screening(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate and apply the saved production early-cycle anomaly screen."""
    content = await file.read()
    validation = validate_csv_upload(file.filename, content)
    if not validation["valid"]:
        raise HTTPException(
            status_code=422,
            detail={"message": "The uploaded CSV does not satisfy the normalized dataset contract.", "validation": validation},
        )

    frame = _read_normalized_csv(content)
    try:
        feature_result = extract_early_features(frame)
        skipped_engines = feature_result.skipped_engines
        if skipped_engines:
            raise ValueError("Each engine must provide at least 30 unique cycles in the early-cycle window.")

        features = feature_result.features
        pipeline = _load_compatible_production_pipeline(features)
        predictions = pipeline.predict(features)
    except ValueError as exc:
        detail: dict[str, Any] = {"message": str(exc)}
        if "skipped_engines" in locals() and skipped_engines:
            detail["insufficient_early_cycles"] = [
                {"engine_id": engine_id, "available_early_cycles": cycle_count}
                for engine_id, cycle_count in sorted(skipped_engines.items())
            ]
        raise HTTPException(status_code=422, detail=detail) from exc

    components = [_component_response(row) for _, row in predictions.sort_values(ENGINE_ID_COLUMN).iterrows()]
    status_counts = {status: sum(component["status"] == status for component in components) for status in ("Normal", "Watchlist", "High Risk")}
    return {
        "components": components,
        "summary": {
            "total_components_screened": len(components),
            "normal_count": status_counts["Normal"],
            "watchlist_count": status_counts["Watchlist"],
            "high_risk_count": status_counts["High Risk"],
        },
    }


def _read_normalized_csv(content: bytes) -> pd.DataFrame:
    """Parse a CSV after the dataset validator has accepted its normalized contract."""
    frame = pd.read_csv(StringIO(content.decode("utf-8-sig")))
    sensors = sensor_columns(frame.columns.tolist())
    frame[ENGINE_ID_COLUMN] = pd.to_numeric(frame[ENGINE_ID_COLUMN], errors="raise").astype("int64")
    frame[CYCLE_COLUMN] = pd.to_numeric(frame[CYCLE_COLUMN], errors="raise").astype("int64")
    for column in [*REQUIRED_NON_SENSOR_COLUMNS[2:], *sensors]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _load_compatible_production_pipeline(features: pd.DataFrame):
    """Load the saved FD001 artifact and reject uploads outside its fixed feature schema."""
    paths = production_artifact_paths()
    if not paths.model.is_file() or not paths.metadata.is_file():
        raise RuntimeError("The saved production model artifacts are unavailable.")
    pipeline = load_production_pipeline()
    received_columns = tuple(column for column in features.columns if column != ENGINE_ID_COLUMN)
    if received_columns != pipeline.preprocessor.input_columns:
        raise ValueError("The saved production model requires the FD001-compatible sensor_1 through sensor_21 feature schema.")
    return pipeline


def _component_response(row: pd.Series) -> dict[str, Any]:
    explanation = row["explanation"]
    return {
        "engine_id": int(row[ENGINE_ID_COLUMN]),
        "risk_score": float(row["risk_score"]),
        "status": str(row["classification"]),
        "z_score": float(row["z_score"]),
        "isolation_score": float(row["isolation_score"]),
        "top_features": explanation["top_features"],
    }
