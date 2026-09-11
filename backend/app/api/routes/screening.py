"""Run the existing real-only anomaly screen for an uploaded sensor CSV."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import math

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.params import Form as FormField
import numpy as np
from sklearn.decomposition import PCA

from app.auth import AuthRepository, User, get_repository, require_role
from app.datasets.contract import CYCLE_COLUMN, ENGINE_ID_COLUMN, REQUIRED_NON_SENSOR_COLUMNS, sensor_columns
from app.datasets.validation import validate_csv_upload
from app.ml.production_training import load_production_pipeline, production_artifact_paths
from app.preparation.cmapss_adapter import CMAPSSFileKind, CMAPSSSubset, CMAPSSVariant, detect_cmapss_filename, load_cmapss, load_cmapss_rul, resolve_cmapss_source
from app.preparation.early_features import extract_early_features

router = APIRouter(prefix="/screening", tags=["screening"])
require_screening_user = require_role("INSPECTOR", "ENGINEER", "ADMIN")


@router.post("/run")
async def run_screening(
    file: UploadFile = File(...),
    rul_file: UploadFile | None = File(default=None),
    watchlist_threshold: float | None = Form(default=None),
    high_risk_threshold: float | None = Form(default=None),
    sensitivity: str = Form(default="Standard"),
    dataset_variant: str | None = Form(default=None),
    dataset_subset: str | None = Form(default=None),
    current_user: User = Depends(require_screening_user),
    repository: AuthRepository = Depends(get_repository),
) -> dict[str, Any]:
    """Validate and apply the saved production early-cycle anomaly screen."""
    # FastAPI resolves multipart parameters to plain Python values. Direct
    # function calls used by focused tests retain the Form(...) defaults, so
    # normalize only those declaration objects at this boundary.
    watchlist_threshold = _unwrap_form_default(watchlist_threshold)
    high_risk_threshold = _unwrap_form_default(high_risk_threshold)
    sensitivity = _unwrap_form_default(sensitivity)
    dataset_variant = _unwrap_form_default(dataset_variant)
    dataset_subset = _unwrap_form_default(dataset_subset)

    content = await file.read()
    rul_file = _unwrap_form_default(rul_file)
    rul_content = await rul_file.read() if rul_file is not None else None
    frame, validation, source = _prepare_screening_upload(file.filename, content, dataset_variant, dataset_subset)
    evaluation = _prepare_test_rul_evaluation(file.filename, frame, rul_file.filename if rul_file else None, rul_content, source)
    try:
        feature_result = extract_early_features(frame)
        skipped_engines = feature_result.skipped_engines
        features = feature_result.features
        if features.empty:
            raise ValueError("No engine has the required 30 unique cycles in the production early-cycle window.")
        pipeline = _load_compatible_production_pipeline(features)
        predictions = pipeline.predict(features)
        resolved_watchlist, resolved_high_risk = _resolve_screening_thresholds(
            watchlist_threshold,
            high_risk_threshold,
            pipeline.thresholds.watchlist,
            pipeline.thresholds.high_risk,
            sensitivity,
        )
        predictions = _apply_decision_thresholds(predictions, resolved_watchlist, resolved_high_risk)
        coordinates = _visualization_coordinates(pipeline, features)
    except ValueError as exc:
        detail: dict[str, Any] = {"message": str(exc)}
        if "skipped_engines" in locals() and skipped_engines:
            detail["insufficient_early_cycles"] = [
                {"engine_id": engine_id, "available_early_cycles": cycle_count}
                for engine_id, cycle_count in sorted(skipped_engines.items())
            ]
        raise HTTPException(status_code=422, detail=detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "The production screening model is unavailable."}) from exc

    components = [
        _component_response(row, coordinates.get(int(row[ENGINE_ID_COLUMN])))
        for _, row in predictions.sort_values(ENGINE_ID_COLUMN).iterrows()
    ]
    status_counts = {status: sum(component["status"] == status for component in components) for status in ("Normal", "Watchlist", "High Risk")}
    repository.log_event(
        "SCREENING_RUN",
        actor_user_id=current_user.id,
        metadata={
            "component_count": len(components),
            "custom_thresholds": watchlist_threshold is not None or high_risk_threshold is not None,
            "sensitivity": sensitivity,
        },
    )
    return {
        "components": components,
        "summary": {
            "total_components_screened": len(components),
            "normal_count": status_counts["Normal"],
            "watchlist_count": status_counts["Watchlist"],
            "high_risk_count": status_counts["High Risk"],
        },
        "screening_parameters": {
            "watchlist_threshold": resolved_watchlist,
            "high_risk_threshold": resolved_high_risk,
            "sensitivity": sensitivity,
            "validated_defaults": {
                "watchlist_threshold": pipeline.thresholds.watchlist,
                "high_risk_threshold": pipeline.thresholds.high_risk,
            },
        },
        "source": {
            **source,
            "screened_engine_count": len(components),
            "excluded_insufficient_early_cycles": [
                {"engine_id": engine_id, "available_early_cycles": cycle_count}
                for engine_id, cycle_count in sorted(skipped_engines.items())
            ],
        },
        "evaluation": evaluation,
    }


def _prepare_test_rul_evaluation(
    test_filename: str | None,
    frame: pd.DataFrame,
    rul_filename: str | None,
    rul_content: bytes | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Strictly align an optional matching TEST RUL file for post-hoc reporting.

    The saved anomaly model never receives RUL values. Binary evaluation is
    reported unavailable here when the existing training-derived horizon is not
    bundled with the uploaded test file; this avoids inventing a threshold.
    """
    if source.get("file_type") != "TEST":
        if rul_filename is not None:
            raise HTTPException(status_code=422, detail={"message": "A matching RUL file may only accompany a C-MAPSS TEST file."})
        return {"mode": "prediction-only", "available": False, "message": "Prediction-only demonstration — matching RUL ground truth was not supplied."}
    if rul_filename is None:
        return {"mode": "prediction-only", "available": False, "message": "Prediction-only demonstration — matching RUL ground truth was not supplied."}
    test_descriptor = detect_cmapss_filename(test_filename)
    rul_descriptor = detect_cmapss_filename(rul_filename)
    if test_descriptor is None or test_descriptor.kind is not CMAPSSFileKind.TEST or rul_descriptor is None or rul_descriptor.kind is not CMAPSSFileKind.RUL or rul_descriptor.variant is not test_descriptor.variant:
        raise HTTPException(status_code=422, detail={"message": "The RUL file must match the uploaded C-MAPSS TEST dataset exactly."})
    try:
        rul_values = load_cmapss_rul(BytesIO(rul_content or b""))
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    engine_ids = sorted(int(value) for value in frame[ENGINE_ID_COLUMN].dropna().unique())
    if len(rul_values) != len(engine_ids):
        raise HTTPException(status_code=422, detail={"message": f"{rul_filename} contains {len(rul_values)} RUL values but the TEST file contains {len(engine_ids)} engines."})
    aligned = [{"engine_id": engine_id, "rul": float(rul)} for engine_id, rul in zip(engine_ids, rul_values)]
    return {
        "mode": "evaluation",
        "available": False,
        "dataset": test_descriptor.variant.value,
        "test_filename": test_descriptor.filename,
        "rul_filename": rul_descriptor.filename,
        "aligned_engine_count": len(aligned),
        "rul_by_engine": aligned,
        "message": "Matching RUL values were aligned for post-hoc evaluation. Binary metrics are unavailable because the existing training-derived target horizon is not part of this upload.",
        "target_definition": "Existing near-term degradation target requires a training-derived horizon; RUL values were not used for model fitting or threshold calibration.",
    }


def _prepare_screening_upload(
    filename: str | None,
    content: bytes,
    dataset_variant: str | None,
    dataset_subset: str | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Accept either the existing normalized CSV or official raw C-MAPSS TXT."""
    name = filename or ""
    lower_name = name.lower()
    if lower_name.endswith(".csv"):
        validation = validate_csv_upload(name, content)
        if not validation["valid"]:
            raise HTTPException(
                status_code=422,
                detail={"message": "The uploaded CSV does not satisfy the normalized dataset contract.", "validation": validation},
            )
        return _read_normalized_csv(content), validation, {
            "format": "normalized_csv",
            "filename": name,
            "dataset": None,
            "file_type": None,
            "input_engine_count": int(validation.get("summary", {}).get("engine_count", 0)),
            "domain_warning": None,
        }
    if not lower_name.endswith(".txt"):
        raise HTTPException(status_code=422, detail={"message": "Upload a normalized .csv or an official C-MAPSS train_*/test_*.txt file."})
    try:
        variant, subset, _ = resolve_cmapss_source(
            name,
            selected_variant=dataset_variant,
            selected_subset=dataset_subset,
        )
        frame = load_cmapss(BytesIO(content), variant=variant, subset=subset)
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc

    # Raw files are parsed by the strict 26-column C-MAPSS adapter, then run
    # through the unchanged normalized contract validation before screening.
    validation = validate_csv_upload(f"{Path(name).stem}.csv", frame.to_csv(index=False).encode("utf-8"))
    validation["filename"] = name
    if not validation["valid"]:
        raise HTTPException(
            status_code=422,
            detail={"message": "The parsed C-MAPSS file does not satisfy the normalized dataset contract.", "validation": validation},
        )
    return frame, validation, {
        "format": "raw_cmapss_txt",
        "filename": name,
        "dataset": variant.value,
        "file_type": subset.value.upper(),
        "input_engine_count": int(frame[ENGINE_ID_COLUMN].nunique()),
        "domain_warning": None if variant is CMAPSSVariant.FD001 else "Domain-shift demonstration — production model trained on FD001.",
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


def _unwrap_form_default(value: Any) -> Any:
    """Return a Form declaration's default for direct endpoint calls."""
    return value.default if isinstance(value, FormField) else value


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


def _component_response(row: pd.Series, coordinates: tuple[float, float] | None = None) -> dict[str, Any]:
    explanation = row["explanation"]
    response = {
        "engine_id": int(row[ENGINE_ID_COLUMN]),
        "risk_score": float(row["risk_score"]),
        "status": str(row["classification"]),
        "z_score": float(row["z_score"]),
        "isolation_score": float(row["isolation_score"]),
        "top_features": explanation["top_features"],
    }
    if coordinates is not None:
        response["pca_1"], response["pca_2"] = coordinates
    return response


def _resolve_screening_thresholds(
    watchlist: float | None,
    high_risk: float | None,
    default_watchlist: float,
    default_high_risk: float,
    sensitivity: str,
) -> tuple[float, float]:
    if sensitivity not in {"Standard", "Sensitive", "Conservative"}:
        raise ValueError("Sensitivity must be Standard, Sensitive, or Conservative.")
    resolved_watchlist = default_watchlist if watchlist is None else watchlist
    resolved_high_risk = default_high_risk if high_risk is None else high_risk
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (resolved_watchlist, resolved_high_risk)):
        raise ValueError("Screening thresholds must be finite values between 0 and 1.")
    if resolved_high_risk <= resolved_watchlist:
        raise ValueError("High Risk threshold must be greater than the Watchlist threshold.")
    return float(resolved_watchlist), float(resolved_high_risk)


def _apply_decision_thresholds(predictions: pd.DataFrame, watchlist: float, high_risk: float) -> pd.DataFrame:
    """Reclassify existing fixed-model risk scores without fitting or recalibrating."""
    result = predictions.copy()
    result["classification"] = result["risk_score"].map(
        lambda score: "High Risk" if score >= high_risk else "Watchlist" if score >= watchlist else "Normal"
    )
    return result


def _visualization_coordinates(pipeline: Any, features: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Apply PCA only to display existing early-cycle features in two dimensions."""
    # PCA intentionally receives the unchanged 168 extracted input features.
    # It is fit per displayed upload only and is never used by the saved model.
    matrix = features.loc[:, pipeline.preprocessor.input_columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        medians = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
    if len(matrix) == 1:
        projected = [[0.0, 0.0]]
    else:
        projected = PCA(n_components=2, random_state=42).fit_transform(matrix)
    return {
        int(engine_id): (float(point[0]), float(point[1]))
        for engine_id, point in zip(features[ENGINE_ID_COLUMN], projected)
    }
