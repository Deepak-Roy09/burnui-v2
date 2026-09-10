"""CSV validation for the normalized C-MAPSS-compatible data contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from app.datasets.contract import (
    CYCLE_COLUMN,
    ENGINE_ID_COLUMN,
    REQUIRED_NON_SENSOR_COLUMNS,
    SENSOR_COLUMN_PATTERN,
    sensor_columns,
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    rows: list[int] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if not self.rows:
            result.pop("rows")
        if not self.columns:
            result.pop("columns")
        return result


def _issue(code: str, message: str, *, rows: list[int] | None = None, columns: list[str] | None = None) -> ValidationIssue:
    return ValidationIssue(code, message, rows or [], columns or [])


def _sample_rows(mask: pd.Series, limit: int = 10) -> list[int]:
    """Return one-based CSV data-row numbers, limited for an API response."""
    return [int(index) + 2 for index in mask[mask].index[:limit]]


def validate_csv_upload(filename: str | None, content: bytes) -> dict[str, Any]:
    """Validate an uploaded normalized sensor CSV without storing or modelling it."""
    errors: list[ValidationIssue] = []

    if not filename or not filename.lower().endswith(".csv"):
        errors.append(_issue("invalid_file_type", "Upload a CSV file with a .csv extension."))
        return _report(filename, errors)
    if not content.strip():
        errors.append(_issue("empty_file", "The uploaded CSV is empty."))
        return _report(filename, errors)
    if len(content) > MAX_UPLOAD_BYTES:
        errors.append(_issue("file_too_large", f"The CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."))
        return _report(filename, errors)

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        errors.append(_issue("invalid_encoding", "The CSV must use UTF-8 encoding."))
        return _report(filename, errors)

    try:
        frame = pd.read_csv(StringIO(decoded))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        errors.append(_issue("unreadable_csv", f"The CSV could not be parsed: {exc}"))
        return _report(filename, errors)

    if frame.empty:
        errors.append(_issue("no_data_rows", "The CSV must contain at least one data row."))

    columns = frame.columns.tolist()
    missing = [column for column in REQUIRED_NON_SENSOR_COLUMNS if column not in columns]
    if missing:
        errors.append(_issue("missing_required_columns", "Required columns are missing.", columns=missing))

    sensors = sensor_columns(columns)
    if not sensors:
        errors.append(_issue("missing_sensor_columns", "Provide one or more sequential sensor columns: sensor_1 through sensor_N."))
    else:
        sensor_indices = [int(SENSOR_COLUMN_PATTERN.fullmatch(column).group(1)) for column in sensors]
        expected_indices = list(range(1, len(sensor_indices) + 1))
        if sensor_indices != expected_indices:
            errors.append(_issue("non_sequential_sensor_columns", "Sensor columns must be numbered continuously from sensor_1 to sensor_N.", columns=sensors))

    allowed = set(REQUIRED_NON_SENSOR_COLUMNS) | set(sensors)
    unexpected = [column for column in columns if column not in allowed]
    if unexpected:
        errors.append(_issue("unexpected_columns", "Only engine_id, cycle, setting_1 through setting_3, and sensor_1 through sensor_N are allowed in the normalized contract.", columns=unexpected))

    if missing or not sensors or frame.empty:
        return _report(filename, errors, frame=frame, sensors=sensors)

    required_columns = [*REQUIRED_NON_SENSOR_COLUMNS, *sensors]
    missing_values = frame[required_columns].isna()
    for column in required_columns:
        if missing_values[column].any():
            errors.append(_issue("missing_values", f"{column} contains missing values.", rows=_sample_rows(missing_values[column]), columns=[column]))

    numeric_columns = [CYCLE_COLUMN, *REQUIRED_NON_SENSOR_COLUMNS[2:], *sensors]
    numeric: dict[str, pd.Series] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric[column] = values
        invalid = values.isna() | ~np.isfinite(values)
        if invalid.any():
            errors.append(_issue("non_numeric_values", f"{column} must contain finite numeric values.", rows=_sample_rows(invalid), columns=[column]))

    engine_values = pd.to_numeric(frame[ENGINE_ID_COLUMN], errors="coerce")
    invalid_engine = engine_values.isna() | ~np.isfinite(engine_values) | (engine_values <= 0) | (engine_values % 1 != 0)
    if invalid_engine.any():
        errors.append(_issue("invalid_engine_id", "engine_id must be a positive integer.", rows=_sample_rows(invalid_engine), columns=[ENGINE_ID_COLUMN]))

    cycles = numeric[CYCLE_COLUMN]
    invalid_cycle = cycles.isna() | ~np.isfinite(cycles) | (cycles <= 0) | (cycles % 1 != 0)
    if invalid_cycle.any():
        errors.append(_issue("invalid_cycle", "cycle must be a positive integer.", rows=_sample_rows(invalid_cycle), columns=[CYCLE_COLUMN]))

    if not invalid_engine.any() and not invalid_cycle.any():
        key_frame = pd.DataFrame({ENGINE_ID_COLUMN: engine_values.astype("int64"), CYCLE_COLUMN: cycles.astype("int64")})
        duplicates = key_frame.duplicated(keep=False)
        if duplicates.any():
            errors.append(_issue("duplicate_engine_cycle", "Each engine_id/cycle pair must appear only once.", rows=_sample_rows(duplicates), columns=[ENGINE_ID_COLUMN, CYCLE_COLUMN]))

        non_monotonic = key_frame.groupby(ENGINE_ID_COLUMN, sort=False)[CYCLE_COLUMN].diff().le(0).fillna(False)
        if non_monotonic.any():
            errors.append(_issue("non_monotonic_cycles", "Cycles must be strictly increasing for each engine in CSV row order.", rows=_sample_rows(non_monotonic), columns=[ENGINE_ID_COLUMN, CYCLE_COLUMN]))

    return _report(filename, errors, frame=frame, sensors=sensors, engine_values=engine_values, cycles=cycles)


def _report(
    filename: str | None,
    errors: list[ValidationIssue],
    *,
    frame: pd.DataFrame | None = None,
    sensors: list[str] | None = None,
    engine_values: pd.Series | None = None,
    cycles: pd.Series | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if frame is not None:
        summary["row_count"] = int(len(frame))
        summary["column_count"] = int(len(frame.columns))
    if sensors:
        summary["sensor_count"] = len(sensors)
    if engine_values is not None and engine_values.notna().any():
        summary["engine_count"] = int(engine_values.dropna().nunique())
    if cycles is not None and cycles.notna().any():
        summary["cycle_range"] = {"min": int(cycles.min()), "max": int(cycles.max())}
    return {
        "filename": filename,
        "valid": not errors,
        "contract": "cmapss-normalized-v1",
        "errors": [issue.to_dict() for issue in errors],
        "summary": summary,
    }
