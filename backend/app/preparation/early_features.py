"""Early-cycle feature extraction and train-fitted preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from app.datasets.contract import sensor_columns


@dataclass(frozen=True)
class EarlyWindowConfig:
    early_start_cycle: int = 1
    early_end_cycle: int = 30
    minimum_early_cycles: int = 30

    def __post_init__(self) -> None:
        if self.early_start_cycle < 1 or self.early_end_cycle < self.early_start_cycle:
            raise ValueError("early cycle bounds must be positive and ordered.")
        if self.minimum_early_cycles < 1:
            raise ValueError("minimum_early_cycles must be positive.")


@dataclass(frozen=True)
class EarlyFeatureResult:
    features: pd.DataFrame
    skipped_engines: dict[int, int]
    config: EarlyWindowConfig


def extract_early_features(frame: pd.DataFrame, config: EarlyWindowConfig = EarlyWindowConfig()) -> EarlyFeatureResult:
    """Create compact per-engine sensor summaries using no cycles outside the window."""
    required = {"engine_id", "cycle"}
    if not required.issubset(frame.columns):
        raise ValueError("frame must contain engine_id and cycle")
    sensors = sensor_columns(frame.columns.tolist())
    if not sensors:
        raise ValueError("frame must contain at least one normalized sensor column")

    window = frame.loc[frame["cycle"].between(config.early_start_cycle, config.early_end_cycle), ["engine_id", "cycle", *sensors]].copy()
    records: list[dict[str, float | int]] = []
    skipped: dict[int, int] = {}
    for engine_id in sorted(frame["engine_id"].dropna().unique()):
        engine_window = window.loc[window["engine_id"] == engine_id]
        engine_window = engine_window.sort_values("cycle", kind="stable")
        unique_cycles = int(engine_window["cycle"].nunique())
        if unique_cycles < config.minimum_early_cycles:
            skipped[int(engine_id)] = unique_cycles
            continue
        row: dict[str, float | int] = {"engine_id": int(engine_id)}
        cycles = engine_window["cycle"].to_numpy(dtype=float)
        for sensor in sensors:
            values = pd.to_numeric(engine_window[sensor], errors="coerce").to_numpy(dtype=float)
            row[f"{sensor}_mean"] = float(np.nanmean(values))
            row[f"{sensor}_std"] = float(np.nanstd(values))
            row[f"{sensor}_min"] = float(np.nanmin(values))
            row[f"{sensor}_max"] = float(np.nanmax(values))
            row[f"{sensor}_first"] = float(values[0])
            row[f"{sensor}_last"] = float(values[-1])
            row[f"{sensor}_delta"] = float(values[-1] - values[0])
            row[f"{sensor}_slope"] = float(np.polyfit(cycles, values, deg=1)[0])
        records.append(row)

    feature_columns = [f"{sensor}_{stat}" for sensor in sensors for stat in ("mean", "std", "min", "max", "first", "last", "delta", "slope")]
    features = pd.DataFrame(records, columns=["engine_id", *feature_columns])
    return EarlyFeatureResult(features=features, skipped_engines=skipped, config=config)


@dataclass
class FittedFeaturePreprocessor:
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    fit_engine_ids: tuple[int, ...]
    imputer: SimpleImputer
    variance_filter: VarianceThreshold
    scaler: StandardScaler

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        _validate_feature_columns(features, self.input_columns)
        raw = features.loc[:, self.input_columns].replace([np.inf, -np.inf], np.nan)
        imputed = self.imputer.transform(raw)
        selected = self.variance_filter.transform(imputed)
        scaled = self.scaler.transform(selected)
        if not np.isfinite(scaled).all():
            raise ValueError("Preprocessing produced non-finite values.")
        return pd.DataFrame(scaled, columns=self.output_columns, index=features["engine_id"].to_numpy())


def fit_feature_preprocessor(train_features: pd.DataFrame) -> tuple[FittedFeaturePreprocessor, pd.DataFrame]:
    """Fit imputation, constant-feature removal, and scaling on training engines only."""
    if "engine_id" not in train_features.columns or train_features.empty:
        raise ValueError("train_features must contain at least one engine_id row")
    input_columns = tuple(column for column in train_features.columns if column != "engine_id")
    if not input_columns:
        raise ValueError("train_features contains no feature columns")
    raw = train_features.loc[:, input_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(raw)
    variance_filter = VarianceThreshold(threshold=0.0)
    selected = variance_filter.fit_transform(imputed)
    support = variance_filter.get_support()
    output_columns = tuple(column for column, keep in zip(input_columns, support) if keep)
    if not output_columns:
        raise ValueError("All training features are constant or unusable.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(selected)
    preprocessor = FittedFeaturePreprocessor(
        input_columns=input_columns,
        output_columns=output_columns,
        fit_engine_ids=tuple(int(value) for value in train_features["engine_id"]),
        imputer=imputer,
        variance_filter=variance_filter,
        scaler=scaler,
    )
    train_matrix = pd.DataFrame(scaled, columns=output_columns, index=train_features["engine_id"].to_numpy())
    return preprocessor, train_matrix


def save_feature_preprocessor(preprocessor: FittedFeaturePreprocessor, path: str) -> None:
    """Persist the train-fitted object for later inference; no data is refit."""
    joblib.dump(preprocessor, path)


def _validate_feature_columns(features: pd.DataFrame, expected_columns: tuple[str, ...]) -> None:
    received = tuple(column for column in features.columns if column != "engine_id")
    if received != expected_columns:
        raise ValueError("Feature columns must exactly match the training feature columns and order.")
