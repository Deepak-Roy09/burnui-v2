"""Isolated C-MAPSS drift-prediction evaluation.

C-MAPSS is a publicly available time-series degradation/prognostics proxy. It
is not ISRO component burn-in data and this module is not used by BurnUI's
production anomaly-screening decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.ml.production_training import load_fd001_training_data
from app.preparation.early_features import (
    EarlyWindowConfig,
    FittedFeaturePreprocessor,
    extract_early_features,
    fit_feature_preprocessor,
)
from app.preparation.splitting import split_by_engine


TARGET_SENSOR = "sensor_12"
TARGET_CYCLE = 50
EARLY_START_CYCLE = 1
EARLY_END_CYCLE = 30
MINIMUM_EARLY_CYCLES = 30
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
MODEL_NAME = "Ridge regression (alpha=1.0)"
BASELINE_MODEL_NAME = "Training-target mean baseline"
DEFAULT_CMAPSS_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")


@dataclass(frozen=True)
class FittedDriftPredictor:
    """A train-fitted preprocessing object and the isolated drift regressor."""

    preprocessor: FittedFeaturePreprocessor
    model: Ridge
    train_engine_ids: tuple[int, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(features)
        return self.model.predict(matrix.to_numpy(dtype=float))


@dataclass(frozen=True)
class DriftPredictionEvaluation:
    """Final held-out prediction results and metadata suitable for later UI use."""

    target_sensor: str
    target_cycle: int
    input_window_start_cycle: int
    input_window_end_cycle: int
    model_name: str
    train_engine_count: int
    validation_engine_count: int
    test_engine_count: int
    validation_mae: float
    mae: float
    recalculated_mae: float
    baseline_model_name: str
    baseline_validation_mae: float
    baseline_mae: float
    training_target_mean: float
    usable_engine_count: int
    excluded_engine_count: int
    exclusion_reason: str
    engine_ids: tuple[int, ...]
    predicted_values: tuple[float, ...]
    actual_values: tuple[float, ...]
    train_engine_ids: tuple[int, ...]
    validation_engine_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "description": (
                "C-MAPSS is a publicly available time-series degradation/prognostics proxy, "
                "not actual ISRO component burn-in data. This drift evaluation is separate from anomaly screening."
            ),
            "target_sensor": self.target_sensor,
            "target_cycle": self.target_cycle,
            "input_window": {"start_cycle": self.input_window_start_cycle, "end_cycle": self.input_window_end_cycle},
            "model_name": self.model_name,
            "train_engine_count": self.train_engine_count,
            "validation_engine_count": self.validation_engine_count,
            "test_engine_count": self.test_engine_count,
            "validation_mae": self.validation_mae,
            "mae": self.mae,
            "recalculated_mae": self.recalculated_mae,
            "baseline_model_name": self.baseline_model_name,
            "baseline_validation_mae": self.baseline_validation_mae,
            "baseline_mae": self.baseline_mae,
            "usable_engine_count": self.usable_engine_count,
            "excluded_engine_count": self.excluded_engine_count,
            "exclusion_reason": self.exclusion_reason,
            "engine_ids": list(self.engine_ids),
            "predicted_values": list(self.predicted_values),
            "actual_values": list(self.actual_values),
        }


def extract_drift_input_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create existing feature summaries from cycles 1-30 only.

    The future target at cycle 50 is deliberately outside this input window.
    """
    result = extract_early_features(
        frame,
        EarlyWindowConfig(
            early_start_cycle=EARLY_START_CYCLE,
            early_end_cycle=EARLY_END_CYCLE,
            minimum_early_cycles=MINIMUM_EARLY_CYCLES,
        ),
    )
    if result.skipped_engines:
        raise ValueError(f"Every engine requires cycles 1-{EARLY_END_CYCLE}; insufficient data: {result.skipped_engines}")
    return result.features


def future_sensor_targets(
    frame: pd.DataFrame,
    *,
    target_sensor: str = TARGET_SENSOR,
    target_cycle: int = TARGET_CYCLE,
) -> pd.Series:
    """Read actual later sensor values for evaluation targets, never model inputs."""
    if target_sensor not in frame.columns:
        raise ValueError(f"Frame does not contain the target sensor {target_sensor}.")
    engine_ids = set(int(value) for value in frame["engine_id"].dropna().unique())
    at_target_cycle = frame.loc[frame["cycle"] == target_cycle, ["engine_id", target_sensor]].copy()
    if at_target_cycle["engine_id"].duplicated().any():
        raise ValueError(f"More than one {target_sensor} value exists at cycle {target_cycle} for an engine.")
    targets = pd.to_numeric(at_target_cycle.set_index("engine_id")[target_sensor], errors="raise")
    if not np.isfinite(targets.to_numpy(dtype=float)).all():
        raise ValueError(f"Target sensor {target_sensor} contains non-finite values at cycle {target_cycle}.")
    missing = engine_ids - set(int(value) for value in targets.index)
    if missing:
        raise ValueError(f"Target cycle {target_cycle} is unavailable for engines: {sorted(missing)}")
    return targets.sort_index()


def calculate_mae(predicted_values: np.ndarray, actual_values: np.ndarray) -> float:
    """Calculate mean(abs(predicted future value - actual future value))."""
    predicted = np.asarray(predicted_values, dtype=float)
    actual = np.asarray(actual_values, dtype=float)
    if len(predicted) != len(actual) or not len(predicted):
        raise ValueError("Predicted and actual arrays must be non-empty and equal in length.")
    return float(np.mean(np.abs(predicted - actual)))


def fit_drift_predictor(train_features: pd.DataFrame, train_targets: pd.Series) -> FittedDriftPredictor:
    """Fit preprocessing and Ridge only on real training-engine feature rows."""
    train_ids = tuple(int(value) for value in train_features["engine_id"])
    targets = _align_targets(train_targets, train_ids)
    preprocessor, train_matrix = fit_feature_preprocessor(train_features)
    model = Ridge(alpha=RIDGE_ALPHA).fit(train_matrix.to_numpy(dtype=float), targets.to_numpy(dtype=float))
    return FittedDriftPredictor(preprocessor=preprocessor, model=model, train_engine_ids=train_ids)


def evaluate_drift_prediction(frame: pd.DataFrame) -> DriftPredictionEvaluation:
    """Run a leakage-safe train/validation/test evaluation on FD001-style trajectories."""
    split = split_by_engine(frame, seed=RANDOM_STATE)
    _assert_disjoint_split(split.metadata.train_engine_ids, split.metadata.validation_engine_ids, split.metadata.test_engine_ids)

    # Training and validation are prepared and evaluated before test features or
    # future test targets are accessed.
    train_input_trajectories = split.train.loc[split.train["cycle"].between(EARLY_START_CYCLE, EARLY_END_CYCLE)].copy()
    validation_input_trajectories = split.validation.loc[
        split.validation["cycle"].between(EARLY_START_CYCLE, EARLY_END_CYCLE)
    ].copy()
    train_features = extract_drift_input_features(train_input_trajectories)
    validation_features = extract_drift_input_features(validation_input_trajectories)
    train_targets = future_sensor_targets(split.train)
    validation_targets = future_sensor_targets(split.validation)
    predictor = fit_drift_predictor(train_features, train_targets)
    if set(predictor.preprocessor.fit_engine_ids) != set(split.metadata.train_engine_ids):
        raise AssertionError("Drift preprocessing must be fitted on training engines only.")
    validation_predictions = predictor.predict(validation_features)
    validation_actuals = _align_targets(validation_targets, tuple(int(value) for value in validation_features["engine_id"]))
    validation_mae = calculate_mae(validation_predictions, validation_actuals.to_numpy(dtype=float))
    training_target_mean = float(train_targets.mean())
    baseline_validation_predictions = np.full(len(validation_actuals), training_target_mean, dtype=float)
    baseline_validation_mae = calculate_mae(baseline_validation_predictions, validation_actuals.to_numpy(dtype=float))

    # Held-out test engines are first used here, after model fitting and the
    # validation-only diagnostic have completed.
    test_input_trajectories = split.test.loc[split.test["cycle"].between(EARLY_START_CYCLE, EARLY_END_CYCLE)].copy()
    test_features = extract_drift_input_features(test_input_trajectories)
    test_targets = future_sensor_targets(split.test)
    test_ids = tuple(int(value) for value in test_features["engine_id"])
    test_predictions = predictor.predict(test_features)
    test_actuals = _align_targets(test_targets, test_ids)
    mae = calculate_mae(test_predictions, test_actuals.to_numpy(dtype=float))
    recalculated_mae = float(np.mean(np.abs(np.asarray(test_predictions, dtype=float) - test_actuals.to_numpy(dtype=float))))
    if not np.isclose(mae, recalculated_mae):
        raise AssertionError("Reported MAE does not match the independent mean absolute error calculation.")
    baseline_test_predictions = np.full(len(test_actuals), training_target_mean, dtype=float)
    baseline_mae = calculate_mae(baseline_test_predictions, test_actuals.to_numpy(dtype=float))

    return DriftPredictionEvaluation(
        target_sensor=TARGET_SENSOR,
        target_cycle=TARGET_CYCLE,
        input_window_start_cycle=EARLY_START_CYCLE,
        input_window_end_cycle=EARLY_END_CYCLE,
        model_name=MODEL_NAME,
        train_engine_count=len(split.metadata.train_engine_ids),
        validation_engine_count=len(split.metadata.validation_engine_ids),
        test_engine_count=len(split.metadata.test_engine_ids),
        validation_mae=validation_mae,
        mae=mae,
        recalculated_mae=recalculated_mae,
        baseline_model_name=BASELINE_MODEL_NAME,
        baseline_validation_mae=baseline_validation_mae,
        baseline_mae=baseline_mae,
        training_target_mean=training_target_mean,
        usable_engine_count=len(split.metadata.train_engine_ids)
        + len(split.metadata.validation_engine_ids)
        + len(split.metadata.test_engine_ids),
        excluded_engine_count=0,
        exclusion_reason=(
            f"No engines excluded: every evaluated engine has cycles {EARLY_START_CYCLE}-{EARLY_END_CYCLE} "
            f"and a finite {TARGET_SENSOR} value at cycle {TARGET_CYCLE}."
        ),
        engine_ids=test_ids,
        predicted_values=tuple(float(value) for value in test_predictions),
        actual_values=tuple(float(value) for value in test_actuals),
        train_engine_ids=split.metadata.train_engine_ids,
        validation_engine_ids=split.metadata.validation_engine_ids,
    )


def run_fd001_drift_evaluation(zip_path: Path = DEFAULT_CMAPSS_ZIP) -> DriftPredictionEvaluation:
    """Load FD001 and execute this isolated research/evaluation workflow."""
    return evaluate_drift_prediction(load_fd001_training_data(zip_path))


def _align_targets(targets: pd.Series, engine_ids: tuple[int, ...]) -> pd.Series:
    missing = set(engine_ids) - set(int(value) for value in targets.index)
    if missing:
        raise ValueError(f"Targets are missing feature engines: {sorted(missing)}")
    return targets.loc[list(engine_ids)]


def _assert_disjoint_split(
    train_engine_ids: tuple[int, ...],
    validation_engine_ids: tuple[int, ...],
    test_engine_ids: tuple[int, ...],
) -> None:
    train_ids, validation_ids, test_ids = map(set, (train_engine_ids, validation_engine_ids, test_engine_ids))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Drift evaluation requires disjoint engine-level splits.")
