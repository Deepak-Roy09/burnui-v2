"""Held-out evaluation for the real-only anomaly model.

C-MAPSS is a prognostics/degradation benchmark, not a burn-in screening dataset.
For this prototype, the evaluation target is therefore a *near-term observed
run-to-failure* target: after the early window, an engine is positive when its
actual remaining cycles to the observed end of its C-MAPSS training trajectory
are at or below a horizon derived from training trajectories only. Future
cycles are never supplied to the anomaly model or its feature matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from app.models.real_only_anomaly import RealOnlyAnomalyPipeline


@dataclass(frozen=True)
class FutureDegradationTarget:
    early_end_cycle: int
    training_quantile: float
    remaining_cycle_horizon: float
    training_engine_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_future_degradation_target(
    training_trajectories: pd.DataFrame,
    *,
    early_end_cycle: int = 30,
    training_quantile: float = 0.20,
) -> FutureDegradationTarget:
    """Calibrate a near-term RUL horizon exclusively from training trajectories.

    The observed last cycle in C-MAPSS ``train`` trajectories is a run-to-failure
    endpoint. This function uses it only to create an evaluation target—not as
    an anomaly-model input, feature, score, or classification threshold.
    """
    if not 0 < training_quantile < 1:
        raise ValueError("training_quantile must be between 0 and 1.")
    remaining = _remaining_cycles(training_trajectories, early_end_cycle)
    if remaining.empty:
        raise ValueError("No training engines have an observed endpoint after the early window.")
    return FutureDegradationTarget(
        early_end_cycle=early_end_cycle,
        training_quantile=training_quantile,
        remaining_cycle_horizon=float(np.quantile(remaining["remaining_cycles"], training_quantile)),
        training_engine_ids=tuple(int(engine_id) for engine_id in remaining["engine_id"]),
    )


def construct_future_degradation_truth(
    trajectories: pd.DataFrame,
    target: FutureDegradationTarget,
) -> pd.DataFrame:
    """Construct labels from actual future cycle availability, never predictions."""
    remaining = _remaining_cycles(trajectories, target.early_end_cycle)
    remaining["near_term_degradation"] = remaining["remaining_cycles"] <= target.remaining_cycle_horizon
    return remaining


def evaluate_held_out_test_set(
    pipeline: RealOnlyAnomalyPipeline,
    test_features: pd.DataFrame,
    test_trajectories: pd.DataFrame,
    target: FutureDegradationTarget,
) -> dict[str, Any]:
    """Evaluate once on held-out engines using a pre-fit model and target rule."""
    test_engine_ids = tuple(int(engine_id) for engine_id in test_features["engine_id"])
    forbidden = set(pipeline.train_engine_ids) | set(pipeline.validation_engine_ids)
    if set(test_engine_ids) & forbidden:
        raise ValueError("Test engines overlap with model fitting or validation engines.")
    if set(test_engine_ids) & set(target.training_engine_ids):
        raise ValueError("Test engines overlap with engines used to define the target horizon.")

    truth = construct_future_degradation_truth(test_trajectories, target).set_index("engine_id")
    missing_truth = set(test_engine_ids) - set(int(value) for value in truth.index)
    if missing_truth:
        raise ValueError(f"Missing observed endpoints for test engines: {sorted(missing_truth)}")

    predictions = pipeline.predict(test_features).set_index("engine_id")
    aligned_truth = truth.loc[predictions.index]
    y_true = aligned_truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = predictions["risk_score"].to_numpy(dtype=float)
    y_pred = predictions["classification"].ne("Normal").to_numpy(dtype=bool)
    metrics = calculate_binary_metrics(y_true, y_pred, y_score)
    class_counts = {name: int((predictions["classification"] == name).sum()) for name in ("Normal", "Watchlist", "High Risk")}

    return {
        "target": target.to_dict(),
        "test_engine_ids": [int(engine_id) for engine_id in predictions.index],
        "test_engine_count": len(predictions),
        "prediction_class_counts": class_counts,
        "confusion_matrix": {
            "labels": ["not_near_term", "near_term"],
            "matrix": [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        },
        "metrics": metrics,
    }


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, int | float | None]:
    """Calculate transparent binary metrics; undefined AUC values are ``None``."""
    true = np.asarray(y_true, dtype=bool)
    predicted = np.asarray(y_pred, dtype=bool)
    scores = np.asarray(y_score, dtype=float)
    if not (len(true) == len(predicted) == len(scores)):
        raise ValueError("Truth, predictions, and scores must have equal lengths.")
    if not np.isfinite(scores).all():
        raise ValueError("Risk scores must be finite for evaluation.")

    tp = int(np.logical_and(true, predicted).sum())
    tn = int(np.logical_and(~true, ~predicted).sum())
    fp = int(np.logical_and(~true, predicted).sum())
    fn = int(np.logical_and(true, ~predicted).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    fpr = _ratio(fp, fp + tn)
    fnr = _ratio(fn, fn + tp)
    auc_available = len(np.unique(true)) == 2
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "pr_auc": float(average_precision_score(true, scores)) if auc_available else None,
        "roc_auc": float(roc_auc_score(true, scores)) if auc_available else None,
    }


def _remaining_cycles(trajectories: pd.DataFrame, early_end_cycle: int) -> pd.DataFrame:
    if not {"engine_id", "cycle"}.issubset(trajectories.columns):
        raise ValueError("trajectories must contain engine_id and cycle")
    endpoints = trajectories.groupby("engine_id", as_index=False)["cycle"].max().rename(columns={"cycle": "observed_final_cycle"})
    endpoints["remaining_cycles"] = endpoints["observed_final_cycle"] - early_end_cycle
    return endpoints.loc[endpoints["remaining_cycles"] >= 0].copy()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
