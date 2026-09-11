"""Validation-only threshold optimization for the saved BurnUI model.

This module changes no risk scores or production artifacts. Test engines are
not feature-extracted until after a validation threshold is locked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.evaluation.real_only_evaluation import calculate_binary_metrics, construct_future_degradation_truth, fit_future_degradation_target
from app.ml.production_training import load_production_pipeline, production_artifact_paths
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import EngineSplitMetadata, split_by_engine

RISK_GRID = tuple(round(value, 2) for value in np.arange(0.50, 1.00, 0.05))
MAX_VALIDATION_ALERT_RATE = 0.50
MIN_VALIDATION_NORMAL_RATE = 0.50


@dataclass(frozen=True)
class ThresholdEvaluation:
    evaluation_split: str
    model: str
    watchlist_threshold: float
    high_risk_threshold: float
    metrics: dict[str, int | float | None]
    normal_count: int
    watchlist_count: int
    high_risk_count: int
    alert_rate_defensible: bool

    def report_row(self) -> dict[str, Any]:
        return {"evaluation_split": self.evaluation_split, "model": self.model, "watchlist_threshold": self.watchlist_threshold, "high_risk_threshold": self.high_risk_threshold, **self.metrics, "normal_count": self.normal_count, "watchlist_count": self.watchlist_count, "high_risk_count": self.high_risk_count, "alert_rate_defensible": self.alert_rate_defensible}


@dataclass(frozen=True)
class ThresholdDiagnostic:
    baseline_validation: ThresholdEvaluation
    validation_candidates: tuple[ThresholdEvaluation, ...]
    selected_validation: ThresholdEvaluation
    baseline_test: ThresholdEvaluation
    selected_test: ThresholdEvaluation
    split_metadata: EngineSplitMetadata
    artifact_sha256_before: str
    artifact_sha256_after: str


def validate_thresholds(watchlist: float, high_risk: float) -> None:
    if not np.isfinite(watchlist) or not np.isfinite(high_risk) or not 0.0 <= watchlist < high_risk <= 1.0:
        raise ValueError("High Risk threshold must be greater than Watchlist and both must be finite values in [0, 1].")


def classify_risk_scores(scores: np.ndarray, watchlist: float, high_risk: float) -> np.ndarray:
    validate_thresholds(watchlist, high_risk)
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Risk scores must be finite.")
    return np.where(values >= high_risk, "High Risk", np.where(values >= watchlist, "Watchlist", "Normal"))


def evaluate_threshold(*, split_name: str, model_name: str, scores: np.ndarray, truth: np.ndarray, watchlist: float, high_risk: float) -> ThresholdEvaluation:
    statuses = classify_risk_scores(scores, watchlist, high_risk)
    normal_count = int(np.sum(statuses == "Normal"))
    watchlist_count = int(np.sum(statuses == "Watchlist"))
    high_risk_count = int(np.sum(statuses == "High Risk"))
    total = len(statuses)
    alert_rate = (watchlist_count + high_risk_count) / total if total else 1.0
    normal_rate = normal_count / total if total else 0.0
    metrics = calculate_binary_metrics(np.asarray(truth, dtype=bool), statuses != "Normal", np.asarray(scores, dtype=float))
    return ThresholdEvaluation(split_name, model_name, float(watchlist), float(high_risk), metrics, normal_count, watchlist_count, high_risk_count, alert_rate <= MAX_VALIDATION_ALERT_RATE and normal_rate >= MIN_VALIDATION_NORMAL_RATE)


def select_validation_threshold(candidates: list[ThresholdEvaluation]) -> ThresholdEvaluation:
    eligible = [candidate for candidate in candidates if candidate.alert_rate_defensible]
    if not eligible:
        raise ValueError("No non-degenerate validation threshold configuration was found.")

    def metric(candidate: ThresholdEvaluation, name: str, missing: float = -1.0) -> float:
        value = candidate.metrics[name]
        return missing if value is None else float(value)

    return max(eligible, key=lambda candidate: (metric(candidate, "recall"), -metric(candidate, "fnr", 1.0), metric(candidate, "precision"), metric(candidate, "f1"), -metric(candidate, "fpr", 1.0)))


def run_threshold_diagnostic(frame: pd.DataFrame) -> ThresholdDiagnostic:
    """Search validation cutoffs, then make one locked test scoring pass."""
    artifact = production_artifact_paths().model
    before = sha256_file(artifact)
    split = split_by_engine(frame, seed=42)
    config = EarlyWindowConfig(early_start_cycle=1, early_end_cycle=30, minimum_early_cycles=30)
    train_features = extract_early_features(split.train, config).features
    validation_features = extract_early_features(split.validation, config).features
    if train_features.empty or validation_features.empty:
        raise ValueError("Train and validation engines must provide early-cycle features.")

    target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=0.20)
    validation_truth = construct_future_degradation_truth(split.validation, target).set_index("engine_id")
    validation_ids = validation_features["engine_id"].astype(int).to_numpy()
    y_validation = validation_truth.loc[validation_ids, "near_term_degradation"].to_numpy(dtype=bool)
    pipeline = load_production_pipeline()
    validation_scores = pipeline.predict(validation_features)["risk_score"].to_numpy(dtype=float)
    baseline_watchlist = float(pipeline.thresholds.watchlist)
    baseline_high = float(pipeline.thresholds.high_risk)
    baseline_validation = evaluate_threshold(
        split_name="VALIDATION", model_name="production_baseline", scores=validation_scores,
        truth=y_validation, watchlist=baseline_watchlist, high_risk=baseline_high,
    )

    candidates: list[ThresholdEvaluation] = []
    pairs = [(baseline_watchlist, baseline_high)] + [(watchlist, high) for watchlist in RISK_GRID for high in RISK_GRID if watchlist < high]
    seen: set[tuple[float, float]] = set()
    for watchlist, high in pairs:
        pair = (round(float(watchlist), 6), round(float(high), 6))
        if pair in seen:
            continue
        seen.add(pair)
        candidates.append(evaluate_threshold(
            split_name="VALIDATION", model_name="threshold_candidate", scores=validation_scores,
            truth=y_validation, watchlist=watchlist, high_risk=high,
        ))
    selected = select_validation_threshold(candidates)

    # Test features and labels are first touched only after the validation winner is locked.
    test_features = extract_early_features(split.test, config).features
    test_ids = test_features["engine_id"].astype(int).to_numpy()
    test_truth = construct_future_degradation_truth(split.test, target).set_index("engine_id")
    y_test = test_truth.loc[test_ids, "near_term_degradation"].to_numpy(dtype=bool)
    test_scores = pipeline.predict(test_features)["risk_score"].to_numpy(dtype=float)
    baseline_test = evaluate_threshold(
        split_name="TEST", model_name="production_baseline", scores=test_scores, truth=y_test,
        watchlist=baseline_watchlist, high_risk=baseline_high,
    )
    selected_test = evaluate_threshold(
        split_name="TEST", model_name="validation_selected", scores=test_scores, truth=y_test,
        watchlist=selected.watchlist_threshold, high_risk=selected.high_risk_threshold,
    )
    after = sha256_file(artifact)
    if before != after:
        raise RuntimeError("Production model artifact changed during threshold diagnostic.")
    return ThresholdDiagnostic(baseline_validation, tuple(candidates), selected, baseline_test, selected_test, split.metadata, before, after)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()
