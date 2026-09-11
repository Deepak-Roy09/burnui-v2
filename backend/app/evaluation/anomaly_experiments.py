"""Validation-only anomaly detector experiments, isolated from production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from app.evaluation.real_only_evaluation import (
    FutureDegradationTarget,
    calculate_binary_metrics,
    construct_future_degradation_truth,
    fit_future_degradation_target,
)
from app.models.real_only_anomaly import IsolationForestConfig, RealOnlyAnomalyPipeline, _empirical_percentile
from app.preparation.early_features import EarlyWindowConfig, FittedFeaturePreprocessor, extract_early_features, fit_feature_preprocessor
from app.preparation.splitting import EngineSplitMetadata, split_by_engine


EXPERIMENT_RANDOM_STATE = 42
EARLY_WINDOW = EarlyWindowConfig(early_start_cycle=1, early_end_cycle=30, minimum_early_cycles=30)
WATCHLIST_QUANTILE = 0.85


@dataclass(frozen=True)
class ValidationExperimentData:
    """The test split is purposefully not present in this experiment input."""

    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    validation_truth: np.ndarray
    target: FutureDegradationTarget
    split_metadata: EngineSplitMetadata


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    validation_threshold: float
    threshold_quantile: float
    threshold_calibration_engine_ids: tuple[int, ...]
    model_fit_engine_ids: tuple[int, ...]
    preprocessor_fit_engine_ids: tuple[int, ...]
    metrics: dict[str, int | float | None]
    notes: str

    def report_row(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "validation_threshold": self.validation_threshold,
            "threshold_quantile": self.threshold_quantile,
            **self.metrics,
            "notes": self.notes,
        }


class _PipelineScorer:
    def __init__(self, name: str, notes: str, config: IsolationForestConfig) -> None:
        self.name, self.notes, self.config = name, notes, config

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> None:
        self.pipeline = RealOnlyAnomalyPipeline(isolation_config=self.config).fit(train, validation)
        self.preprocessor = self.pipeline.preprocessor
        self.fit_engine_ids = self.pipeline.train_engine_ids
        self.validation_threshold = self.pipeline.thresholds.watchlist

    def risk_scores(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(features)["risk_score"].to_numpy(dtype=float)


class _RobustMadScorer:
    name = "robust_mad_max_deviation"
    notes = "Train-fitted median/MAD detector; validation 85th-percentile alert threshold."

    def fit(self, train: pd.DataFrame) -> None:
        self.preprocessor, matrix = fit_feature_preprocessor(train)
        values = matrix.to_numpy(dtype=float)
        self.center_ = np.median(values, axis=0)
        scale = 1.4826 * np.median(np.abs(values - self.center_), axis=0)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        self.reference_ = np.sort(self._raw(values))
        self.fit_engine_ids = self.preprocessor.fit_engine_ids

    def _raw(self, values: np.ndarray) -> np.ndarray:
        return np.max(np.abs((values - self.center_) / self.scale_), axis=1)

    def risk_scores(self, features: pd.DataFrame) -> np.ndarray:
        values = self.preprocessor.transform(features).to_numpy(dtype=float)
        return _empirical_percentile(self._raw(values), self.reference_)


class _LocalOutlierFactorScorer:
    name = "local_outlier_factor_novelty"
    notes = "Single additional novelty detector (20 neighbours); train fit and validation 85th-percentile calibration."

    def fit(self, train: pd.DataFrame) -> None:
        self.preprocessor, matrix = fit_feature_preprocessor(train)
        values = matrix.to_numpy(dtype=float)
        self.detector_ = LocalOutlierFactor(n_neighbors=min(20, len(values) - 1), novelty=True, n_jobs=1).fit(values)
        self.reference_ = np.sort(-self.detector_.score_samples(values))
        self.fit_engine_ids = self.preprocessor.fit_engine_ids

    def risk_scores(self, features: pd.DataFrame) -> np.ndarray:
        values = self.preprocessor.transform(features).to_numpy(dtype=float)
        return _empirical_percentile(-self.detector_.score_samples(values), self.reference_)


def prepare_validation_experiment(frame: pd.DataFrame) -> ValidationExperimentData:
    """Build train/validation-only data; test engines are never feature-extracted here."""
    split = split_by_engine(frame, seed=EXPERIMENT_RANDOM_STATE)
    train_result = extract_early_features(split.train, EARLY_WINDOW)
    validation_result = extract_early_features(split.validation, EARLY_WINDOW)
    if train_result.skipped_engines or validation_result.skipped_engines:
        raise ValueError("Every train and validation engine must provide cycles 1–30.")
    _validate_features(train_result.features)
    _validate_features(validation_result.features)
    target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=0.20)
    truth = construct_future_degradation_truth(split.validation, target).set_index("engine_id")
    validation_ids = validation_result.features["engine_id"].astype(int).to_numpy()
    if not set(validation_ids).issubset(set(int(value) for value in truth.index)):
        raise ValueError("A validation engine lacks a post-window target endpoint.")
    _assert_disjoint(split.metadata, train_result.features, validation_result.features, target)
    return ValidationExperimentData(
        train_features=train_result.features,
        validation_features=validation_result.features,
        validation_truth=truth.loc[validation_ids, "near_term_degradation"].to_numpy(dtype=bool),
        target=target,
        split_metadata=split.metadata,
    )


def run_validation_experiments(frame: pd.DataFrame) -> tuple[ValidationExperimentData, list[ExperimentResult]]:
    """Fit/evaluate fixed candidates using train and validation only."""
    data = prepare_validation_experiment(frame)
    candidates: list[object] = [
        _PipelineScorer(
            "production_baseline_zscore_isolation_forest",
            "Exact production 50/50 Z-score + Isolation Forest configuration.",
            IsolationForestConfig(),
        ),
        _PipelineScorer(
            "isolation_forest_400_estimators_80pct_samples",
            "Same 50/50 fusion; deterministic 400-tree, 80%-sample IF comparison.",
            IsolationForestConfig(n_estimators=400, max_samples=0.8, contamination="auto", random_state=EXPERIMENT_RANDOM_STATE),
        ),
        _RobustMadScorer(),
        _LocalOutlierFactorScorer(),
    ]
    results: list[ExperimentResult] = []
    for candidate in candidates:
        if isinstance(candidate, _PipelineScorer):
            candidate.fit(data.train_features, data.validation_features)
            threshold = candidate.validation_threshold
        else:
            candidate.fit(data.train_features)
            threshold = float(np.quantile(candidate.risk_scores(data.validation_features), WATCHLIST_QUANTILE))
        scores = candidate.risk_scores(data.validation_features)
        if not np.isfinite(scores).all():
            raise ValueError(f"{candidate.name} produced non-finite validation scores.")
        metrics = calculate_binary_metrics(data.validation_truth, scores >= threshold, scores)
        results.append(ExperimentResult(
            model_name=candidate.name,
            validation_threshold=float(threshold),
            threshold_quantile=WATCHLIST_QUANTILE,
            threshold_calibration_engine_ids=tuple(int(value) for value in data.validation_features["engine_id"]),
            model_fit_engine_ids=tuple(int(value) for value in candidate.fit_engine_ids),
            preprocessor_fit_engine_ids=tuple(int(value) for value in candidate.preprocessor.fit_engine_ids),
            metrics=metrics,
            notes=candidate.notes,
        ))
    return data, results


def select_validation_candidate(results: list[ExperimentResult]) -> ExperimentResult:
    """Apply the declared recall/FNR-first criterion; preserve order on exact ties."""
    if not results:
        raise ValueError("At least one result is required.")

    def value(metric: int | float | None, missing: float = -1.0) -> float:
        return missing if metric is None else float(metric)

    return max(results, key=lambda result: (
        value(result.metrics["recall"]),
        -value(result.metrics["fnr"], 1.0),
        value(result.metrics["precision"]),
        value(result.metrics["f1"]),
        value(result.metrics["pr_auc"]),
        -value(result.metrics["fpr"], 1.0),
        value(result.metrics["roc_auc"]),
    ))


def promotion_recommendation(results: list[ExperimentResult]) -> str:
    """Give a conservative validation-only result; this never changes artifacts."""
    baseline = next(item for item in results if item.model_name == "production_baseline_zscore_isolation_forest")
    selected = select_validation_candidate(results)
    if selected.model_name == baseline.model_name:
        return "Retain the production baseline: no candidate outranked it on validation."
    if float(selected.metrics["recall"] or 0.0) <= float(baseline.metrics["recall"] or 0.0):
        return "Retain the production baseline: the selected candidate did not improve validation recall."
    return f"{selected.model_name} merits separate Part 2 review only; production remains unchanged pending pre-registered test evaluation."


def _validate_features(features: pd.DataFrame) -> None:
    if features.shape[1] != 169:
        raise ValueError("Expected existing engine_id plus 168 early-cycle features.")
    if not np.isfinite(features.drop(columns="engine_id").to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix contains non-finite values.")


def _assert_disjoint(metadata: EngineSplitMetadata, train: pd.DataFrame, validation: pd.DataFrame, target: FutureDegradationTarget) -> None:
    train_ids = set(int(value) for value in train["engine_id"])
    validation_ids = set(int(value) for value in validation["engine_id"])
    test_ids = set(metadata.test_engine_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Experiment engine split overlap detected.")
    if validation_ids & set(target.training_engine_ids):
        raise ValueError("Validation engines must not calibrate the target horizon.")
