"""Leakage-safe, unsupervised anomaly screening for early-cycle engine features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.preparation.early_features import FittedFeaturePreprocessor, fit_feature_preprocessor


@dataclass(frozen=True)
class IsolationForestConfig:
    n_estimators: int = 200
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    random_state: int = 42


@dataclass(frozen=True)
class Thresholds:
    watchlist: float
    high_risk: float
    watchlist_quantile: float
    high_risk_quantile: float


class RealOnlyAnomalyPipeline:
    """Train-only anomaly model with validation-only classification calibration.

    The pipeline receives feature tables from Step 3. Its ``fit`` method has no
    test-data argument by design: it can only fit on training engines and
    calibrate class cutoffs from validation engines.
    """

    def __init__(
        self,
        *,
        isolation_config: IsolationForestConfig = IsolationForestConfig(),
        z_weight: float = 0.5,
        isolation_weight: float = 0.5,
        watchlist_quantile: float = 0.85,
        high_risk_quantile: float = 0.95,
    ) -> None:
        if z_weight < 0 or isolation_weight < 0 or z_weight + isolation_weight <= 0:
            raise ValueError("Detector weights must be non-negative with a positive total.")
        if not 0 < watchlist_quantile < high_risk_quantile < 1:
            raise ValueError("Quantiles must satisfy 0 < watchlist < high_risk < 1.")
        self.isolation_config = isolation_config
        self.z_weight = z_weight / (z_weight + isolation_weight)
        self.isolation_weight = isolation_weight / (z_weight + isolation_weight)
        self.watchlist_quantile = watchlist_quantile
        self.high_risk_quantile = high_risk_quantile

    def fit(self, train_features: pd.DataFrame, validation_features: pd.DataFrame) -> "RealOnlyAnomalyPipeline":
        """Fit exclusively on training engines, then calibrate thresholds on validation."""
        self.preprocessor, train_matrix = fit_feature_preprocessor(train_features)
        validation_matrix = self.preprocessor.transform(validation_features)
        self.train_engine_ids = tuple(int(value) for value in train_features["engine_id"])
        self.validation_engine_ids = tuple(int(value) for value in validation_features["engine_id"])
        if set(self.train_engine_ids) & set(self.validation_engine_ids):
            raise ValueError("Training and validation engine IDs must be disjoint.")

        self.z_center_ = train_matrix.mean(axis=0).to_numpy(dtype=float)
        self.z_scale_ = train_matrix.std(axis=0, ddof=0).to_numpy(dtype=float)
        self.z_scale_[self.z_scale_ < 1e-12] = 1.0
        self.isolation_forest_ = IsolationForest(
            n_estimators=self.isolation_config.n_estimators,
            max_samples=self.isolation_config.max_samples,
            contamination=self.isolation_config.contamination,
            random_state=self.isolation_config.random_state,
            n_jobs=1,
        ).fit(train_matrix.to_numpy(dtype=float))

        train_z, train_iso = self._raw_scores(train_matrix)
        self.train_z_reference_ = np.sort(train_z)
        self.train_isolation_reference_ = np.sort(train_iso)
        validation_risk = self._combined_risk(*self._raw_scores(validation_matrix))[-1]
        self.thresholds = Thresholds(
            watchlist=float(np.quantile(validation_risk, self.watchlist_quantile)),
            high_risk=float(np.quantile(validation_risk, self.high_risk_quantile)),
            watchlist_quantile=self.watchlist_quantile,
            high_risk_quantile=self.high_risk_quantile,
        )
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score engine features without fitting or recalibrating any parameter."""
        self._require_fitted()
        matrix = self.preprocessor.transform(features)
        z_raw, isolation_raw = self._raw_scores(matrix)
        z_risk, isolation_risk, combined = self._combined_risk(z_raw, isolation_raw)
        classifications = [self._classify(value) for value in combined]
        explanations = [
            self._explain(features.iloc[index], matrix.iloc[index], float(z_raw[index]), float(isolation_raw[index]))
            for index in range(len(features))
        ]
        return pd.DataFrame(
            {
                "engine_id": features["engine_id"].to_numpy(),
                "z_score": z_raw,
                "z_risk": z_risk,
                "isolation_score": isolation_raw,
                "isolation_risk": isolation_risk,
                "risk_score": combined,
                "classification": classifications,
                "explanation": explanations,
            }
        )

    def _raw_scores(self, matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = matrix.to_numpy(dtype=float)
        z_raw = np.max(np.abs((values - self.z_center_) / self.z_scale_), axis=1)
        isolation_raw = -self.isolation_forest_.score_samples(values)
        return z_raw, isolation_raw

    def _combined_risk(self, z_raw: np.ndarray, isolation_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_risk = _empirical_percentile(z_raw, self.train_z_reference_)
        isolation_risk = _empirical_percentile(isolation_raw, self.train_isolation_reference_)
        combined = np.clip(self.z_weight * z_risk + self.isolation_weight * isolation_risk, 0.0, 1.0)
        return z_risk, isolation_risk, combined

    def _classify(self, risk_score: float) -> str:
        if risk_score >= self.thresholds.high_risk:
            return "High Risk"
        if risk_score >= self.thresholds.watchlist:
            return "Watchlist"
        return "Normal"

    def _explain(self, feature_row: pd.Series, transformed_row: pd.Series, z_raw: float, isolation_raw: float) -> dict[str, Any]:
        contributions = np.abs((transformed_row.to_numpy(dtype=float) - self.z_center_) / self.z_scale_)
        positions = np.argsort(contributions)[::-1][:3]
        top_features = []
        for position in positions:
            name = str(transformed_row.index[position])
            top_features.append(
                {
                    "feature": name,
                    "value": float(feature_row[name]),
                    "z_contribution": float(contributions[position]),
                }
            )
        return {
            "top_features": top_features,
            "z_score": z_raw,
            "isolation_score": isolation_raw,
            "summary": "Top features are ordered by their calculated training-baseline Z-score contribution.",
        }

    def _require_fitted(self) -> None:
        required = ("preprocessor", "z_center_", "z_scale_", "isolation_forest_", "train_z_reference_", "train_isolation_reference_", "thresholds")
        if not all(hasattr(self, attribute) for attribute in required):
            raise RuntimeError("Fit the anomaly pipeline before prediction.")


def save_anomaly_pipeline(pipeline: RealOnlyAnomalyPipeline, path: str | Path) -> None:
    """Persist the fitted, train-calibrated pipeline for later inference."""
    pipeline._require_fitted()
    joblib.dump(pipeline, path)


def load_anomaly_pipeline(path: str | Path) -> RealOnlyAnomalyPipeline:
    """Load a persisted pipeline without fitting or recalibrating it."""
    pipeline = joblib.load(path)
    if not isinstance(pipeline, RealOnlyAnomalyPipeline):
        raise TypeError("Saved object is not a RealOnlyAnomalyPipeline.")
    pipeline._require_fitted()
    return pipeline


def _empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map detector scores to [0, 1] using the training score distribution."""
    return np.searchsorted(reference, values, side="right") / len(reference)
