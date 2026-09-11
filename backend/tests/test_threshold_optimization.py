"""Focused tests for validation-only threshold optimization."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.evaluation.threshold_optimization import (
    classify_risk_scores,
    evaluate_threshold,
    run_threshold_diagnostic,
    select_validation_threshold,
    validate_thresholds,
)


def _synthetic_frame(engine_count: int = 20):
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, 41):
            row = {"engine_id": engine_id, "cycle": cycle}
            for sensor in range(1, 22):
                row[f"sensor_{sensor}"] = engine_id * 0.1 + cycle * sensor * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


class _DeterministicThresholdPipeline:
    """Test-only score source with a controlled, non-degenerate validation spread."""

    class _Thresholds:
        watchlist = 0.807143
        high_risk = 0.885714

    thresholds = _Thresholds()

    def predict(self, features):
        scores = np.linspace(0.1, 0.9, len(features), dtype=float)
        return pd.DataFrame({"risk_score": scores})


class ThresholdOptimizationTests(unittest.TestCase):
    def test_threshold_order_and_degenerate_configurations_are_rejected(self) -> None:
        validate_thresholds(0.50, 0.80)
        with self.assertRaises(ValueError):
            validate_thresholds(0.80, 0.50)
        with self.assertRaises(ValueError):
            validate_thresholds(-0.1, 0.80)

    def test_statuses_are_derived_from_scores_and_thresholds(self) -> None:
        scores = np.array([0.2, 0.6, 0.9])
        np.testing.assert_array_equal(classify_risk_scores(scores, 0.5, 0.8), ["Normal", "Watchlist", "High Risk"])

    def test_risk_scores_are_unchanged_by_threshold_evaluation(self) -> None:
        scores = np.array([0.2, 0.6, 0.9], dtype=float)
        truth = np.array([False, True, True])
        before = scores.copy()
        evaluate_threshold(split_name="VALIDATION", model_name="test", scores=scores, truth=truth, watchlist=0.5, high_risk=0.8)
        np.testing.assert_array_equal(scores, before)

    def test_selection_is_deterministic_and_ignores_degenerate_candidates(self) -> None:
        scores = np.array([0.05, 0.15, 0.25, 0.55, 0.9, 0.95])
        truth = np.array([False, False, False, True, True, False])
        eligible = evaluate_threshold(split_name="VALIDATION", model_name="candidate", scores=scores, truth=truth, watchlist=0.55, high_risk=0.8)
        degenerate = evaluate_threshold(split_name="VALIDATION", model_name="degenerate", scores=scores, truth=truth, watchlist=0.1, high_risk=0.2)
        self.assertTrue(eligible.alert_rate_defensible)
        self.assertFalse(degenerate.alert_rate_defensible)
        first = select_validation_threshold([eligible, degenerate])
        second = select_validation_threshold([eligible, degenerate])
        self.assertEqual(first, second)
        self.assertEqual(first.model, "candidate")

    def test_test_engine_values_cannot_change_validation_selection(self) -> None:
        frame = _synthetic_frame()
        with patch(
            "app.evaluation.threshold_optimization.load_production_pipeline",
            return_value=_DeterministicThresholdPipeline(),
        ):
            first = run_threshold_diagnostic(frame)
            changed = frame.copy()
            test_ids = set(first.split_metadata.test_engine_ids)
            sensors = [column for column in changed.columns if column.startswith("sensor_")]
            changed.loc[changed["engine_id"].isin(test_ids) & changed["cycle"].le(30), sensors] = 1e8
            second = run_threshold_diagnostic(changed)
        self.assertEqual(first.selected_validation, second.selected_validation)
