import unittest

import numpy as np
import pandas as pd

from app.evaluation.real_only_evaluation import (
    calculate_binary_metrics,
    construct_future_degradation_truth,
    evaluate_held_out_test_set,
    fit_future_degradation_target,
)
from app.models.real_only_anomaly import IsolationForestConfig, RealOnlyAnomalyPipeline
from app.preparation.early_features import extract_early_features
from app.preparation.splitting import split_by_engine
from .test_real_only_anomaly import make_sensor_frame


class RealOnlyEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_sensor_frame()
        self.split = split_by_engine(self.frame, seed=42)
        self.train_features = extract_early_features(self.split.train).features
        self.validation_features = extract_early_features(self.split.validation).features
        self.test_features = extract_early_features(self.split.test).features
        self.pipeline = RealOnlyAnomalyPipeline(
            isolation_config=IsolationForestConfig(n_estimators=80, random_state=17)
        ).fit(self.train_features, self.validation_features)
        self.target = fit_future_degradation_target(self.split.train)

    def test_metric_formulas(self) -> None:
        metrics = calculate_binary_metrics(
            np.array([True, True, False, False]),
            np.array([True, False, True, False]),
            np.array([0.9, 0.2, 0.8, 0.1]),
        )

        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertEqual(metrics["fnr"], 0.5)
        self.assertIsNotNone(metrics["pr_auc"])
        self.assertIsNotNone(metrics["roc_auc"])

    def test_ground_truth_uses_observed_future_endpoint(self) -> None:
        trajectories = pd.DataFrame({"engine_id": [1, 1, 2, 2], "cycle": [1, 50, 1, 100]})
        target = fit_future_degradation_target(trajectories, early_end_cycle=30, training_quantile=0.5)
        truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")

        self.assertEqual(target.remaining_cycle_horizon, 45.0)
        self.assertTrue(truth.loc[1, "near_term_degradation"])
        self.assertFalse(truth.loc[2, "near_term_degradation"])

    def test_test_set_isolation_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_held_out_test_set(self.pipeline, self.train_features, self.split.train, self.target)

    def test_future_sensor_changes_do_not_change_test_features_or_predictions(self) -> None:
        changed = self.split.test.copy()
        changed.loc[changed["cycle"] > 30, "sensor_1"] = 999999.0
        changed_features = extract_early_features(changed).features

        pd.testing.assert_frame_equal(self.test_features, changed_features)
        np.testing.assert_allclose(
            self.pipeline.predict(self.test_features)["risk_score"],
            self.pipeline.predict(changed_features)["risk_score"],
        )

    def test_evaluation_is_deterministic(self) -> None:
        first = evaluate_held_out_test_set(self.pipeline, self.test_features, self.split.test, self.target)
        second = evaluate_held_out_test_set(self.pipeline, self.test_features, self.split.test, self.target)

        self.assertEqual(first, second)
