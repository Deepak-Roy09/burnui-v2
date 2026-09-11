import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.api.routes.drift_evaluation import run_drift_evaluation
from app.evaluation.drift_prediction import (
    TARGET_CYCLE,
    calculate_mae,
    evaluate_drift_prediction,
    extract_drift_input_features,
    fit_drift_predictor,
    future_sensor_targets,
)
from app.main import create_app
from app.preparation.splitting import split_by_engine


def make_drift_frame(engine_count: int = 100) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, TARGET_CYCLE + 1):
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            for sensor_number in range(1, 22):
                row[f"sensor_{sensor_number}"] = (
                    engine_id * sensor_number * 0.02
                    + cycle * sensor_number * 0.003
                    + ((engine_id + cycle) % (sensor_number + 3)) * 0.001
                )
            rows.append(row)
    return pd.DataFrame(rows)


class DriftPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = make_drift_frame()
        cls.result = evaluate_drift_prediction(cls.frame)

    def test_engine_level_split_is_separate_and_test_remains_held_out(self) -> None:
        train_ids = set(self.result.train_engine_ids)
        validation_ids = set(self.result.validation_engine_ids)
        test_ids = set(self.result.engine_ids)

        self.assertEqual((len(train_ids), len(validation_ids), len(test_ids)), (70, 15, 15))
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_all_evaluated_engines_have_a_finite_target_and_none_are_excluded(self) -> None:
        self.assertEqual(self.result.usable_engine_count, 100)
        self.assertEqual(self.result.excluded_engine_count, 0)
        self.assertIn("No engines excluded", self.result.exclusion_reason)
        self.assertEqual(len(self.result.engine_ids), self.result.test_engine_count)
        self.assertTrue(np.isfinite(self.result.actual_values).all())

    def test_future_target_cycle_is_not_in_input_features(self) -> None:
        baseline = extract_drift_input_features(self.frame)
        changed = self.frame.copy()
        changed.loc[changed["cycle"] == TARGET_CYCLE, "sensor_12"] = 999999.0
        changed_features = extract_drift_input_features(changed)

        pd.testing.assert_frame_equal(baseline, changed_features)

    def test_preprocessing_is_fitted_on_training_engines_only(self) -> None:
        split = split_by_engine(self.frame, seed=42)
        train_features = extract_drift_input_features(split.train)
        predictor = fit_drift_predictor(train_features, future_sensor_targets(split.train))

        self.assertEqual(set(predictor.preprocessor.fit_engine_ids), set(split.metadata.train_engine_ids))
        self.assertTrue(set(predictor.preprocessor.fit_engine_ids).isdisjoint(split.metadata.validation_engine_ids))
        self.assertTrue(set(predictor.preprocessor.fit_engine_ids).isdisjoint(split.metadata.test_engine_ids))

    def test_predictions_are_deterministic(self) -> None:
        repeated = evaluate_drift_prediction(self.frame)

        np.testing.assert_allclose(self.result.predicted_values, repeated.predicted_values)
        np.testing.assert_allclose(self.result.actual_values, repeated.actual_values)
        self.assertAlmostEqual(self.result.mae, repeated.mae)

    def test_mae_uses_mean_absolute_prediction_error(self) -> None:
        self.assertEqual(calculate_mae(np.array([1.0, 5.0]), np.array([3.0, 2.0])), 2.5)
        independent = float(np.mean(np.abs(np.asarray(self.result.predicted_values) - np.asarray(self.result.actual_values))))
        self.assertAlmostEqual(self.result.mae, independent)
        self.assertAlmostEqual(self.result.mae, self.result.recalculated_mae)

    def test_fixed_training_mean_baseline_uses_no_validation_or_test_targets(self) -> None:
        split = split_by_engine(self.frame, seed=42)
        training_mean = float(future_sensor_targets(split.train).mean())
        test_actuals = future_sensor_targets(split.test).loc[list(self.result.engine_ids)].to_numpy(dtype=float)
        expected_mae = calculate_mae(np.full(len(test_actuals), training_mean), test_actuals)

        self.assertEqual(self.result.baseline_model_name, "Training-target mean baseline")
        self.assertAlmostEqual(self.result.training_target_mean, training_mean)
        self.assertAlmostEqual(self.result.baseline_mae, expected_mae)

    def test_api_returns_isolated_drift_evaluation_structure(self) -> None:
        self.assertIn("/api/drift-evaluation/run", create_app().openapi()["paths"])

        with patch("app.api.routes.drift_evaluation.run_fd001_drift_evaluation", return_value=self.result):
            response = run_drift_evaluation()

        expected_keys = {
            "description",
            "target_sensor",
            "target_cycle",
            "input_window",
            "model_name",
            "train_engine_count",
            "validation_engine_count",
            "test_engine_count",
            "validation_mae",
            "mae",
            "recalculated_mae",
            "baseline_model_name",
            "baseline_validation_mae",
            "baseline_mae",
            "usable_engine_count",
            "excluded_engine_count",
            "exclusion_reason",
            "engine_ids",
            "predicted_values",
            "actual_values",
        }
        self.assertEqual(set(response), expected_keys)
        self.assertEqual(len(response["engine_ids"]), response["test_engine_count"])
        self.assertEqual(len(response["predicted_values"]), len(response["actual_values"]))
