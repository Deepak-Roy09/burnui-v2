"""Focused validation-only tests for controlled anomaly-model experiments."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.evaluation.anomaly_experiments import prepare_validation_experiment, run_validation_experiments
from app.ml.production_training import load_production_pipeline, production_artifact_paths


def _synthetic_fd001_like_frame(engine_count: int = 20) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, 41):
            row: dict[str, float | int] = {"engine_id": engine_id, "cycle": cycle}
            for sensor_number in range(1, 22):
                row[f"sensor_{sensor_number}"] = engine_id * 0.1 + cycle * sensor_number * 0.001
            records.append(row)
    return pd.DataFrame.from_records(records)


class ValidationOnlyAnomalyExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = _synthetic_fd001_like_frame()

    def test_engine_split_is_disjoint_and_test_features_are_absent(self) -> None:
        data = prepare_validation_experiment(self.frame)
        metadata = data.split_metadata
        train_ids, validation_ids, test_ids = map(set, (metadata.train_engine_ids, metadata.validation_engine_ids, metadata.test_engine_ids))
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertTrue(set(data.train_features["engine_id"]).isdisjoint(test_ids))
        self.assertTrue(set(data.validation_features["engine_id"]).isdisjoint(test_ids))

    def test_future_cycles_do_not_enter_early_features(self) -> None:
        changed = self.frame.copy()
        sensors = [column for column in changed.columns if column.startswith("sensor_")]
        changed.loc[changed["cycle"] > 30, sensors] = 1e20
        original_data = prepare_validation_experiment(self.frame)
        changed_data = prepare_validation_experiment(changed)
        pd.testing.assert_frame_equal(original_data.train_features, changed_data.train_features)
        pd.testing.assert_frame_equal(original_data.validation_features, changed_data.validation_features)

    def test_results_are_deterministic_and_fit_preprocessing_on_train_only(self) -> None:
        first_data, first = run_validation_experiments(self.frame)
        second_data, second = run_validation_experiments(self.frame)
        self.assertEqual(first_data.split_metadata, second_data.split_metadata)
        self.assertEqual([item.report_row() for item in first], [item.report_row() for item in second])
        train_ids = tuple(int(value) for value in first_data.train_features["engine_id"])
        validation_ids = tuple(int(value) for value in first_data.validation_features["engine_id"])
        for result in first:
            self.assertEqual(result.model_fit_engine_ids, train_ids)
            self.assertEqual(result.preprocessor_fit_engine_ids, train_ids)
            self.assertEqual(result.threshold_calibration_engine_ids, validation_ids)

    def test_feature_matrix_and_calibrated_thresholds_are_finite(self) -> None:
        data, results = run_validation_experiments(self.frame)
        self.assertEqual(data.train_features.shape[1], 169)
        self.assertEqual(data.validation_features.shape[1], 169)
        for features in (data.train_features, data.validation_features):
            self.assertTrue(np.isfinite(features.drop(columns="engine_id").to_numpy(dtype=float)).all())
        for result in results:
            self.assertTrue(np.isfinite(result.validation_threshold))

    def test_saved_production_artifact_still_matches_existing_feature_schema(self) -> None:
        paths = production_artifact_paths()
        self.assertTrue(paths.model.is_file())
        pipeline = load_production_pipeline()
        self.assertEqual(len(pipeline.preprocessor.input_columns), 168)
        self.assertEqual(pipeline.z_weight, 0.5)
        self.assertEqual(pipeline.isolation_weight, 0.5)
