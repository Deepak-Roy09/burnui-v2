import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.production_training import (
    FEATURE_COUNT,
    load_production_pipeline,
    production_artifact_paths,
    save_production_artifacts,
    train_production_pipeline,
)


def make_fd001_like_frame(engine_count: int = 100) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, 31):
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            for sensor_number in range(1, 22):
                row[f"sensor_{sensor_number}"] = (
                    engine_id * sensor_number * 0.01
                    + cycle * sensor_number * 0.001
                    + ((engine_id + cycle) % (sensor_number + 2)) * 0.0001
                )
            rows.append(row)
    return pd.DataFrame(rows)


class ProductionTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = train_production_pipeline(make_fd001_like_frame())

    def test_saved_model_loads_and_predictions_match_fitted_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_directory = Path(directory)
            paths = save_production_artifacts(self.result, artifact_directory)
            loaded = load_production_pipeline(artifact_directory)

            self.assertTrue(paths.model.is_file())
            self.assertTrue(paths.metadata.is_file())
            expected = self.result.pipeline.predict(self.result.validation_features)
            actual = loaded.predict(self.result.validation_features)

        np.testing.assert_allclose(expected["risk_score"], actual["risk_score"])
        self.assertListEqual(expected["classification"].tolist(), actual["classification"].tolist())

    def test_metadata_records_the_expected_168_feature_production_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_directory = Path(directory)
            save_production_artifacts(self.result, artifact_directory)
            metadata_path = production_artifact_paths(artifact_directory).metadata
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["dataset"], "FD001")
        self.assertEqual(metadata["early_end_cycle"], 30)
        self.assertEqual(metadata["feature_count"], FEATURE_COUNT)
        self.assertEqual(len(metadata["feature_columns"]), FEATURE_COUNT)
        self.assertEqual(metadata["random_state"], 42)
        self.assertEqual(metadata["model_configuration"]["risk_fusion"], {"z_weight": 0.5, "isolation_weight": 0.5})
        self.assertEqual(metadata["model_configuration"]["isolation_forest"]["random_state"], 42)

    def test_training_validation_and_test_engine_ids_remain_disjoint(self) -> None:
        split = self.result.split_metadata
        train_ids = set(split.train_engine_ids)
        validation_ids = set(split.validation_engine_ids)
        test_ids = set(split.test_engine_ids)

        self.assertEqual((len(train_ids), len(validation_ids), len(test_ids)), (70, 15, 15))
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertTrue(test_ids.isdisjoint(self.result.pipeline.train_engine_ids))
        self.assertTrue(test_ids.isdisjoint(self.result.pipeline.validation_engine_ids))
