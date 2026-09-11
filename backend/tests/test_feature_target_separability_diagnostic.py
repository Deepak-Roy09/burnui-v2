import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml_feature_target_separability_diagnostic import (
    aggregate_validation_probe,
    feature_statistics,
    fit_probe_preprocessor,
    rank_by_effect,
    run_dataset,
    transform_probe_features,
)


class FeatureTargetSeparabilityDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame({"engine_id": [1, 2, 3, 4], "sensor_1_mean": [0., 0.1, 2., 2.1], "sensor_2_mean": [1., 1., 1., 1.]})
        self.validation = pd.DataFrame({"engine_id": [5, 6, 7, 8], "sensor_1_mean": [0.2, 0.3, 1.9, 2.2], "sensor_2_mean": [1., 1., 1., 1.]})
        self.labels = np.array([False, False, True, True])

    def test_train_effect_ranking_does_not_use_validation_values(self):
        first = rank_by_effect(feature_statistics(self.train, self.labels, self.validation, self.labels))
        changed_validation = self.validation.copy()
        changed_validation["sensor_1_mean"] = changed_validation["sensor_1_mean"] * -1000
        second = rank_by_effect(feature_statistics(self.train, self.labels, changed_validation, self.labels))
        self.assertEqual([row["feature"] for row in first], [row["feature"] for row in second])
        self.assertEqual(first[0]["feature"], "sensor_1_mean")

    def test_probe_preprocessing_is_fitted_on_train_only(self):
        result = aggregate_validation_probe(self.train, self.labels, self.validation, self.labels, ["sensor_1_mean"])
        self.assertEqual(result["preprocessor_fit_engine_ids"], [1, 2, 3, 4])
        self.assertIsNotNone(result["validation_roc_auc"])

    def test_validation_uses_frozen_train_probe_preprocessing(self):
        preprocessor = fit_probe_preprocessor(self.train, ["sensor_1_mean"])
        first = transform_probe_features(preprocessor, self.validation)
        changed = self.validation.copy()
        changed["sensor_1_mean"] = changed["sensor_1_mean"] + 100.0
        second = transform_probe_features(preprocessor, changed)
        self.assertTrue(np.allclose(second - first, 100.0 / preprocessor.scales[0]))
        self.assertEqual(preprocessor.fit_engine_ids, (1, 2, 3, 4))

    def test_internal_holdout_transform_uses_the_same_train_fitted_state(self):
        result = aggregate_validation_probe(
            self.train,
            self.labels,
            self.validation,
            self.labels,
            ["sensor_1_mean"],
            self.validation,
        )
        self.assertEqual(result["internal_holdout_transformed_count"], len(self.validation))
        self.assertEqual(result["preprocessor_fit_engine_ids"], [1, 2, 3, 4])

    def test_zero_variance_subset_is_reported_without_crashing(self):
        result = aggregate_validation_probe(self.train, self.labels, self.validation, self.labels, ["sensor_2_mean"])
        self.assertEqual(result["usable_feature_count"], 0)
        self.assertEqual(result["probe_status"], "no_usable_train_variance")
        self.assertIsNone(result["validation_roc_auc"])

    def test_runner_never_opens_official_test_or_rul_files(self):
        runner_source = inspect.getsource(run_dataset)
        self.assertIn("load_train_member", runner_source)
        self.assertNotIn("test_FD", runner_source)
        self.assertNotIn("RUL_", runner_source)
        loader_source = inspect.getsource(__import__("ml_feature_target_separability_diagnostic").load_train_member)
        self.assertIn("train_{dataset}", loader_source)
        self.assertNotIn("CMAPSSSubset.TEST", loader_source)
        self.assertNotIn("load_cmapss_rul", loader_source)

    def test_diagnostic_probe_is_deterministic(self):
        first = aggregate_validation_probe(self.train, self.labels, self.validation, self.labels, ["sensor_1_mean"])
        second = aggregate_validation_probe(self.train, self.labels, self.validation, self.labels, ["sensor_1_mean"])
        self.assertEqual(first, second)

    def test_production_artifact_hash_remains_expected(self):
        artifact = Path(__file__).resolve().parents[1] / "model_artifacts" / "burnui_production_model.joblib"
        expected = "24B69C7C216C72F6D5940FAA70E4551764F145DDFB528CD924D8431274CFC0B1"
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest().upper(), expected)


if __name__ == "__main__":
    unittest.main()
