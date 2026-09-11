import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml_augmentation_diagnostic import generate_synthetic_training_features
from app.preparation.cmapss_adapter import CMAPSSFileKind, CMAPSSVariant, detect_cmapss_filename
from app.preparation.splitting import split_by_engine


class AugmentationDatasetTests(unittest.TestCase):
    def test_production_artifact_hash_is_unchanged(self):
        import hashlib

        artifact = Path(__file__).resolve().parents[1] / "model_artifacts" / "burnui_production_model.joblib"
        self.assertEqual(
            hashlib.sha256(artifact.read_bytes()).hexdigest().upper(),
            "24B69C7C216C72F6D5940FAA70E4551764F145DDFB528CD924D8431274CFC0B1",
        )

    def test_all_supported_train_and_test_names_are_detected(self):
        for variant in CMAPSSVariant:
            for kind in (CMAPSSFileKind.TRAIN, CMAPSSFileKind.TEST):
                descriptor = detect_cmapss_filename(f"{kind.value}_{variant.value}.txt")
                self.assertIsNotNone(descriptor)
                self.assertEqual((descriptor.variant, descriptor.kind), (variant, kind))

    def test_rul_file_is_not_a_train_trajectory(self):
        descriptor = detect_cmapss_filename("RUL_FD004.txt")
        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.kind, CMAPSSFileKind.RUL)

    def test_synthetic_generator_sees_training_feature_values_only(self):
        train = pd.DataFrame({"engine_id": [1, 2], **{f"sensor_1_{index}": [1.0 + index, 2.0 + index] for index in range(168)}})
        observed = {}

        class Synthesizer:
            def __init__(self, metadata):
                observed["metadata"] = metadata

            def fit(self, values):
                observed["fit"] = values.copy()

            def sample(self, num_rows):
                return observed["fit"].iloc[[0] * num_rows].reset_index(drop=True)

        generated = generate_synthetic_training_features(train, Synthesizer, lambda values: tuple(values.columns))
        self.assertNotIn("engine_id", observed["fit"])
        pd.testing.assert_frame_equal(observed["fit"], train.drop(columns=["engine_id"]))
        self.assertEqual(tuple(generated.columns), tuple(train.columns))
        self.assertTrue(np.isfinite(generated.drop(columns=["engine_id"]).to_numpy(dtype=float)).all())
        self.assertTrue(set(generated["engine_id"]).isdisjoint(set(train["engine_id"])))

    def test_engine_split_is_deterministic_and_disjoint(self):
        frame = pd.DataFrame({"engine_id": np.repeat(np.arange(1, 21), 30), "cycle": np.tile(np.arange(1, 31), 20)})
        first, second = split_by_engine(frame, seed=42), split_by_engine(frame, seed=42)
        self.assertEqual(first.metadata, second.metadata)
        train, validation, test = map(set, (first.metadata.train_engine_ids, first.metadata.validation_engine_ids, first.metadata.test_engine_ids))
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

    def test_each_dataset_split_is_local_and_disjoint(self):
        for offset in (0, 1000, 2000, 3000):
            frame = pd.DataFrame({"engine_id": np.repeat(np.arange(1, 101) + offset, 30), "cycle": np.tile(np.arange(1, 31), 100)})
            split = split_by_engine(frame, seed=42)
            train, validation, test = map(set, (split.metadata.train_engine_ids, split.metadata.validation_engine_ids, split.metadata.test_engine_ids))
            self.assertTrue(train.isdisjoint(validation))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(validation.isdisjoint(test))
            self.assertEqual(len(train) + len(validation) + len(test), 100)


if __name__ == "__main__":
    unittest.main()
