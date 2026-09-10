import unittest

import numpy as np
import pandas as pd

from app.preparation.early_features import EarlyWindowConfig, extract_early_features, fit_feature_preprocessor
from app.preparation.splitting import split_by_engine


def make_frame(engine_count: int = 10, max_cycle: int = 40) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, max_cycle + 1):
            rows.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
                "sensor_1": engine_id + cycle * 0.1,
                "sensor_2": engine_id * 2 + cycle * 0.2,
            })
    return pd.DataFrame(rows)


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_frame()

    def test_engine_splits_are_disjoint_and_deterministic(self) -> None:
        first = split_by_engine(self.frame, seed=7)
        second = split_by_engine(self.frame, seed=7)
        metadata = first.metadata

        self.assertEqual(metadata, second.metadata)
        self.assertFalse(set(metadata.train_engine_ids) & set(metadata.validation_engine_ids))
        self.assertFalse(set(metadata.train_engine_ids) & set(metadata.test_engine_ids))
        self.assertFalse(set(metadata.validation_engine_ids) & set(metadata.test_engine_ids))
        self.assertEqual(len(first.train) + len(first.validation) + len(first.test), len(self.frame))

    def test_future_cycles_do_not_change_early_features(self) -> None:
        baseline = extract_early_features(self.frame).features
        changed = self.frame.copy()
        changed.loc[changed["cycle"] > 30, "sensor_1"] = 999999.0
        changed_features = extract_early_features(changed).features

        pd.testing.assert_frame_equal(baseline, changed_features)

    def test_insufficient_early_cycles_are_reported(self) -> None:
        result = extract_early_features(make_frame(engine_count=2, max_cycle=29))

        self.assertTrue(result.features.empty)
        self.assertEqual(result.skipped_engines, {1: 29, 2: 29})

    def test_preprocessing_is_train_only_and_finite_with_identical_columns(self) -> None:
        split = split_by_engine(self.frame, seed=3)
        config = EarlyWindowConfig()
        train = extract_early_features(split.train, config).features
        validation = extract_early_features(split.validation, config).features
        test = extract_early_features(split.test, config).features
        preprocessor, train_matrix = fit_feature_preprocessor(train)
        validation_matrix = preprocessor.transform(validation)
        test_matrix = preprocessor.transform(test)

        self.assertEqual(set(preprocessor.fit_engine_ids), set(split.metadata.train_engine_ids))
        self.assertTrue(set(preprocessor.fit_engine_ids).isdisjoint(split.metadata.validation_engine_ids))
        self.assertTrue(set(preprocessor.fit_engine_ids).isdisjoint(split.metadata.test_engine_ids))
        self.assertEqual(list(train_matrix.columns), list(validation_matrix.columns))
        self.assertEqual(list(train_matrix.columns), list(test_matrix.columns))
        self.assertTrue(np.isfinite(train_matrix.to_numpy()).all())
        self.assertTrue(np.isfinite(validation_matrix.to_numpy()).all())
        self.assertTrue(np.isfinite(test_matrix.to_numpy()).all())
