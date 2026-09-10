import tempfile
import unittest

import numpy as np
import pandas as pd

from app.models.real_only_anomaly import (
    IsolationForestConfig,
    RealOnlyAnomalyPipeline,
    load_anomaly_pipeline,
    save_anomaly_pipeline,
)
from app.preparation.early_features import extract_early_features
from app.preparation.splitting import split_by_engine


def make_sensor_frame() -> pd.DataFrame:
    rows = []
    for engine_id in range(1, 21):
        for cycle in range(1, 41):
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


class RealOnlyAnomalyTests(unittest.TestCase):
    def setUp(self) -> None:
        split = split_by_engine(make_sensor_frame(), seed=42)
        self.train = extract_early_features(split.train).features
        self.validation = extract_early_features(split.validation).features
        self.test = extract_early_features(split.test).features
        self.split = split

    def _pipeline(self) -> RealOnlyAnomalyPipeline:
        return RealOnlyAnomalyPipeline(isolation_config=IsolationForestConfig(n_estimators=80, random_state=17))

    def test_fit_uses_train_and_validation_only(self) -> None:
        pipeline = self._pipeline().fit(self.train, self.validation)

        self.assertEqual(set(pipeline.train_engine_ids), set(self.split.metadata.train_engine_ids))
        self.assertEqual(set(pipeline.validation_engine_ids), set(self.split.metadata.validation_engine_ids))
        self.assertTrue(set(self.test["engine_id"]).isdisjoint(pipeline.train_engine_ids))
        self.assertTrue(set(self.test["engine_id"]).isdisjoint(pipeline.validation_engine_ids))

    def test_results_are_deterministic_and_normalized(self) -> None:
        first = self._pipeline().fit(self.train, self.validation).predict(self.validation)
        second = self._pipeline().fit(self.train, self.validation).predict(self.validation)

        np.testing.assert_allclose(first["risk_score"], second["risk_score"])
        self.assertTrue(np.isfinite(first[["z_score", "isolation_score", "risk_score"]].to_numpy()).all())
        self.assertTrue(first["risk_score"].between(0.0, 1.0).all())
        self.assertTrue(set(first["classification"]).issubset({"Normal", "Watchlist", "High Risk"}))
        self.assertEqual(len(first["classification"]), len(self.validation))

    def test_explanations_reference_real_feature_values(self) -> None:
        changed = self.validation.copy()
        changed.loc[changed.index[0], "sensor_1_mean"] = 10000.0
        result = self._pipeline().fit(self.train, self.validation).predict(changed).iloc[0]

        top = result["explanation"]["top_features"]
        self.assertTrue(top)
        self.assertIn(top[0]["feature"], changed.columns)
        self.assertEqual(top[0]["value"], float(changed.iloc[0][top[0]["feature"]]))
        self.assertGreaterEqual(top[0]["z_contribution"], top[-1]["z_contribution"])

    def test_save_load_keeps_predictions_equivalent(self) -> None:
        pipeline = self._pipeline().fit(self.train, self.validation)
        expected = pipeline.predict(self.validation)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/pipeline.joblib"
            save_anomaly_pipeline(pipeline, path)
            actual = load_anomaly_pipeline(path).predict(self.validation)

        np.testing.assert_allclose(expected["risk_score"], actual["risk_score"])
        self.assertListEqual(expected["classification"].tolist(), actual["classification"].tolist())
