import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from app.api.routes import screening
from app.api.routes.screening import _read_normalized_csv, run_screening
from app.main import create_app
from app.ml.production_training import load_production_pipeline, production_artifact_paths
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.early_features import extract_early_features


def make_screening_csv(engine_count: int = 6) -> bytes:
    sensors = [f"sensor_{number}" for number in range(1, 22)]
    rows = [",".join(["engine_id", "cycle", "setting_1", "setting_2", "setting_3", *sensors])]
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, 31):
            values = [engine_id * sensor * 0.01 + cycle * sensor * 0.001 for sensor in range(1, 22)]
            rows.append(",".join(str(value) for value in [engine_id, cycle, 0.0, 0.0, 100.0, *values]))
    return "\n".join(rows).encode("utf-8")


def upload(content: bytes, filename: str = "screening.csv") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content))


class ScreeningEndpointTests(unittest.TestCase):
    def test_registers_and_screens_every_valid_engine(self) -> None:
        paths = create_app().openapi()["paths"]
        self.assertIn("/api/screening/run", paths)

        response = asyncio.run(run_screening(upload(make_screening_csv())))

        self.assertEqual(response["summary"]["total_components_screened"], 6)
        self.assertEqual(
            response["summary"]["total_components_screened"],
            response["summary"]["normal_count"]
            + response["summary"]["watchlist_count"]
            + response["summary"]["high_risk_count"],
        )
        self.assertEqual([component["engine_id"] for component in response["components"]], [1, 2, 3, 4, 5, 6])
        for component in response["components"]:
            self.assertTrue(0.0 <= component["risk_score"] <= 1.0)
            self.assertIn(component["status"], {"Normal", "Watchlist", "High Risk"})
            self.assertTrue(component["top_features"])

    def test_uses_saved_production_model_without_fitting_during_upload(self) -> None:
        paths = production_artifact_paths()
        self.assertTrue(paths.model.is_file())
        self.assertTrue(paths.metadata.is_file())

        content = make_screening_csv()
        features = extract_early_features(_read_normalized_csv(content)).features
        expected = load_production_pipeline().predict(features).sort_values("engine_id").reset_index(drop=True)

        with patch.object(screening, "load_production_pipeline", wraps=load_production_pipeline) as load_model, patch.object(
            RealOnlyAnomalyPipeline,
            "fit",
            side_effect=AssertionError("The screening endpoint must not fit during an upload."),
        ):
            response = asyncio.run(run_screening(upload(content)))

        load_model.assert_called_once_with()
        self.assertEqual(response["summary"]["total_components_screened"], len(expected))
        self.assertEqual([component["engine_id"] for component in response["components"]], expected["engine_id"].tolist())
        for component, (_, prediction) in zip(response["components"], expected.iterrows()):
            self.assertAlmostEqual(component["risk_score"], prediction["risk_score"])
            self.assertAlmostEqual(component["z_score"], prediction["z_score"])
            self.assertAlmostEqual(component["isolation_score"], prediction["isolation_score"])
            self.assertEqual(component["status"], prediction["classification"])
            self.assertEqual(component["top_features"], prediction["explanation"]["top_features"])

    def test_rejects_an_upload_that_fails_existing_validation(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(run_screening(upload(b"engine_id,cycle\n1,1\n")))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertFalse(raised.exception.detail["validation"]["valid"])
