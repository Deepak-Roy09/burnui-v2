import asyncio
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException, UploadFile

from app.api.routes import screening
from app.api.routes.screening import _read_normalized_csv, _resolve_screening_thresholds, run_screening
from app.auth import User
from app.main import create_app
from app.ml.production_training import load_production_pipeline, production_artifact_paths
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.early_features import extract_early_features
from app.preparation.cmapss_adapter import CMAPSSVariant
from .test_cmapss_uploads import raw_cmapss_text


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


def rul_upload(values: list[int], filename: str) -> UploadFile:
    return upload(("\n".join(str(value) for value in values) + "\n").encode("utf-8"), filename)


def screening_user() -> User:
    return User(
        id=1, full_name="Test Inspector", email="test@example.com", role="INSPECTOR", department="Test",
        password_hash=None, is_active=True, session_version=0, failed_login_count=0, locked_until=None,
        activation_token_hash=None, activation_expires_at=None, activation_used_at=None, created_at="2026-01-01T00:00:00+00:00",
    )


class ScreeningEndpointTests(unittest.TestCase):
    def test_test_file_without_rul_is_prediction_only(self) -> None:
        response = asyncio.run(run_screening(upload(raw_cmapss_text(), "test_FD001.txt"), current_user=screening_user(), repository=Mock()))
        self.assertEqual(response["evaluation"]["mode"], "prediction-only")
        self.assertFalse(response["evaluation"]["available"])

    def test_matching_rul_file_is_aligned_to_sorted_test_engines(self) -> None:
        response = asyncio.run(run_screening(upload(raw_cmapss_text(), "test_FD001.txt"), rul_file=rul_upload([12, 34], "RUL_FD001.txt"), current_user=screening_user(), repository=Mock()))
        self.assertEqual(response["evaluation"]["mode"], "evaluation")
        self.assertEqual(response["evaluation"]["rul_by_engine"], [{"engine_id": 1, "rul": 12.0}, {"engine_id": 2, "rul": 34.0}])

    def test_mismatched_rul_file_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(run_screening(upload(raw_cmapss_text(), "test_FD002.txt"), rul_file=rul_upload([12, 34], "RUL_FD001.txt"), current_user=screening_user(), repository=Mock()))
        self.assertEqual(raised.exception.status_code, 422)
    def test_screens_official_raw_fd002_test_file_and_returns_domain_warning(self) -> None:
        response = asyncio.run(run_screening(upload(raw_cmapss_text(), "test_FD002.txt"), current_user=screening_user(), repository=Mock()))
        self.assertEqual(response["summary"]["total_components_screened"], 2)
        self.assertEqual(response["source"]["dataset"], CMAPSSVariant.FD002.value)
        self.assertEqual(response["source"]["file_type"], "TEST")
        self.assertIn("Domain-shift demonstration", response["source"]["domain_warning"])

    def test_registers_and_screens_every_valid_engine(self) -> None:
        paths = create_app().openapi()["paths"]
        self.assertIn("/api/screening/run", paths)

        response = asyncio.run(run_screening(upload(make_screening_csv()), current_user=screening_user(), repository=Mock()))

        self.assertEqual(response["summary"]["total_components_screened"], 6)
        self.assertEqual(
            response["summary"]["total_components_screened"],
            response["summary"]["normal_count"]
            + response["summary"]["watchlist_count"]
            + response["summary"]["high_risk_count"],
        )
        self.assertEqual([component["engine_id"] for component in response["components"]], [1, 2, 3, 4, 5, 6])
        self.assertEqual(response["screening_parameters"]["sensitivity"], "Standard")
        self.assertAlmostEqual(response["screening_parameters"]["validated_defaults"]["watchlist_threshold"], 0.8071428571428572)
        for component in response["components"]:
            self.assertTrue(0.0 <= component["risk_score"] <= 1.0)
            self.assertIn(component["status"], {"Normal", "Watchlist", "High Risk"})
            self.assertTrue(component["top_features"])
            self.assertIn("pca_1", component)
            self.assertIn("pca_2", component)

    def test_parameter_threshold_validation_preserves_default_decision_configuration(self) -> None:
        self.assertEqual(
            _resolve_screening_thresholds(None, None, 0.807143, 0.885714, "Standard"),
            (0.807143, 0.885714),
        )
        with self.assertRaises(ValueError):
            _resolve_screening_thresholds(0.9, 0.8, 0.807143, 0.885714, "Standard")

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
            response = asyncio.run(run_screening(upload(content), current_user=screening_user(), repository=Mock()))

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
            asyncio.run(run_screening(upload(b"engine_id,cycle\n1,1\n"), current_user=screening_user(), repository=Mock()))

        self.assertEqual(raised.exception.status_code, 422)
        self.assertFalse(raised.exception.detail["validation"]["valid"])
