import unittest
from unittest.mock import patch

from app.api.routes.evaluation import anomaly_evidence
from app.evaluation.evidence import build_anomaly_evidence
from app.evaluation.drift_prediction import evaluate_drift_prediction
from app.auth import User
from .test_drift_prediction import make_drift_frame


class EvidenceTests(unittest.TestCase):
    def test_anomaly_evidence_route_returns_actual_curve_payload(self) -> None:
        payload = {
            "curve_split": "Validation",
            "validation_curve": {"precision": [1.0, 0.5], "recall": [0.0, 1.0]},
            "validation_pr_auc": 0.5,
            "test_engine_count": 15,
            "test_metrics": {"tp": 0, "tn": 9, "fp": 3, "fn": 3, "precision": 0.0, "recall": 0.0, "f1": None, "fpr": 0.25, "fnr": 1.0, "pr_auc": 0.21, "roc_auc": 0.278},
            "target": {"early_end_cycle": 30, "training_quantile": 0.2, "remaining_cycle_horizon": 135.8},
            "disclaimer": "C-MAPSS is used as a public degradation/prognostics proxy.",
        }
        user = User(1, "Inspector", "inspector@example.com", "INSPECTOR", None, None, True, 0, 0, None, None, None, None, "now")
        with patch("app.api.routes.evaluation.build_anomaly_evidence", return_value=payload):
            self.assertEqual(anomaly_evidence(user), payload)

    def test_drift_graph_data_is_prediction_aligned(self) -> None:
        result = evaluate_drift_prediction(make_drift_frame()).to_dict()
        self.assertEqual(len(result["engine_ids"]), 15)
        self.assertEqual(len(result["engine_ids"]), len(result["predicted_values"]))
        self.assertEqual(len(result["predicted_values"]), len(result["actual_values"]))
