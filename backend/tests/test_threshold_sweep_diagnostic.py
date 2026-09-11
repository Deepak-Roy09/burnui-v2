import hashlib
import inspect
import unittest
from pathlib import Path

import numpy as np

from ml_threshold_sweep_diagnostic import (
    PRODUCTION_SHA256,
    SELECTED_FUSION_WEIGHTS,
    evaluate_threshold,
    run_dataset,
    select_validation_threshold,
    threshold_candidates,
)


class ThresholdSweepDiagnosticTests(unittest.TestCase):
    def test_candidates_and_selection_are_deterministic(self):
        scores = np.array([0.10, 0.20, 0.30, 0.40, 0.80, 0.90])
        truth = np.array([False, False, False, True, True, True])
        first = [evaluate_threshold(scores, truth, threshold) for threshold, _ in threshold_candidates(scores)]
        second = [evaluate_threshold(scores, truth, threshold) for threshold, _ in threshold_candidates(scores)]
        self.assertEqual(first, second)
        self.assertEqual(select_validation_threshold(first), select_validation_threshold(second))

    def test_selection_prioritizes_recall_then_f1(self):
        candidates = [
            {"threshold": 0.8, "recall": 0.5, "f1": 0.40, "precision": 0.5, "fpr": 0.2, "alert_rate": 0.2},
            {"threshold": 0.4, "recall": 0.75, "f1": 0.50, "precision": 0.4, "fpr": 0.5, "alert_rate": 0.6},
            {"threshold": 0.3, "recall": 1.0, "f1": 0.45, "precision": 0.3, "fpr": 0.7, "alert_rate": 0.8},
        ]
        selected = select_validation_threshold(candidates)
        self.assertEqual(selected["threshold"], 0.4)
        self.assertEqual(selected["selection_status"], "recall_target_met")

    def test_selection_uses_recall_when_no_candidate_reaches_the_gate(self):
        candidates = [
            {"threshold": 0.8, "recall": 0.25, "f1": 0.30, "precision": 0.8, "fpr": 0.1, "alert_rate": 0.1},
            {"threshold": 0.4, "recall": 0.40, "f1": 0.20, "precision": 0.2, "fpr": 0.7, "alert_rate": 0.8},
        ]
        selected = select_validation_threshold(candidates)
        self.assertEqual(selected["threshold"], 0.4)
        self.assertEqual(selected["selection_status"], "best_available_recall_below_target")

    def test_zero_denominators_are_safe(self):
        result = evaluate_threshold(np.array([0.2, 0.3]), np.array([False, False]), 0.9)
        self.assertEqual(result["alert_count"], 0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])

    def test_frozen_validation_threshold_is_reused_for_holdout_and_test_scoring(self):
        validation_scores = np.array([0.1, 0.2, 0.8, 0.9])
        validation_truth = np.array([False, False, True, True])
        candidates = [evaluate_threshold(validation_scores, validation_truth, threshold) for threshold, _ in threshold_candidates(validation_scores)]
        frozen = select_validation_threshold(candidates)
        holdout = evaluate_threshold(np.array([0.3, 0.7]), np.array([False, True]), frozen["threshold"])
        official_test = evaluate_threshold(np.array([0.4, 0.8]), np.array([False, True]), frozen["threshold"])
        self.assertEqual(holdout["threshold"], frozen["threshold"])
        self.assertEqual(official_test["threshold"], frozen["threshold"])

    def test_runner_freezes_validation_choice_before_opening_official_test_rul(self):
        source = inspect.getsource(run_dataset)
        self.assertLess(source.index("frozen = select_validation_threshold"), source.index("test_frame = load_from_zip"))
        self.assertLess(source.index("frozen = select_validation_threshold"), source.index("rul_values = load_from_zip"))

    def test_selected_fusions_match_the_prior_diagnostic(self):
        self.assertEqual(SELECTED_FUSION_WEIGHTS, {"FD002": 0.0, "FD003": 0.0, "FD004": 0.2})

    def test_production_artifact_hash_is_protected(self):
        artifact = Path(__file__).resolve().parents[1] / "model_artifacts" / "burnui_production_model.joblib"
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest().upper(), PRODUCTION_SHA256)


if __name__ == "__main__":
    unittest.main()
