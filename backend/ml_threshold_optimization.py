"""Run validation-only threshold search and one locked test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.evaluation.threshold_optimization import run_threshold_diagnostic
from app.ml.production_training import load_fd001_training_data


DEFAULT_CMAPSS_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")
DEFAULT_REPORT = Path(__file__).resolve().parent / "ml_threshold_optimization_results.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="BurnUI validation-only threshold optimization.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_CMAPSS_ZIP)
    parser.add_argument("--csv", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostic = run_threshold_diagnostic(load_fd001_training_data(args.zip))
    validation = pd.DataFrame([item.report_row() for item in diagnostic.validation_candidates])
    validation.to_csv(args.csv, index=False)
    print("BURNUI THRESHOLD OPTIMIZATION — VALIDATION-FIRST, TEST LOCKED")
    print("Continuous anomaly risk scores are unchanged; only decision thresholds were compared.")
    print("Validation threshold grid was fixed at 0.50–0.95 in 0.05 increments, plus current production cutoffs.")
    print("Degenerate candidates require at least 50% Normal and no more than 50% alerted validation engines.")
    print(validation.to_string(index=False))
    print("\nSELECTED VALIDATION THRESHOLD")
    print(pd.DataFrame([diagnostic.selected_validation.report_row()]).to_string(index=False))
    print("\nFINAL TEST COMPARISON (ONE LOCKED TEST SCORING PASS)")
    print(pd.DataFrame([diagnostic.baseline_test.report_row(), diagnostic.selected_test.report_row()]).to_string(index=False))
    print(f"Split: train={len(diagnostic.split_metadata.train_engine_ids)}, validation={len(diagnostic.split_metadata.validation_engine_ids)}, test={len(diagnostic.split_metadata.test_engine_ids)}")
    print(f"Production model SHA-256 before: {diagnostic.artifact_sha256_before}")
    print(f"Production model SHA-256 after:  {diagnostic.artifact_sha256_after}")
    print(f"Wrote validation report: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
