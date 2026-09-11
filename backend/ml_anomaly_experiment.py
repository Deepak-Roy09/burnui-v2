"""Run validation-only anomaly-detector experiments without touching production."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.evaluation.anomaly_experiments import promotion_recommendation, run_validation_experiments, select_validation_candidate
from app.ml.production_training import load_fd001_training_data, load_production_pipeline, production_artifact_paths


DEFAULT_CMAPSS_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only BurnUI anomaly experiment. Production artifacts are read only.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_CMAPSS_ZIP, help="ZIP containing train_FD001.txt")
    parser.add_argument("--csv", type=Path, help="Optional output CSV for validation results")
    args = parser.parse_args()

    paths = production_artifact_paths()
    pipeline = load_production_pipeline()
    if len(pipeline.preprocessor.input_columns) != 168:
        raise RuntimeError("Production artifact does not have the expected 168-feature schema.")
    frame = load_fd001_training_data(args.zip)
    data, results = run_validation_experiments(frame)
    report = pd.DataFrame([result.report_row() for result in results])

    print("BURNUI VALIDATION-ONLY ANOMALY EXPERIMENT")
    print("C-MAPSS is a public degradation/prognostics proxy, not ISRO burn-in data.")
    print(f"Production artifact (read only): {paths.model}")
    print(f"Engine split: train={len(data.split_metadata.train_engine_ids)}, validation={len(data.split_metadata.validation_engine_ids)}, test=untouched ({len(data.split_metadata.test_engine_ids)})")
    print(f"Model inputs: cycles 1–30, {data.train_features.shape[1] - 1} existing features")
    print(f"Training-derived target horizon: {data.target.remaining_cycle_horizon:.3f}")
    print(report.to_string(index=False))
    print(f"Best validation-only FN-first candidate: {select_validation_candidate(results).model_name}")
    print(promotion_recommendation(results))
    if args.csv:
        report.to_csv(args.csv, index=False)
        print(f"Wrote validation-only report: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
