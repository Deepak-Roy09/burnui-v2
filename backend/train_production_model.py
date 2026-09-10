"""Train and save the approved BurnUI FD001 production anomaly-screening model."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ml.production_training import production_artifact_paths, train_and_save_production_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"),
        help="Path to the original C-MAPSS ZIP containing train_FD001.txt.",
    )
    parser.add_argument("--artifact-directory", type=Path, default=None)
    args = parser.parse_args()

    result = train_and_save_production_model(args.dataset, args.artifact_directory)
    paths = production_artifact_paths(args.artifact_directory)
    metadata = result.metadata
    print("BurnUI production model trained and saved.")
    print(f"Model: {paths.model}")
    print(f"Metadata: {paths.metadata}")
    print(
        "Split: "
        f"train={metadata['train_engine_count']}, validation={metadata['validation_engine_count']}, "
        f"test={metadata['test_engine_count']}"
    )
    print(
        "Thresholds: "
        f"watchlist={float(metadata['watchlist_threshold']):.6f}, "
        f"high_risk={float(metadata['high_risk_threshold']):.6f}"
    )


if __name__ == "__main__":
    main()
