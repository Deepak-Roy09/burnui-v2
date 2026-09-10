"""Read-only SDV GaussianCopula augmentation diagnostic for BurnUI.

This script compares the existing FD001 baseline against the same pipeline
trained on real 1-30 training features plus one equal-sized synthetic feature
table. SDV is fitted only on the real training feature values. Validation and
test remain real; test data is accessed only after both pipelines are fitted.
No production code, data, thresholds, or tests are changed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from app.evaluation.real_only_evaluation import (
    calculate_binary_metrics,
    construct_future_degradation_truth,
    fit_future_degradation_target,
)
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import split_by_engine


METRICS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")
TARGET_EARLY_END_CYCLE = 30
TARGET_TRAINING_QUANTILE = 0.20
EXPECTED_TARGET_HORIZON = 135.800
SYNTHETIC_TO_REAL_TRAINING_RATIO = 1.0
SYNTHETIC_ENGINE_ID_START = -1


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Read the original FD001 train trajectory file through the project adapter."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the supplied C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the unchanged production extractor for exactly cycles 1-30."""
    result = extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=30, minimum_early_cycles=30),
    )
    if result.skipped_engines:
        raise AssertionError(f"The 1-30 window skipped engines: {result.skipped_engines}")
    if result.features.shape[1] - 1 != 168:
        raise AssertionError("The 1-30 production extractor must return exactly 168 sensor-derived features.")
    return result.features


def import_sdv_gaussian_copula() -> tuple[Any, Any] | None:
    """Import the supported SDV single-table API without installing anything."""
    try:
        import sdv
        from sdv.single_table import GaussianCopulaSynthesizer
    except ImportError:
        print("SDV is not installed in the interpreter running this diagnostic.")
        print("No requirements file or dependency was changed. Install SDV separately, then rerun this diagnostic.")
        return None

    try:
        from sdv.metadata import Metadata

        def build_metadata(frame: pd.DataFrame):
            return Metadata.detect_from_dataframe(frame)

    except ImportError:
        # Older SDV releases expose the equivalent single-table metadata class.
        from sdv.metadata import SingleTableMetadata

        def build_metadata(frame: pd.DataFrame):
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(frame)
            return metadata

    print(f"SDV available: version {sdv.__version__}; synthesizer=GaussianCopulaSynthesizer")
    return GaussianCopulaSynthesizer, build_metadata


def generate_synthetic_training_features(
    real_train_features: pd.DataFrame,
    synthesizer_class,
    metadata_builder,
) -> pd.DataFrame:
    """Fit SDV only on real training feature values and sample one synthetic row per real engine."""
    feature_columns = [column for column in real_train_features.columns if column != "engine_id"]
    if len(feature_columns) != 168:
        raise AssertionError("Synthetic generation must receive the full existing 168-feature table.")
    real_values = real_train_features.loc[:, feature_columns].copy()
    # SDV receives neither engine IDs nor any validation/test data or labels.
    if not real_values.index.equals(pd.RangeIndex(len(real_values))):
        real_values = real_values.reset_index(drop=True)
    metadata = metadata_builder(real_values)
    np.random.seed(42)
    synthesizer = synthesizer_class(metadata)
    synthesizer.fit(real_values)
    synthetic_count = int(len(real_values) * SYNTHETIC_TO_REAL_TRAINING_RATIO)
    synthetic_values = synthesizer.sample(num_rows=synthetic_count)
    synthetic_values = synthetic_values.loc[:, feature_columns].copy()
    if len(synthetic_values) != synthetic_count or tuple(synthetic_values.columns) != tuple(feature_columns):
        raise AssertionError("SDV did not return the expected synthetic training feature shape.")

    synthetic = synthetic_values.copy()
    synthetic.insert(0, "engine_id", np.arange(SYNTHETIC_ENGINE_ID_START, SYNTHETIC_ENGINE_ID_START - synthetic_count, -1))
    if set(synthetic["engine_id"]) & set(real_train_features["engine_id"]):
        raise AssertionError("Synthetic engine identifiers overlap real training engine identifiers.")
    return synthetic.loc[:, ["engine_id", *feature_columns]]


def align_truth(features: pd.DataFrame, trajectories: pd.DataFrame, target) -> pd.DataFrame:
    """Use the existing post-hoc C-MAPSS truth construction without changing it."""
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")
    engine_ids = features["engine_id"].astype(int).tolist()
    if set(engine_ids) - set(truth.index.astype(int)):
        raise AssertionError("Truth is missing engines with real feature rows.")
    return truth.loc[engine_ids]


def evaluate_at_current_watchlist(pipeline: RealOnlyAnomalyPipeline, features: pd.DataFrame, trajectories: pd.DataFrame, target):
    """Evaluate a real validation or held-out table at the unchanged watchlist threshold."""
    scores = pipeline.predict(features)
    truth = align_truth(scores, trajectories, target)
    y_true = truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = scores["risk_score"].to_numpy(dtype=float)
    metrics = calculate_binary_metrics(y_true, y_score >= pipeline.thresholds.watchlist, y_score)
    return scores, metrics


def result_record(
    *,
    scope: str,
    configuration: str,
    pipeline: RealOnlyAnomalyPipeline,
    metrics: dict[str, int | float | None],
    target,
    real_training_count: int,
    synthetic_training_count: int,
    leakage_checks: str,
) -> dict[str, object]:
    return {
        "scope": scope,
        "configuration": configuration,
        "early_window": "1-30",
        "feature_count": 168,
        "detector": "existing 50/50 Z-score + Isolation Forest",
        "threshold_name": "existing 85th percentile watchlist",
        "threshold": pipeline.thresholds.watchlist,
        "real_training_count": real_training_count,
        "synthetic_training_count": synthetic_training_count,
        "target_early_end_cycle": target.early_end_cycle,
        "target_training_quantile": target.training_quantile,
        "target_remaining_cycle_horizon": target.remaining_cycle_horizon,
        "confusion_matrix": f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]",
        "leakage_checks": leakage_checks,
        **{name: metrics[name] for name in METRICS},
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "scope", "configuration", "early_window", "feature_count", "detector", "threshold_name", "threshold",
        "real_training_count", "synthetic_training_count", "target_early_end_cycle", "target_training_quantile",
        "target_remaining_cycle_horizon", "confusion_matrix", "leakage_checks", *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def print_metrics(label: str, pipeline: RealOnlyAnomalyPipeline, metrics: dict[str, int | float | None]) -> None:
    values = ", ".join(f"{name.upper()}={format_value(metrics[name])}" for name in METRICS)
    print(f"{label}: current watchlist threshold={pipeline.thresholds.watchlist:.6f}; {values}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ml_augmentation_diagnostic_results.csv"),
    )
    args = parser.parse_args()

    sdv_api = import_sdv_gaussian_copula()
    if sdv_api is None:
        return
    synthesizer_class, metadata_builder = sdv_api

    frame = load_fd001(args.dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    # Test data is intentionally not accessed until both pipelines are fitted.
    if (len(train_ids), len(validation_ids)) != (70, 15) or train_ids & validation_ids:
        raise AssertionError("The deterministic split must contain disjoint 70-train and 15-validation engine sets.")
    target = fit_future_degradation_target(
        split.train,
        early_end_cycle=TARGET_EARLY_END_CYCLE,
        training_quantile=TARGET_TRAINING_QUANTILE,
    )
    if not np.isclose(target.remaining_cycle_horizon, EXPECTED_TARGET_HORIZON):
        raise AssertionError(
            f"Expected training-derived target horizon {EXPECTED_TARGET_HORIZON:.3f}, "
            f"found {target.remaining_cycle_horizon:.3f}."
        )

    print("BURNUI SDV AUGMENTATION DIAGNOSTIC — DESCRIPTIVE ONLY")
    print("Split before final scoring: train=70, validation=15, test=untouched")
    print(
        f"Target: early_end_cycle={target.early_end_cycle}, training_quantile={target.training_quantile:.2f}, "
        f"training-derived horizon={target.remaining_cycle_horizon:.3f}"
    )

    # Training and validation only: real 1-30 features, existing 168 columns.
    train_features = extract_features(split.train)
    validation_features = extract_features(split.validation)
    baseline_pipeline = RealOnlyAnomalyPipeline().fit(train_features, validation_features)
    synthetic_features = generate_synthetic_training_features(train_features, synthesizer_class, metadata_builder)
    augmented_train_features = pd.concat([train_features, synthetic_features], ignore_index=True)
    if len(augmented_train_features) != len(train_features) + len(synthetic_features):
        raise AssertionError("The augmented training table has an unexpected row count.")
    if set(synthetic_features["engine_id"]) & validation_ids:
        raise AssertionError("Synthetic training identifiers overlap real validation engine identifiers.")
    augmented_pipeline = RealOnlyAnomalyPipeline().fit(augmented_train_features, validation_features)

    baseline_validation_scores, baseline_validation_metrics = evaluate_at_current_watchlist(
        baseline_pipeline, validation_features, split.validation, target
    )
    augmented_validation_scores, augmented_validation_metrics = evaluate_at_current_watchlist(
        augmented_pipeline, validation_features, split.validation, target
    )
    _ = (baseline_validation_scores, augmented_validation_scores)  # Scores are reported only through aggregate validation metrics.
    print("\nREAL VALIDATION RESULTS")
    print_metrics("Baseline", baseline_pipeline, baseline_validation_metrics)
    print_metrics("Augmented candidate", augmented_pipeline, augmented_validation_metrics)

    # Final evaluation: this is the first access to held-out test trajectories.
    test_ids = set(split.metadata.test_engine_ids)
    if len(test_ids) != 15 or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("The held-out split must contain 15 engines disjoint from train and validation.")
    if set(target.training_engine_ids) & test_ids:
        raise AssertionError("Target fitting engines overlap the held-out test engines.")
    # Only cycles 1-30 enter the test feature extractor; later cycles are used
    # once, after scoring, when the existing observed-endpoint truth is built.
    early_test_rows = split.test.loc[split.test["cycle"] <= 30].copy()
    test_features = extract_features(early_test_rows)
    if set(test_features["engine_id"].astype(int)) != test_ids:
        raise AssertionError("The held-out early-feature table does not contain exactly the test engines.")
    _, baseline_test_metrics = evaluate_at_current_watchlist(baseline_pipeline, test_features, split.test, target)
    _, augmented_test_metrics = evaluate_at_current_watchlist(augmented_pipeline, test_features, split.test, target)

    leakage_checks = (
        "GaussianCopula fit and sample source: real 70-engine training features only; synthetic IDs are new negative IDs; "
        "validation and test are real only; no validation/test data or labels are passed to SDV; test data is first accessed "
        "after both models are fit; test feature inputs contain only cycles 1-30; future test trajectories are used only for truth."
    )
    print("\nFINAL REAL HELD-OUT TEST RESULTS")
    print_metrics("Baseline", baseline_pipeline, baseline_test_metrics)
    print_metrics("Augmented candidate", augmented_pipeline, augmented_test_metrics)
    print("\nLEAKAGE CHECKS")
    print(f"- {leakage_checks}")

    records = [
        result_record(
            scope="validation",
            configuration="Baseline: real training only",
            pipeline=baseline_pipeline,
            metrics=baseline_validation_metrics,
            target=target,
            real_training_count=len(train_features),
            synthetic_training_count=0,
            leakage_checks=leakage_checks,
        ),
        result_record(
            scope="validation",
            configuration="Augmented candidate: real + SDV GaussianCopula training",
            pipeline=augmented_pipeline,
            metrics=augmented_validation_metrics,
            target=target,
            real_training_count=len(train_features),
            synthetic_training_count=len(synthetic_features),
            leakage_checks=leakage_checks,
        ),
        result_record(
            scope="final_held_out_test",
            configuration="Baseline: real training only",
            pipeline=baseline_pipeline,
            metrics=baseline_test_metrics,
            target=target,
            real_training_count=len(train_features),
            synthetic_training_count=0,
            leakage_checks=leakage_checks,
        ),
        result_record(
            scope="final_held_out_test",
            configuration="Augmented candidate: real + SDV GaussianCopula training",
            pipeline=augmented_pipeline,
            metrics=augmented_test_metrics,
            target=target,
            real_training_count=len(train_features),
            synthetic_training_count=len(synthetic_features),
            leakage_checks=leakage_checks,
        ),
    ]
    write_csv(args.output, records)
    print(f"\nWrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
