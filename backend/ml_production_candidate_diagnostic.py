"""Read-only comparison of a validation-calibrated BurnUI production candidate.

The candidate uses 1-50 early-cycle features restricted to six existing sensor
families. The C-MAPSS evaluation target remains the existing 1-30 target and
is fitted only from training trajectories. This script changes no production
code, data, model configuration, or threshold implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
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


FIXED_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
METRICS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")
RECURRING_SENSORS = ("sensor_2", "sensor_3", "sensor_4", "sensor_12", "sensor_20", "sensor_21")
TARGET_EARLY_END_CYCLE = 30
TARGET_TRAINING_QUANTILE = 0.20
EXPECTED_TARGET_HORIZON = 135.800


@dataclass(frozen=True)
class DiagnosticConfiguration:
    name: str
    early_window: int
    use_recurring_sensor_features: bool
    expected_feature_count: int


@dataclass(frozen=True)
class PreparedConfiguration:
    config: DiagnosticConfiguration
    pipeline: RealOnlyAnomalyPipeline
    validation_scores: pd.DataFrame
    validation_truth: pd.DataFrame
    test_features: pd.DataFrame


CONFIGURATIONS = (
    DiagnosticConfiguration(
        name="Baseline: 1-30 all current features",
        early_window=30,
        use_recurring_sensor_features=False,
        expected_feature_count=168,
    ),
    DiagnosticConfiguration(
        name="Candidate: 1-50 recurring-signal sensors",
        early_window=50,
        use_recurring_sensor_features=True,
        expected_feature_count=48,
    ),
)


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Load the original FD001 train file through the production adapter."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the supplied C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    result = extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )
    if result.skipped_engines:
        raise AssertionError(f"Window 1-{window} skipped engines: {result.skipped_engines}")
    if result.features.shape[1] - 1 != 168:
        raise AssertionError(f"Window 1-{window} must produce 168 existing sensor-derived features.")
    return result.features


def select_feature_family(features: pd.DataFrame, config: DiagnosticConfiguration) -> pd.DataFrame:
    """Keep either all production columns or only the six requested sensor families."""
    derived_columns = [column for column in features.columns if column != "engine_id"]
    if config.use_recurring_sensor_features:
        prefixes = tuple(f"{sensor}_" for sensor in RECURRING_SENSORS)
        derived_columns = [column for column in derived_columns if column.startswith(prefixes)]
    if len(derived_columns) != config.expected_feature_count:
        raise AssertionError(
            f"{config.name} expected {config.expected_feature_count} existing features, found {len(derived_columns)}."
        )
    return features.loc[:, ["engine_id", *derived_columns]].copy()


def feature_tables(split, config: DiagnosticConfiguration) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create feature tables without allowing cycles after a window into model inputs."""
    unfiltered_train = extract_features(split.train, config.early_window)
    unfiltered_validation = extract_features(split.validation, config.early_window)
    unfiltered_test = extract_features(split.test, config.early_window)

    # Independently confirm test values are unchanged when every future cycle
    # is removed before the existing feature extractor sees the trajectories.
    truncated_test = split.test.loc[split.test["cycle"] <= config.early_window].copy()
    pd.testing.assert_frame_equal(unfiltered_test, extract_features(truncated_test, config.early_window))

    return (
        select_feature_family(unfiltered_train, config),
        select_feature_family(unfiltered_validation, config),
        select_feature_family(unfiltered_test, config),
    )


def align_truth(scores: pd.DataFrame, trajectories: pd.DataFrame, target) -> pd.DataFrame:
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")
    engine_ids = scores["engine_id"].astype(int).tolist()
    if set(engine_ids) - set(truth.index.astype(int)):
        raise AssertionError("The evaluation target is missing engines with risk scores.")
    return truth.loc[engine_ids]


def threshold_metrics(scores: pd.DataFrame, truth: pd.DataFrame, threshold: float) -> dict[str, int | float | None]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Risk threshold must lie in [0, 1], got {threshold}.")
    y_true = truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = scores["risk_score"].to_numpy(dtype=float)
    return calculate_binary_metrics(y_true, y_score >= threshold, y_score)


def candidate_thresholds(pipeline: RealOnlyAnomalyPipeline) -> list[tuple[str, float]]:
    """Match the earlier threshold diagnostic's validation-only candidate set and order."""
    return [
        ("current 85th percentile watchlist", pipeline.thresholds.watchlist),
        ("current 95th percentile high-risk", pipeline.thresholds.high_risk),
        *((f"fixed {value:.2f}", value) for value in FIXED_THRESHOLDS),
    ]


def select_from_validation(records: list[dict[str, object]]) -> dict[str, object]:
    """Use the exact existing diagnostic priority and deterministic first-tie behavior."""
    def key(record: dict[str, object]) -> tuple[float, float, float]:
        recall = -1.0 if record["recall"] is None else float(record["recall"])
        fpr = 1.0 if record["fpr"] is None else float(record["fpr"])
        f1 = -1.0 if record["f1"] is None else float(record["f1"])
        return recall, -fpr, f1

    return max(records, key=key)


def record(
    *,
    scope: str,
    config: DiagnosticConfiguration,
    threshold_name: str,
    threshold: float,
    target,
    metrics: dict[str, int | float | None],
    selected_from_validation: bool,
) -> dict[str, object]:
    return {
        "scope": scope,
        "configuration": config.name,
        "feature_window": f"1-{config.early_window}",
        "feature_family": "recurring-signal sensors" if config.use_recurring_sensor_features else "all current features",
        "feature_count": config.expected_feature_count,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "target_early_end_cycle": target.early_end_cycle,
        "target_training_quantile": target.training_quantile,
        "target_remaining_cycle_horizon": target.remaining_cycle_horizon,
        "selected_from_validation": selected_from_validation,
        **{name: metrics[name] for name in METRICS},
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "scope", "configuration", "feature_window", "feature_family", "feature_count", "threshold_name", "threshold",
        "target_early_end_cycle", "target_training_quantile", "target_remaining_cycle_horizon",
        "selected_from_validation", *METRICS,
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


def print_metrics(label: str, metrics: dict[str, object]) -> None:
    values = ", ".join(f"{name.upper()}={format_value(metrics[name])}" for name in METRICS)
    print(f"{label}: threshold={float(metrics['threshold']):.6f}; {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ml_production_candidate_diagnostic_results.csv"),
    )
    args = parser.parse_args()

    frame = load_fd001(args.dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    test_ids = set(split.metadata.test_engine_ids)
    if (len(train_ids), len(validation_ids), len(test_ids)) != (70, 15, 15):
        raise AssertionError("FD001 split must contain exactly 70 train, 15 validation, and 15 test engines.")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train, validation, and test engine IDs must be disjoint.")

    # The target deliberately remains tied to cycle 30 even for the 1-50
    # candidate feature window. It is calibrated entirely from training engines.
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
    if set(target.training_engine_ids) & test_ids:
        raise AssertionError("Target horizon training engines overlap the held-out test engines.")

    print("BURNUI PRODUCTION-CANDIDATE DIAGNOSTIC")
    print(f"Engine split: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")
    print(
        "Evaluation target: "
        f"early_end_cycle={target.early_end_cycle}, training_quantile={target.training_quantile:.2f}, "
        f"training-derived horizon={target.remaining_cycle_horizon:.3f}"
    )

    prepared: list[PreparedConfiguration] = []
    validation_records: list[dict[str, object]] = []
    current_production_records: list[dict[str, object]] = []
    selected_records: list[dict[str, object]] = []

    # All threshold candidates are scored and selected using validation data
    # before any held-out test score or label is examined.
    for config in CONFIGURATIONS:
        train_features, validation_features, test_features = feature_tables(split, config)
        if set(test_features["engine_id"].astype(int)) != test_ids:
            raise AssertionError(f"{config.name} changed the held-out test engine IDs.")
        pipeline = RealOnlyAnomalyPipeline().fit(train_features, validation_features)
        validation_scores = pipeline.predict(validation_features)
        validation_truth = align_truth(validation_scores, split.validation, target)
        if set(validation_truth.index.astype(int)) & test_ids:
            raise AssertionError("Validation truth overlaps held-out test engines.")

        config_records: list[dict[str, object]] = []
        for threshold_name, threshold in candidate_thresholds(pipeline):
            metrics = threshold_metrics(validation_scores, validation_truth, threshold)
            entry = record(
                scope="validation",
                config=config,
                threshold_name=threshold_name,
                threshold=threshold,
                target=target,
                metrics=metrics,
                selected_from_validation=False,
            )
            validation_records.append(entry)
            config_records.append(entry)
            if threshold_name == "current 85th percentile watchlist":
                current_production_records.append(entry)

        selected = select_from_validation(config_records)
        selected["selected_from_validation"] = True
        selected_records.append(selected)
        prepared.append(PreparedConfiguration(config, pipeline, validation_scores, validation_truth, test_features))

    # Each configuration and its threshold are now locked from validation. Test
    # scores are evaluated once per locked configuration for descriptive only.
    locked_test_records: list[dict[str, object]] = []
    if len(prepared) != len(selected_records):
        raise AssertionError("Each prepared configuration must have exactly one validation-selected threshold.")
    for item, selected in zip(prepared, selected_records):
        test_scores = item.pipeline.predict(item.test_features)
        test_truth = align_truth(test_scores, split.test, target)
        metrics = threshold_metrics(test_scores, test_truth, float(selected["threshold"]))
        locked_test_records.append(
            record(
                scope="locked_test",
                config=item.config,
                threshold_name=f"locked validation selection: {selected['threshold_name']}",
                threshold=float(selected["threshold"]),
                target=target,
                metrics=metrics,
                selected_from_validation=True,
            )
        )

    write_csv(args.output, [*validation_records, *locked_test_records])

    print("\nCURRENT PRODUCTION WATCHLIST THRESHOLD ON VALIDATION")
    for entry in current_production_records:
        print_metrics(str(entry["configuration"]), entry)

    print("\nVALIDATION-SELECTED THRESHOLD PER CONFIGURATION")
    for entry in selected_records:
        print_metrics(f"{entry['configuration']} | {entry['threshold_name']}", entry)

    print("\nLOCKED HELD-OUT TEST RESULTS (descriptive, low-confidence: 15 engines)")
    for entry in locked_test_records:
        print_metrics(str(entry["configuration"]), entry)
    print(
        "\nThresholds were selected from validation recall, then validation FPR, then validation F1. "
        "Held-out test outcomes were not used to select a feature family, detector, or threshold."
    )
    print(f"Wrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
