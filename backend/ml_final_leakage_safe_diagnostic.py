"""Final leakage-safe, descriptive comparison of the BurnUI baseline and candidate.

The candidate selects sensor families from training data only, using the
existing 1-50 early-window features and the existing training-derived C-MAPSS
proxy target. Validation is reserved for threshold selection. Held-out test
engines are not accessed until both candidate sensors and threshold are frozen.
No production code, data, model configuration, or evaluation code is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
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
TARGET_EARLY_END_CYCLE = 30
TARGET_TRAINING_QUANTILE = 0.20
EXPECTED_TARGET_HORIZON = 135.800
CANDIDATE_SENSOR_COUNT = 6


@dataclass(frozen=True)
class Configuration:
    name: str
    window: int
    feature_columns: tuple[str, ...]
    selected_sensors: tuple[str, ...]
    threshold_name: str
    threshold: float


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Load the original FD001 train file directly from the source ZIP."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_window_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Use only the existing extractor and require all engines for a window."""
    result = extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )
    if result.skipped_engines:
        raise AssertionError(f"Window 1-{window} skipped engines: {result.skipped_engines}")
    if result.features.shape[1] - 1 != 168:
        raise AssertionError(f"Window 1-{window} must yield exactly 168 existing sensor-derived features.")
    return result.features


def align_truth(features: pd.DataFrame, trajectories: pd.DataFrame, target) -> pd.DataFrame:
    """Align existing post-hoc truth to scored engines without changing its logic."""
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")
    engine_ids = features["engine_id"].astype(int).tolist()
    if set(engine_ids) - set(truth.index.astype(int)):
        raise AssertionError("The existing target is missing engines that have feature rows.")
    return truth.loc[engine_ids]


def feature_sensor(feature_name: str) -> str:
    sensor, _, statistic = feature_name.rpartition("_")
    if not sensor.startswith("sensor_") or statistic not in {"mean", "std", "min", "max", "first", "last", "delta", "slope"}:
        raise AssertionError(f"Unexpected existing feature name: {feature_name}")
    return sensor


def sensor_number(sensor: str) -> int:
    return int(sensor.removeprefix("sensor_"))


def select_candidate_sensors(train_features: pd.DataFrame, train_truth: pd.DataFrame) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Select six sensor families from *training data only*.

    Rule: for each existing 1-50 feature, calculate absolute Pearson
    association with the existing binary training proxy target. A sensor's
    score is the mean of its eight feature associations, with non-finite or
    constant-feature associations treated as zero. Select the six highest
    scoring sensors; resolve ties by ascending sensor number.

    The target is used only for this diagnostic feature subset selection. The
    anomaly pipeline remains the existing unsupervised Z-score/Isolation Forest
    implementation and receives no target labels.
    """
    truth = train_truth.loc[train_features["engine_id"].astype(int), "near_term_degradation"].to_numpy(dtype=float)
    feature_columns = [column for column in train_features.columns if column != "engine_id"]
    by_sensor: dict[str, list[tuple[str, float]]] = {}
    for column in feature_columns:
        values = pd.to_numeric(train_features[column], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values) & np.isfinite(truth)
        association = 0.0
        if usable.sum() >= 2 and np.unique(values[usable]).size > 1 and np.unique(truth[usable]).size > 1:
            coefficient = float(np.corrcoef(values[usable], truth[usable])[0, 1])
            association = 0.0 if not np.isfinite(coefficient) else abs(coefficient)
        by_sensor.setdefault(feature_sensor(column), []).append((column, association))

    sensor_rows: list[dict[str, object]] = []
    for sensor, values in by_sensor.items():
        if len(values) != 8:
            raise AssertionError(f"{sensor} must have the existing eight feature types, found {len(values)}.")
        sensor_rows.append(
            {
                "sensor": sensor,
                "feature_count": len(values),
                "mean_absolute_training_association": float(np.mean([score for _, score in values])),
                "feature_associations": {name: score for name, score in values},
            }
        )
    if len(sensor_rows) != 21:
        raise AssertionError(f"Expected 21 existing sensor families, found {len(sensor_rows)}.")
    sensor_rows.sort(key=lambda row: (-float(row["mean_absolute_training_association"]), sensor_number(str(row["sensor"]))))
    for rank, row in enumerate(sensor_rows, start=1):
        row["training_selection_rank"] = rank
        row["selected"] = rank <= CANDIDATE_SENSOR_COUNT

    selected = tuple(str(row["sensor"]) for row in sensor_rows[:CANDIDATE_SENSOR_COUNT])
    return selected, sensor_rows


def feature_columns_for_sensors(features: pd.DataFrame, sensors: tuple[str, ...]) -> tuple[str, ...]:
    prefixes = tuple(f"{sensor}_" for sensor in sensors)
    selected = tuple(column for column in features.columns if column != "engine_id" and column.startswith(prefixes))
    expected = CANDIDATE_SENSOR_COUNT * 8
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} existing candidate features, found {len(selected)}.")
    return selected


def with_columns(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return features.loc[:, ["engine_id", *columns]].copy()


def metrics_at_threshold(scores: pd.DataFrame, truth: pd.DataFrame, threshold: float) -> dict[str, int | float | None]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be within the existing 0-1 risk-score scale, got {threshold}.")
    y_true = truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = scores["risk_score"].to_numpy(dtype=float)
    return calculate_binary_metrics(y_true, y_score >= threshold, y_score)


def validation_thresholds(pipeline: RealOnlyAnomalyPipeline) -> list[tuple[str, float]]:
    """Use the same validation-only threshold candidates as the prior diagnostic."""
    return [
        ("current 85th percentile watchlist", pipeline.thresholds.watchlist),
        ("current 95th percentile high-risk", pipeline.thresholds.high_risk),
        *((f"fixed {value:.2f}", value) for value in FIXED_THRESHOLDS),
    ]


def select_validation_threshold(records: list[dict[str, object]]) -> dict[str, object]:
    """Prioritize recall, then lower FPR, then F1 with deterministic first-tie behavior."""
    def key(record: dict[str, object]) -> tuple[float, float, float]:
        recall = -1.0 if record["recall"] is None else float(record["recall"])
        fpr = 1.0 if record["fpr"] is None else float(record["fpr"])
        f1 = -1.0 if record["f1"] is None else float(record["f1"])
        return recall, -fpr, f1

    return max(records, key=key)


def configuration_record(
    *,
    record_type: str,
    configuration: Configuration,
    metrics: dict[str, int | float | None] | None,
    target,
    feature_selection_rule: str,
    leakage_checks: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "record_type": record_type,
        "configuration": configuration.name,
        "feature_window": f"1-{configuration.window}",
        "feature_count": len(configuration.feature_columns),
        "selected_sensors": ";".join(configuration.selected_sensors),
        "selected_features": ";".join(configuration.feature_columns),
        "threshold_name": configuration.threshold_name,
        "threshold": configuration.threshold,
        "target_early_end_cycle": target.early_end_cycle,
        "target_training_quantile": target.training_quantile,
        "target_remaining_cycle_horizon": target.remaining_cycle_horizon,
        "feature_selection_rule": feature_selection_rule,
        "leakage_checks": leakage_checks,
        "confusion_matrix": "",
        "training_selection_rank": "",
        "training_sensor_score": "",
        "selected_for_candidate": "",
        **{metric: "" for metric in METRICS},
    }
    if metrics is not None:
        record.update({metric: metrics[metric] for metric in METRICS})
        record["confusion_matrix"] = json.dumps(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        )
    return record


def sensor_record(sensor_row: dict[str, object]) -> dict[str, object]:
    return {
        "record_type": "training_sensor_selection",
        "configuration": "Candidate: 1-50 training-selected sensors",
        "feature_window": "1-50",
        "feature_count": sensor_row["feature_count"],
        "selected_sensors": sensor_row["sensor"],
        "selected_features": json.dumps(sensor_row["feature_associations"], sort_keys=True),
        "threshold_name": "",
        "threshold": "",
        "target_early_end_cycle": TARGET_EARLY_END_CYCLE,
        "target_training_quantile": TARGET_TRAINING_QUANTILE,
        "target_remaining_cycle_horizon": "",
        "feature_selection_rule": "Mean absolute training-only feature/target Pearson association; rank ties by sensor number.",
        "leakage_checks": f"training_selection_rank={sensor_row['training_selection_rank']}; selected={sensor_row['selected']}",
        "confusion_matrix": "",
        "training_selection_rank": sensor_row["training_selection_rank"],
        "training_sensor_score": sensor_row["mean_absolute_training_association"],
        "selected_for_candidate": sensor_row["selected"],
        "tp": "",
        "tn": "",
        "fp": "",
        "fn": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "fpr": "",
        "fnr": "",
        "pr_auc": "",
        "roc_auc": "",
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "record_type", "configuration", "feature_window", "feature_count", "selected_sensors", "selected_features",
        "threshold_name", "threshold", "target_early_end_cycle", "target_training_quantile",
        "target_remaining_cycle_horizon", "feature_selection_rule", "leakage_checks", "confusion_matrix", *METRICS,
        "training_selection_rank", "training_sensor_score", "selected_for_candidate",
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def format_value(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def print_metrics(label: str, metrics: dict[str, int | float | None]) -> None:
    measures = ", ".join(f"{metric.upper()}={format_value(metrics[metric])}" for metric in METRICS)
    print(f"{label}: {measures}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("ml_final_leakage_safe_diagnostic_results.csv"),
    )
    args = parser.parse_args()

    frame = load_fd001(args.dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    # Intentionally do not access test IDs, trajectories, features, rankings,
    # contributions, or labels until after candidate selection is frozen.
    if (len(train_ids), len(validation_ids)) != (70, 15):
        raise AssertionError("The required deterministic split must have 70 train and 15 validation engines.")
    if train_ids & validation_ids:
        raise AssertionError("Training and validation engine IDs overlap.")

    target = fit_future_degradation_target(
        split.train,
        early_end_cycle=TARGET_EARLY_END_CYCLE,
        training_quantile=TARGET_TRAINING_QUANTILE,
    )
    if not np.isclose(target.remaining_cycle_horizon, EXPECTED_TARGET_HORIZON):
        raise AssertionError(
            f"Expected training-derived horizon {EXPECTED_TARGET_HORIZON:.3f}, "
            f"found {target.remaining_cycle_horizon:.3f}."
        )
    if set(target.training_engine_ids) != train_ids:
        raise AssertionError("The target must be fitted using exactly the training-engine trajectories.")

    print("FINAL LEAKAGE-SAFE DIAGNOSTIC — DESCRIPTIVE, LOW-CONFIDENCE DUE TO 15-ENGINE TEST SET")
    print(f"Engine split before final evaluation: train={len(train_ids)}, validation={len(validation_ids)}, test=untouched")
    print(
        f"Target: early_end_cycle={target.early_end_cycle}, training_quantile={target.training_quantile:.2f}, "
        f"training-derived horizon={target.remaining_cycle_horizon:.3f}"
    )

    # ----- Training/validation-only phase: baseline and candidate are frozen. -----
    baseline_train_all = extract_window_features(split.train, window=30)
    baseline_validation_all = extract_window_features(split.validation, window=30)
    baseline_columns = tuple(column for column in baseline_train_all.columns if column != "engine_id")
    if len(baseline_columns) != 168:
        raise AssertionError("Baseline must retain all 168 existing features.")
    baseline_pipeline = RealOnlyAnomalyPipeline().fit(baseline_train_all, baseline_validation_all)
    baseline_validation_scores = baseline_pipeline.predict(baseline_validation_all)
    baseline_validation_truth = align_truth(baseline_validation_scores, split.validation, target)
    baseline_configuration = Configuration(
        name="Baseline: 1-30 all 168 features, existing 50/50 fusion",
        window=30,
        feature_columns=baseline_columns,
        selected_sensors=tuple(f"sensor_{number}" for number in range(1, 22)),
        threshold_name="existing 85th percentile watchlist",
        threshold=baseline_pipeline.thresholds.watchlist,
    )
    baseline_validation_metrics = metrics_at_threshold(
        baseline_validation_scores, baseline_validation_truth, baseline_configuration.threshold
    )

    candidate_train_all = extract_window_features(split.train, window=50)
    candidate_validation_all = extract_window_features(split.validation, window=50)
    candidate_train_truth = align_truth(candidate_train_all, split.train, target)
    selected_sensors, sensor_rows = select_candidate_sensors(candidate_train_all, candidate_train_truth)
    candidate_columns = feature_columns_for_sensors(candidate_train_all, selected_sensors)
    candidate_train = with_columns(candidate_train_all, candidate_columns)
    candidate_validation = with_columns(candidate_validation_all, candidate_columns)
    candidate_pipeline = RealOnlyAnomalyPipeline().fit(candidate_train, candidate_validation)
    candidate_validation_scores = candidate_pipeline.predict(candidate_validation)
    candidate_validation_truth = align_truth(candidate_validation_scores, split.validation, target)
    threshold_records: list[dict[str, object]] = []
    for threshold_name, threshold in validation_thresholds(candidate_pipeline):
        threshold_records.append(
            {
                "threshold_name": threshold_name,
                "threshold": threshold,
                **metrics_at_threshold(candidate_validation_scores, candidate_validation_truth, threshold),
            }
        )
    frozen_threshold = select_validation_threshold(threshold_records)
    candidate_configuration = Configuration(
        name="Candidate: 1-50 training-selected sensor features, existing 50/50 fusion",
        window=50,
        feature_columns=candidate_columns,
        selected_sensors=selected_sensors,
        threshold_name=str(frozen_threshold["threshold_name"]),
        threshold=float(frozen_threshold["threshold"]),
    )
    candidate_validation_metrics = metrics_at_threshold(
        candidate_validation_scores, candidate_validation_truth, candidate_configuration.threshold
    )

    selection_rule = (
        "Training-only: rank the 21 sensors by mean absolute Pearson association between each sensor's eight existing "
        "1-50 features and the existing training proxy target; select the top six, ties by ascending sensor number."
    )
    print("\nCANDIDATE FEATURE SELECTION — TRAINING ONLY")
    print(selection_rule)
    print(f"Selected sensors ({len(selected_sensors)}): {', '.join(selected_sensors)}")
    print(f"Selected existing features: {len(candidate_columns)}")
    print("Sensor ranking (mean absolute training association):")
    for row in sensor_rows:
        marker = " selected" if bool(row["selected"]) else ""
        print(
            f"  {int(row['training_selection_rank']):>2}. {row['sensor']}: "
            f"{float(row['mean_absolute_training_association']):.6f}{marker}"
        )
    print("\nVALIDATION RESULTS — TEST STILL UNTOUCHED")
    print_metrics(f"Baseline | {baseline_configuration.threshold_name} ({baseline_configuration.threshold:.6f})", baseline_validation_metrics)
    print_metrics(f"Candidate | frozen {candidate_configuration.threshold_name} ({candidate_configuration.threshold:.6f})", candidate_validation_metrics)

    # ----- Final held-out phase: first access to test engines and their future data. -----
    test_ids = set(split.metadata.test_engine_ids)
    if len(test_ids) != 15 or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Final held-out split must contain 15 engines disjoint from train and validation.")
    if set(target.training_engine_ids) & test_ids:
        raise AssertionError("Target-fitting engines overlap held-out test engines.")

    def final_test_metrics(configuration: Configuration, pipeline: RealOnlyAnomalyPipeline) -> dict[str, int | float | None]:
        # Pass only early-window test rows to feature extraction. Future test
        # observations are intentionally withheld from all model features.
        early_test_rows = split.test.loc[split.test["cycle"] <= configuration.window].copy()
        if not early_test_rows["cycle"].between(1, configuration.window).all():
            raise AssertionError("Test feature rows contain cycles outside the frozen early window.")
        test_all = extract_window_features(early_test_rows, configuration.window)
        test_features = with_columns(test_all, configuration.feature_columns)
        if set(test_features["engine_id"].astype(int)) != test_ids:
            raise AssertionError("Final test feature extraction altered the held-out test engine set.")
        test_scores = pipeline.predict(test_features)
        # This is the first and only post-hoc use of future test trajectories:
        # constructing the existing observed-endpoint truth.
        test_truth = align_truth(test_scores, split.test, target)
        return metrics_at_threshold(test_scores, test_truth, configuration.threshold)

    baseline_test_metrics = final_test_metrics(baseline_configuration, baseline_pipeline)
    candidate_test_metrics = final_test_metrics(candidate_configuration, candidate_pipeline)

    leakage_checks = (
        "Target horizon fitted from train only; candidate sensors selected from train only; candidate threshold selected "
        "from validation only; no test features, labels, rankings, or contributions accessed before both were frozen; "
        "test model inputs used only cycles within each early window; future test trajectories used only post-hoc for truth."
    )
    print("\nFINAL HELD-OUT TEST RESULTS")
    print_metrics("Baseline", baseline_test_metrics)
    print_metrics("Candidate", candidate_test_metrics)
    print("\nLEAKAGE CHECKS")
    print(f"- {leakage_checks}")
    print("- Detector difference: none. Both configurations use the existing default 50/50 Z-score + Isolation Forest pipeline.")

    results: list[dict[str, object]] = [
        *(sensor_record(row) for row in sensor_rows),
        configuration_record(
            record_type="validation",
            configuration=baseline_configuration,
            metrics=baseline_validation_metrics,
            target=target,
            feature_selection_rule="Baseline retains all existing features; no feature selection.",
            leakage_checks=leakage_checks,
        ),
        configuration_record(
            record_type="validation",
            configuration=candidate_configuration,
            metrics=candidate_validation_metrics,
            target=target,
            feature_selection_rule=selection_rule,
            leakage_checks=leakage_checks,
        ),
        configuration_record(
            record_type="final_held_out_test",
            configuration=baseline_configuration,
            metrics=baseline_test_metrics,
            target=target,
            feature_selection_rule="Baseline retains all existing features; no feature selection.",
            leakage_checks=leakage_checks,
        ),
        configuration_record(
            record_type="final_held_out_test",
            configuration=candidate_configuration,
            metrics=candidate_test_metrics,
            target=target,
            feature_selection_rule=selection_rule,
            leakage_checks=leakage_checks,
        ),
    ]
    write_csv(args.output, results)
    print(f"\nWrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
