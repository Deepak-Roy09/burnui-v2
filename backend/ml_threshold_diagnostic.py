"""Read-only validation-threshold diagnostic for the existing BurnUI pipeline.

Threshold selection uses validation data only. Exactly one configuration is
locked from validation and then scored once on the untouched held-out test set.
No production model, threshold, preprocessing, or evaluation code is changed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

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


WINDOWS = (30, 50, 75)
DETECTORS = (
    ("Z-score only", 1.0, 0.0),
    ("Isolation Forest only", 0.0, 1.0),
    ("50/50 fusion", 0.5, 0.5),
)
FIXED_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
METRICS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")


@dataclass(frozen=True)
class FittedExperiment:
    window: int
    detector: str
    pipeline: RealOnlyAnomalyPipeline
    target: object
    validation_scores: pd.DataFrame
    validation_truth: pd.DataFrame
    test_features: pd.DataFrame
    test_trajectories: pd.DataFrame


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Read train_FD001.txt from the supplied ZIP via the production adapter."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_features(frame: pd.DataFrame, window: int):
    return extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )


def prepared_features(split, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce production feature tables and assert no future test features leak in."""
    train = extract_features(split.train, window)
    validation = extract_features(split.validation, window)
    test = extract_features(split.test, window)
    for name, result in (("train", train), ("validation", validation), ("test", test)):
        if result.skipped_engines:
            raise AssertionError(f"Window 1-{window} skipped {name} engines: {result.skipped_engines}")
        if result.features.shape[1] - 1 != 168:
            raise AssertionError(f"Window 1-{window} {name} features are not the expected 168 sensor-derived features.")

    truncated_test = split.test.loc[split.test["cycle"] <= window].copy()
    pd.testing.assert_frame_equal(test.features, extract_features(truncated_test, window).features)
    return train.features, validation.features, test.features


def align_truth(predictions: pd.DataFrame, trajectories: pd.DataFrame, target) -> pd.DataFrame:
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")
    engine_ids = predictions["engine_id"].astype(int).tolist()
    if set(engine_ids) - set(truth.index):
        raise AssertionError("The current evaluation target is missing scored engines.")
    return truth.loc[engine_ids]


def threshold_metrics(scores: pd.DataFrame, truth: pd.DataFrame, threshold: float) -> dict[str, int | float | None]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Risk threshold must be in [0, 1], got {threshold}.")
    y_true = truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = scores["risk_score"].to_numpy(dtype=float)
    return calculate_binary_metrics(y_true, y_score >= threshold, y_score)


def result_record(
    *, scope: str, window: int, detector: str, threshold_name: str, threshold: float,
    metrics: dict[str, int | float | None], target, positive_count: int, selected: bool,
) -> dict[str, object]:
    return {
        "scope": scope,
        "window": f"1-{window}",
        "detector": detector,
        "threshold_name": threshold_name,
        "threshold": threshold,
        "target_remaining_cycle_horizon": target.remaining_cycle_horizon,
        "positive_count": positive_count,
        "selected_from_validation": selected,
        **{name: metrics[name] for name in METRICS},
    }


def candidate_thresholds(pipeline: RealOnlyAnomalyPipeline) -> list[tuple[str, float]]:
    """Keep the current production cuts and requested fixed 0–1 risk thresholds."""
    return [
        ("current 85th percentile watchlist", pipeline.thresholds.watchlist),
        ("current 95th percentile high-risk", pipeline.thresholds.high_risk),
        *((f"fixed {value:.2f}", value) for value in FIXED_THRESHOLDS),
    ]


def select_from_validation(records: list[dict[str, object]]) -> dict[str, object]:
    """Maximize recall, then minimize FPR, then maximize F1—only on validation."""
    def key(record: dict[str, object]) -> tuple[float, float, float]:
        recall = -1.0 if record["recall"] is None else float(record["recall"])
        fpr = 1.0 if record["fpr"] is None else float(record["fpr"])
        f1 = -1.0 if record["f1"] is None else float(record["f1"])
        return recall, -fpr, f1

    return max(records, key=key)


def format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def print_metrics(label: str, record: dict[str, object]) -> None:
    metrics = ", ".join(f"{name.upper()}={format_value(record[name])}" for name in METRICS)
    print(f"{label}: threshold={float(record['threshold']):.6f}; {metrics}")


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "scope", "window", "detector", "threshold_name", "threshold", "target_remaining_cycle_horizon",
        "positive_count", "selected_from_validation", *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_threshold_diagnostic_results.csv"))
    args = parser.parse_args()

    frame = load_fd001(args.dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    test_ids = set(split.metadata.test_engine_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train, validation, and test engine IDs must be disjoint.")
    print("BURNUI VALIDATION THRESHOLD DIAGNOSTIC")
    print(f"Engine split: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")

    validation_records: list[dict[str, object]] = []
    experiments: dict[tuple[str, str], FittedExperiment] = {}
    for window in WINDOWS:
        train_features, validation_features, test_features = prepared_features(split, window)
        if set(test_features["engine_id"].astype(int)) != test_ids:
            raise AssertionError(f"Window 1-{window} altered held-out test engine IDs.")
        target = fit_future_degradation_target(split.train, early_end_cycle=window)
        if set(target.training_engine_ids) & test_ids:
            raise AssertionError("Target training engines overlap held-out test engines.")

        for detector, z_weight, isolation_weight in DETECTORS:
            # The production pipeline fits preprocessing and Isolation Forest on
            # train only, then calibrates its current thresholds on validation.
            pipeline = RealOnlyAnomalyPipeline(z_weight=z_weight, isolation_weight=isolation_weight).fit(
                train_features, validation_features
            )
            validation_scores = pipeline.predict(validation_features)
            validation_truth = align_truth(validation_scores, split.validation, target)
            experiments[(f"1-{window}", detector)] = FittedExperiment(
                window, detector, pipeline, target, validation_scores, validation_truth, test_features, split.test
            )
            for threshold_name, threshold in candidate_thresholds(pipeline):
                metrics = threshold_metrics(validation_scores, validation_truth, threshold)
                validation_records.append(
                    result_record(
                        scope="validation", window=window, detector=detector, threshold_name=threshold_name,
                        threshold=threshold, metrics=metrics, target=target,
                        positive_count=int(validation_truth["near_term_degradation"].sum()), selected=False,
                    )
                )

    production_validation = next(
        record for record in validation_records
        if record["window"] == "1-30" and record["detector"] == "50/50 fusion"
        and record["threshold_name"] == "current 85th percentile watchlist"
    )
    selected = select_from_validation(validation_records)
    selected["selected_from_validation"] = True
    selected_experiment = experiments[(str(selected["window"]), str(selected["detector"]))]

    # First and only use of held-out test scores: configuration and threshold
    # have already been locked exclusively from validation records above.
    locked_test_scores = selected_experiment.pipeline.predict(selected_experiment.test_features)
    locked_test_truth = align_truth(locked_test_scores, selected_experiment.test_trajectories, selected_experiment.target)
    locked_test_metrics = threshold_metrics(locked_test_scores, locked_test_truth, float(selected["threshold"]))
    locked_test_record = result_record(
        scope="locked_test", window=selected_experiment.window, detector=selected_experiment.detector,
        threshold_name=f"locked validation selection: {selected['threshold_name']}", threshold=float(selected["threshold"]),
        metrics=locked_test_metrics, target=selected_experiment.target,
        positive_count=int(locked_test_truth["near_term_degradation"].sum()), selected=True,
    )
    write_csv(args.output, [*validation_records, locked_test_record])

    print("\nCURRENT PRODUCTION THRESHOLD ON VALIDATION")
    print_metrics("1-30 + 50/50 fusion watchlist", production_validation)
    print("\nVALIDATION-SELECTED THRESHOLD")
    print(
        f"Configuration: {selected['window']} + {selected['detector']} | "
        f"rule: {selected['threshold_name']} | validation positives: {selected['positive_count']}"
    )
    print_metrics("Validation", selected)
    print("\nLOCKED HELD-OUT TEST (descriptive, low-confidence: 15 engines)")
    print(f"Test positives: {locked_test_record['positive_count']}")
    print_metrics("Locked test", locked_test_record)
    print(
        "\nDifference: selection prioritized validation recall, then validation FPR, then validation F1; "
        "held-out test outcomes were not used for selection."
    )
    print(
        "Threshold calibration is a plausible binary-recall factor only if the validation-selected rule materially "
        "improves validation recall. The locked test outcome remains descriptive and low-confidence."
    )
    print(f"\nWrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
