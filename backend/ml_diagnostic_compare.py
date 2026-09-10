"""Read-only FD001 diagnostic comparison for BurnUI's existing anomaly pipeline.

This script does not alter production configuration or fit on held-out engines.
It is intentionally separate from the application and writes only the requested
CSV summary next to itself.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import pandas as pd

from app.evaluation.real_only_evaluation import (
    construct_future_degradation_truth,
    evaluate_held_out_test_set,
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
METRIC_COLUMNS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Load the FD001 train file directly from the supplied archive."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the supplied C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def feature_result(frame: pd.DataFrame, window: int):
    """Use the production feature extractor with a predefined early window."""
    return extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )


def validate_window_data(split, window: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assert the requested no-leakage and feature-shape invariants."""
    train_result = feature_result(split.train, window)
    validation_result = feature_result(split.validation, window)
    test_result = feature_result(split.test, window)
    results = (train_result, validation_result, test_result)

    if any(result.skipped_engines for result in results):
        skipped = {name: result.skipped_engines for name, result in zip(("train", "validation", "test"), results)}
        raise AssertionError(f"Window 1-{window} skipped engines: {skipped}")
    if any(result.features.shape[1] - 1 != 168 for result in results):
        shapes = [result.features.shape for result in results]
        raise AssertionError(f"Window 1-{window} did not yield 168 sensor-derived features: {shapes}")

    # The extractor receives full trajectories, but its output must be identical
    # to one built from data truncated at the early-window endpoint.
    truncated_test = split.test.loc[split.test["cycle"] <= window].copy()
    pd.testing.assert_frame_equal(test_result.features, feature_result(truncated_test, window).features)
    return train_result.features, validation_result.features, test_result.features


def metric_record(window: int, detector: str, metrics: dict[str, int | float | None], target: dict) -> dict[str, object]:
    record: dict[str, object] = {
        "window": f"1-{window}",
        "detector": detector,
        "target_remaining_cycle_horizon": target["remaining_cycle_horizon"],
    }
    record.update({name: metrics[name] for name in METRIC_COLUMNS})
    return record


def top_feature_summary(
    pipeline: RealOnlyAnomalyPipeline,
    test_features: pd.DataFrame,
    test_trajectories: pd.DataFrame,
    target,
) -> list[tuple[str, float, int, int]]:
    """Summarize the pipeline's existing Z-contribution explanations on test positives.

    Isolation Forest has no built-in per-feature attribution, so this reports the
    production explanation signal rather than inventing one for the forest.
    """
    predictions = pipeline.predict(test_features).set_index("engine_id")
    truth = construct_future_degradation_truth(test_trajectories, target).set_index("engine_id")
    positive_ids = truth.index[truth["near_term_degradation"]].tolist()
    contribution_by_feature: dict[str, list[float]] = defaultdict(list)
    appearances: dict[str, int] = defaultdict(int)

    for engine_id in positive_ids:
        for feature in predictions.loc[engine_id, "explanation"]["top_features"]:
            contribution_by_feature[feature["feature"]].append(float(feature["z_contribution"]))
            appearances[feature["feature"]] += 1

    return sorted(
        (
            (feature, mean(contributions), appearances[feature], len(positive_ids))
            for feature, contributions in contribution_by_feature.items()
        ),
        key=lambda item: (-item[1], -item[2], item[0]),
    )[:5]


def print_dataset_summary(frame: pd.DataFrame) -> None:
    endpoints = frame.groupby("engine_id")["cycle"].max()
    print("BURNUI ML COMPARISON DIAGNOSTIC\n")
    print("DATASET SUMMARY")
    print(f"- rows: {len(frame)}")
    print(f"- engines: {frame['engine_id'].nunique()}")
    print(
        "- trajectory end cycles: "
        f"min={int(endpoints.min())}, p25={endpoints.quantile(0.25):.2f}, "
        f"median={endpoints.median():.2f}, p75={endpoints.quantile(0.75):.2f}, max={int(endpoints.max())}"
    )
    for window in WINDOWS:
        usable = int((endpoints >= window).sum())
        print(f"- engines usable for 1-{window}: {usable}; excluded: {len(endpoints) - usable}")


def print_results(records: list[dict[str, object]]) -> None:
    print("\nRESULTS")
    columns = ("Window", "Detector", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "FPR", "FNR", "PR-AUC", "ROC-AUC")
    print(" | ".join(columns))
    for record in records:
        values = [
            record["window"], record["detector"], record["tp"], record["tn"], record["fp"], record["fn"],
            *(_format_metric(record[name]) for name in ("precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")),
        ]
        print(" | ".join(str(value) for value in values))


def _format_metric(value: object) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def print_rankings(records: list[dict[str, object]]) -> None:
    def best(metric: str) -> dict[str, object] | None:
        available = [record for record in records if record[metric] is not None]
        return max(available, key=lambda record: float(record[metric])) if available else None

    print("\nHELD-OUT DESCRIPTIVE RANKINGS (not model-selection results)")
    for label, metric in (("best Recall", "recall"), ("best PR-AUC", "pr_auc"), ("best F1", "f1")):
        record = best(metric)
        result = "N/A" if record is None else f"{record['window']} + {record['detector']} ({_format_metric(record[metric])})"
        print(f"- {label}: {result}")

    production = next(record for record in records if record["window"] == "1-30" and record["detector"] == "50/50 fusion")
    print(
        "- existing production configuration (1-30 + 50/50 fusion): "
        f"Recall={_format_metric(production['recall'])}, PR-AUC={_format_metric(production['pr_auc'])}, "
        f"F1={_format_metric(production['f1'])}"
    )
    print("- interpretation: fixed-split held-out diagnostics are descriptive, not a basis for production model selection.")


def write_results(path: Path, records: list[dict[str, object]]) -> None:
    fields = ["window", "detector", "target_remaining_cycle_horizon", *METRIC_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_diagnostic_results.csv"))
    args = parser.parse_args()

    frame = load_fd001(args.dataset)
    print_dataset_summary(frame)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    expected_test_ids = set(split.metadata.test_engine_ids)
    if train_ids & validation_ids or train_ids & expected_test_ids or validation_ids & expected_test_ids:
        raise AssertionError("Engine split is not disjoint.")
    print("\nENGINE SPLIT")
    print(f"- train count: {len(train_ids)}")
    print(f"- validation count: {len(validation_ids)}")
    print(f"- test count: {len(expected_test_ids)}")

    records: list[dict[str, object]] = []
    for window in WINDOWS:
        train_features, validation_features, test_features = validate_window_data(split, window)
        if set(test_features["engine_id"]) != expected_test_ids:
            raise AssertionError(f"Window 1-{window} changed the held-out test engine IDs.")
        target = fit_future_degradation_target(split.train, early_end_cycle=window)
        if set(target.training_engine_ids) & expected_test_ids:
            raise AssertionError("Target calibration engines overlap test engines.")

        print(f"\nWINDOW 1-{window}: {train_features.shape[1] - 1} sensor-derived features; no engines excluded")
        fusion_for_features: RealOnlyAnomalyPipeline | None = None
        for detector, z_weight, isolation_weight in DETECTORS:
            pipeline = RealOnlyAnomalyPipeline(z_weight=z_weight, isolation_weight=isolation_weight).fit(
                train_features,
                validation_features,
            )
            evaluation = evaluate_held_out_test_set(pipeline, test_features, split.test, target)
            if set(evaluation["test_engine_ids"]) != expected_test_ids:
                raise AssertionError("Evaluation did not preserve the held-out test engine IDs.")
            records.append(metric_record(window, detector, evaluation["metrics"], evaluation["target"]))
            if detector == "50/50 fusion":
                fusion_for_features = pipeline

        assert fusion_for_features is not None
        feature_summary = top_feature_summary(fusion_for_features, test_features, split.test, target)
        print("- strongest returned Z-contribution features among held-out target-positive engines:")
        if feature_summary:
            for feature, average_contribution, appearances, positive_count in feature_summary:
                print(f"  {feature}: mean contribution={average_contribution:.3f}, top-three appearances={appearances}/{positive_count}")
        else:
            print("  none (the held-out target contains no positives)")

    write_results(args.output, records)
    print_results(records)
    print_rankings(records)
    print(f"\nWrote diagnostic metrics CSV: {args.output}")


if __name__ == "__main__":
    main()
