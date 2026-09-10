"""Read-only feature-family diagnostic for BurnUI's existing anomaly pipeline.

This compares subsets of production-generated feature columns. It does not add
features, alter pipeline behavior, calibrate on held-out data, or select a
production model from the held-out comparison.
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
    evaluate_held_out_test_set,
    fit_future_degradation_target,
)
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import split_by_engine


WINDOWS = (30, 50, 75)
RECURRING_SENSORS = ("sensor_12", "sensor_3", "sensor_20", "sensor_21", "sensor_2", "sensor_4")
EXPECTED_POSITIVE_IDS = (13, 36, 65)
METRICS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")


@dataclass(frozen=True)
class Experiment:
    window: int
    family: str
    feature_count: int
    pipeline: RealOnlyAnomalyPipeline
    validation_features: pd.DataFrame
    test_features: pd.DataFrame
    target: object


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Read the original FD001 train data directly from the ZIP archive."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the supplied ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_window_features(frame: pd.DataFrame, window: int):
    return extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )


def feature_families(features: pd.DataFrame) -> dict[str, list[str]]:
    """Select only existing feature columns by their production naming scheme."""
    columns = [column for column in features.columns if column != "engine_id"]
    if len(columns) != 168:
        raise AssertionError(f"Expected 168 production sensor-derived features, found {len(columns)}.")
    temporal_suffixes = {"first", "last", "delta", "slope"}
    distribution_suffixes = {"mean", "std", "min", "max"}
    families = {
        "All current features": columns,
        "Temporal only": [column for column in columns if column.rsplit("_", 1)[-1] in temporal_suffixes],
        "Distribution/statistical only": [column for column in columns if column.rsplit("_", 1)[-1] in distribution_suffixes],
        "Recurring-signal sensors": [
            column for column in columns if any(column.startswith(f"{sensor}_") for sensor in RECURRING_SENSORS)
        ],
    }
    expected_counts = {
        "All current features": 168,
        "Temporal only": 84,
        "Distribution/statistical only": 84,
        "Recurring-signal sensors": 48,
    }
    if {name: len(columns) for name, columns in families.items()} != expected_counts:
        raise AssertionError(f"Unexpected feature family sizes: { {name: len(columns) for name, columns in families.items()} }")
    return families


def select_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return features.loc[:, ["engine_id", *columns]].copy()


def align_truth(predictions: pd.DataFrame, trajectories: pd.DataFrame, target) -> pd.DataFrame:
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id")
    ids = predictions["engine_id"].astype(int).tolist()
    if set(ids) - set(truth.index):
        raise AssertionError("Evaluation target is missing scored engines.")
    return truth.loc[ids]


def validation_metrics(pipeline: RealOnlyAnomalyPipeline, validation_features: pd.DataFrame, validation_trajectories: pd.DataFrame, target):
    predictions = pipeline.predict(validation_features)
    truth = align_truth(predictions, validation_trajectories, target)
    y_true = truth["near_term_degradation"].to_numpy(dtype=bool)
    y_score = predictions["risk_score"].to_numpy(dtype=float)
    y_pred = predictions["classification"].ne("Normal").to_numpy(dtype=bool)
    return calculate_binary_metrics(y_true, y_pred, y_score)


def descending_rank(values: pd.Series, engine_id: int) -> tuple[int, float]:
    ordered = values.sort_values(ascending=False, kind="stable").index.astype(int).tolist()
    rank = ordered.index(engine_id) + 1
    percentile = 100.0 if len(ordered) == 1 else 100.0 * (len(ordered) - rank) / (len(ordered) - 1)
    return rank, percentile


def positive_details(predictions: pd.DataFrame, truth: pd.DataFrame, pipeline: RealOnlyAnomalyPipeline) -> tuple[list[dict[str, object]], float]:
    indexed = predictions.set_index("engine_id")
    risk_values = indexed["risk_score"]
    details: list[dict[str, object]] = []
    for engine_id in EXPECTED_POSITIVE_IDS:
        row = indexed.loc[engine_id]
        rank, percentile = descending_rank(risk_values, engine_id)
        negatives = truth.index[~truth["near_term_degradation"]]
        lower_negative_count = int((risk_values.loc[negatives] < row["risk_score"]).sum())
        details.append(
            {
                "engine_id": engine_id,
                "risk_score": float(row["risk_score"]),
                "rank_descending": rank,
                "rank_percentile": percentile,
                "negative_engines_ranked_below": lower_negative_count,
                "crosses_watchlist": bool(row["risk_score"] >= pipeline.thresholds.watchlist),
                "crosses_high_risk": bool(row["risk_score"] >= pipeline.thresholds.high_risk),
            }
        )
    return details, float(np.mean([detail["rank_percentile"] for detail in details]))


def format_metric(value: object) -> str:
    return "N/A" if value is None else (str(value) if isinstance(value, int) else f"{float(value):.3f}")


def record(
    experiment: Experiment,
    validation: dict[str, int | float | None],
    test: dict[str, int | float | None],
    details: list[dict[str, object]],
    mean_positive_rank_percentile: float,
) -> dict[str, object]:
    return {
        "window": f"1-{experiment.window}",
        "feature_family": experiment.family,
        "feature_count": experiment.feature_count,
        **{f"validation_{name}": validation[name] for name in METRICS},
        **{f"test_{name}": test[name] for name in METRICS},
        "positive_mean_rank_percentile": mean_positive_rank_percentile,
        "positive_engine_details": json.dumps(details),
    }


def best_validation(records: list[dict[str, object]], metric: str) -> dict[str, object] | None:
    available = [record for record in records if record[f"validation_{metric}"] is not None]
    return max(available, key=lambda record: float(record[f"validation_{metric}"])) if available else None


def print_validation_ranking(records: list[dict[str, object]]) -> None:
    print("\nVALIDATION-ONLY RANKING (not a production model-selection decision)")
    for metric in ("recall", "pr_auc", "roc_auc"):
        best = best_validation(records, metric)
        if best is not None:
            print(
                f"- best validation {metric}: {best['window']} + {best['feature_family']} "
                f"({format_metric(best[f'validation_{metric}'])})"
            )


def print_test_report(records: list[dict[str, object]]) -> None:
    print("\nHELD-OUT TEST COMPARISON (descriptive and low-confidence: 15 engines, 3 positives)")
    print("Window | Family | Features | TP | TN | FP | FN | Recall | PR-AUC | ROC-AUC | Mean positive rank percentile")
    for item in records:
        print(
            f"{item['window']} | {item['feature_family']} | {item['feature_count']} | "
            f"{item['test_tp']} | {item['test_tn']} | {item['test_fp']} | {item['test_fn']} | "
            f"{format_metric(item['test_recall'])} | {format_metric(item['test_pr_auc'])} | "
            f"{format_metric(item['test_roc_auc'])} | {item['positive_mean_rank_percentile']:.1f}"
        )


def print_answers(records: list[dict[str, object]]) -> None:
    strong_profiles: list[str] = []
    for item in records:
        details = json.loads(str(item["positive_engine_details"]))
        engines = [str(detail["engine_id"]) for detail in details if detail["negative_engines_ranked_below"] > 6]
        if engines:
            strong_profiles.append(f"{item['window']} + {item['feature_family']}: engines {', '.join(engines)}")
    by_family: dict[str, dict[str, float]] = {}
    for item in records:
        by_family.setdefault(str(item["feature_family"]), {})[str(item["window"])] = float(item["positive_mean_rank_percentile"])
    improves_50 = all(values.get("1-50", -np.inf) > values.get("1-30", np.inf) for values in by_family.values())
    improves_75 = all(values.get("1-75", -np.inf) > values.get("1-30", np.inf) for values in by_family.values())

    print("\nDESCRIPTIVE ANSWERS")
    print("1–3. See validation-only ranking above; it is the only ranking suitable for follow-up investigation.")
    print("4. Positive profiles above more than half of negative engines: " + ("; ".join(strong_profiles) if strong_profiles else "none"))
    print(f"5. 1-50 consistently improves mean positive ranking over 1-30 across families: {improves_50}")
    print(f"   1-75 consistently improves mean positive ranking over 1-30 across families: {improves_75}")
    print(
        "6. This is not enough evidence to isolate feature representation, Isolation Forest, calibration, or early-signal limitations. "
        "Comparable validation and low-confidence test results would support a combination of weak early signal and representation/detector limits; "
        "threshold calibration is held fixed here and is not tested by this diagnostic."
    )


def write_results(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_feature_family_diagnostic_results.csv"))
    args = parser.parse_args()

    frame = load_fd001(args.dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    test_ids = set(split.metadata.test_engine_ids)
    if len(train_ids) != 70 or len(validation_ids) != 15 or len(test_ids) != 15:
        raise AssertionError("Expected the production 70/15/15 engine split.")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Production split engine IDs are not disjoint.")
    target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=0.20)
    if set(target.training_engine_ids) & test_ids:
        raise AssertionError("Target-calibration engines overlap test engines.")
    test_truth = construct_future_degradation_truth(split.test, target).set_index("engine_id")
    positive_ids = tuple(sorted(int(engine_id) for engine_id in test_truth.index[test_truth["near_term_degradation"]]))
    if positive_ids != EXPECTED_POSITIVE_IDS:
        raise AssertionError(f"Expected held-out target positives {EXPECTED_POSITIVE_IDS}, found {positive_ids}")

    print("BURNUI FEATURE-FAMILY DIAGNOSTIC")
    print(f"Split: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")
    print(f"Target: early_end_cycle=30, training_quantile=0.20, horizon={target.remaining_cycle_horizon:.3f}")
    print(f"Held-out positives: {positive_ids}")

    validation_records: list[dict[str, object]] = []
    experiments: list[Experiment] = []
    for window in WINDOWS:
        train_result = extract_window_features(split.train, window)
        validation_result = extract_window_features(split.validation, window)
        test_result = extract_window_features(split.test, window)
        for result in (train_result, validation_result, test_result):
            if result.skipped_engines or result.features.shape[1] - 1 != 168:
                raise AssertionError(f"Window 1-{window} must retain every engine and 168 production features.")
        if set(test_result.features["engine_id"].astype(int)) != test_ids:
            raise AssertionError(f"Window 1-{window} changed test engine IDs.")
        truncated_test = split.test.loc[split.test["cycle"] <= window].copy()
        pd.testing.assert_frame_equal(test_result.features, extract_window_features(truncated_test, window).features)

        for family, columns in feature_families(train_result.features).items():
            train_features = select_features(train_result.features, columns)
            validation_features = select_features(validation_result.features, columns)
            test_features = select_features(test_result.features, columns)
            pipeline = RealOnlyAnomalyPipeline().fit(train_features, validation_features)
            validation = validation_metrics(pipeline, validation_features, split.validation, target)
            experiment = Experiment(window, family, len(columns), pipeline, validation_features, test_features, target)
            experiments.append(experiment)
            validation_records.append(
                {
                    "window": f"1-{window}",
                    "feature_family": family,
                    "feature_count": len(columns),
                    **{f"validation_{name}": validation[name] for name in METRICS},
                }
            )

    print_validation_ranking(validation_records)

    # Held-out scores are produced only after all validation diagnostics above;
    # no test result is used to select a feature family or model.
    complete_records: list[dict[str, object]] = []
    for experiment, validation_record in zip(experiments, validation_records):
        evaluation = evaluate_held_out_test_set(
            experiment.pipeline, experiment.test_features, split.test, experiment.target
        )
        predictions = experiment.pipeline.predict(experiment.test_features)
        truth = align_truth(predictions, split.test, experiment.target)
        details, mean_rank = positive_details(predictions, truth, experiment.pipeline)
        complete_records.append(record(experiment, {name: validation_record[f"validation_{name}"] for name in METRICS}, evaluation["metrics"], details, mean_rank))

    print_test_report(complete_records)
    print_answers(complete_records)
    write_results(args.output, complete_records)
    print(f"\nWrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
