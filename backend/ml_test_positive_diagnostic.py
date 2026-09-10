"""Read-only diagnostic for held-out FD001 target-positive engine scores.

The script uses the production split, feature extraction, target construction,
and 50/50 anomaly pipeline without changing fitting or threshold behavior.
It describes the untouched held-out engines; it does not tune or select a model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from app.evaluation.real_only_evaluation import (
    construct_future_degradation_truth,
    fit_future_degradation_target,
)
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import split_by_engine


WINDOWS = (30, 50, 75)
PRIMARY_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Important\CMAPSSData.zip")
FALLBACK_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")


def resolve_dataset(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"Dataset ZIP does not exist: {requested}")
        return requested
    if PRIMARY_ZIP.exists():
        return PRIMARY_ZIP
    if FALLBACK_ZIP.exists():
        return FALLBACK_ZIP
    raise FileNotFoundError(f"Neither {PRIMARY_ZIP} nor {FALLBACK_ZIP} exists.")


def load_fd001(zip_path: Path) -> pd.DataFrame:
    """Load train_FD001.txt directly from the ZIP via the production adapter."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the dataset ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def extract_window_features(frame: pd.DataFrame, window: int):
    return extract_early_features(
        frame,
        EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window),
    )


def feature_contributions(
    pipeline: RealOnlyAnomalyPipeline,
    features: pd.DataFrame,
    engine_id: int,
    limit: int,
) -> list[dict[str, float | str]]:
    """Return the existing pipeline's Z-contribution formula for more than top 3.

    ``predict`` returns three feature explanations. For a top-5/top-10
    diagnostic, this repeats exactly the pipeline's private ``_explain``
    calculation; it does not create an Isolation Forest attribution.
    """
    feature_row = features.loc[features["engine_id"] == engine_id].iloc[0]
    transformed_row = pipeline.preprocessor.transform(features.loc[features["engine_id"] == engine_id]).iloc[0]
    contributions = np.abs(
        (transformed_row.to_numpy(dtype=float) - pipeline.z_center_) / pipeline.z_scale_
    )
    positions = np.argsort(contributions)[::-1][:limit]
    return [
        {
            "feature": str(transformed_row.index[position]),
            "value": float(feature_row[str(transformed_row.index[position])]),
            "z_contribution": float(contributions[position]),
        }
        for position in positions
    ]


def verify_returned_explanations(
    prediction: pd.Series,
    calculated: list[dict[str, float | str]],
) -> None:
    """Ensure the diagnostic extension matches the existing returned top 3."""
    returned = prediction["explanation"]["top_features"]
    for supplied, computed in zip(returned, calculated[: len(returned)]):
        if supplied["feature"] != computed["feature"] or not np.isclose(supplied["z_contribution"], computed["z_contribution"]):
            raise AssertionError("Diagnostic feature contribution differs from the production explanation formula.")


def percentile_from_descending_rank(rank: int, total: int) -> float:
    return 100.0 if total == 1 else 100.0 * (total - rank) / (total - 1)


def group_summary(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, object]:
    merged = predictions.set_index("engine_id").join(truth[["near_term_degradation", "observed_final_cycle", "remaining_cycles"]])
    positive = merged.loc[merged["near_term_degradation"]]
    negative = merged.loc[~merged["near_term_degradation"]]
    return {
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "positive_risk_mean": float(positive["risk_score"].mean()),
        "positive_risk_median": float(positive["risk_score"].median()),
        "negative_risk_mean": float(negative["risk_score"].mean()),
        "negative_risk_median": float(negative["risk_score"].median()),
        "positive_z_mean": float(positive["z_score"].abs().mean()),
        "positive_z_median": float(positive["z_score"].abs().median()),
        "negative_z_mean": float(negative["z_score"].abs().mean()),
        "negative_z_median": float(negative["z_score"].abs().median()),
        "positive_isolation_mean": float(positive["isolation_score"].mean()),
        "positive_isolation_median": float(positive["isolation_score"].median()),
        "negative_isolation_mean": float(negative["isolation_score"].mean()),
        "negative_isolation_median": float(negative["isolation_score"].median()),
    }


def output_row(
    *,
    window: int,
    engine_id: int,
    prediction: pd.Series,
    truth: pd.Series,
    rank: int,
    total: int,
    pipeline: RealOnlyAnomalyPipeline,
    returned_top: list[dict[str, float | str]],
    top_five: list[dict[str, float | str]],
    top_ten: list[dict[str, float | str]],
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "window": f"1-{window}",
        "engine_id": engine_id,
        "actual_final_observed_cycle": int(truth["observed_final_cycle"]),
        "remaining_cycles_after_early_window": int(truth["observed_final_cycle"] - window),
        "ground_truth_near_term_degradation": bool(truth["near_term_degradation"]),
        "z_score": float(prediction["z_score"]),
        "z_risk": float(prediction["z_risk"]),
        "isolation_score": float(prediction["isolation_score"]),
        "isolation_risk": float(prediction["isolation_risk"]),
        "risk_score": float(prediction["risk_score"]),
        "classification": str(prediction["classification"]),
        "risk_rank_descending": rank,
        "risk_rank_percentile": percentile_from_descending_rank(rank, total),
        "watchlist_threshold": float(pipeline.thresholds.watchlist),
        "high_risk_threshold": float(pipeline.thresholds.high_risk),
        "exceeds_watchlist": bool(prediction["risk_score"] >= pipeline.thresholds.watchlist),
        "exceeds_high_risk": bool(prediction["risk_score"] >= pipeline.thresholds.high_risk),
        "returned_top_features": json.dumps(returned_top),
        "top_5_z_contributions": json.dumps(top_five),
        "top_10_z_contributions": json.dumps(top_ten),
        **summary,
    }


def print_window_report(window: int, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    print(f"\nWINDOW 1-{window}")
    print(
        "Positive vs negative risk: "
        f"mean={summary['positive_risk_mean']:.3f} vs {summary['negative_risk_mean']:.3f}; "
        f"median={summary['positive_risk_median']:.3f} vs {summary['negative_risk_median']:.3f}"
    )
    for row in rows:
        print(
            f"- Engine {row['engine_id']}: risk={row['risk_score']:.3f}, rank={row['risk_rank_descending']}/15 "
            f"({row['risk_rank_percentile']:.1f}th percentile), "
            f"watchlist={row['exceeds_watchlist']}, high-risk={row['exceeds_high_risk']}, "
            f"final cycle={row['actual_final_observed_cycle']}"
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_test_positive_diagnostic_results.csv"))
    args = parser.parse_args()

    dataset = resolve_dataset(args.dataset)
    frame = load_fd001(dataset)
    split = split_by_engine(frame, seed=42)
    train_ids = set(split.metadata.train_engine_ids)
    validation_ids = set(split.metadata.validation_engine_ids)
    test_ids = set(split.metadata.test_engine_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Production engine split is not disjoint.")

    # Target stays exactly as requested: calibrated on training trajectories only
    # with the production 30-cycle endpoint and 0.20 training quantile.
    target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=0.20)
    if set(target.training_engine_ids) & test_ids:
        raise AssertionError("Target-training engines overlap held-out test engines.")
    test_truth = construct_future_degradation_truth(split.test, target).set_index("engine_id")
    positive_ids = sorted(int(engine_id) for engine_id in test_truth.index[test_truth["near_term_degradation"]])
    if len(positive_ids) != 3:
        raise AssertionError(f"Expected three held-out target positives; found {len(positive_ids)}: {positive_ids}")

    print("BURNUI HELD-OUT TEST-POSITIVE DIAGNOSTIC")
    print(f"Dataset: {dataset}")
    print(f"Split: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}")
    print(f"Target: early_end_cycle=30, training_quantile=0.20, remaining-cycle horizon={target.remaining_cycle_horizon:.3f}")
    print(f"Held-out target-positive engines: {positive_ids}")
    print("Feature contribution note: the pipeline returns top 3; top 5/top 10 use its same Z-contribution formula. No IF attribution is invented.")

    expected_test_ids = test_ids
    all_rows: list[dict[str, object]] = []
    separation: list[tuple[int, float, float]] = []
    repeated_features: dict[str, int] = {}
    for window in WINDOWS:
        config = EarlyWindowConfig(early_start_cycle=1, early_end_cycle=window, minimum_early_cycles=window)
        train_result = extract_early_features(split.train, config)
        validation_result = extract_early_features(split.validation, config)
        test_result = extract_early_features(split.test, config)
        for result in (train_result, validation_result, test_result):
            if result.skipped_engines or result.features.shape[1] - 1 != 168:
                raise AssertionError(f"Window 1-{window} did not preserve all engines and 168 sensor-derived features.")
        if set(test_result.features["engine_id"].astype(int)) != expected_test_ids:
            raise AssertionError(f"Window 1-{window} changed held-out test engines.")
        truncated_test = split.test.loc[split.test["cycle"] <= window].copy()
        pd.testing.assert_frame_equal(test_result.features, extract_window_features(truncated_test, window).features)

        pipeline = RealOnlyAnomalyPipeline().fit(train_result.features, validation_result.features)
        predictions = pipeline.predict(test_result.features).set_index("engine_id")
        ranked_ids = predictions["risk_score"].sort_values(ascending=False, kind="stable").index.astype(int).tolist()
        ranks = {engine_id: position + 1 for position, engine_id in enumerate(ranked_ids)}
        summary = group_summary(predictions.reset_index(), test_truth)
        rows: list[dict[str, object]] = []
        for engine_id in positive_ids:
            prediction = predictions.loc[engine_id]
            calculated_top_ten = feature_contributions(pipeline, test_result.features, engine_id, limit=10)
            verify_returned_explanations(prediction, calculated_top_ten)
            returned_top = prediction["explanation"]["top_features"]
            for feature in calculated_top_ten:
                repeated_features[feature["feature"]] = repeated_features.get(feature["feature"], 0) + 1
            row = output_row(
                window=window,
                engine_id=engine_id,
                prediction=prediction,
                truth=test_truth.loc[engine_id],
                rank=ranks[engine_id],
                total=len(ranked_ids),
                pipeline=pipeline,
                returned_top=returned_top,
                top_five=calculated_top_ten[:5],
                top_ten=calculated_top_ten,
                summary=summary,
            )
            rows.append(row)
            all_rows.append(row)
        separation.append((window, float(summary["positive_risk_mean"] - summary["negative_risk_mean"]), float(np.mean([row["risk_rank_percentile"] for row in rows]))))
        print_window_report(window, rows, summary)

    write_csv(args.output, all_rows)
    print("\nFEATURE REPETITION ACROSS POSITIVE-ENGINE TOP-10 Z CONTRIBUTIONS")
    for feature, count in sorted(repeated_features.items(), key=lambda item: (-item[1], item[0]))[:10]:
        print(f"- {feature}: {count} appearances across 9 positive-engine/window profiles")
    print("\nDESCRIPTIVE WINDOW SEPARATION")
    for window, risk_gap, mean_rank_percentile in separation:
        print(f"- 1-{window}: positive-minus-negative mean risk={risk_gap:.3f}; mean positive rank percentile={mean_rank_percentile:.1f}")
    print("A–D answer: inspect the per-window positive scores, ranks, and threshold crossings above. These are descriptive results from three positives and are not statistically significant or a basis for model/threshold selection.")
    print(f"\nWrote diagnostic results: {args.output}")


if __name__ == "__main__":
    main()
