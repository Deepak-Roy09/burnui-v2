"""Validation-only threshold sweep for FD002--FD004 experimental models.

This diagnostic does not write production artifacts.  Official TEST/RUL data
is opened only after the validation decision is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from app.evaluation.dataset_specific_experiments import feature_table, official_test_truth, train_dataset_specific
from app.evaluation.real_only_evaluation import calculate_binary_metrics, construct_future_degradation_truth
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss, load_cmapss_rul
from app.preparation.splitting import split_by_engine

SELECTED_FUSION_WEIGHTS = {"FD002": 0.0, "FD003": 0.0, "FD004": 0.2}
ALERT_RATE_TARGETS = tuple(rate / 100 for rate in range(10, 71, 5))
FINE_GRID_POINTS = 201
PRODUCTION_SHA256 = "24B69C7C216C72F6D5940FAA70E4551764F145DDFB528CD924D8431274CFC0B1"


def load_from_zip(zip_path: Path, filename: str, variant: CMAPSSVariant, subset: CMAPSSSubset | None):
    """Load one dataset member without extracting the archive."""
    with ZipFile(zip_path) as archive:
        member = next(item for item in archive.namelist() if Path(item).name.lower() == filename.lower())
        with archive.open(member) as handle:
            return load_cmapss_rul(handle) if subset is None else load_cmapss(handle, variant=variant, subset=subset)


def rebuild_training_split(train_frame, metadata):
    """Rebuild the deterministic TRAIN-only split represented by metadata."""
    split = split_by_engine(train_frame, seed=metadata.seed)
    if split.metadata != metadata:
        raise AssertionError("Rebuilt split differs from trained model metadata.")
    return split


def fused_risk(predictions, z_weight: float) -> np.ndarray:
    """Reuse pipeline component risks; do not fit or alter the detectors."""
    return np.clip(
        z_weight * predictions.z_risk.to_numpy(dtype=float)
        + (1.0 - z_weight) * predictions.isolation_risk.to_numpy(dtype=float),
        0.0,
        1.0,
    )


def threshold_candidates(scores: np.ndarray) -> tuple[tuple[float, str], ...]:
    """Build alert-rate targets and a fine score-range grid deterministically."""
    values = np.asarray(scores, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Validation risk scores must be non-empty and finite.")
    candidates: dict[float, set[str]] = {}
    for rate in ALERT_RATE_TARGETS:
        candidates.setdefault(float(np.quantile(values, 1.0 - rate)), set()).add(f"validation_alert_rate_{int(rate * 100)}pct")
    for threshold in np.linspace(float(values.min()), float(values.max()), FINE_GRID_POINTS):
        candidates.setdefault(float(threshold), set()).add("validation_fine_grid")
    for threshold in np.unique(values):
        candidates.setdefault(float(threshold), set()).add("validation_observed_score")
    return tuple((threshold, "+".join(sorted(sources))) for threshold, sources in sorted(candidates.items()))


def evaluate_threshold(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate metrics for one decision threshold without changing scores."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(truth, dtype=bool)
    metrics = calculate_binary_metrics(labels, values >= threshold, values)
    alerts = int((values >= threshold).sum())
    return {
        **metrics,
        "threshold": float(threshold),
        "evaluated_count": int(len(values)),
        "alert_count": alerts,
        "alert_rate": float(alerts / len(values)) if len(values) else 0.0,
    }


def _value(metric: float | None, fallback: float) -> float:
    return fallback if metric is None else float(metric)


def select_validation_threshold(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the requested recall-first rule to validation candidates only."""
    if not candidates:
        raise ValueError("No validation threshold candidates were supplied.")
    satisfactory = [item for item in candidates if _value(item["recall"], -1.0) >= 0.50]
    pool = satisfactory or candidates
    if satisfactory:
        # The recall target is a gate.  Once it is met, use the requested F1,
        # precision, FPR, then alert-rate order rather than rewarding extra
        # recall at the expense of the stated trade-off.
        selected = max(
            pool,
            key=lambda item: (
                _value(item["f1"], -1.0),
                _value(item["precision"], -1.0),
                -_value(item["fpr"], 1.0),
                -float(item["alert_rate"]),
                -float(item["threshold"]),
            ),
        )
    else:
        selected = max(
            pool,
            key=lambda item: (
                _value(item["recall"], -1.0),
                _value(item["f1"], -1.0),
                _value(item["precision"], -1.0),
                -_value(item["fpr"], 1.0),
                -float(item["alert_rate"]),
                -float(item["threshold"]),
            ),
        )
    return {**selected, "selection_status": "recall_target_met" if satisfactory else "best_available_recall_below_target"}


def _row(dataset: str, scope: str, z_weight: float, source: str, horizon: float, metrics: dict[str, Any], status: str = "not_selected") -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scope": scope,
        "fusion_z_weight": z_weight,
        "fusion_if_weight": 1.0 - z_weight,
        "threshold_source": source,
        "selection_status": status,
        "target_early_end_cycle": 30,
        "target_training_quantile": 0.20,
        "training_derived_remaining_cycle_horizon": horizon,
        **metrics,
    }


def _truth_for(frame, target, engine_ids) -> np.ndarray:
    truth = construct_future_degradation_truth(frame, target).set_index("engine_id")
    return truth.loc[engine_ids, "near_term_degradation"].to_numpy(dtype=bool)


def run_dataset(zip_path: Path, dataset: str) -> list[dict[str, Any]]:
    """Use TRAIN/validation for selection before any internal/official testing."""
    variant = CMAPSSVariant(dataset)
    train_frame = load_from_zip(zip_path, f"train_{dataset}.txt", variant, CMAPSSSubset.TRAIN)
    pipeline, target, metadata, _ = train_dataset_specific(train_frame)
    split = rebuild_training_split(train_frame, metadata)
    validation_features, _ = feature_table(split.validation)
    validation_predictions = pipeline.predict(validation_features)
    validation_truth = _truth_for(split.validation, target, validation_predictions.engine_id)
    horizon = float(target.remaining_cycle_horizon)
    rows: list[dict[str, Any]] = []

    baseline_scores = fused_risk(validation_predictions, 0.5)
    rows.append(_row(dataset, "VALIDATION_BASELINE_50_50", 0.5, "existing_validation_85th_percentile", horizon,
                     evaluate_threshold(baseline_scores, validation_truth, pipeline.thresholds.watchlist)))

    z_weight = SELECTED_FUSION_WEIGHTS[dataset]
    selected_scores = fused_risk(validation_predictions, z_weight)
    standard_threshold = float(np.quantile(selected_scores, 0.85))
    rows.append(_row(dataset, "VALIDATION_BEST_FUSION_85TH_PERCENTILE", z_weight,
                     "best_fusion_validation_85th_percentile", horizon,
                     evaluate_threshold(selected_scores, validation_truth, standard_threshold)))

    sweep: list[dict[str, Any]] = []
    for threshold, source in threshold_candidates(selected_scores):
        candidate = _row(dataset, "VALIDATION_THRESHOLD_SWEEP", z_weight, source, horizon,
                         evaluate_threshold(selected_scores, validation_truth, threshold))
        rows.append(candidate)
        sweep.append(candidate)
    frozen = select_validation_threshold(sweep)
    rows.append({**frozen, "scope": "VALIDATION_SELECTED"})
    frozen_threshold = float(frozen["threshold"])

    # Fixed selection is used unchanged below.  No decision follows holdout/test inspection.
    holdout_features, _ = feature_table(split.test)
    holdout_predictions = pipeline.predict(holdout_features)
    holdout_truth = _truth_for(split.test, target, holdout_predictions.engine_id)
    rows.append(_row(dataset, "INTERNAL_HOLDOUT_LOCKED", z_weight, "frozen_validation_selected", horizon,
                     evaluate_threshold(fused_risk(holdout_predictions, z_weight), holdout_truth, frozen_threshold),
                     frozen["selection_status"]))

    # TEST/RUL is opened only after validation selection is irreversibly frozen.
    test_frame = load_from_zip(zip_path, f"test_{dataset}.txt", variant, CMAPSSSubset.TEST)
    rul_values = load_from_zip(zip_path, f"RUL_{dataset}.txt", variant, None)
    test_features, _ = feature_table(test_frame)
    test_predictions = pipeline.predict(test_features)
    test_truth = official_test_truth(test_frame, rul_values, target).loc[test_predictions.engine_id, "near_term_degradation"].to_numpy(dtype=bool)
    test_metrics = evaluate_threshold(fused_risk(test_predictions, z_weight), test_truth, frozen_threshold)
    test_metrics["positive_count"] = int(test_truth.sum())
    test_metrics["negative_count"] = int((~test_truth).sum())
    rows.append(_row(dataset, "OFFICIAL_TEST_RUL_LOCKED", z_weight, "frozen_validation_selected", horizon,
                     test_metrics, frozen["selection_status"]))
    return rows


def run_diagnostic(zip_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Run FD002--FD004 while asserting that the protected FD001 artifact is immutable."""
    production = Path(__file__).resolve().parent / "model_artifacts" / "burnui_production_model.joblib"
    before = hashlib.sha256(production.read_bytes()).hexdigest().upper()
    if before != PRODUCTION_SHA256:
        raise RuntimeError("Protected FD001 artifact hash is unexpected; refusing to run.")
    rows: list[dict[str, Any]] = []
    for dataset in ("FD002", "FD003", "FD004"):
        rows.extend(run_dataset(zip_path, dataset))
    fields = sorted({field for row in rows for field in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    after = hashlib.sha256(production.read_bytes()).hexdigest().upper()
    if after != before:
        raise AssertionError("Protected FD001 artifact changed during diagnostic.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_threshold_sweep_diagnostic_results.csv"))
    args = parser.parse_args()
    rows = run_diagnostic(args.zip, args.output)
    for dataset in ("FD002", "FD003", "FD004"):
        selected = next(row for row in rows if row["dataset"] == dataset and row["scope"] == "VALIDATION_SELECTED")
        print(f"{dataset}: threshold={selected['threshold']:.6f}; recall={selected['recall']}; f1={selected['f1']}; "
              f"precision={selected['precision']}; fpr={selected['fpr']}; alert_rate={selected['alert_rate']:.3f}")
    print(f"FD001 SHA before/after: {PRODUCTION_SHA256}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
