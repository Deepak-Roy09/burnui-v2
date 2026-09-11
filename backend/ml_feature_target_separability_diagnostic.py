"""Read-only FD002--FD004 early-feature/target separability diagnostic.

The script builds the existing cycles 1--30, 168-feature table before it
constructs the established post-hoc target.  It never opens official TEST/RUL
files and never writes, fits, or promotes an anomaly-screening artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from app.evaluation.dataset_specific_experiments import feature_table
from app.evaluation.real_only_evaluation import construct_future_degradation_truth, fit_future_degradation_target
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.splitting import split_by_engine


PRODUCTION_SHA256 = "24B69C7C216C72F6D5940FAA70E4551764F145DDFB528CD924D8431274CFC0B1"
FEATURE_COUNTS = (5, 10, 20)
LOGISTIC_PROBE_DESCRIPTION = "Fixed diagnostic L2 logistic regression (C=1.0, liblinear, random_state=42); not a production model."


def load_train_member(zip_path: Path, dataset: str) -> pd.DataFrame:
    """Load only a TRAIN member; this diagnostic deliberately has no TEST/RUL path."""
    filename = f"train_{dataset}.txt"
    with ZipFile(zip_path) as archive:
        member = next(item for item in archive.namelist() if Path(item).name.lower() == filename.lower())
        with archive.open(member) as handle:
            return load_cmapss(handle, variant=CMAPSSVariant(dataset), subset=CMAPSSSubset.TRAIN)


def _truth_for(frame: pd.DataFrame, target, engine_ids: pd.Series) -> np.ndarray:
    truth = construct_future_degradation_truth(frame, target).set_index("engine_id")
    return truth.loc[engine_ids, "near_term_degradation"].to_numpy(dtype=bool)


def prepare_dataset(train_frame: pd.DataFrame) -> dict[str, Any]:
    """Extract early features first, then construct the train-derived target."""
    split = split_by_engine(train_frame, seed=42)
    feature_sets: dict[str, pd.DataFrame] = {}
    for name, frame in (("train", split.train), ("validation", split.validation), ("internal_holdout", split.test)):
        features, skipped = feature_table(frame)
        if skipped:
            raise ValueError(f"{name} has engines without the required cycles 1-30: {sorted(skipped)}")
        if features.shape[1] != 169:
            raise AssertionError("Expected engine_id plus the existing 168 features.")
        feature_sets[name] = features

    # This uses observed future trajectory ends only for the established,
    # post-feature-construction evaluation target.  It is never model input.
    target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=0.20)
    labels = {
        "train": _truth_for(split.train, target, feature_sets["train"]["engine_id"]),
        "validation": _truth_for(split.validation, target, feature_sets["validation"]["engine_id"]),
        "internal_holdout": _truth_for(split.test, target, feature_sets["internal_holdout"]["engine_id"]),
    }
    train_ids, validation_ids, holdout_ids = map(set, (
        split.metadata.train_engine_ids,
        split.metadata.validation_engine_ids,
        split.metadata.test_engine_ids,
    ))
    if train_ids & validation_ids or train_ids & holdout_ids or validation_ids & holdout_ids:
        raise AssertionError("Engine-level split must remain disjoint.")
    if set(target.training_engine_ids) & (validation_ids | holdout_ids):
        raise AssertionError("Target horizon must be fitted from training engines only.")
    return {"split": split, "features": feature_sets, "labels": labels, "target": target}


def _safe_mean(values: np.ndarray) -> float | None:
    return None if not len(values) else float(np.mean(values))


def _safe_std(values: np.ndarray) -> float | None:
    return None if not len(values) else float(np.std(values, ddof=0))


def _point_biserial(values: np.ndarray, labels: np.ndarray) -> float | None:
    if len(values) < 2 or len(np.unique(labels)) < 2 or float(np.std(values, ddof=0)) == 0.0:
        return None
    return float(np.corrcoef(values, labels.astype(float))[0, 1])


def _auc_max_direction(values: np.ndarray, labels: np.ndarray) -> tuple[float | None, str | None]:
    if not len(values) or len(np.unique(labels)) < 2:
        return None, None
    raw_auc = float(roc_auc_score(labels, values))
    return max(raw_auc, 1.0 - raw_auc), "positive" if raw_auc >= 0.5 else "negative"


def feature_statistics(train_features: pd.DataFrame, train_labels: np.ndarray, validation_features: pd.DataFrame, validation_labels: np.ndarray) -> list[dict[str, Any]]:
    """Calculate TRAIN-only effect sizes and validation-only directional AUCs."""
    rows: list[dict[str, Any]] = []
    for feature in (column for column in train_features.columns if column != "engine_id"):
        train_values = pd.to_numeric(train_features[feature], errors="coerce").to_numpy(dtype=float)
        validation_values = pd.to_numeric(validation_features[feature], errors="coerce").to_numpy(dtype=float)
        train_finite = np.isfinite(train_values)
        validation_finite = np.isfinite(validation_values)
        positive = train_values[train_finite & train_labels]
        negative = train_values[train_finite & ~train_labels]
        positive_mean, negative_mean = _safe_mean(positive), _safe_mean(negative)
        positive_std, negative_std = _safe_std(positive), _safe_std(negative)
        if positive_mean is None or negative_mean is None or positive_std is None or negative_std is None:
            effect_size: float | None = None
        else:
            pooled = float(np.sqrt((positive_std ** 2 + negative_std ** 2) / 2.0))
            if pooled == 0.0:
                effect_size = 0.0 if positive_mean == negative_mean else float(np.sign(positive_mean - negative_mean) * np.inf)
            else:
                effect_size = float((positive_mean - negative_mean) / pooled)
        validation_auc, validation_direction = _auc_max_direction(
            validation_values[validation_finite], validation_labels[validation_finite]
        )
        rows.append({
            "feature": feature,
            "train_positive_mean": positive_mean,
            "train_negative_mean": negative_mean,
            "train_positive_std": positive_std,
            "train_negative_std": negative_std,
            "train_standardized_mean_difference": effect_size,
            "train_abs_standardized_mean_difference": None if effect_size is None else abs(effect_size),
            "train_point_biserial": _point_biserial(train_values[train_finite], train_labels[train_finite]),
            "train_missing_count": int(pd.isna(train_features[feature]).sum()),
            "train_nonfinite_count": int((~train_finite).sum()),
            "validation_missing_count": int(pd.isna(validation_features[feature]).sum()),
            "validation_nonfinite_count": int((~validation_finite).sum()),
            "validation_univariate_auc_max_direction": validation_auc,
            "validation_auc_direction": validation_direction,
        })
    return rows


def rank_by_effect(statistics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank from TRAIN-only effects; unavailable values sort last."""
    return sorted(
        statistics,
        key=lambda row: (-float(row["train_abs_standardized_mean_difference"])
                         if row["train_abs_standardized_mean_difference"] is not None else float("inf"), row["feature"]),
    )


def rank_by_validation_auc(statistics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        statistics,
        key=lambda row: (-float(row["validation_univariate_auc_max_direction"])
                         if row["validation_univariate_auc_max_direction"] is not None else float("inf"), row["feature"]),
    )


@dataclass(frozen=True)
class ProbePreprocessor:
    """Diagnostic-only subset preprocessing fitted strictly on TRAIN features."""

    selected_features: tuple[str, ...]
    kept_features: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    fit_engine_ids: tuple[int, ...]


def fit_probe_preprocessor(train_features: pd.DataFrame, selected_features: list[str]) -> ProbePreprocessor:
    """Fit finite-value imputation, variance filtering, and scaling on TRAIN only.

    This intentionally does not reuse the production preprocessor because a
    diagnostic top-k subset can legitimately be entirely constant.  In that
    case an empty, reportable fitted state is returned instead of raising.
    """
    if not selected_features:
        raise ValueError("A diagnostic probe requires at least one selected feature.")
    selected = tuple(selected_features)
    raw = train_features.loc[:, selected].to_numpy(dtype=float)
    finite = np.isfinite(raw)
    medians = np.array(
        [float(np.median(raw[finite[:, index], index])) if finite[:, index].any() else 0.0 for index in range(raw.shape[1])],
        dtype=float,
    )
    imputed = np.where(finite, raw, medians)
    variances = np.var(imputed, axis=0)
    keep_mask = np.isfinite(variances) & (variances > 1e-12)
    kept = tuple(feature for feature, keep in zip(selected, keep_mask) if keep)
    if not kept:
        return ProbePreprocessor(selected, (), medians, np.empty(0), np.empty(0), tuple(int(value) for value in train_features["engine_id"]))
    kept_values = imputed[:, keep_mask]
    means = kept_values.mean(axis=0)
    scales = kept_values.std(axis=0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 1e-12):
        raise AssertionError("Probe variance filtering must leave finite positive TRAIN scales.")
    return ProbePreprocessor(selected, kept, medians, means, scales, tuple(int(value) for value in train_features["engine_id"]))


def transform_probe_features(preprocessor: ProbePreprocessor, features: pd.DataFrame) -> np.ndarray:
    """Transform any split solely with the frozen TRAIN-fitted probe state."""
    if not preprocessor.kept_features:
        return np.empty((len(features), 0), dtype=float)
    raw = features.loc[:, preprocessor.selected_features].to_numpy(dtype=float)
    imputed = np.where(np.isfinite(raw), raw, preprocessor.medians)
    kept_positions = [preprocessor.selected_features.index(feature) for feature in preprocessor.kept_features]
    transformed = (imputed[:, kept_positions] - preprocessor.means) / preprocessor.scales
    if not np.isfinite(transformed).all():
        raise ValueError("Frozen TRAIN probe preprocessing produced non-finite values.")
    return transformed


def aggregate_validation_probe(train_features: pd.DataFrame, train_labels: np.ndarray, validation_features: pd.DataFrame, validation_labels: np.ndarray, selected_features: list[str], internal_holdout_features: pd.DataFrame | None = None) -> dict[str, Any]:
    """Evaluate a fixed train-fitted linear probe; validation data never fits it."""
    preprocessor = fit_probe_preprocessor(train_features, selected_features)
    base = {
        "selected_feature_count": len(selected_features),
        "usable_feature_count": len(preprocessor.kept_features),
        "dropped_zero_variance_features": ";".join(feature for feature in preprocessor.selected_features if feature not in preprocessor.kept_features),
        "preprocessor_fit_engine_ids": list(preprocessor.fit_engine_ids),
    }
    # This transform is intentionally label-free.  It verifies the same frozen
    # TRAIN state can be applied to the internal holdout without making it a
    # feature-selection, model-selection, or threshold-selection input.
    holdout_count = 0
    if internal_holdout_features is not None:
        holdout_count = int(transform_probe_features(preprocessor, internal_holdout_features).shape[0])
    base["internal_holdout_transformed_count"] = holdout_count
    if not preprocessor.kept_features:
        return {**base, "probe_status": "no_usable_train_variance", "validation_roc_auc": None, "validation_pr_auc": None}
    if len(np.unique(train_labels)) < 2 or len(np.unique(validation_labels)) < 2:
        return {**base, "probe_status": "single_class_unavailable", "validation_roc_auc": None, "validation_pr_auc": None}
    train_matrix = transform_probe_features(preprocessor, train_features)
    validation_matrix = transform_probe_features(preprocessor, validation_features)
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42).fit(
        train_matrix, train_labels
    )
    probabilities = model.predict_proba(validation_matrix)[:, 1]
    return {
        **base,
        "probe_status": "evaluated",
        "validation_roc_auc": float(roc_auc_score(validation_labels, probabilities)),
        "validation_pr_auc": float(average_precision_score(validation_labels, probabilities)),
    }


def run_dataset(zip_path: Path, dataset: str) -> list[dict[str, Any]]:
    prepared = prepare_dataset(load_train_member(zip_path, dataset))
    feature_sets, labels, target, split = prepared["features"], prepared["labels"], prepared["target"], prepared["split"]
    train_features, validation_features = feature_sets["train"], feature_sets["validation"]
    train_labels, validation_labels = labels["train"], labels["validation"]
    statistics = feature_statistics(train_features, train_labels, validation_features, validation_labels)
    by_effect, by_auc = rank_by_effect(statistics), rank_by_validation_auc(statistics)
    top_effect = [row["feature"] for row in by_effect[:10]]
    top_auc = [row["feature"] for row in by_auc[:10]]
    overlap = sorted(set(top_effect) & set(top_auc))
    base = {
        "dataset": dataset,
        "feature_count": 168,
        "early_window": "1-30",
        "target_early_end_cycle": target.early_end_cycle,
        "target_training_quantile": target.training_quantile,
        "training_derived_remaining_cycle_horizon": target.remaining_cycle_horizon,
        "train_engine_count": len(split.metadata.train_engine_ids),
        "validation_engine_count": len(split.metadata.validation_engine_ids),
        "internal_holdout_engine_count": len(split.metadata.test_engine_ids),
    }
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "internal_holdout"):
        labels_for_split = labels[split_name]
        rows.append({**base, "row_type": "class_count", "split": split_name,
                     "positive_count": int(labels_for_split.sum()), "negative_count": int((~labels_for_split).sum()),
                     "evaluated_count": int(len(labels_for_split))})
    for rank, row in enumerate(by_effect, start=1):
        rows.append({**base, "row_type": "feature_statistic", "effect_rank_train_only": rank,
                     "validation_auc_rank": next(index for index, candidate in enumerate(by_auc, start=1) if candidate["feature"] == row["feature"]),
                     **row})
    finite_effects = np.asarray([row["train_abs_standardized_mean_difference"] for row in statistics if row["train_abs_standardized_mean_difference"] is not None and np.isfinite(row["train_abs_standardized_mean_difference"])], dtype=float)
    rows.append({**base, "row_type": "ranking_summary", "top10_train_effect_features": ";".join(top_effect),
                 "top10_validation_auc_features": ";".join(top_auc), "top10_overlap_features": ";".join(overlap),
                 "top10_overlap_count": len(overlap), "median_abs_effect": float(np.median(finite_effects)) if len(finite_effects) else None,
                 "p90_abs_effect": float(np.quantile(finite_effects, .90)) if len(finite_effects) else None,
                 "abs_effect_at_least_0_5_count": int((finite_effects >= .5).sum()),
                 "abs_effect_at_least_0_8_count": int((finite_effects >= .8).sum())})
    all_features = [column for column in train_features.columns if column != "engine_id"]
    for count in (len(all_features), *FEATURE_COUNTS):
        selected = all_features if count == len(all_features) else [row["feature"] for row in by_effect[:count]]
        probe = aggregate_validation_probe(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            selected,
            feature_sets["internal_holdout"],
        )
        rows.append({**base, "row_type": "aggregate_validation_probe", "selection_scope": "TRAIN_ONLY",
                     "feature_subset": "all_168" if count == len(all_features) else f"top_{count}_train_abs_effect",
                     "selected_feature_count": len(selected), "selected_features": ";".join(selected),
                     "probe": LOGISTIC_PROBE_DESCRIPTION, **probe})
    return rows


def run_diagnostic(zip_path: Path, output_path: Path) -> list[dict[str, Any]]:
    production = Path(__file__).resolve().parent / "model_artifacts" / "burnui_production_model.joblib"
    before = hashlib.sha256(production.read_bytes()).hexdigest().upper()
    if before != PRODUCTION_SHA256:
        raise RuntimeError("Protected FD001 artifact hash is unexpected; refusing to run.")
    rows = [row for dataset in ("FD002", "FD003", "FD004") for row in run_dataset(zip_path, dataset)]
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
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("ml_feature_target_separability_diagnostic_results.csv"))
    args = parser.parse_args()
    rows = run_diagnostic(args.zip, args.output)
    for dataset in ("FD002", "FD003", "FD004"):
        summary = next(row for row in rows if row["dataset"] == dataset and row["row_type"] == "ranking_summary")
        print(f"{dataset}: top effect features={summary['top10_train_effect_features']}; validation-AUC overlap={summary['top10_overlap_count']}")
    print(f"FD001 SHA before/after: {PRODUCTION_SHA256}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
