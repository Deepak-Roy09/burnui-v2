"""Leakage-safe SDV augmentation diagnostic for C-MAPSS FD001--FD004."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from app.evaluation.real_only_evaluation import calculate_binary_metrics, construct_future_degradation_truth, fit_future_degradation_target
from app.models.real_only_anomaly import RealOnlyAnomalyPipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import split_by_engine
from ml_augmentation_diagnostic import generate_synthetic_training_features, import_sdv_gaussian_copula

METRICS = ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fpr", "fnr", "pr_auc", "roc_auc")


def load_train(zip_path: Path, variant: CMAPSSVariant) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        expected = f"train_{variant.value}.txt".lower()
        member = next((item for item in archive.namelist() if Path(item).name.lower() == expected), None)
        if member is None:
            raise FileNotFoundError(f"{expected} not found in C-MAPSS ZIP")
        with archive.open(member) as source:
            return load_cmapss(source, variant=variant, subset=CMAPSSSubset.TRAIN)


def early_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = extract_early_features(frame, EarlyWindowConfig(1, 30, 30))
    if result.skipped_engines or result.features.shape[1] != 169:
        raise ValueError(f"Expected all engines and 168 features; got {result.features.shape}, skipped={result.skipped_engines}")
    if not np.isfinite(result.features.drop(columns=["engine_id"]).to_numpy(dtype=float)).all():
        raise ValueError("Early feature matrix contains non-finite values")
    return result.features


def evaluate(pipeline: RealOnlyAnomalyPipeline, features: pd.DataFrame, trajectories: pd.DataFrame, target):
    predictions = pipeline.predict(features)
    truth = construct_future_degradation_truth(trajectories, target).set_index("engine_id").loc[predictions["engine_id"].astype(int)]
    risk = predictions["risk_score"].to_numpy(dtype=float)
    metrics = calculate_binary_metrics(truth["near_term_degradation"].to_numpy(dtype=bool), risk >= pipeline.thresholds.watchlist, risk)
    counts = {label: int((predictions["classification"] == label).sum()) for label in ("Normal", "Watchlist", "High Risk")}
    return metrics, counts


def rank(metrics):
    return (metrics["recall"], -(metrics["fnr"] or 1), metrics["precision"] or 0, metrics["f1"] or 0, -(metrics["fpr"] or 1))


def row(dataset, scope, configuration, pipeline, metrics, counts, synthetic_count, target, selected):
    return {"dataset": dataset, "scope": scope, "configuration": configuration, "early_window": "1-30", "feature_count": 168, "detector": "existing 50/50 Z-score + Isolation Forest", "watchlist_threshold": pipeline.thresholds.watchlist, "high_risk_threshold": pipeline.thresholds.high_risk, "real_training_count": len(pipeline.train_engine_ids), "synthetic_training_count": synthetic_count, "target_horizon": target.remaining_cycle_horizon, "selected_by_validation": selected, "normal_count": counts["Normal"], "watchlist_count": counts["Watchlist"], "high_risk_count": counts["High Risk"], **metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")
    parser.add_argument("--output", default="ml_augmentation_all_datasets_results.csv")
    parser.add_argument("--variants", nargs="+", choices=[item.value for item in CMAPSSVariant], default=[item.value for item in CMAPSSVariant])
    args = parser.parse_args()
    sdv_api = import_sdv_gaussian_copula()
    if sdv_api is None:
        return 0
    synthesizer_class, metadata_builder = sdv_api
    results = []
    for dataset in args.variants:
        split = split_by_engine(load_train(Path(args.dataset), CMAPSSVariant(dataset)), seed=42)
        engine_sets = [set(part["engine_id"]) for part in (split.train, split.validation, split.test)]
        if any(engine_sets[i] & engine_sets[j] for i in range(3) for j in range(i)):
            raise AssertionError(f"{dataset} engine split is not disjoint")
        if any(not ids for ids in engine_sets):
            raise AssertionError(f"{dataset} engine split contains an empty partition")
        train, validation = early_features(split.train), early_features(split.validation)
        target = fit_future_degradation_target(split.train, early_end_cycle=30, training_quantile=.20)
        real = RealOnlyAnomalyPipeline().fit(train, validation)
        synthetic = generate_synthetic_training_features(train, synthesizer_class, metadata_builder)
        augmented = RealOnlyAnomalyPipeline().fit(pd.concat([train, synthetic], ignore_index=True), validation)
        real_metrics, real_counts = evaluate(real, validation, split.validation, target)
        augmented_metrics, augmented_counts = evaluate(augmented, validation, split.validation, target)
        choose_augmented = rank(augmented_metrics) > rank(real_metrics)  # ties retain real-only
        results.extend((row(dataset, "VALIDATION", "real_only", real, real_metrics, real_counts, 0, target, not choose_augmented), row(dataset, "VALIDATION", "real_plus_sdv", augmented, augmented_metrics, augmented_counts, len(synthetic), target, choose_augmented)))
        # Held-out test data is first touched after the validation decision is frozen.
        selected = augmented if choose_augmented else real
        test_metrics, test_counts = evaluate(selected, early_features(split.test), split.test, target)
        results.append(row(dataset, "TEST_LOCKED", "selected_validation_configuration", selected, test_metrics, test_counts, len(synthetic) if choose_augmented else 0, target, choose_augmented))
        print(f"{dataset}: validation recall real={real_metrics['recall']:.3f}, SDV={augmented_metrics['recall']:.3f}; selected={'SDV' if choose_augmented else 'real-only'}; locked test recall={test_metrics['recall']:.3f}")
    fields = ["dataset", "scope", "configuration", "early_window", "feature_count", "detector", "watchlist_threshold", "high_risk_threshold", "real_training_count", "synthetic_training_count", "target_horizon", "selected_by_validation", "normal_count", "watchlist_count", "high_risk_count", *METRICS]
    with Path(args.output).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote diagnostic results: {Path(args.output).resolve()}")
    print("Production model artifact was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
