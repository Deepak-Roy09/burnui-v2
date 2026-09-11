"""Read-only evidence data for BurnUI visualizations.

This module reuses the validated production artifact and existing held-out
evaluation helpers. It is never part of live component classification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from app.evaluation.real_only_evaluation import (
    construct_future_degradation_truth,
    evaluate_held_out_test_set,
    fit_future_degradation_target,
)
from app.ml.production_training import load_fd001_training_data, load_production_pipeline
from app.preparation.early_features import extract_early_features
from app.preparation.splitting import split_by_engine


DEFAULT_CMAPSS_ZIP = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")


def build_anomaly_evidence(zip_path: Path = DEFAULT_CMAPSS_ZIP) -> dict[str, object]:
    """Return actual validation PR points plus compact untouched-test metrics."""
    frame = load_fd001_training_data(zip_path)
    split = split_by_engine(frame, seed=42)
    pipeline = load_production_pipeline()
    train_features = extract_early_features(split.train).features
    validation_features = extract_early_features(split.validation).features
    test_features = extract_early_features(split.test).features
    if train_features.empty or validation_features.empty or test_features.empty:
        raise ValueError("FD001 evidence requires usable early-cycle features for every split.")

    target = fit_future_degradation_target(split.train)
    validation_truth = construct_future_degradation_truth(split.validation, target).set_index("engine_id")
    validation_predictions = pipeline.predict(validation_features).set_index("engine_id")
    y_true = validation_truth.loc[validation_predictions.index, "near_term_degradation"].to_numpy(dtype=bool)
    y_score = validation_predictions["risk_score"].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        precision, recall = np.array([1.0]), np.array([0.0])
        pr_auc: float | None = None
    else:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        from sklearn.metrics import average_precision_score

        pr_auc = float(average_precision_score(y_true, y_score))
    held_out = evaluate_held_out_test_set(pipeline, test_features, split.test, target)
    return {
        "curve_split": "Validation",
        "validation_curve": {"precision": [float(value) for value in precision], "recall": [float(value) for value in recall]},
        "validation_pr_auc": pr_auc,
        "test_engine_count": held_out["test_engine_count"],
        "test_metrics": held_out["metrics"],
        "target": held_out["target"],
        "disclaimer": (
            "C-MAPSS is used as a public degradation/prognostics proxy. It is not actual ISRO component burn-in data."
        ),
    }
