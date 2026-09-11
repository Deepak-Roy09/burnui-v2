"""Train/evaluate separate real-data FD002-FD004 models; never changes FD001."""
import argparse, csv, hashlib, json
from pathlib import Path
from zipfile import ZipFile
from app.evaluation.dataset_specific_experiments import train_dataset_specific, evaluate_official_test
from app.ml.production_training import load_production_pipeline
from app.models.real_only_anomaly import save_anomaly_pipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss, load_cmapss_rul

def load_member(path, name, variant, subset):
    with ZipFile(path) as z:
        member = next(item for item in z.namelist() if Path(item).name.lower() == name.lower())
        with z.open(member) as handle:
            return load_cmapss(handle, variant=variant, subset=subset) if subset else load_cmapss_rul(handle)

def main():
    root = Path(__file__).resolve().parent; production = root / "model_artifacts" / "burnui_production_model.joblib"; experiments = root / "ml_models" / "experimental"
    parser = argparse.ArgumentParser(); parser.add_argument("--zip", type=Path, default=Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")); parser.add_argument("--output", type=Path, default=root / "ml_dataset_specific_experiment_results.csv"); args = parser.parse_args()
    before = hashlib.sha256(production.read_bytes()).hexdigest().upper(); baseline = load_production_pipeline(); rows = []
    for name in ("FD002", "FD003", "FD004"):
        variant = CMAPSSVariant(name); train = load_member(args.zip, f"train_{name}.txt", variant, CMAPSSSubset.TRAIN); test = load_member(args.zip, f"test_{name}.txt", variant, CMAPSSSubset.TEST); rul = load_member(args.zip, f"RUL_{name}.txt", variant, None)
        model, target, split, validation = train_dataset_specific(train); specific = evaluate_official_test(model, test, rul, target); base = evaluate_official_test(baseline, test, rul, target)
        artifact_dir = experiments / name.lower(); artifact_dir.mkdir(parents=True, exist_ok=True)
        save_anomaly_pipeline(model, artifact_dir / f"anomaly_{name.lower()}_experimental.joblib")
        (artifact_dir / "metadata.json").write_text(json.dumps({"dataset": name, "experimental": True, "feature_count": 168, "split": split.as_dict(), "target": {"early_end_cycle": target.early_end_cycle, "training_quantile": target.training_quantile, "remaining_cycle_horizon": target.remaining_cycle_horizon}, "watchlist_threshold": model.thresholds.watchlist, "high_risk_threshold": model.thresholds.high_risk}, indent=2), encoding="utf-8")
        for label, result, pipe in (("FD001 production baseline", base, baseline), ("dataset-specific experimental", specific, model)):
            rows.append({"dataset": name, "model": label, "validation_metrics": json.dumps(validation) if label.startswith("dataset") else None, "watchlist_threshold": pipe.thresholds.watchlist, "high_risk_threshold": pipe.thresholds.high_risk, "horizon": target.remaining_cycle_horizon, **result["metrics"], "evaluated_engines": result["evaluated_engine_count"], "positives": result["positive_count"], "negatives": result["negative_count"], "normal_count": result["status_counts"]["Normal"], "watchlist_count": result["status_counts"]["Watchlist"], "high_risk_count": result["status_counts"]["High Risk"], "risk_score_distribution": json.dumps(result["risk_score_distribution"]), "z_score_distribution": json.dumps(result["z_score_distribution"]), "isolation_score_distribution": json.dumps(result["isolation_score_distribution"]), "decision": "KEEP EXPERIMENTAL"})
        print(f"{name}: train={len(split.train_engine_ids)}, validation={len(split.validation_engine_ids)}, internal_holdout={len(split.test_engine_ids)}, validation_recall={validation['recall']}")
    fields = list(rows[0]);
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    after = hashlib.sha256(production.read_bytes()).hexdigest().upper(); assert before == after; print(f"FD001 SHA before/after: {before}"); print(f"Wrote {args.output}")

if __name__ == "__main__": main()
