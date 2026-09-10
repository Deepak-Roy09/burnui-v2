"""Production training and artifact loading for the existing BurnUI anomaly pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from app.models.real_only_anomaly import RealOnlyAnomalyPipeline, load_anomaly_pipeline, save_anomaly_pipeline
from app.preparation.cmapss_adapter import CMAPSSSubset, CMAPSSVariant, load_cmapss
from app.preparation.early_features import EarlyWindowConfig, extract_early_features
from app.preparation.splitting import EngineSplitMetadata, split_by_engine


DATASET_NAME = "FD001"
EARLY_START_CYCLE = 1
EARLY_END_CYCLE = 30
FEATURE_COUNT = 168
SPLIT_RANDOM_STATE = 42
DEFAULT_ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[2] / "model_artifacts"
PRODUCTION_MODEL_FILENAME = "burnui_production_model.joblib"
PRODUCTION_METADATA_FILENAME = "burnui_production_metadata.json"


@dataclass(frozen=True)
class ProductionArtifactPaths:
    model: Path
    metadata: Path


@dataclass(frozen=True)
class ProductionTrainingResult:
    pipeline: RealOnlyAnomalyPipeline
    metadata: dict[str, object]
    split_metadata: EngineSplitMetadata
    validation_features: pd.DataFrame


def production_artifact_paths(artifact_directory: Path | None = None) -> ProductionArtifactPaths:
    """Return the requested standard model and metadata locations."""
    directory = DEFAULT_ARTIFACT_DIRECTORY if artifact_directory is None else Path(artifact_directory)
    return ProductionArtifactPaths(
        model=directory / PRODUCTION_MODEL_FILENAME,
        metadata=directory / PRODUCTION_METADATA_FILENAME,
    )


def load_fd001_training_data(zip_path: Path) -> pd.DataFrame:
    """Load only the original FD001 training trajectories using the C-MAPSS adapter."""
    with ZipFile(zip_path) as archive:
        member = next((name for name in archive.namelist() if name.endswith("train_FD001.txt")), None)
        if member is None:
            raise FileNotFoundError("train_FD001.txt was not found in the supplied C-MAPSS ZIP.")
        with archive.open(member) as source:
            return load_cmapss(source, variant=CMAPSSVariant.FD001, subset=CMAPSSSubset.TRAIN)


def train_production_pipeline(frame: pd.DataFrame) -> ProductionTrainingResult:
    """Fit the unchanged 1-30, 168-feature production pipeline without using test data."""
    split = split_by_engine(frame, seed=SPLIT_RANDOM_STATE)
    _assert_split_metadata(split.metadata)

    config = EarlyWindowConfig(
        early_start_cycle=EARLY_START_CYCLE,
        early_end_cycle=EARLY_END_CYCLE,
        minimum_early_cycles=EARLY_END_CYCLE,
    )
    train_result = extract_early_features(split.train, config)
    validation_result = extract_early_features(split.validation, config)
    if train_result.skipped_engines or validation_result.skipped_engines:
        raise ValueError(
            "Production training requires every training and validation engine to provide 30 early-cycle observations."
        )
    train_features = train_result.features
    validation_features = validation_result.features
    if train_features.shape[1] - 1 != FEATURE_COUNT or validation_features.shape[1] - 1 != FEATURE_COUNT:
        raise AssertionError(f"Expected exactly {FEATURE_COUNT} existing sensor-derived features.")

    # RealOnlyAnomalyPipeline fits preprocessing and Isolation Forest on train
    # only and derives class cutoffs solely from the real validation features.
    pipeline = RealOnlyAnomalyPipeline().fit(train_features, validation_features)
    metadata = _build_metadata(pipeline, split.metadata, tuple(column for column in train_features.columns if column != "engine_id"))
    return ProductionTrainingResult(
        pipeline=pipeline,
        metadata=metadata,
        split_metadata=split.metadata,
        validation_features=validation_features,
    )


def save_production_artifacts(result: ProductionTrainingResult, artifact_directory: Path | None = None) -> ProductionArtifactPaths:
    """Save the fitted production pipeline and its JSON metadata beside each other."""
    paths = production_artifact_paths(artifact_directory)
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    save_anomaly_pipeline(result.pipeline, paths.model)
    paths.metadata.write_text(json.dumps(result.metadata, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def train_and_save_production_model(zip_path: Path, artifact_directory: Path | None = None) -> ProductionTrainingResult:
    """Run the approved FD001 production-training flow and save its artifacts."""
    result = train_production_pipeline(load_fd001_training_data(zip_path))
    save_production_artifacts(result, artifact_directory)
    return result


def load_production_pipeline(artifact_directory: Path | None = None) -> RealOnlyAnomalyPipeline:
    """Load the production artifact without refitting or recalibrating it."""
    return load_anomaly_pipeline(production_artifact_paths(artifact_directory).model)


def _assert_split_metadata(metadata: EngineSplitMetadata) -> None:
    train_ids = set(metadata.train_engine_ids)
    validation_ids = set(metadata.validation_engine_ids)
    test_ids = set(metadata.test_engine_ids)
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Production train, validation, and test engine IDs must remain disjoint.")


def _build_metadata(
    pipeline: RealOnlyAnomalyPipeline,
    split_metadata: EngineSplitMetadata,
    feature_columns: tuple[str, ...],
) -> dict[str, object]:
    if len(feature_columns) != FEATURE_COUNT:
        raise AssertionError(f"Expected {FEATURE_COUNT} production feature columns.")
    return {
        "dataset": DATASET_NAME,
        "early_start_cycle": EARLY_START_CYCLE,
        "early_end_cycle": EARLY_END_CYCLE,
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "train_engine_count": len(split_metadata.train_engine_ids),
        "validation_engine_count": len(split_metadata.validation_engine_ids),
        "test_engine_count": len(split_metadata.test_engine_ids),
        "random_state": SPLIT_RANDOM_STATE,
        "watchlist_threshold": pipeline.thresholds.watchlist,
        "high_risk_threshold": pipeline.thresholds.high_risk,
        "model_configuration": {
            "risk_fusion": {
                "z_weight": pipeline.z_weight,
                "isolation_weight": pipeline.isolation_weight,
            },
            "isolation_forest": asdict(pipeline.isolation_config),
            "threshold_quantiles": {
                "watchlist": pipeline.thresholds.watchlist_quantile,
                "high_risk": pipeline.thresholds.high_risk_quantile,
            },
        },
        "split": split_metadata.as_dict(),
    }
