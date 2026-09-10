"""Deterministic, engine-level train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineSplitMetadata:
    seed: int
    train_engine_ids: tuple[int, ...]
    validation_engine_ids: tuple[int, ...]
    test_engine_ids: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "train_engine_ids": list(self.train_engine_ids),
            "validation_engine_ids": list(self.validation_engine_ids),
            "test_engine_ids": list(self.test_engine_ids),
            "train_engine_count": len(self.train_engine_ids),
            "validation_engine_count": len(self.validation_engine_ids),
            "test_engine_count": len(self.test_engine_ids),
        }


@dataclass(frozen=True)
class EngineSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: EngineSplitMetadata


def split_by_engine(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> EngineSplit:
    """Split whole engines deterministically; individual rows are never sampled."""
    if "engine_id" not in frame.columns:
        raise ValueError("frame must contain engine_id")
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio, validation_ratio, and test_ratio must be positive and sum to 1.")

    engine_ids = np.sort(frame["engine_id"].dropna().unique())
    if len(engine_ids) < 3:
        raise ValueError("At least three engines are required for train/validation/test splitting.")
    ordered = np.random.default_rng(seed).permutation(engine_ids)
    train_count = max(1, int(np.floor(len(ordered) * train_ratio)))
    validation_count = max(1, int(np.floor(len(ordered) * validation_ratio)))
    test_count = len(ordered) - train_count - validation_count
    if test_count < 1:
        raise ValueError("Ratios leave no engine for the test split.")

    train_ids = tuple(int(value) for value in ordered[:train_count])
    validation_ids = tuple(int(value) for value in ordered[train_count : train_count + validation_count])
    test_ids = tuple(int(value) for value in ordered[train_count + validation_count :])
    _assert_disjoint(train_ids, validation_ids, test_ids)
    metadata = EngineSplitMetadata(seed, train_ids, validation_ids, test_ids)
    return EngineSplit(
        train=frame[frame["engine_id"].isin(train_ids)].copy(),
        validation=frame[frame["engine_id"].isin(validation_ids)].copy(),
        test=frame[frame["engine_id"].isin(test_ids)].copy(),
        metadata=metadata,
    )


def _assert_disjoint(train_ids: tuple[int, ...], validation_ids: tuple[int, ...], test_ids: tuple[int, ...]) -> None:
    train, validation, test = map(set, (train_ids, validation_ids, test_ids))
    assert train.isdisjoint(validation), "Train and validation engine IDs overlap."
    assert train.isdisjoint(test), "Train and test engine IDs overlap."
    assert validation.isdisjoint(test), "Validation and test engine IDs overlap."
