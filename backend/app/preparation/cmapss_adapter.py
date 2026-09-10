"""Read original NASA C-MAPSS files into BurnUI's normalized contract."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd


class CMAPSSSubset(str, Enum):
    TRAIN = "train"
    TEST = "test"


class CMAPSSVariant(str, Enum):
    """All current C-MAPSS variants share the same 26 data columns."""

    FD001 = "FD001"
    FD002 = "FD002"
    FD003 = "FD003"
    FD004 = "FD004"


SOURCE_COLUMNS = [
    "unit_number",
    "time_cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *(f"sensor_{index}" for index in range(1, 22)),
]
NORMALIZED_COLUMNS = ["engine_id", "cycle", *SOURCE_COLUMNS[2:]]


def load_cmapss(
    source: str | Path | BinaryIO | TextIO,
    *,
    variant: CMAPSSVariant = CMAPSSVariant.FD001,
    subset: CMAPSSSubset = CMAPSSSubset.TRAIN,
) -> pd.DataFrame:
    """Load one original whitespace-separated C-MAPSS data file.

    ``variant`` and ``subset`` document the source and make adding data-source
    selection later straightforward. No values are altered other than mapping
    the first two positional columns to the normalized ID names and sorting.
    """
    _ = (variant, subset)
    raw = pd.read_csv(source, sep=r"\s+", header=None, engine="python")
    raw = raw.dropna(axis=1, how="all")
    if raw.shape[1] != len(SOURCE_COLUMNS):
        raise ValueError(
            f"Expected {len(SOURCE_COLUMNS)} C-MAPSS data columns after removing empty trailing columns; found {raw.shape[1]}."
        )
    raw.columns = SOURCE_COLUMNS
    normalized = raw.rename(columns={"unit_number": "engine_id", "time_cycle": "cycle"})
    normalized = normalized[NORMALIZED_COLUMNS].sort_values(["engine_id", "cycle"], kind="stable").reset_index(drop=True)
    return normalized
