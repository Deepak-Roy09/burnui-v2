"""Read original NASA C-MAPSS files into BurnUI's normalized contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import BinaryIO, TextIO

import pandas as pd
import numpy as np


class CMAPSSSubset(str, Enum):
    TRAIN = "train"
    TEST = "test"


class CMAPSSFileKind(str, Enum):
    """NASA C-MAPSS distribution file kinds."""

    TRAIN = "train"
    TEST = "test"
    RUL = "rul"


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
CMAPSS_FILENAME_PATTERN = re.compile(r"^(?P<kind>train|test|rul)_(?P<variant>FD00[1-4])\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class CMAPSSFileDescriptor:
    """Reliable filename-derived source metadata, without opening the file."""

    filename: str
    kind: CMAPSSFileKind
    variant: CMAPSSVariant

    @property
    def subset(self) -> CMAPSSSubset | None:
        if self.kind is CMAPSSFileKind.TRAIN:
            return CMAPSSSubset.TRAIN
        if self.kind is CMAPSSFileKind.TEST:
            return CMAPSSSubset.TEST
        return None


def detect_cmapss_filename(filename: str | None) -> CMAPSSFileDescriptor | None:
    """Identify official NASA train/test/RUL filenames without guessing variants."""
    if not filename:
        return None
    match = CMAPSS_FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        return None
    return CMAPSSFileDescriptor(
        filename=Path(filename).name,
        kind=CMAPSSFileKind(match.group("kind").lower()),
        variant=CMAPSSVariant(match.group("variant").upper()),
    )


def resolve_cmapss_source(
    filename: str | None,
    *,
    selected_variant: str | None = None,
    selected_subset: str | None = None,
) -> tuple[CMAPSSVariant, CMAPSSSubset, CMAPSSFileDescriptor | None]:
    """Resolve raw upload source safely; ambiguous names require explicit selection."""
    descriptor = detect_cmapss_filename(filename)
    if descriptor and descriptor.kind is CMAPSSFileKind.RUL:
        raise ValueError("RUL files are recognized as ground-truth files but cannot be screened because they contain no sensor trajectories.")
    explicit_variant = _coerce_variant(selected_variant)
    explicit_subset = _coerce_subset(selected_subset)
    if descriptor:
        if explicit_variant and explicit_variant is not descriptor.variant:
            raise ValueError("The selected C-MAPSS dataset does not match the uploaded filename.")
        if explicit_subset and explicit_subset is not descriptor.subset:
            raise ValueError("The selected C-MAPSS file type does not match the uploaded filename.")
        assert descriptor.subset is not None
        return descriptor.variant, descriptor.subset, descriptor
    if explicit_variant is None or explicit_subset is None:
        raise ValueError("Ambiguous raw C-MAPSS filenames require explicit dataset and file-type selection.")
    return explicit_variant, explicit_subset, None


def _coerce_variant(value: str | None) -> CMAPSSVariant | None:
    if value is None or not value.strip() or value.strip().upper() == "AUTO":
        return None
    try:
        return CMAPSSVariant(value.strip().upper())
    except ValueError as exc:
        raise ValueError("Dataset selection must be FD001, FD002, FD003, or FD004.") from exc


def _coerce_subset(value: str | None) -> CMAPSSSubset | None:
    if value is None or not value.strip() or value.strip().upper() == "AUTO":
        return None
    try:
        return CMAPSSSubset(value.strip().lower())
    except ValueError as exc:
        raise ValueError("File-type selection must be TRAIN or TEST.") from exc


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


def load_cmapss_rul(source: BinaryIO | TextIO) -> list[float]:
    """Load one NASA C-MAPSS RUL file as one value per test-engine row."""
    values = pd.read_csv(source, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    if values.shape[1] != 1 or values.empty:
        raise ValueError("A C-MAPSS RUL file must contain exactly one numeric value per engine.")
    numeric = pd.to_numeric(values.iloc[:, 0], errors="coerce")
    if numeric.isna().any() or (~np.isfinite(numeric.to_numpy(dtype=float))).any() or (numeric < 0).any():
        raise ValueError("C-MAPSS RUL values must be finite non-negative numbers.")
    return [float(value) for value in numeric]
