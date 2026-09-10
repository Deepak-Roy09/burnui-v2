from pathlib import Path
import numpy as np
import pandas as pd

from app.preparation.cmapss_adapter import (
    load_cmapss,
    CMAPSSVariant,
    CMAPSSSubset,
)
from app.preparation.early_features import (
    EarlyWindowConfig,
    extract_early_features,
)


ZIP_PATH = Path(r"C:\Users\Deepak Roy\OneDrive\Desktop\CMAPSSData.zip")


print("=" * 70)
print("BURNUI ML DIAGNOSTIC")
print("=" * 70)

print("\nLoading FD001...")

with __import__("zipfile").ZipFile(ZIP_PATH) as z:
    with z.open("train_FD001.txt") as f:
        df = load_cmapss(
            f,
            variant=CMAPSSVariant.FD001,
            subset=CMAPSSSubset.TRAIN,
        )

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Engines:", df["engine_id"].nunique())

lengths = df.groupby("engine_id")["cycle"].max()

print("\nTRAJECTORY LENGTHS")
print("Min:", lengths.min())
print("25%:", lengths.quantile(.25))
print("Median:", lengths.median())
print("75%:", lengths.quantile(.75))
print("Max:", lengths.max())


print("\nEARLY WINDOW AVAILABILITY")

for window in [30, 50, 75]:
    usable = int((lengths >= window).sum())
    print(f"1-{window}: {usable}/{len(lengths)} engines")


print("\nFEATURE EXTRACTION")

for window in [30, 50, 75]:

    config = EarlyWindowConfig(
        early_start_cycle=1,
        early_end_cycle=window,
        minimum_early_cycles=window,
    )

    result = extract_early_features(df, config)

    features = result.features

    feature_names = [
        c for c in features.columns
        if c != "engine_id"
    ]

    print(f"\nWindow 1-{window}")
    print("Engines:", len(features))
    print("Features:", len(feature_names))
    print("Shape:", features.shape)
    print("Skipped:", len(result.skipped_engines))
    print("All finite:", np.isfinite(
        features[feature_names].to_numpy(dtype=float)
    ).all())

    print("Feature types:")

    for suffix in [
        "_mean",
        "_std",
        "_min",
        "_max",
        "_first",
        "_last",
        "_delta",
        "_slope",
    ]:
        count = sum(c.endswith(suffix) for c in feature_names)
        print(f"  {suffix}: {count}")


print("\nTEMPORAL FEATURE CHECK")

config = EarlyWindowConfig(
    early_start_cycle=1,
    early_end_cycle=30,
    minimum_early_cycles=30,
)

result = extract_early_features(df, config)

feature_names = [
    c for c in result.features.columns
    if c != "engine_id"
]

for keyword in ["delta", "slope", "first", "last"]:
    matches = [c for c in feature_names if keyword in c]
    print(f"{keyword}: {len(matches)} features")


print("\nSENSOR VARIABILITY")

sensor_cols = [
    c for c in df.columns
    if c.startswith("sensor_")
]

variability = []

for sensor in sensor_cols:
    values = pd.to_numeric(df[sensor], errors="coerce")

    variability.append({
        "sensor": sensor,
        "std": float(values.std()),
        "unique": int(values.nunique()),
    })

variability = pd.DataFrame(variability)

print(
    variability
    .sort_values("std", ascending=False)
    .head(10)
    .to_string(index=False)
)


print("\nEARLY SLOPE MAGNITUDE")

slopes = []

for engine_id, group in df.groupby("engine_id"):

    early = group[group["cycle"] <= 30].sort_values("cycle")

    if len(early) < 2:
        continue

    x = early["cycle"].to_numpy(dtype=float)

    for sensor in sensor_cols:

        y = pd.to_numeric(
            early[sensor],
            errors="coerce"
        ).to_numpy(dtype=float)

        if np.isfinite(y).all():

            slope = np.polyfit(x, y, 1)[0]

            slopes.append({
                "engine_id": int(engine_id),
                "sensor": sensor,
                "slope": slope,
                "abs_slope": abs(slope),
            })

slopes = pd.DataFrame(slopes)

if not slopes.empty:

    top = (
        slopes
        .groupby("sensor")["abs_slope"]
        .median()
        .sort_values(ascending=False)
        .head(10)
    )

    print(top.to_string())


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("No application files were modified.")
print("=" * 70)