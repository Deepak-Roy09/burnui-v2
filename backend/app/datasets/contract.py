"""Normalized internal CSV contract for C-MAPSS-style sensor trajectories."""

from __future__ import annotations

import re

ENGINE_ID_COLUMN = "engine_id"
CYCLE_COLUMN = "cycle"
SETTING_COLUMNS = ("setting_1", "setting_2", "setting_3")
SENSOR_COLUMN_PATTERN = re.compile(r"^sensor_([1-9]\d*)$")
REQUIRED_NON_SENSOR_COLUMNS = (ENGINE_ID_COLUMN, CYCLE_COLUMN, *SETTING_COLUMNS)


def sensor_columns(columns: list[str]) -> list[str]:
    """Return normalized sensor columns in numeric order.

    The contract supports any positive number of sensors, but requires them to
    be numbered continuously from ``sensor_1`` through ``sensor_N``.
    """
    matches: list[tuple[int, str]] = []
    for column in columns:
        match = SENSOR_COLUMN_PATTERN.fullmatch(column)
        if match:
            matches.append((int(match.group(1)), column))
    return [column for _, column in sorted(matches)]


def expected_columns(sensor_count: int) -> list[str]:
    """Build the exact normalized header for a chosen sensor count."""
    if sensor_count < 1:
        raise ValueError("sensor_count must be at least 1")
    return [*REQUIRED_NON_SENSOR_COLUMNS, *(f"sensor_{index}" for index in range(1, sensor_count + 1))]
