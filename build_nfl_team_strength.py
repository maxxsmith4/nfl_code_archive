#!/usr/bin/env python
"""Build canonical 2026 NFL preseason team-strength ratings.

Required SQLite inputs
----------------------
- nfl_team_unit_ratings_2026
- nfl_position_group_ratings_2026
- nfl_team_unit_completeness_2026
- nfl_projected_depth_chart_2026

Outputs
-------
- nfl_team_strength_2026
- nfl_team_strength_component_audit_2026
- nfl_team_strength_qb_audit_2026
- nfl_team_strength_completeness_audit_2026

Design rules
------------
1. Preserve replacement-anchored offense, defense, and special-teams scales.
2. Do not standardize individual units or force any component variance.
3. Do not add a separate QB adjustment. QB is already embedded in offense.
4. Apply OL continuity exactly once, only through the OL share of offense.
5. Weight the OL continuity context by its explicit confidence and availability,
   with a deliberately small cap so individual blocking talent remains primary.
6. Do not re-merge player performance. The depth chart is the QB audit contract.
7. Publish a centered raw index and an overall z-score for downstream historical
   point calibration; neither is itself claimed to be spread points.
8. Keep injuries, home field, rest, travel, weather, and weekly form outside the
   permanent preseason base-strength layer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2026
BUILD_ID = "NFL_TEAM_STRENGTH_2026_CANONICAL_V3"
VERSION = "v3_pff_ol_talent_primary_bounded_continuity_no_double_count"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

TEAM_UNIT_TABLE = "nfl_team_unit_ratings_2026"
POSITION_UNIT_TABLE = "nfl_position_group_ratings_2026"
COMPLETENESS_TABLE = "nfl_team_unit_completeness_2026"
DEPTH_TABLE = "nfl_projected_depth_chart_2026"

OUTPUT_TABLE = "nfl_team_strength_2026"
COMPONENT_AUDIT_TABLE = "nfl_team_strength_component_audit_2026"
QB_AUDIT_TABLE = "nfl_team_strength_qb_audit_2026"
COMPLETENESS_AUDIT_TABLE = "nfl_team_strength_completeness_audit_2026"

# Preserve the same macro weights used by the canonical unit builder.
OFFENSE_OVERALL_WEIGHT = 0.52
DEFENSE_OVERALL_WEIGHT = 0.44
SPECIAL_TEAMS_OVERALL_WEIGHT = 0.04

# OL carries 27% of offense in the canonical unit builder.
OL_WEIGHT_WITHIN_OFFENSE = 0.27

# Continuity is context, not blocking quality. At the theoretical extremes,
# continuity can move the OL contribution by at most two unit-rating points.
# The resulting maximum overall-team impact is:
# 2.0 * 0.27 * 0.52 = 0.2808 replacement-anchored strength units.
OL_CONTINUITY_CENTER = 50.0
OL_CONTINUITY_MAX_OL_UNIT_ADJUSTMENT = 2.0

EXPECTED_UNIT_NAMES = {
    "QB", "RB", "WR_TE", "OL", "DL_EDGE", "LB", "DB", "ST"
}

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}


# =============================================================================
# ARGUMENTS AND LOGGING
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical 2026 NFL preseason team strength."
    )
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nfl_team_strength_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = (
        str(value)
        .strip()
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_id(value: Any) -> str:
    return re.sub(r"\.0$", "", clean_scalar(value))


def normalize_team(value: Any) -> str:
    team = clean_scalar(value).upper().replace(".", "")
    return TEAM_ALIASES.get(team, team)


def numeric(
    frame: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return pd.to_numeric(
        frame[column], errors="coerce"
    ).fillna(default)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            LIMIT 1
            """,
            (table_name,),
        ).fetchone()
        is not None
    )


def read_table(
    conn: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    if not table_exists(conn, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")

    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(
        f'SELECT * FROM "{escaped}"', conn
    )
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]
    return frame


def reject_collision_columns(
    frame: pd.DataFrame,
    frame_name: str,
) -> None:
    collisions = [
        column
        for column in frame.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if collisions:
        raise RuntimeError(
            f"{frame_name} contains merge-collision columns: {collisions}"
        )


def safe_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))

    if not np.isfinite(std) or std <= 1e-12:
        raise RuntimeError(
            "Overall team strength has no usable cross-team variation."
        )

    return (values - mean) / std


# =============================================================================
# INPUT PREPARATION
# =============================================================================


def prepare_team_units(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "team",
        "offense_unit_rating",
        "defense_unit_rating",
        "special_teams_unit_rating",
        "overall_team_unit_rating",
        "offense_unit_confidence",
        "defense_unit_confidence",
        "special_teams_unit_confidence",
        "overall_team_unit_confidence",
        "overall_quality_adjusted_completeness",
        "qb_unit_rating",
        "ol_unit_rating",
        "ol_continuity_score",
        "ol_continuity_confidence",
        "ol_continuity_available",
        "ol_continuity_prior_snap_season",
        "ol_complete_five_man_unit_flag",
        "returning_snap_pct",
        "returning_starter_pct",
        "retained_pair_pct",
        "projected_starter_quality_score",
        "projected_starter_availability_score",
        "projected_starter_confidence",
        "continuity_applied_to_quality_flag",
        "normalization_method",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(
            f"{TEAM_UNIT_TABLE} is missing required columns: {missing}"
        )

    frame = raw.copy()
    reject_collision_columns(frame, TEAM_UNIT_TABLE)

    frame["season"] = pd.to_numeric(
        frame["season"], errors="coerce"
    )
    frame = frame[frame["season"].fillna(SEASON).eq(SEASON)].copy()
    frame["season"] = SEASON
    frame["team"] = frame["team"].map(normalize_team)

    numeric_columns = [
        "offense_unit_rating",
        "defense_unit_rating",
        "special_teams_unit_rating",
        "overall_team_unit_rating",
        "offense_unit_confidence",
        "defense_unit_confidence",
        "special_teams_unit_confidence",
        "overall_team_unit_confidence",
        "overall_quality_adjusted_completeness",
        "qb_unit_rating",
        "ol_unit_rating",
        "ol_continuity_score",
        "ol_continuity_confidence",
        "ol_continuity_available",
        "ol_continuity_prior_snap_season",
        "ol_complete_five_man_unit_flag",
        "returning_snap_pct",
        "returning_starter_pct",
        "retained_pair_pct",
        "projected_starter_quality_score",
        "projected_starter_availability_score",
        "projected_starter_confidence",
        "continuity_applied_to_quality_flag",
    ]
    for column in numeric_columns:
        frame[column] = numeric(frame, column, 0.0)

    confidence_columns = [
        "offense_unit_confidence",
        "defense_unit_confidence",
        "special_teams_unit_confidence",
        "overall_team_unit_confidence",
        "overall_quality_adjusted_completeness",
        "ol_continuity_confidence",
        "projected_starter_confidence",
    ]
    for column in confidence_columns:
        frame[column] = frame[column].clip(0.0, 1.0)

    percentage_columns = [
        "returning_snap_pct",
        "returning_starter_pct",
        "retained_pair_pct",
    ]
    for column in percentage_columns:
        frame[column] = frame[column].clip(0.0, 1.0)

    if len(frame) != 32 or frame["team"].nunique() != 32:
        raise RuntimeError(
            f"{TEAM_UNIT_TABLE} must contain 32 unique teams. "
            f"Rows={len(frame)}, teams={frame['team'].nunique()}"
        )

    if frame["team"].eq("").any():
        raise RuntimeError(f"{TEAM_UNIT_TABLE} contains blank teams.")

    if not frame["continuity_applied_to_quality_flag"].eq(0).all():
        raise RuntimeError(
            "OL continuity was already applied in the unit builder. "
            "Team strength would double count it."
        )

    normalization_methods = set(
        frame["normalization_method"].map(clean_scalar)
    )
    if normalization_methods != {"replacement_anchor_no_variance_forcing"}:
        raise RuntimeError(
            "Team units are not exclusively replacement-anchored. "
            f"Found normalization methods: {sorted(normalization_methods)}"
        )

    prior_seasons = set(
        frame["ol_continuity_prior_snap_season"]
        .round()
        .astype(int)
    )
    if prior_seasons != {2025}:
        raise RuntimeError(
            "OL continuity must use 2025 as the prior snap season. "
            f"Found: {sorted(prior_seasons)}"
        )

    return frame.sort_values("team").reset_index(drop=True)


def prepare_position_units(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "team",
        "unit_name",
        "unit_rating",
        "unit_confidence",
        "unit_depth_completeness",
        "unit_usable_grade_weight",
        "replacement_slots",
        "normalization_method",
        "continuity_applied_to_quality_flag",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(
            f"{POSITION_UNIT_TABLE} is missing required columns: {missing}"
        )

    frame = raw.copy()
    reject_collision_columns(frame, POSITION_UNIT_TABLE)

    frame["season"] = pd.to_numeric(
        frame["season"], errors="coerce"
    )
    frame = frame[frame["season"].fillna(SEASON).eq(SEASON)].copy()
    frame["season"] = SEASON
    frame["team"] = frame["team"].map(normalize_team)
    frame["unit_name"] = (
        frame["unit_name"]
        .map(clean_scalar)
        .str.upper()
    )

    for column in [
        "unit_rating",
        "unit_confidence",
        "unit_depth_completeness",
        "unit_usable_grade_weight",
        "replacement_slots",
        "continuity_applied_to_quality_flag",
    ]:
        frame[column] = numeric(frame, column, 0.0)

    frame["unit_confidence"] = frame["unit_confidence"].clip(0.0, 1.0)
    frame["unit_depth_completeness"] = frame[
        "unit_depth_completeness"
    ].clip(0.0, 1.0)
    frame["unit_usable_grade_weight"] = frame[
        "unit_usable_grade_weight"
    ].clip(0.0, 1.0)

    expected_rows = 32 * len(EXPECTED_UNIT_NAMES)
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{POSITION_UNIT_TABLE} must contain {expected_rows} rows. "
            f"Found {len(frame)}."
        )

    if frame.duplicated(["team", "unit_name"]).any():
        raise RuntimeError(
            f"{POSITION_UNIT_TABLE} contains duplicate team/unit rows."
        )

    units_present = set(frame["unit_name"])
    if units_present != EXPECTED_UNIT_NAMES:
        raise RuntimeError(
            "Unexpected unit set. "
            f"Expected={sorted(EXPECTED_UNIT_NAMES)}, "
            f"found={sorted(units_present)}"
        )

    if not frame["continuity_applied_to_quality_flag"].eq(0).all():
        raise RuntimeError(
            "Position-unit table already contains a continuity quality adjustment."
        )

    if set(frame["normalization_method"].map(clean_scalar)) != {
        "replacement_anchor_no_variance_forcing"
    }:
        raise RuntimeError(
            "Position-unit table contains noncanonical normalization methods."
        )

    return frame.sort_values(["team", "unit_name"]).reset_index(drop=True)


def prepare_completeness(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "team",
        "unit_name",
        "required_slots",
        "filled_slots",
        "replacement_slots",
        "depth_completeness",
        "usable_grade_weight",
        "unit_confidence",
        "unit_availability_context",
        "quality_adjusted_completeness",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(
            f"{COMPLETENESS_TABLE} is missing required columns: {missing}"
        )

    frame = raw.copy()
    reject_collision_columns(frame, COMPLETENESS_TABLE)

    frame["season"] = pd.to_numeric(
        frame["season"], errors="coerce"
    )
    frame = frame[frame["season"].fillna(SEASON).eq(SEASON)].copy()
    frame["season"] = SEASON
    frame["team"] = frame["team"].map(normalize_team)
    frame["unit_name"] = (
        frame["unit_name"].map(clean_scalar).str.upper()
    )

    numeric_columns = [
        "required_slots",
        "filled_slots",
        "replacement_slots",
        "depth_completeness",
        "usable_grade_weight",
        "unit_confidence",
        "unit_availability_context",
        "quality_adjusted_completeness",
    ]
    for column in numeric_columns:
        frame[column] = numeric(frame, column, 0.0)

    for column in [
        "depth_completeness",
        "usable_grade_weight",
        "unit_confidence",
        "quality_adjusted_completeness",
    ]:
        frame[column] = frame[column].clip(0.0, 1.0)

    frame["unit_availability_context"] = frame[
        "unit_availability_context"
    ].clip(0.0, 100.0)

    expected_rows = 32 * len(EXPECTED_UNIT_NAMES)
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"{COMPLETENESS_TABLE} must contain {expected_rows} rows. "
            f"Found {len(frame)}."
        )

    if frame.duplicated(["team", "unit_name"]).any():
        raise RuntimeError(
            f"{COMPLETENESS_TABLE} contains duplicate team/unit rows."
        )

    return frame.sort_values(["team", "unit_name"]).reset_index(drop=True)


def prepare_depth(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "team",
        "player_id",
        "player_name",
        "canonical_role",
        "depth_rank",
        "is_projected_starter",
        "starter_slot",
        "projected_snap_share",
        "likely_unavailable",
        "performance_grade",
        "unit_quality_grade",
        "position_confidence_score",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(
            f"{DEPTH_TABLE} is missing required columns: {missing}"
        )

    frame = raw.copy()
    reject_collision_columns(frame, DEPTH_TABLE)

    frame["season"] = pd.to_numeric(
        frame["season"], errors="coerce"
    )
    frame = frame[frame["season"].fillna(SEASON).eq(SEASON)].copy()
    frame["season"] = SEASON
    frame["team"] = frame["team"].map(normalize_team)
    frame["player_id"] = frame["player_id"].map(clean_id)
    frame["player_name"] = frame["player_name"].map(clean_scalar)
    frame["canonical_role"] = (
        frame["canonical_role"].map(clean_scalar).str.upper()
    )
    frame["starter_slot"] = (
        frame["starter_slot"].map(clean_scalar).str.upper()
    )

    for column in [
        "depth_rank",
        "is_projected_starter",
        "projected_snap_share",
        "likely_unavailable",
        "performance_grade",
        "unit_quality_grade",
        "position_confidence_score",
    ]:
        frame[column] = numeric(frame, column, 0.0)

    frame["projected_snap_share"] = frame[
        "projected_snap_share"
    ].clip(0.0, 1.0)
    frame["position_confidence_score"] = frame[
        "position_confidence_score"
    ].clip(0.0, 1.0)

    if frame["player_id"].duplicated().any():
        duplicate_ids = sorted(
            frame.loc[
                frame["player_id"].duplicated(keep=False), "player_id"
            ].unique()
        )
        raise RuntimeError(
            "Depth table contains duplicate player IDs: "
            f"{duplicate_ids[:20]}"
        )

    return frame


# =============================================================================
# QB AUDIT — NO SEPARATE QB VALUE ADJUSTMENT
# =============================================================================


def select_authoritative_qbs(
    depth: pd.DataFrame,
    team_units: pd.DataFrame,
) -> pd.DataFrame:
    qbs = depth[depth["canonical_role"].eq("QB")].copy()
    if qbs.empty:
        raise RuntimeError("Depth table contains no quarterbacks.")

    qbs["starter_slot_flag"] = qbs["starter_slot"].eq("QB1").astype(int)
    qbs["starter_flag"] = qbs["is_projected_starter"].gt(0).astype(int)
    qbs["available_flag"] = qbs["likely_unavailable"].le(0).astype(int)

    qbs["starter_resolution_score"] = (
        1000.0 * qbs["starter_flag"]
        + 500.0 * qbs["starter_slot_flag"]
        + 100.0 * qbs["available_flag"]
        + 10.0 * qbs["projected_snap_share"]
        + qbs["unit_quality_grade"]
        - qbs["depth_rank"]
    )

    selected = (
        qbs.sort_values(
            [
                "team",
                "starter_resolution_score",
                "unit_quality_grade",
                "position_confidence_score",
                "player_name",
            ],
            ascending=[True, False, False, False, True],
        )
        .groupby("team", as_index=False)
        .head(1)
        .copy()
    )

    if len(selected) != 32 or selected["team"].nunique() != 32:
        raise RuntimeError(
            "Authoritative QB selection did not produce one QB per team."
        )

    unit_lookup = team_units.set_index("team")
    selected["qb_unit_rating"] = selected["team"].map(
        unit_lookup["qb_unit_rating"]
    )
    selected["offense_unit_rating"] = selected["team"].map(
        unit_lookup["offense_unit_rating"]
    )

    selected["qb_context_adjustment"] = 0.0
    selected["separate_qb_adjustment_applied_flag"] = 0
    selected["qb_double_count_prevention_flag"] = 1
    selected["qb_value_location"] = "embedded_once_in_qb_unit_and_offense"
    selected["team_strength_version"] = VERSION
    selected["build_id"] = BUILD_ID
    selected["date_imported"] = dt.datetime.now().isoformat(
        timespec="seconds"
    )

    output_columns = [
        "season",
        "team",
        "player_id",
        "player_name",
        "depth_rank",
        "starter_slot",
        "is_projected_starter",
        "projected_snap_share",
        "likely_unavailable",
        "performance_grade",
        "unit_quality_grade",
        "position_confidence_score",
        "qb_unit_rating",
        "offense_unit_rating",
        "qb_context_adjustment",
        "separate_qb_adjustment_applied_flag",
        "qb_double_count_prevention_flag",
        "qb_value_location",
        "team_strength_version",
        "build_id",
        "date_imported",
    ]

    return selected[output_columns].sort_values("team").reset_index(drop=True)


# =============================================================================
# TEAM STRENGTH
# =============================================================================


def build_team_strength(
    team_units: pd.DataFrame,
    qb_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = team_units.copy()

    # Convert continuity to an OL-unit context adjustment. This is the only
    # place continuity enters permanent team quality.
    continuity_relative = (
        frame["ol_continuity_score"] - OL_CONTINUITY_CENTER
    ) / OL_CONTINUITY_CENTER

    frame["ol_continuity_ol_unit_adjustment_uncapped"] = (
        continuity_relative
        * OL_CONTINUITY_MAX_OL_UNIT_ADJUSTMENT
        * frame["ol_continuity_confidence"]
        * frame["ol_continuity_available"].clip(0.0, 1.0)
    )
    frame["ol_continuity_ol_unit_adjustment"] = frame[
        "ol_continuity_ol_unit_adjustment_uncapped"
    ].clip(
        -OL_CONTINUITY_MAX_OL_UNIT_ADJUSTMENT,
        OL_CONTINUITY_MAX_OL_UNIT_ADJUSTMENT,
    )

    # Continuity changes only the OL share of offense.
    frame["ol_continuity_offense_adjustment"] = (
        OL_WEIGHT_WITHIN_OFFENSE
        * frame["ol_continuity_ol_unit_adjustment"]
    )

    # Compatibility field: actual total impact on overall team strength.
    frame["ol_continuity_strength_adjustment"] = (
        OFFENSE_OVERALL_WEIGHT
        * frame["ol_continuity_offense_adjustment"]
    )

    frame["offense_strength_base"] = frame["offense_unit_rating"]
    frame["defense_strength_base"] = frame["defense_unit_rating"]
    frame["special_teams_strength_base"] = frame[
        "special_teams_unit_rating"
    ]

    frame["offense_strength"] = (
        frame["offense_strength_base"]
        + frame["ol_continuity_offense_adjustment"]
    )
    frame["defense_strength"] = frame["defense_strength_base"]
    frame["special_teams_strength"] = frame[
        "special_teams_strength_base"
    ]

    frame["overall_strength_base"] = (
        OFFENSE_OVERALL_WEIGHT * frame["offense_strength_base"]
        + DEFENSE_OVERALL_WEIGHT * frame["defense_strength_base"]
        + SPECIAL_TEAMS_OVERALL_WEIGHT
        * frame["special_teams_strength_base"]
    )

    frame["overall_strength"] = (
        OFFENSE_OVERALL_WEIGHT * frame["offense_strength"]
        + DEFENSE_OVERALL_WEIGHT * frame["defense_strength"]
        + SPECIAL_TEAMS_OVERALL_WEIGHT
        * frame["special_teams_strength"]
    )

    # No target variance is imposed. The centered index preserves the natural
    # replacement-anchored cross-team dispersion. The z-score exists only as a
    # dimensionless input to the separate historical point-calibration stage.
    frame["overall_strength_centered"] = (
        frame["overall_strength"] - frame["overall_strength"].mean()
    )
    frame["point_ready_index"] = frame["overall_strength_centered"]
    frame["overall_strength_z"] = safe_zscore(
        frame["overall_strength"]
    )

    frame["overall_strength_rank"] = (
        frame["overall_strength"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Confidence remains an audit/reliability dimension. Player grades were
    # already confidence-shrunk upstream, so ratings are not shrunk a second time.
    frame["offense_strength_confidence"] = frame[
        "offense_unit_confidence"
    ]
    frame["defense_strength_confidence"] = frame[
        "defense_unit_confidence"
    ]
    frame["special_teams_strength_confidence"] = frame[
        "special_teams_unit_confidence"
    ]
    frame["overall_strength_confidence"] = (
        OFFENSE_OVERALL_WEIGHT
        * frame["offense_strength_confidence"]
        + DEFENSE_OVERALL_WEIGHT
        * frame["defense_strength_confidence"]
        + SPECIAL_TEAMS_OVERALL_WEIGHT
        * frame["special_teams_strength_confidence"]
    )

    qb_lookup = qb_audit.set_index("team")
    frame["qb_player_id"] = frame["team"].map(qb_lookup["player_id"])
    frame["player_name"] = frame["team"].map(qb_lookup["player_name"])
    frame["qb_starter_confidence"] = frame["team"].map(
        qb_lookup["position_confidence_score"]
    )
    frame["qb_context_adjustment"] = 0.0
    frame["separate_qb_adjustment_applied_flag"] = 0
    frame["qb_double_count_prevention_flag"] = 1

    frame["prior_snap_season"] = frame[
        "ol_continuity_prior_snap_season"
    ].round().astype(int)
    frame["stale_history_flag"] = (
        frame["prior_snap_season"].ne(2025)
    ).astype(int)
    frame["ol_continuity_applied_once_flag"] = 1
    frame["unit_variance_forcing_applied_flag"] = 0
    frame["overall_variance_forcing_applied_flag"] = 0
    frame["second_confidence_shrink_applied_flag"] = 0
    frame["rating_excludes_home_field_flag"] = 1
    frame["rating_excludes_weekly_injuries_flag"] = 1
    frame["rating_excludes_rest_travel_flag"] = 1
    frame["rating_excludes_weather_flag"] = 1
    frame["strength_method"] = (
        "replacement_anchor_plus_single_confidence_weighted_ol_continuity"
    )
    frame["point_ready_index_method"] = (
        "league_centered_raw_strength_not_calibrated_points"
    )
    frame["team_strength_version"] = VERSION
    frame["build_id"] = BUILD_ID
    frame["date_imported"] = dt.datetime.now().isoformat(
        timespec="seconds"
    )

    component_rows: list[dict[str, Any]] = []
    component_specs = [
        (
            "OFFENSE",
            "offense_strength_base",
            "ol_continuity_offense_adjustment",
            "offense_strength",
            "offense_strength_confidence",
            OFFENSE_OVERALL_WEIGHT,
        ),
        (
            "DEFENSE",
            "defense_strength_base",
            None,
            "defense_strength",
            "defense_strength_confidence",
            DEFENSE_OVERALL_WEIGHT,
        ),
        (
            "SPECIAL_TEAMS",
            "special_teams_strength_base",
            None,
            "special_teams_strength",
            "special_teams_strength_confidence",
            SPECIAL_TEAMS_OVERALL_WEIGHT,
        ),
    ]

    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        for (
            component_name,
            base_column,
            adjustment_column,
            final_column,
            confidence_column,
            overall_weight,
        ) in component_specs:
            adjustment = (
                float(row_dict[adjustment_column])
                if adjustment_column is not None
                else 0.0
            )
            final_strength = float(row_dict[final_column])
            component_rows.append(
                {
                    "season": SEASON,
                    "team": row_dict["team"],
                    "component_name": component_name,
                    "base_strength": float(row_dict[base_column]),
                    "context_adjustment": adjustment,
                    "final_strength": final_strength,
                    "component_confidence": float(
                        row_dict[confidence_column]
                    ),
                    "overall_weight": overall_weight,
                    "weighted_strength_contribution": (
                        overall_weight * final_strength
                    ),
                    "confidence_shrink_applied_flag": 0,
                    "fixed_variance_normalization_flag": 0,
                    "qb_separate_adjustment_flag": 0,
                    "ol_continuity_adjustment_flag": int(
                        component_name == "OFFENSE"
                    ),
                    "team_strength_version": VERSION,
                    "build_id": BUILD_ID,
                    "date_imported": row_dict["date_imported"],
                }
            )

    component_audit = pd.DataFrame(component_rows)

    output_columns = [
        "season",
        "team",
        "player_name",
        "qb_player_id",
        "qb_unit_rating",
        "qb_starter_confidence",
        "offense_strength_base",
        "defense_strength_base",
        "special_teams_strength_base",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength_base",
        "overall_strength",
        "overall_strength_centered",
        "overall_strength_z",
        "point_ready_index",
        "overall_strength_rank",
        "offense_strength_confidence",
        "defense_strength_confidence",
        "special_teams_strength_confidence",
        "overall_strength_confidence",
        "overall_quality_adjusted_completeness",
        "qb_context_adjustment",
        "separate_qb_adjustment_applied_flag",
        "qb_double_count_prevention_flag",
        "ol_unit_rating",
        "ol_continuity_score",
        "ol_continuity_confidence",
        "ol_continuity_available",
        "returning_snap_pct",
        "returning_starter_pct",
        "retained_pair_pct",
        "ol_continuity_ol_unit_adjustment_uncapped",
        "ol_continuity_ol_unit_adjustment",
        "ol_continuity_offense_adjustment",
        "ol_continuity_strength_adjustment",
        "ol_continuity_applied_once_flag",
        "projected_starter_quality_score",
        "projected_starter_availability_score",
        "projected_starter_confidence",
        "prior_snap_season",
        "stale_history_flag",
        "unit_variance_forcing_applied_flag",
        "overall_variance_forcing_applied_flag",
        "second_confidence_shrink_applied_flag",
        "rating_excludes_home_field_flag",
        "rating_excludes_weekly_injuries_flag",
        "rating_excludes_rest_travel_flag",
        "rating_excludes_weather_flag",
        "strength_method",
        "point_ready_index_method",
        "team_strength_version",
        "build_id",
        "date_imported",
    ]

    strength = frame[output_columns].sort_values(
        "overall_strength_rank"
    ).reset_index(drop=True)

    return strength, component_audit.sort_values(
        ["team", "component_name"]
    ).reset_index(drop=True)


# =============================================================================
# COMPLETENESS AUDIT
# =============================================================================


def build_completeness_audit(
    completeness: pd.DataFrame,
    position_units: pd.DataFrame,
) -> pd.DataFrame:
    unit_context = position_units[
        [
            "team",
            "unit_name",
            "unit_rating",
            "unit_confidence",
            "unit_depth_completeness",
            "unit_usable_grade_weight",
            "replacement_slots",
        ]
    ].rename(
        columns={
            "unit_confidence": "position_unit_confidence",
            "unit_depth_completeness": (
                "position_unit_depth_completeness"
            ),
            "unit_usable_grade_weight": (
                "position_unit_usable_grade_weight"
            ),
            "replacement_slots": "position_unit_replacement_slots",
        }
    )

    audit = completeness.merge(
        unit_context,
        on=["team", "unit_name"],
        how="left",
        validate="one_to_one",
    )
    reject_collision_columns(audit, COMPLETENESS_AUDIT_TABLE)

    audit["confidence_reconciliation_difference"] = (
        audit["unit_confidence"]
        - audit["position_unit_confidence"]
    )
    audit["depth_reconciliation_difference"] = (
        audit["depth_completeness"]
        - audit["position_unit_depth_completeness"]
    )
    audit["usable_grade_reconciliation_difference"] = (
        audit["usable_grade_weight"]
        - audit["position_unit_usable_grade_weight"]
    )
    audit["replacement_slot_reconciliation_difference"] = (
        audit["replacement_slots"]
        - audit["position_unit_replacement_slots"]
    )
    audit["second_confidence_shrink_applied_flag"] = 0
    audit["team_strength_version"] = VERSION
    audit["build_id"] = BUILD_ID
    audit["date_imported"] = dt.datetime.now().isoformat(
        timespec="seconds"
    )

    return audit.sort_values(["team", "unit_name"]).reset_index(drop=True)


# =============================================================================
# VALIDATION, SAVING, AND REPORTING
# =============================================================================


def validate_outputs(
    strength: pd.DataFrame,
    component_audit: pd.DataFrame,
    qb_audit: pd.DataFrame,
    completeness_audit: pd.DataFrame,
) -> None:
    frames = {
        OUTPUT_TABLE: strength,
        COMPONENT_AUDIT_TABLE: component_audit,
        QB_AUDIT_TABLE: qb_audit,
        COMPLETENESS_AUDIT_TABLE: completeness_audit,
    }
    for name, frame in frames.items():
        if len(frame.columns) == 0:
            raise RuntimeError(f"{name} has no columns.")
        reject_collision_columns(frame, name)

    if len(strength) != 32 or strength["team"].nunique() != 32:
        raise RuntimeError(
            "Team-strength output does not contain 32 unique teams."
        )

    if strength["team"].duplicated().any():
        raise RuntimeError("Duplicate team rows in team-strength output.")

    if len(component_audit) != 96:
        raise RuntimeError(
            f"Expected 96 component rows, found {len(component_audit)}."
        )

    if len(qb_audit) != 32 or qb_audit["team"].nunique() != 32:
        raise RuntimeError("QB audit does not contain one row per team.")

    expected_completeness_rows = 32 * len(EXPECTED_UNIT_NAMES)
    if len(completeness_audit) != expected_completeness_rows:
        raise RuntimeError(
            "Completeness audit row count is incorrect: "
            f"{len(completeness_audit)}"
        )

    required_numeric = [
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_strength_z",
        "point_ready_index",
        "overall_strength_confidence",
        "overall_quality_adjusted_completeness",
        "ol_continuity_strength_adjustment",
    ]
    if strength[required_numeric].isna().any().any():
        missing = strength[required_numeric].isna().sum()
        raise RuntimeError(
            "Missing strength values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )

    if abs(float(strength["overall_strength_z"].mean())) > 1e-10:
        raise RuntimeError("overall_strength_z is not centered at zero.")

    if not np.isclose(
        float(strength["overall_strength_z"].std(ddof=1)),
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError("overall_strength_z does not have unit variance.")

    if abs(float(strength["point_ready_index"].mean())) > 1e-10:
        raise RuntimeError("point_ready_index is not centered at zero.")

    if not np.allclose(
        strength["point_ready_index"],
        strength["overall_strength_centered"],
        atol=1e-12,
    ):
        raise RuntimeError(
            "point_ready_index does not preserve centered raw strength."
        )

    if not strength["qb_context_adjustment"].eq(0.0).all():
        raise RuntimeError("A separate QB adjustment was applied.")

    if not strength["separate_qb_adjustment_applied_flag"].eq(0).all():
        raise RuntimeError("QB double-count prevention flag failed.")

    if not strength["qb_double_count_prevention_flag"].eq(1).all():
        raise RuntimeError("QB double-count prevention is not active.")

    if not strength["ol_continuity_applied_once_flag"].eq(1).all():
        raise RuntimeError("OL continuity single-application flag failed.")

    if not strength["unit_variance_forcing_applied_flag"].eq(0).all():
        raise RuntimeError("Unit variance forcing is active.")

    if not strength["overall_variance_forcing_applied_flag"].eq(0).all():
        raise RuntimeError("Overall variance forcing is active.")

    if not strength["second_confidence_shrink_applied_flag"].eq(0).all():
        raise RuntimeError("A second confidence shrink was applied.")

    if set(strength["prior_snap_season"].astype(int)) != {2025}:
        raise RuntimeError(
            "Team strength does not consistently use 2025 OL history."
        )

    if not strength["stale_history_flag"].eq(0).all():
        raise RuntimeError("Team strength contains stale OL history.")

    max_total_impact = (
        OL_CONTINUITY_MAX_OL_UNIT_ADJUSTMENT
        * OL_WEIGHT_WITHIN_OFFENSE
        * OFFENSE_OVERALL_WEIGHT
    )
    if strength["ol_continuity_strength_adjustment"].abs().max() > (
        max_total_impact + 1e-12
    ):
        raise RuntimeError(
            "OL continuity exceeds the configured total-strength cap."
        )

    expected_overall = (
        OFFENSE_OVERALL_WEIGHT * strength["offense_strength"]
        + DEFENSE_OVERALL_WEIGHT * strength["defense_strength"]
        + SPECIAL_TEAMS_OVERALL_WEIGHT
        * strength["special_teams_strength"]
    )
    if not np.allclose(
        strength["overall_strength"],
        expected_overall,
        atol=1e-10,
    ):
        raise RuntimeError("Overall-strength component reconciliation failed.")

    component_sums = component_audit.groupby("team")[
        "weighted_strength_contribution"
    ].sum()
    strength_lookup = strength.set_index("team")["overall_strength"]
    if not np.allclose(
        component_sums.sort_index(),
        strength_lookup.sort_index(),
        atol=1e-10,
    ):
        raise RuntimeError("Component-audit contribution reconciliation failed.")

    reconciliation_columns = [
        "confidence_reconciliation_difference",
        "depth_reconciliation_difference",
        "usable_grade_reconciliation_difference",
        "replacement_slot_reconciliation_difference",
    ]
    if completeness_audit[reconciliation_columns].abs().max().max() > 1e-10:
        raise RuntimeError(
            "Completeness and position-unit tables do not reconcile."
        )


def save_outputs(
    conn: sqlite3.Connection,
    outputs: dict[str, pd.DataFrame],
    output_dir: Path,
    write_csv: bool,
    logger: logging.Logger,
) -> None:
    for table_name, frame in outputs.items():
        frame.to_sql(
            table_name,
            conn,
            if_exists="replace",
            index=False,
        )

        if write_csv:
            path = output_dir / f"{table_name}.csv"
            frame.to_csv(
                path,
                index=False,
                encoding="utf-8-sig",
            )
            logger.info("[STRENGTH] CSV saved: %s", path)

        logger.info(
            "[STRENGTH] Saved %s rows to %s",
            f"{len(frame):,}",
            table_name,
        )


def add_indexes(conn: sqlite3.Connection) -> None:
    statements = [
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_team
        ON {OUTPUT_TABLE}(team)
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{COMPONENT_AUDIT_TABLE}_team_component
        ON {COMPONENT_AUDIT_TABLE}(team, component_name)
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{QB_AUDIT_TABLE}_team
        ON {QB_AUDIT_TABLE}(team)
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{COMPLETENESS_AUDIT_TABLE}_team_unit
        ON {COMPLETENESS_AUDIT_TABLE}(team, unit_name)
        """,
    ]

    for statement in statements:
        conn.execute(statement)


def print_report(
    logger: logging.Logger,
    strength: pd.DataFrame,
    component_audit: pd.DataFrame,
    qb_audit: pd.DataFrame,
) -> None:
    logger.info("")
    logger.info("=" * 100)
    logger.info("[STRENGTH] 2026 NFL PRESEASON TEAM-STRENGTH SUMMARY")
    logger.info("=" * 100)
    logger.info("[STRENGTH] Teams: %s", len(strength))
    logger.info(
        "[STRENGTH] Raw overall mean/std: %.6f / %.6f",
        strength["overall_strength"].mean(),
        strength["overall_strength"].std(ddof=1),
    )
    logger.info(
        "[STRENGTH] Point-ready mean/std: %.6f / %.6f",
        strength["point_ready_index"].mean(),
        strength["point_ready_index"].std(ddof=1),
    )
    logger.info(
        "[STRENGTH] Overall z mean/std: %.6f / %.6f",
        strength["overall_strength_z"].mean(),
        strength["overall_strength_z"].std(ddof=1),
    )
    logger.info(
        "[STRENGTH] Separate QB adjustment applied: NO"
    )
    logger.info(
        "[STRENGTH] Fixed-variance normalization applied: NO"
    )
    logger.info(
        "[STRENGTH] Second confidence shrink applied: NO"
    )
    logger.info(
        "[STRENGTH] OL continuity applied exactly once: YES"
    )
    logger.info(
        "[STRENGTH] Maximum absolute OL overall impact: %.6f",
        strength["ol_continuity_strength_adjustment"].abs().max(),
    )

    display_columns = [
        "overall_strength_rank",
        "team",
        "player_name",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "point_ready_index",
        "ol_continuity_score",
        "ol_continuity_strength_adjustment",
        "overall_strength_confidence",
    ]

    logger.info("")
    logger.info(
        "[STRENGTH] Top 12 teams:\n%s",
        strength.sort_values(
            "overall_strength", ascending=False
        )[display_columns]
        .head(12)
        .to_string(index=False),
    )

    logger.info("")
    logger.info(
        "[STRENGTH] Bottom 12 teams:\n%s",
        strength.sort_values(
            "overall_strength", ascending=False
        )[display_columns]
        .tail(12)
        .to_string(index=False),
    )

    logger.info("")
    logger.info(
        "[STRENGTH] OL continuity impact audit:\n%s",
        strength.sort_values(
            "ol_continuity_strength_adjustment",
            ascending=False,
        )[
            [
                "team",
                "ol_continuity_score",
                "ol_continuity_confidence",
                "ol_continuity_ol_unit_adjustment",
                "ol_continuity_offense_adjustment",
                "ol_continuity_strength_adjustment",
            ]
        ].to_string(index=False),
    )

    logger.info("")
    logger.info(
        "[STRENGTH] QB inclusion audit:\n%s",
        qb_audit[
            [
                "team",
                "player_name",
                "qb_unit_rating",
                "unit_quality_grade",
                "position_confidence_score",
                "qb_context_adjustment",
                "qb_value_location",
            ]
        ].to_string(index=False),
    )

    component_distribution = component_audit.groupby(
        "component_name"
    ).agg(
        mean_strength=("final_strength", "mean"),
        std_strength=("final_strength", "std"),
        min_strength=("final_strength", "min"),
        max_strength=("final_strength", "max"),
        avg_confidence=("component_confidence", "mean"),
    ).reset_index()

    logger.info("")
    logger.info(
        "[STRENGTH] Natural component distribution:\n%s",
        component_distribution.to_string(index=False),
    )
    logger.info("=" * 100)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    db_path = args.db_path.resolve()
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(
        log_dir / "build_nfl_team_strength.log"
    )

    logger.info(
        "[STRENGTH] Building canonical 2026 NFL preseason team strength"
    )
    logger.info("[STRENGTH] Build ID: %s", BUILD_ID)
    logger.info("[STRENGTH] Version: %s", VERSION)
    logger.info("[STRENGTH] Database: %s", db_path)

    try:
        with sqlite3.connect(db_path) as conn:
            team_units = prepare_team_units(
                read_table(conn, TEAM_UNIT_TABLE)
            )
            position_units = prepare_position_units(
                read_table(conn, POSITION_UNIT_TABLE)
            )
            completeness = prepare_completeness(
                read_table(conn, COMPLETENESS_TABLE)
            )
            depth = prepare_depth(
                read_table(conn, DEPTH_TABLE)
            )

            qb_audit = select_authoritative_qbs(
                depth,
                team_units,
            )
            strength, component_audit = build_team_strength(
                team_units,
                qb_audit,
            )
            completeness_audit = build_completeness_audit(
                completeness,
                position_units,
            )

            validate_outputs(
                strength,
                component_audit,
                qb_audit,
                completeness_audit,
            )

            outputs = {
                OUTPUT_TABLE: strength,
                COMPONENT_AUDIT_TABLE: component_audit,
                QB_AUDIT_TABLE: qb_audit,
                COMPLETENESS_AUDIT_TABLE: completeness_audit,
            }

            save_outputs(
                conn,
                outputs,
                output_dir,
                not args.no_csv,
                logger,
            )
            add_indexes(conn)
            conn.commit()

        print_report(
            logger,
            strength,
            component_audit,
            qb_audit,
        )

        return 0

    except Exception as exc:  # noqa: BLE001
        logger.error("[STRENGTH] FAILED: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
