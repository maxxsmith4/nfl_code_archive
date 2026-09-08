#!/usr/bin/env python
"""Build canonical 2026 NFL player-performance inputs.

Required SQLite inputs
----------------------
1. nfl_player_master_2026
2. nfl_player_advanced_stats_current_roster_2026
3. nfl_player_advanced_stats_2022_2025
4. nfl_qb_rbsdm_ratings_2026
5. nfl_defensive_player_metrics_current_roster_2026

Required local OL input
-----------------------
- PFF offense-blocking exports for 2022, 2023, 2024 and 2025. The default
  filenames are offense_blocking (3).csv, offense_blocking (2).csv,
  offense_blocking (1).csv and offense_blocking.csv, respectively.

Primary outputs
---------------
1. nfl_player_performance_inputs_2026
2. nfl_player_performance_seasonal_2022_2025
3. nfl_ol_pff_player_season_2022_2025

Audit outputs
-------------
1. nfl_player_performance_position_summary_2026
2. nfl_player_performance_component_audit_2026
3. nfl_player_performance_unresolved_master_audit_2026
4. nfl_ol_pff_identity_audit_2026

Design principles
-----------------
- Join every upstream source by canonical GSIS player_id only.
- Preserve 2022-2025 as a true multi-year prior and score 2025 separately.
- Use the dedicated QB efficiency model as the authoritative QB talent source.
- Compute percentiles only among players with relevant samples.
- Renormalize component weights when an optional metric is unavailable.
- Make actual PFF blocking performance the authoritative OL talent signal.
- Treat OL snaps as sample confidence and availability, never as talent.
- Carry qualified pre-injury OL performance forward when a recent season is
  missing; an injury absence does not turn an established player into a
  replacement-level blocker.
- Apply one confidence shrink toward a data-derived position replacement
  baseline.
- Keep QB durability and age as metadata; do not mix them into QB talent.
- Do not globally re-standardize final player grades.
- Keep positional scarcity as metadata; team/unit weighting owns positional value.
- Retain rookies and low-history players at conservative replacement grades.

This file is the canonical full-file replacement for
build_nfl_player_performance.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import re
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sql


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2026
HISTORICAL_SEASONS = (2022, 2023, 2024, 2025)
RECENT_SEASON = 2025
BUILD_ID = "NFL_PLAYER_PERFORMANCE_2026_CANONICAL_V4"
PERFORMANCE_VERSION = "v4_1_pff_row_identity_collision_safe"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

MASTER_TABLE = "nfl_player_master_2026"
ADVANCED_CURRENT_TABLE = "nfl_player_advanced_stats_current_roster_2026"
ADVANCED_HISTORY_TABLE = "nfl_player_advanced_stats_2022_2025"
QB_TABLE = "nfl_qb_rbsdm_ratings_2026"
DEFENSE_TABLE = "nfl_defensive_player_metrics_current_roster_2026"

OUTPUT_TABLE = "nfl_player_performance_inputs_2026"
SEASONAL_OUTPUT_TABLE = "nfl_player_performance_seasonal_2022_2025"
POSITION_SUMMARY_TABLE = "nfl_player_performance_position_summary_2026"
COMPONENT_AUDIT_TABLE = "nfl_player_performance_component_audit_2026"
UNRESOLVED_AUDIT_TABLE = "nfl_player_performance_unresolved_master_audit_2026"
PFF_OL_HISTORY_TABLE = "nfl_ol_pff_player_season_2022_2025"
PFF_OL_IDENTITY_AUDIT_TABLE = "nfl_ol_pff_identity_audit_2026"

PFF_DEFAULT_FILENAMES = {
    2022: "offense_blocking (3).csv",
    2023: "offense_blocking (2).csv",
    2024: "offense_blocking (1).csv",
    2025: "offense_blocking.csv",
}
PFF_CANONICAL_FILENAMES = {
    season: f"nfl_pff_offense_blocking_{season}.csv"
    for season in HISTORICAL_SEASONS
}

# PFF's overall blocking grade is the most stable year-over-year signal in the
# supplied archive. Pass and run grades add style/context without overriding
# the more reliable overall evaluation.
OL_PFF_GRADE_WEIGHTS = {
    "overall_grade": 0.60,
    "pass_block_grade": 0.25,
    "run_block_grade": 0.15,
}
OL_PFF_SEASON_WEIGHTS = {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40}
OL_PFF_FULL_RELIABILITY_SNAPS = 500.0
OL_PFF_CONFIDENCE_PRIOR_SNAPS = 500.0

# Fixed from the supplied 2022-2025 PFF archive: 20th percentile overall
# blocking grade among player-seasons with at least 200 offensive snaps.
# These are replacement anchors, not league-average grades.
OL_PFF_REPLACEMENT_BY_POSITION = {
    "T": 55.4,
    "OT": 55.4,
    "LT": 55.4,
    "RT": 55.4,
    "G": 52.3,
    "OG": 52.3,
    "LG": 52.3,
    "RG": 52.3,
    "C": 55.8,
    "OL": 54.0,
}

OL_NAME_ALIASES = {
    "olufashanu": "olumuyiwafashanu",
}

POSITION_GROUPS = (
    "QB", "RB", "WR_TE", "OL", "EDGE", "DL", "LB_EDGE", "LB", "DB", "ST", "OTHER"
)
DEFENSIVE_GROUPS = {"EDGE", "DL", "LB_EDGE", "LB", "DB"}

POSITION_SCARCITY = {
    "QB": 100.0,
    "EDGE": 82.0,
    "OL": 76.0,
    "WR_TE": 70.0,
    "DB": 66.0,
    "DL": 64.0,
    "LB_EDGE": 58.0,
    "LB": 55.0,
    "RB": 42.0,
    "ST": 10.0,
    "OTHER": 10.0,
}

REPLACEMENT_BASELINE = {
    "QB": 30.0,
    "RB": 28.0,
    "WR_TE": 28.0,
    "OL": 30.0,
    "EDGE": 30.0,
    "DL": 30.0,
    "LB_EDGE": 30.0,
    "LB": 30.0,
    "DB": 30.0,
    "ST": 20.0,
    "OTHER": 18.0,
}

AGE_PEAKS = {
    "QB": (26, 34),
    "RB": (22, 26),
    "WR_TE": (23, 29),
    "OL": (24, 31),
    "EDGE": (24, 30),
    "DL": (24, 31),
    "LB_EDGE": (24, 30),
    "LB": (24, 30),
    "DB": (23, 29),
    "ST": (24, 35),
    "OTHER": (24, 30),
}

# Base recent-season influence before recent sample confidence is applied.
RECENT_BASE_WEIGHT = {
    "QB": 0.35,
    "RB": 0.50,
    "WR_TE": 0.45,
    "OL": 0.35,
    "EDGE": 0.40,
    "DL": 0.40,
    "LB_EDGE": 0.40,
    "LB": 0.40,
    "DB": 0.45,
    "ST": 0.35,
    "OTHER": 0.25,
}

# Half-saturation volumes used by confidence = volume / (volume + half).
MULTI_YEAR_HALF_VOLUME = {
    "QB": 600.0,
    "RB": 250.0,
    "WR_TE": 180.0,
    "OL": 1000.0,
    "EDGE": 1000.0,
    "DL": 1000.0,
    "LB_EDGE": 1000.0,
    "LB": 1000.0,
    "DB": 1000.0,
    "ST": 500.0,
    "OTHER": 500.0,
}

RECENT_HALF_VOLUME = {
    "QB": 300.0,
    "RB": 120.0,
    "WR_TE": 80.0,
    "OL": 600.0,
    "EDGE": 500.0,
    "DL": 500.0,
    "LB_EDGE": 500.0,
    "LB": 500.0,
    "DB": 500.0,
    "ST": 250.0,
    "OTHER": 250.0,
}

RECENT_QUALIFY_VOLUME = {
    "QB": 75.0,
    "RB": 40.0,
    "WR_TE": 25.0,
    "OL": 200.0,
    "EDGE": 150.0,
    "DL": 150.0,
    "LB_EDGE": 150.0,
    "LB": 150.0,
    "DB": 150.0,
    "ST": 100.0,
    "OTHER": 100.0,
}

# The dedicated efficiency model is the complete QB talent source whenever it
# is available. Recent defense intentionally uses the true 2025 advanced proxy
# because the defensive summary is itself multi-year.
QB_DEDICATED_WEIGHT = 1.00
DEFENSE_DEDICATED_WEIGHT = 0.75

MIN_PERCENTILE_PEERS = 5

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB", "HST": "HOU",
    "JAC": "JAX", "KAN": "KC", "KCC": "KC", "LA": "LAR", "STL": "LAR",
    "SD": "LAC", "SDG": "LAC", "LVR": "LV", "OAK": "LV", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}

ADVANCED_NUMERIC_COLUMNS = [
    "games", "total_snaps", "offense_snaps", "defense_snaps", "st_snaps",
    "qb_dropbacks", "qb_epa_per_dropback", "qb_success_rate", "qb_cpoe",
    "qb_sack_rate", "qb_int_rate", "qb_yards_per_dropback",
    "rush_attempts", "rush_yards", "rush_epa", "rush_success_rate",
    "rush_yards_per_carry", "rush_epa_per_carry", "rush_first_down_rate",
    "targets", "receptions", "receiving_yards", "receiving_epa",
    "receiving_success_rate", "yards_per_target", "receiving_epa_per_target",
    "catch_rate", "receiving_first_down_rate", "target_share_avg",
    "air_yards_share_avg", "wopr_avg", "ngs_avg_separation",
    "ngs_avg_cushion", "ngs_avg_intended_air_yards",
    "ngs_yac_above_expectation", "ngs_rush_yards_over_expected",
    "ngs_rush_yards_over_expected_per_att", "ngs_time_to_throw",
    "ngs_completion_pct_above_expectation", "def_tackles", "def_sacks",
    "def_half_sacks", "def_tfl", "def_qb_hits", "def_interceptions",
    "def_pass_defended", "def_forced_fumbles", "def_fumble_recoveries",
    "pressure_proxy", "coverage_playmaking_proxy", "run_defense_proxy",
    "defensive_playmaking_score", "seasons_observed", "available_season_weight",
    "games_history_sum", "total_snaps_history_sum", "games_recent_2025",
    "total_snaps_recent_2025",
]

DEFENSE_NUMERIC_COLUMNS = [
    "front_event_games", "coverage_event_games", "defensive_event_games",
    "front_history_available", "coverage_history_available",
    "defensive_history_available", "front_component_weight",
    "coverage_component_weight", "defense_snaps", "front_composite_score",
    "front_composite_score_raw", "pressure_score_0_100", "sack_score_0_100",
    "run_defense_score_0_100", "pressure_rate_proxy", "sack_rate_proxy",
    "run_stop_rate_proxy", "pressure_events_proxy", "sacks", "qb_hits", "tfl",
    "coverage_composite_score", "coverage_composite_score_raw",
    "coverage_score_0_100", "ball_hawk_score_0_100", "tackling_score_0_100",
    "coverage_playmaking_rate", "interceptions", "pass_defended",
    "defensive_advanced_score_raw", "defensive_advanced_score",
]

QB_NUMERIC_COLUMNS = [
    "seasons_observed", "qb_plays_3yr", "qb_plays_4yr", "qb_plays_l1",
    "qb_qualified_seasons", "qb_qualified_recent", "qb_multi_year_score",
    "qb_recent_score", "qb_avg_score", "qb_rating_override_raw",
    "qb_rating_override", "qb_multi_year_available", "qb_recent_available",
    "qb_multi_year_rank", "qb_recent_rank", "qb_multi_year_percentile",
    "qb_recent_percentile", "qb_epa_per_play_l1", "qb_success_rate_l1",
    "qb_cpoe_l1", "qb_completion_pct_l1", "qb_epa_cpoe_composite_l1",
]

IDENTITY_COLUMNS = [
    "clean_name", "initial_last_key", "canonical_key", "aliases",
    "identity_quality_flag", "identity_version",
]

# Compatibility schema retained for downstream depth and unit builders.
FINAL_COLUMNS = [
    "player_id", "player_name", "team", "position", "position_group",
    "multi_year_performance_score", "recent_season_performance_score",
    "multi_year_position_score", "recent_position_score",
    "recent_score_available", "recent_games", "recent_total_snaps",
    "recent_volume", "recent_confidence_score", "effective_recent_weight",
    "qb_rbsdm_multi_year_score", "qb_rbsdm_recent_score",
    "qb_rbsdm_multi_year_raw", "qb_rbsdm_recent_raw",
    "qb_rbsdm_multi_year_rank", "qb_rbsdm_recent_rank",
    "qb_rbsdm_multi_year_percentile", "qb_rbsdm_recent_percentile",
    "qb_rbsdm_multi_year_available", "qb_rbsdm_recent_available",
    "durability_score", "positional_scarcity_score",
    "age_curve_score_placeholder", "age_score", "replacement_baseline",
    "ol_pff_history_available", "ol_pff_player_id", "ol_pff_position",
    "ol_pff_match_method", "ol_pff_talent_grade", "ol_pff_recent_grade",
    "ol_pff_latest_season", "ol_pff_seasons_observed",
    "ol_pff_total_snaps", "ol_pff_effective_sample",
    "ol_pff_confidence", "ol_pff_source",
    "history_available", "usable_performance_grade", "history_volume_basis", "performance_input_score_raw",
    "pre_confidence_normalized_score", "position_confidence_score",
    "qb_starter_pool", "position_normalized_score", "performance_input_score",
    "seasons_observed", "games_played_3yr", "total_snaps_3yr", "volume_3yr",
    "qualified_seasons", "games_played_l1", "total_snaps_l1", "volume_l1",
    "qualified_recent", "qb_score", "rb_score", "wr_te_score", "ol_score",
    "edge_score", "dl_score", "lb_score", "db_score", "st_score",
    "position_specific_score", "qb_rbsdm_score", "qb_advanced_score",
    "advanced_efficiency_score", "advanced_volume_score",
    "advanced_playmaking_score", "advanced_score_used", "games", "total_snaps",
    "offense_snaps", "defense_snaps", "st_snaps", "qb_dropbacks",
    "qb_epa_per_dropback", "qb_success_rate", "qb_cpoe", "qb_sack_rate",
    "qb_int_rate", "qb_yards_per_dropback", "rush_attempts", "rush_yards",
    "rush_epa", "rush_success_rate", "rush_yards_per_carry",
    "rush_epa_per_carry", "rush_first_down_rate", "targets", "receptions",
    "receiving_yards", "receiving_epa", "receiving_success_rate",
    "yards_per_target", "receiving_epa_per_target", "catch_rate",
    "receiving_first_down_rate", "target_share_avg", "air_yards_share_avg",
    "wopr_avg", "ngs_avg_separation", "ngs_avg_cushion",
    "ngs_avg_intended_air_yards", "ngs_yac_above_expectation",
    "ngs_rush_yards_over_expected", "ngs_rush_yards_over_expected_per_att",
    "ngs_time_to_throw", "ngs_completion_pct_above_expectation", "def_tackles",
    "def_sacks", "def_half_sacks", "def_tfl", "def_qb_hits",
    "def_interceptions", "def_pass_defended", "def_forced_fumbles",
    "def_fumble_recoveries", "pressure_proxy", "coverage_playmaking_proxy",
    "run_defense_proxy", "defensive_playmaking_score", "defensive_event_games",
    "defensive_advanced_score", "defensive_advanced_score_raw",
    "front_composite_score", "coverage_composite_score", "pressure_score_0_100",
    "sack_score_0_100", "run_defense_score_0_100", "coverage_score_0_100",
    "ball_hawk_score_0_100", "tackling_score_0_100",
    "has_dedicated_defensive_metric", "defensive_metrics_version",
    "master_matched", *IDENTITY_COLUMNS, "recent_season_used",
    "historical_seasons_used", "performance_version", "date_imported",
]

SEASONAL_OUTPUT_COLUMNS = [
    "season", "player_id", "player_name", "team", "position", "position_group",
    "games", "total_snaps", "offense_snaps", "defense_snaps", "st_snaps",
    "season_volume", "season_confidence", "qualified_season",
    "position_specific_score", "score_component_count", "performance_version",
    "date_imported",
]

UNRESOLVED_AUDIT_COLUMNS = [
    "season", "player_name", "team", "position", "position_group", "pfr_id",
    "canonical_key", "identity_quality_flag", "unresolved_reason",
]


# =============================================================================
# ARGUMENTS, LOGGING, DATABASE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    for season in HISTORICAL_SEASONS:
        parser.add_argument(
            f"--pff-ol-{season}-path",
            type=Path,
            default=None,
            help=f"PFF offense-blocking CSV for {season}.",
        )
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_player_performance")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_engine(db_path: Path) -> sql.Engine:
    return sql.create_engine(f"sqlite:///{db_path}", pool_pre_ping=True)


def table_exists(engine: sql.Engine, table_name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"
    with engine.connect() as conn:
        return conn.execute(sql.text(query), {"name": table_name}).fetchone() is not None


def read_table(engine: sql.Engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    with engine.connect() as conn:
        frame = pd.read_sql(sql.text(f'SELECT * FROM "{table_name}"'), conn)
    frame.columns = [clean_column(c) for c in frame.columns]
    return frame


def save_frame(
    frame: pd.DataFrame,
    engine: sql.Engine,
    table_name: str,
    csv_path: Path | None,
) -> None:
    frame.to_sql(table_name, con=engine, if_exists="replace", index=False)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def clean_column(value: object) -> str:
    text = str(value).strip().lower().replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def clean_scalar(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("\u200b", "").replace("\ufeff", "")
    return None if text.lower() in {"", "nan", "none", "null", "na", "<na>"} else text


def clean_id_series(series: pd.Series) -> pd.Series:
    return series.map(clean_scalar)


def normalize_team(value: object) -> str | None:
    text = clean_scalar(value)
    if text is None:
        return None
    text = text.upper()
    return TEAM_ALIASES.get(text, text)


def normalize_position_group(value: object) -> str:
    text = clean_scalar(value)
    if text is None:
        return "OTHER"
    text = text.upper()
    if text in POSITION_GROUPS:
        return text
    return "OTHER"


def ensure_columns(frame: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    return out


def numeric_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def clip_0_100(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=100.0)


def reliability(volume: pd.Series, half_saturation: pd.Series | float) -> pd.Series:
    volume_num = pd.to_numeric(volume, errors="coerce").fillna(0.0).clip(lower=0.0)
    if isinstance(half_saturation, pd.Series):
        half = pd.to_numeric(half_saturation, errors="coerce").fillna(1.0).clip(lower=1.0)
    else:
        half = max(float(half_saturation), 1.0)
    return (volume_num / (volume_num + half)).clip(0.0, 1.0)


def age_curve_score(age: object, position_group: str) -> float:
    if age is None or pd.isna(age):
        return 50.0
    try:
        age_value = float(age)
    except (TypeError, ValueError):
        return 50.0

    peak_start, peak_end = AGE_PEAKS.get(position_group, (24, 30))
    if peak_start <= age_value <= peak_end:
        return 100.0
    if age_value < peak_start:
        return max(50.0, 100.0 - (peak_start - age_value) * 8.0)

    decline = 5.0 if position_group == "QB" else 12.0 if position_group == "RB" else 7.0
    return max(40.0, 100.0 - (age_value - peak_end) * decline)


def weighted_component_score(
    frame: pd.DataFrame,
    components: Sequence[tuple[str, float]],
    output_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    numerator = pd.Series(0.0, index=out.index)
    denominator = pd.Series(0.0, index=out.index)
    count = pd.Series(0, index=out.index, dtype=int)

    for column, weight in components:
        values = pd.to_numeric(out.get(column), errors="coerce")
        available = values.notna()
        numerator = numerator + values.fillna(0.0) * float(weight)
        denominator = denominator + available.astype(float) * float(weight)
        count = count + available.astype(int)

    out[output_column] = np.where(denominator > 0, numerator / denominator, np.nan)
    out[f"{output_column}_component_count"] = count
    return out


def percentile_metric(
    frame: pd.DataFrame,
    group_mask: pd.Series,
    source_column: str,
    output_column: str,
    *,
    higher_is_better: bool,
    available_mask: pd.Series,
) -> pd.DataFrame:
    out = frame.copy()
    out[output_column] = np.nan
    source = pd.to_numeric(out.get(source_column), errors="coerce")
    eligible = group_mask & available_mask & source.notna()
    if int(eligible.sum()) == 0:
        return out

    values = source.loc[eligible]
    if len(values) < MIN_PERCENTILE_PEERS or values.nunique(dropna=True) <= 1:
        scored = pd.Series(50.0, index=values.index)
    else:
        scored = values.rank(method="average", pct=True, ascending=higher_is_better) * 100.0
    out.loc[eligible, output_column] = scored
    return out


def validate_unique_ids(frame: pd.DataFrame, table_name: str, *, allow_null: bool = False) -> None:
    if "player_id" not in frame.columns:
        raise RuntimeError(f"{table_name} is missing player_id.")
    ids = clean_id_series(frame["player_id"])
    if not allow_null and ids.isna().any():
        raise RuntimeError(f"{table_name} contains {int(ids.isna().sum())} null player_id rows.")
    duplicates = ids[ids.notna()].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["player_id"]].head(10).to_dict("records")
        raise RuntimeError(f"{table_name} contains duplicate player_id values: {examples}")


# =============================================================================
# INPUT PREPARATION
# =============================================================================


def prepare_master(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    defaults = {
        "season": SEASON, "player_id": None, "player_name": None, "team": None,
        "position": None, "position_group": "OTHER", "age": np.nan,
        "years_exp": np.nan, "pfr_id": None, "canonical_key": None,
        "identity_quality_flag": None, "clean_name": None,
        "initial_last_key": None, "aliases": None, "identity_version": None,
    }
    frame = ensure_columns(raw, defaults)
    frame["player_id"] = clean_id_series(frame["player_id"])
    frame["team"] = frame["team"].map(normalize_team)
    frame["position_group"] = frame["position_group"].map(normalize_position_group)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["years_exp"] = pd.to_numeric(frame["years_exp"], errors="coerce")

    unresolved = frame[frame["player_id"].isna()].copy()
    unresolved["unresolved_reason"] = "missing_canonical_gsis_id"
    for column in UNRESOLVED_AUDIT_COLUMNS:
        if column not in unresolved.columns:
            unresolved[column] = None
    unresolved = unresolved[UNRESOLVED_AUDIT_COLUMNS].copy()

    canonical = frame[frame["player_id"].notna()].copy()
    canonical = canonical.sort_values(["player_id", "team", "player_name"], na_position="last")
    canonical = canonical.drop_duplicates("player_id", keep="first")
    validate_unique_ids(canonical, MASTER_TABLE)
    return canonical, unresolved


def prepare_advanced(raw: pd.DataFrame, table_name: str, *, require_unique: bool) -> pd.DataFrame:
    frame = raw.copy()
    if "player_id" not in frame.columns:
        raise RuntimeError(f"{table_name} is missing player_id.")
    frame["player_id"] = clean_id_series(frame["player_id"])
    frame = frame[frame["player_id"].notna()].copy()
    if "position_group" not in frame.columns:
        frame["position_group"] = "OTHER"
    if "team" not in frame.columns:
        frame["team"] = None
    frame["position_group"] = frame["position_group"].map(normalize_position_group)
    frame["team"] = frame["team"].map(normalize_team)
    for column in ADVANCED_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if require_unique:
        frame = frame.sort_values(["player_id", "total_snaps"], ascending=[True, False])
        frame = frame.drop_duplicates("player_id", keep="first")
        validate_unique_ids(frame, table_name)
    return frame


def prepare_qb(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "player_id" not in frame.columns:
        raise RuntimeError(f"{QB_TABLE} is missing player_id. Canonical ID joins are required.")
    frame["player_id"] = clean_id_series(frame["player_id"])
    frame = frame[frame["player_id"].notna()].copy()
    for column in QB_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(
        ["player_id", "qb_multi_year_available", "qb_plays_4yr"],
        ascending=[True, False, False],
    ).drop_duplicates("player_id", keep="first")
    validate_unique_ids(frame, QB_TABLE)
    return frame


def prepare_defense(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "player_id" not in frame.columns:
        raise RuntimeError(f"{DEFENSE_TABLE} is missing player_id.")
    frame["player_id"] = clean_id_series(frame["player_id"])
    frame = frame[frame["player_id"].notna()].copy()
    for column in DEFENSE_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(
        ["player_id", "defensive_history_available", "defense_snaps"],
        ascending=[True, False, False],
    ).drop_duplicates("player_id", keep="first")
    validate_unique_ids(frame, DEFENSE_TABLE)
    return frame


# =============================================================================
# PFF OFFENSIVE-LINE TALENT
# =============================================================================


def normalize_person_name(value: object) -> str:
    text = clean_scalar(value)
    if text is None:
        return ""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]+", ascii_text.lower())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens = tokens[:-1]
    key = "".join(tokens)
    return OL_NAME_ALIASES.get(key, key)


def initial_last_name_key(value: object) -> str:
    text = clean_scalar(value)
    if text is None:
        return ""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[a-z0-9]+", ascii_text.lower())
    if tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens = tokens[:-1]
    return (tokens[0][:1] + tokens[-1]) if len(tokens) >= 2 else ""


def normalize_ol_position(value: object) -> str:
    text = re.sub(r"[^A-Z]", "", str(clean_scalar(value) or "").upper())
    if text in {"LT", "RT", "T", "OT"}:
        return "T"
    if text in {"LG", "RG", "G", "OG"}:
        return "G"
    if text == "C":
        return "C"
    return "OL"


def ol_replacement_baseline(position: object) -> float:
    text = re.sub(r"[^A-Z]", "", str(clean_scalar(position) or "").upper())
    return float(OL_PFF_REPLACEMENT_BY_POSITION.get(text, OL_PFF_REPLACEMENT_BY_POSITION[normalize_ol_position(text)]))


def resolve_pff_paths(project_root: Path, args: argparse.Namespace) -> dict[int, Path]:
    search_roots = [
        project_root / "data" / "pff",
        project_root / "data" / "raw",
        project_root,
        Path.home() / "Downloads",
    ]
    resolved: dict[int, Path] = {}
    missing: list[str] = []
    for season in HISTORICAL_SEASONS:
        explicit = getattr(args, f"pff_ol_{season}_path", None)
        if explicit is not None:
            candidate = explicit.expanduser().resolve()
            if not candidate.exists():
                raise RuntimeError(f"Configured {season} PFF OL file does not exist: {candidate}")
            resolved[season] = candidate
            continue

        names = [PFF_CANONICAL_FILENAMES[season], PFF_DEFAULT_FILENAMES[season]]
        found = next(
            (
                root / name
                for root in search_roots
                for name in names
                if (root / name).is_file()
            ),
            None,
        )
        if found is None:
            missing.append(
                f"{season}: expected {PFF_CANONICAL_FILENAMES[season]} or "
                f"{PFF_DEFAULT_FILENAMES[season]}"
            )
        else:
            resolved[season] = found.resolve()
    if missing:
        raise RuntimeError(
            "Missing required PFF OL exports. Place them in the project root, "
            "data\\pff, data\\raw, or Downloads; alternatively pass the explicit "
            "--pff-ol-YYYY-path arguments:\n" + "\n".join(missing)
        )
    return resolved


def load_pff_ol_history(paths: Mapping[int, Path]) -> pd.DataFrame:
    required = {
        "player", "player_id", "position", "team_name", "player_game_count",
        "grades_offense", "grades_pass_block", "grades_run_block",
        "snap_counts_offense", "snap_counts_pass_block", "snap_counts_run_block",
        "pbe", "pressures_allowed", "sacks_allowed", "hits_allowed",
        "hurries_allowed", "penalties",
    }
    parts: list[pd.DataFrame] = []
    for season, path in sorted(paths.items()):
        raw = pd.read_csv(path)
        raw.columns = [clean_column(column) for column in raw.columns]
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise RuntimeError(f"{path.name} is missing required PFF columns: {missing}")
        part = raw.copy()
        part["season"] = int(season)
        parts.append(part)

    frame = pd.concat(parts, ignore_index=True)
    frame["pff_player_id"] = frame["player_id"].map(clean_scalar)
    frame["player_name"] = frame["player"].map(clean_scalar)
    frame["pff_position"] = frame["position"].map(normalize_ol_position)
    frame["pff_team"] = frame["team_name"].map(normalize_team)
    frame = frame.drop(columns=["player_id", "player", "position", "team_name"], errors="ignore")
    frame = frame[
        frame["pff_position"].isin({"T", "G", "C"})
        & frame["pff_player_id"].notna()
        & frame["player_name"].notna()
    ].copy()
    frame["pff_name_key"] = frame["player_name"].map(normalize_person_name)

    rename = {
        "player_game_count": "games",
        "grades_offense": "overall_grade",
        "grades_pass_block": "pass_block_grade",
        "grades_run_block": "run_block_grade",
        "snap_counts_offense": "offense_snaps",
        "snap_counts_pass_block": "pass_block_snaps",
        "snap_counts_run_block": "run_block_snaps",
    }
    frame = frame.rename(columns=rename)
    numeric_columns = [
        "games", "overall_grade", "pass_block_grade", "run_block_grade",
        "offense_snaps", "pass_block_snaps", "run_block_snaps", "pbe",
        "pressures_allowed", "sacks_allowed", "hits_allowed", "hurries_allowed",
        "penalties",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in OL_PFF_GRADE_WEIGHTS.items():
        available = frame[column].notna()
        numerator += frame[column].fillna(0.0) * weight
        denominator += available.astype(float) * weight
    frame["pff_season_talent_grade"] = np.where(denominator > 0, numerator / denominator, np.nan)
    frame["pff_season_sample_reliability"] = (
        frame["offense_snaps"].fillna(0.0).clip(lower=0.0)
        / OL_PFF_FULL_RELIABILITY_SNAPS
    ).clip(upper=1.0)
    frame["gsis_player_id"] = None
    frame["pff_source_file"] = frame["season"].map({season: path.name for season, path in paths.items()})
    frame["performance_version"] = PERFORMANCE_VERSION
    frame["date_imported"] = pd.to_datetime(dt.date.today())

    duplicate = frame.duplicated(["season", "pff_player_id"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["season", "pff_player_id", "player_name"]].head(10)
        raise RuntimeError("Duplicate PFF player-season rows:\n" + examples.to_string(index=False))
    if set(frame["season"].astype(int).unique()) != set(HISTORICAL_SEASONS):
        raise RuntimeError("PFF OL history does not contain all required seasons")
    return frame


def master_alias_keys(row: pd.Series) -> set[str]:
    values: list[object] = [row.get("player_name"), row.get("clean_name")]
    aliases = clean_scalar(row.get("aliases"))
    if aliases:
        values.extend(aliases.split("|"))
    return {key for key in (normalize_person_name(value) for value in values) if key}


def match_pff_ol_to_master(
    pff_history: pd.DataFrame,
    master: pd.DataFrame,
    advanced_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pff = pff_history.copy()

    # Resolve same-name players from facts that existed in the historical
    # archive: identical season, team and normalized player name. This is the
    # authoritative bridge for collisions such as the two active Christian
    # Jones OL records; a current-team/name guess is never allowed to override
    # a unique historical GSIS identity.
    historical = advanced_history.copy()
    required_history = {"season", "team", "player_id", "player_name"}
    missing_history = sorted(required_history.difference(historical.columns))
    if missing_history:
        raise RuntimeError(
            f"{ADVANCED_HISTORY_TABLE} cannot support the PFF identity bridge; "
            f"missing columns: {missing_history}"
        )
    historical["season"] = pd.to_numeric(historical["season"], errors="coerce")
    historical["team"] = historical["team"].map(normalize_team)
    historical["player_id"] = historical["player_id"].map(clean_scalar)
    historical["history_name_key"] = historical["player_name"].map(normalize_person_name)
    if "position_group" in historical.columns:
        historical = historical[
            historical["position_group"].map(normalize_position_group).eq("OL")
        ].copy()
    else:
        historical = historical[
            historical.get("position", pd.Series("", index=historical.index))
            .map(normalize_ol_position).isin({"T", "G", "C"})
        ].copy()
    bridge_rows = pff[["season", "pff_team", "pff_player_id", "pff_name_key"]].merge(
        historical[["season", "team", "player_id", "history_name_key"]],
        how="inner",
        left_on=["season", "pff_team", "pff_name_key"],
        right_on=["season", "team", "history_name_key"],
    )
    def bridge_key(
        season: object,
        team: object,
        pff_player_id: object,
        name_key: object,
    ) -> tuple[int, str, str, str]:
        return (
            int(pd.to_numeric(season, errors="raise")),
            clean_scalar(team) or "",
            clean_scalar(pff_player_id) or "",
            clean_scalar(name_key) or "",
        )

    # A vendor PFF ID is not assumed to represent one person forever. Resolve
    # each PFF season/team/name row first, then use an ID-wide bridge only when
    # every historical row points to the same GSIS identity. This safely
    # handles reused IDs and same-name players without guessing.
    row_bridge_candidates: dict[tuple[int, str, str, str], set[str]] = {}
    pff_bridge_candidates: dict[str, set[str]] = {}
    for record in bridge_rows.to_dict("records"):
        gsis_id = clean_scalar(record.get("player_id"))
        pff_id = clean_scalar(record.get("pff_player_id"))
        if gsis_id is None or pff_id is None:
            continue
        key = bridge_key(
            record.get("season"),
            record.get("pff_team"),
            pff_id,
            record.get("pff_name_key"),
        )
        row_bridge_candidates.setdefault(key, set()).add(gsis_id)
        pff_bridge_candidates.setdefault(pff_id, set()).add(gsis_id)

    row_direct_identity = {
        key: next(iter(gsis_ids))
        for key, gsis_ids in row_bridge_candidates.items()
        if len(gsis_ids) == 1
    }
    direct_identity = {
        pff_id: next(iter(gsis_ids))
        for pff_id, gsis_ids in pff_bridge_candidates.items()
        if len(gsis_ids) == 1
    }
    bridge_collision_ids = {
        pff_id
        for pff_id, gsis_ids in pff_bridge_candidates.items()
        if len(gsis_ids) > 1
    }
    row_links_by_gsis: dict[str, set[str]] = {}
    for key, gsis_id in row_direct_identity.items():
        row_links_by_gsis.setdefault(gsis_id, set()).add(key[2])
    pff_players = (
        pff.sort_values(["pff_player_id", "season"])
        .groupby("pff_player_id", as_index=False)
        .agg(
            player_name=("player_name", "last"),
            pff_name_key=("pff_name_key", "last"),
            pff_initial_last_key=("player_name", lambda values: initial_last_name_key(values.iloc[-1])),
            latest_pff_position=("pff_position", "last"),
            latest_pff_team=("pff_team", "last"),
            latest_pff_season=("season", "max"),
        )
    )
    by_name: dict[str, list[dict[str, Any]]] = {}
    by_initial_last: dict[str, list[dict[str, Any]]] = {}
    for record in pff_players.to_dict("records"):
        by_name.setdefault(record["pff_name_key"], []).append(record)
        by_initial_last.setdefault(record["pff_initial_last_key"], []).append(record)

    ol_master = master[
        master["position_group"].eq("OL")
        | master["position"].map(normalize_ol_position).isin({"T", "G", "C"})
    ].copy()
    master_initial_counts = (
        ol_master["player_name"].map(initial_last_name_key).value_counts().to_dict()
    )
    mapping_claims: dict[str, list[tuple[int, str]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for _, player in ol_master.iterrows():
        master_id = str(player["player_id"])
        row_linked_pff_ids = sorted(row_links_by_gsis.get(master_id, set()))
        if row_linked_pff_ids:
            audit_rows.append({
                "season": SEASON,
                "gsis_player_id": master_id,
                "player_name": player.get("player_name"),
                "team": player.get("team"),
                "position": player.get("position"),
                "pff_player_id": "|".join(row_linked_pff_ids),
                "match_status": "matched",
                "match_method": "historical_season_team_name_gsis_row_bridge",
                "unresolved_reason": "",
                "pff_id_collision_flag": int(
                    any(pff_id in bridge_collision_ids for pff_id in row_linked_pff_ids)
                ),
                "performance_version": PERFORMANCE_VERSION,
                "build_id": BUILD_ID,
                "date_imported": pd.to_datetime(dt.date.today()),
            })
            continue

        keys = master_alias_keys(player)
        candidates: dict[str, dict[str, Any]] = {}
        directly_linked = {
            pff_id for pff_id, gsis_id in direct_identity.items()
            if gsis_id == master_id
        }
        if directly_linked:
            for record in pff_players.to_dict("records"):
                if str(record["pff_player_id"]) in directly_linked:
                    candidates[str(record["pff_player_id"])] = record
        for key in keys:
            for candidate in by_name.get(key, []):
                candidate_pff_id = str(candidate["pff_player_id"])
                if candidate_pff_id in bridge_collision_ids:
                    continue
                linked_gsis = direct_identity.get(candidate_pff_id)
                if linked_gsis is None or linked_gsis == master_id:
                    candidates[candidate_pff_id] = candidate
        candidate_list = list(candidates.values())
        master_position = normalize_ol_position(player.get("position"))
        match_method = (
            "historical_season_team_name_gsis_bridge"
            if directly_linked else "normalized_name_position_team_safe"
        )
        if not candidate_list:
            initial_key = initial_last_name_key(player.get("player_name"))
            fallback = (
                [
                    record
                    for record in by_initial_last.get(initial_key, [])
                    if str(record["pff_player_id"]) not in bridge_collision_ids
                ]
                if initial_key else []
            )
            if (
                master_initial_counts.get(initial_key, 0) == 1
                and len(fallback) == 1
                and (
                    master_position == "OL"
                    or fallback[0]["latest_pff_position"] == master_position
                )
            ):
                candidate_list = fallback
                match_method = "unique_first_initial_last_position_safe"
        if len(candidate_list) > 1 and master_position != "OL":
            compatible = [c for c in candidate_list if c["latest_pff_position"] == master_position]
            if compatible:
                candidate_list = compatible
        if len(candidate_list) > 1:
            same_team = [c for c in candidate_list if c["latest_pff_team"] == player.get("team")]
            if len(same_team) == 1:
                candidate_list = same_team

        if len(candidate_list) == 1:
            candidate = candidate_list[0]
            pff_id = str(candidate["pff_player_id"])
            status = "matched"
            method = match_method
            reason = ""
        elif len(candidate_list) == 0:
            pff_id = ""
            status = "unmatched"
            method = "none"
            reason = "no_2022_2025_pff_history_or_name_alias"
        else:
            pff_id = ""
            status = "ambiguous"
            method = "none"
            reason = "multiple_pff_identity_candidates"
        audit_index = len(audit_rows)
        audit_rows.append({
            "season": SEASON,
            "gsis_player_id": master_id,
            "player_name": player.get("player_name"),
            "team": player.get("team"),
            "position": player.get("position"),
            "pff_player_id": pff_id,
            "match_status": status,
            "match_method": method,
            "unresolved_reason": reason,
            "pff_id_collision_flag": int(pff_id in bridge_collision_ids),
            "performance_version": PERFORMANCE_VERSION,
            "build_id": BUILD_ID,
            "date_imported": pd.to_datetime(dt.date.today()),
        })
        if status == "matched" and pff_id:
            mapping_claims.setdefault(pff_id, []).append((audit_index, master_id))

    # Resolve name-based claims one-to-one. A PFF ID claimed by multiple GSIS
    # rows is retained in the audit but is not assigned to either player.
    mapping: dict[str, str] = {}
    claim_collision_ids: set[str] = set()
    for pff_id, claims in mapping_claims.items():
        claimed_gsis_ids = sorted({master_id for _, master_id in claims})
        if pff_id in bridge_collision_ids or len(claimed_gsis_ids) != 1:
            claim_collision_ids.add(pff_id)
            for audit_index, _ in claims:
                audit_rows[audit_index]["pff_player_id"] = ""
                audit_rows[audit_index]["match_status"] = "ambiguous"
                audit_rows[audit_index]["match_method"] = "none"
                audit_rows[audit_index]["unresolved_reason"] = (
                    "pff_id_claimed_by_multiple_gsis_identities"
                )
                audit_rows[audit_index]["pff_id_collision_flag"] = 1
            continue
        mapping[pff_id] = claimed_gsis_ids[0]

    historical_bridge_ids: list[str | None] = []
    historical_bridge_methods: list[str] = []
    for record in pff.to_dict("records"):
        pff_id = str(record["pff_player_id"])
        key = bridge_key(
            record.get("season"),
            record.get("pff_team"),
            pff_id,
            record.get("pff_name_key"),
        )
        row_gsis_id = row_direct_identity.get(key)
        global_gsis_id = direct_identity.get(pff_id)
        if row_gsis_id is not None:
            historical_bridge_ids.append(row_gsis_id)
            historical_bridge_methods.append(
                "historical_season_team_name_gsis_row_bridge"
            )
        elif global_gsis_id is not None:
            historical_bridge_ids.append(global_gsis_id)
            historical_bridge_methods.append("historical_pff_id_gsis_bridge")
        else:
            historical_bridge_ids.append(None)
            historical_bridge_methods.append("")

    pff["historical_bridge_gsis_player_id"] = historical_bridge_ids
    pff["name_fallback_gsis_player_id"] = pff["pff_player_id"].map(mapping)
    pff["gsis_player_id"] = [
        historical_id
        if clean_scalar(historical_id) is not None
        else fallback_id
        for historical_id, fallback_id in zip(
            pff["historical_bridge_gsis_player_id"],
            pff["name_fallback_gsis_player_id"],
        )
    ]
    pff["pff_identity_match_method"] = historical_bridge_methods
    fallback_mask = (
        pff["pff_identity_match_method"].eq("")
        & pff["name_fallback_gsis_player_id"].notna()
    )
    pff.loc[fallback_mask, "pff_identity_match_method"] = "name_position_team_fallback"
    all_collision_ids = bridge_collision_ids | claim_collision_ids
    pff["pff_id_collision_flag"] = (
        pff["pff_player_id"].astype(str).isin(all_collision_ids).astype(int)
    )
    identity_audit = pd.DataFrame(audit_rows)

    matched = pff[pff["gsis_player_id"].notna()].copy()
    summaries: list[dict[str, Any]] = []
    match_methods = identity_audit.set_index("gsis_player_id")["match_method"].to_dict()
    for gsis_id, group in matched.groupby("gsis_player_id"):
        group = group.sort_values("season")
        grade_available = group["pff_season_talent_grade"].notna()
        weights = (
            group["season"].map(OL_PFF_SEASON_WEIGHTS).fillna(0.0)
            * group["pff_season_sample_reliability"].fillna(0.0)
            * grade_available.astype(float)
        )
        if float(weights.sum()) <= 0:
            continue
        talent = float(np.average(group.loc[weights.gt(0), "pff_season_talent_grade"], weights=weights[weights.gt(0)]))
        recent_row = group[group["pff_season_talent_grade"].notna()].sort_values("season").iloc[-1]
        total_snaps = float(group["offense_snaps"].fillna(0.0).sum())
        best_reliability = float(group["pff_season_sample_reliability"].fillna(0.0).max())
        cumulative_reliability = total_snaps / (total_snaps + OL_PFF_CONFIDENCE_PRIOR_SNAPS)
        confidence = float(np.clip(0.65 * best_reliability + 0.35 * cumulative_reliability, 0.0, 1.0))
        season_weight_available = group["season"].map(OL_PFF_SEASON_WEIGHTS).sum()
        effective_sample = float(weights.sum() / season_weight_available) if season_weight_available > 0 else 0.0
        pff_ids = sorted(set(group["pff_player_id"].astype(str)))
        summaries.append({
            "player_id": str(gsis_id),
            "ol_pff_history_available": 1,
            "ol_pff_player_id": "|".join(pff_ids),
            "ol_pff_position": str(recent_row["pff_position"]),
            "ol_pff_match_method": match_methods.get(str(gsis_id), "normalized_name_position_team_safe"),
            "ol_pff_talent_grade": talent,
            "ol_pff_recent_grade": float(recent_row["pff_season_talent_grade"]),
            "ol_pff_latest_season": int(recent_row["season"]),
            "ol_pff_seasons_observed": int(group["season"].nunique()),
            "ol_pff_total_snaps": total_snaps,
            "ol_pff_effective_sample": effective_sample,
            "ol_pff_confidence": confidence,
            "ol_pff_source": "pff_offense_blocking_2022_2025",
        })
    summary = pd.DataFrame(summaries)
    return pff, summary, identity_audit


# =============================================================================
# SAMPLE CONTEXT
# =============================================================================


def position_volume(frame: pd.DataFrame) -> pd.Series:
    pg = frame["position_group"].map(normalize_position_group)
    qb = numeric_series(frame, "qb_dropbacks", 0.0)
    rb = numeric_series(frame, "rush_attempts", 0.0) + numeric_series(frame, "targets", 0.0)
    wr = numeric_series(frame, "targets", 0.0)
    offense = numeric_series(frame, "offense_snaps", 0.0)
    defense = numeric_series(frame, "defense_snaps", 0.0)
    st = numeric_series(frame, "st_snaps", 0.0)
    total = numeric_series(frame, "total_snaps", 0.0)

    return pd.Series(
        np.select(
            [
                pg.eq("QB"), pg.eq("RB"), pg.eq("WR_TE"), pg.eq("OL"),
                pg.isin(DEFENSIVE_GROUPS), pg.eq("ST"),
            ],
            [qb, rb, wr, offense, defense, st],
            default=total,
        ),
        index=frame.index,
        dtype=float,
    )


def build_history_context(history: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    hist = history.copy()
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    hist = hist[hist["season"].isin(HISTORICAL_SEASONS)].copy()
    hist = hist.drop(columns=["current_position_group"], errors="ignore")
    hist = hist.merge(
        master[["player_id", "position_group"]].rename(columns={"position_group": "current_position_group"}),
        on="player_id", how="inner", validate="many_to_one",
    )
    # Confidence and current depth value should follow the player's current role.
    hist["position_group"] = hist["current_position_group"].map(normalize_position_group)
    hist["season_volume"] = position_volume(hist)

    qualify_threshold = hist["position_group"].map(RECENT_QUALIFY_VOLUME).fillna(100.0)
    hist["qualified_season"] = hist["season_volume"].ge(qualify_threshold).astype(int)

    last_three = hist[hist["season"].isin((2023, 2024, 2025))].copy()
    recent = hist[hist["season"].eq(RECENT_SEASON)].copy()

    aggregate = (
        last_three.groupby("player_id", dropna=False)
        .agg(
            games_played_3yr=("games", "sum"),
            total_snaps_3yr=("total_snaps", "sum"),
            volume_3yr=("season_volume", "sum"),
            qualified_seasons=("qualified_season", "sum"),
        )
        .reset_index()
    )
    observed = (
        hist.groupby("player_id", dropna=False)["season"]
        .nunique().rename("seasons_observed").reset_index()
    )
    recent_agg = (
        recent.groupby("player_id", dropna=False)
        .agg(
            games_played_l1=("games", "sum"),
            total_snaps_l1=("total_snaps", "sum"),
            volume_l1=("season_volume", "sum"),
            qualified_recent=("qualified_season", "max"),
        )
        .reset_index()
    )

    context = master[["player_id", "position_group"]].copy()
    context = context.merge(observed, on="player_id", how="left", validate="one_to_one")
    context = context.merge(aggregate, on="player_id", how="left", validate="one_to_one")
    context = context.merge(recent_agg, on="player_id", how="left", validate="one_to_one")

    numeric_cols = [
        "seasons_observed", "games_played_3yr", "total_snaps_3yr", "volume_3yr",
        "qualified_seasons", "games_played_l1", "total_snaps_l1", "volume_l1",
        "qualified_recent",
    ]
    for column in numeric_cols:
        context[column] = pd.to_numeric(context[column], errors="coerce").fillna(0.0)

    half_multi = context["position_group"].map(MULTI_YEAR_HALF_VOLUME).fillna(500.0)
    half_recent = context["position_group"].map(RECENT_HALF_VOLUME).fillna(250.0)
    volume_conf = reliability(context["volume_3yr"], half_multi)
    games_conf = reliability(context["games_played_3yr"], 17.0)
    season_conf = reliability(context["seasons_observed"], 1.5)
    context["position_confidence_score"] = (
        0.65 * volume_conf + 0.20 * games_conf + 0.15 * season_conf
    ).clip(0.0, 1.0)
    context["recent_confidence_score"] = (
        0.75 * reliability(context["volume_l1"], half_recent)
        + 0.25 * reliability(context["games_played_l1"], 8.0)
    ).clip(0.0, 1.0)
    context["history_available"] = context["seasons_observed"].gt(0).astype(int)
    context["history_volume_basis"] = context["volume_3yr"]
    return context.drop(columns=["position_group"])


# =============================================================================
# POSITION-SPECIFIC ADVANCED SCORING
# =============================================================================


def score_position_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one advanced-only position score without dedicated-source reuse."""
    out = frame.copy()
    out["position_group"] = out["position_group"].map(normalize_position_group)
    out["sample_volume"] = position_volume(out)
    games = numeric_series(out, "games", 0.0)

    # Metric percentile definitions. Core metrics rely on position volume;
    # optional NGS metrics additionally require a nonzero observed value.
    definitions: list[tuple[str, str, str, bool, pd.Series]] = []

    def add(pg: str | Sequence[str], source: str, target: str, higher: bool, availability: pd.Series) -> None:
        groups = {pg} if isinstance(pg, str) else set(pg)
        definitions.append(("|".join(sorted(groups)), source, target, higher, availability))

    qb_mask = out["position_group"].eq("QB")
    qb_sample = qb_mask & numeric_series(out, "qb_dropbacks", 0.0).gt(0)
    add("QB", "qb_epa_per_dropback", "qb_epa_pct", True, qb_sample)
    add("QB", "qb_success_rate", "qb_success_pct", True, qb_sample)
    add("QB", "qb_cpoe", "qb_cpoe_pct", True, qb_sample)
    add("QB", "qb_sack_rate", "qb_sack_avoid_pct", False, qb_sample)
    add("QB", "qb_int_rate", "qb_int_avoid_pct", False, qb_sample)
    add("QB", "qb_yards_per_dropback", "qb_yards_pct", True, qb_sample)

    rb_mask = out["position_group"].eq("RB")
    rb_carry = rb_mask & numeric_series(out, "rush_attempts", 0.0).gt(0)
    rb_target = rb_mask & numeric_series(out, "targets", 0.0).gt(0)
    rb_ngs = rb_carry & numeric_series(out, "ngs_rush_yards_over_expected_per_att", np.nan).notna() & numeric_series(out, "ngs_rush_yards_over_expected_per_att", 0.0).ne(0)
    add("RB", "rush_epa_per_carry", "rb_epa_pct", True, rb_carry)
    add("RB", "rush_success_rate", "rb_success_pct", True, rb_carry)
    add("RB", "rush_yards_per_carry", "rb_ypc_pct", True, rb_carry)
    add("RB", "rush_first_down_rate", "rb_first_down_pct", True, rb_carry)
    add("RB", "receiving_epa_per_target", "rb_receiving_pct", True, rb_target)
    add("RB", "ngs_rush_yards_over_expected_per_att", "rb_ryoe_pct", True, rb_ngs)

    wr_mask = out["position_group"].eq("WR_TE")
    wr_target = wr_mask & numeric_series(out, "targets", 0.0).gt(0)
    wr_sep = wr_target & numeric_series(out, "ngs_avg_separation", np.nan).notna() & numeric_series(out, "ngs_avg_separation", 0.0).ne(0)
    wr_yac = wr_target & numeric_series(out, "ngs_yac_above_expectation", np.nan).notna() & numeric_series(out, "ngs_yac_above_expectation", 0.0).ne(0)
    add("WR_TE", "receiving_epa_per_target", "wr_epa_pct", True, wr_target)
    add("WR_TE", "yards_per_target", "wr_ypt_pct", True, wr_target)
    add("WR_TE", "target_share_avg", "wr_target_share_pct", True, wr_target)
    add("WR_TE", "wopr_avg", "wr_wopr_pct", True, wr_target)
    add("WR_TE", "catch_rate", "wr_catch_pct", True, wr_target)
    add("WR_TE", "receiving_first_down_rate", "wr_first_down_pct", True, wr_target)
    add("WR_TE", "ngs_avg_separation", "wr_separation_pct", True, wr_sep)
    add("WR_TE", "ngs_yac_above_expectation", "wr_yac_pct", True, wr_yac)

    ol_mask = out["position_group"].eq("OL")
    ol_sample = ol_mask & numeric_series(out, "offense_snaps", 0.0).gt(0)
    add("OL", "offense_snaps", "ol_snaps_pct", True, ol_sample)
    add("OL", "games", "ol_games_pct", True, ol_sample)

    defense_groups = ["EDGE", "DL", "LB_EDGE", "LB", "DB"]
    for pg in defense_groups:
        mask = out["position_group"].eq(pg)
        sample = mask & numeric_series(out, "defense_snaps", 0.0).gt(0)
        add(pg, "pressure_proxy", f"{pg.lower()}_pressure_pct", True, sample)
        add(pg, "def_sacks", f"{pg.lower()}_sacks_pct", True, sample)
        add(pg, "def_qb_hits", f"{pg.lower()}_qb_hits_pct", True, sample)
        add(pg, "def_tfl", f"{pg.lower()}_tfl_pct", True, sample)
        add(pg, "run_defense_proxy", f"{pg.lower()}_run_pct", True, sample)
        add(pg, "coverage_playmaking_proxy", f"{pg.lower()}_coverage_pct", True, sample)
        add(pg, "def_tackles", f"{pg.lower()}_tackles_pct", True, sample)
        add(pg, "def_interceptions", f"{pg.lower()}_int_pct", True, sample)
        add(pg, "def_pass_defended", f"{pg.lower()}_pd_pct", True, sample)
        add(pg, "def_forced_fumbles", f"{pg.lower()}_ff_pct", True, sample)

    st_mask = out["position_group"].eq("ST")
    st_sample = st_mask & numeric_series(out, "st_snaps", 0.0).gt(0)
    add("ST", "st_snaps", "st_snaps_pct", True, st_sample)
    add("ST", "games", "st_games_pct", True, st_sample)

    for group_key, source, target, higher, availability in definitions:
        groups = set(group_key.split("|"))
        out = percentile_metric(
            out,
            out["position_group"].isin(groups),
            source,
            target,
            higher_is_better=higher,
            available_mask=availability,
        )

    composites: dict[str, Sequence[tuple[str, float]]] = {
        "QB": [
            ("qb_epa_pct", 0.30), ("qb_success_pct", 0.20),
            ("qb_cpoe_pct", 0.15), ("qb_sack_avoid_pct", 0.15),
            ("qb_int_avoid_pct", 0.10), ("qb_yards_pct", 0.10),
        ],
        "RB": [
            ("rb_epa_pct", 0.25), ("rb_success_pct", 0.20),
            ("rb_ypc_pct", 0.15), ("rb_ryoe_pct", 0.15),
            ("rb_first_down_pct", 0.15), ("rb_receiving_pct", 0.10),
        ],
        "WR_TE": [
            ("wr_epa_pct", 0.25), ("wr_ypt_pct", 0.20),
            ("wr_target_share_pct", 0.15), ("wr_wopr_pct", 0.10),
            ("wr_catch_pct", 0.10), ("wr_first_down_pct", 0.10),
            ("wr_separation_pct", 0.05), ("wr_yac_pct", 0.05),
        ],
        "OL": [("ol_snaps_pct", 0.65), ("ol_games_pct", 0.35)],
        "EDGE": [
            ("edge_pressure_pct", 0.35), ("edge_sacks_pct", 0.25),
            ("edge_qb_hits_pct", 0.15), ("edge_tfl_pct", 0.15),
            ("edge_run_pct", 0.10),
        ],
        "DL": [
            ("dl_pressure_pct", 0.25), ("dl_qb_hits_pct", 0.20),
            ("dl_tfl_pct", 0.20), ("dl_run_pct", 0.25),
            ("dl_sacks_pct", 0.10),
        ],
        "LB_EDGE": [
            ("lb_edge_pressure_pct", 0.25), ("lb_edge_tfl_pct", 0.15),
            ("lb_edge_run_pct", 0.15), ("lb_edge_coverage_pct", 0.15),
            ("lb_edge_tackles_pct", 0.15), ("lb_edge_pd_pct", 0.10),
            ("lb_edge_int_pct", 0.05),
        ],
        "LB": [
            ("lb_tackles_pct", 0.25), ("lb_tfl_pct", 0.15),
            ("lb_pressure_pct", 0.15), ("lb_coverage_pct", 0.15),
            ("lb_pd_pct", 0.15), ("lb_int_pct", 0.10),
            ("lb_run_pct", 0.05),
        ],
        "DB": [
            ("db_coverage_pct", 0.25), ("db_int_pct", 0.20),
            ("db_pd_pct", 0.20), ("db_tackles_pct", 0.15),
            ("db_run_pct", 0.10), ("db_ff_pct", 0.10),
        ],
        "ST": [("st_snaps_pct", 0.70), ("st_games_pct", 0.30)],
    }

    out["position_specific_score"] = np.nan
    out["score_component_count"] = 0
    for pg, components in composites.items():
        mask = out["position_group"].eq(pg)
        if not mask.any():
            continue
        group = weighted_component_score(out.loc[mask].copy(), components, "group_score")
        out.loc[mask, "position_specific_score"] = group["group_score"]
        out.loc[mask, "score_component_count"] = group["group_score_component_count"]

    # A player must have actual position volume to own an advanced score.
    out.loc[out["sample_volume"].le(0), "position_specific_score"] = np.nan
    out["position_specific_score"] = clip_0_100(out["position_specific_score"])
    return out


# =============================================================================
# FINAL ASSEMBLY
# =============================================================================


def merge_sources(
    master: pd.DataFrame,
    advanced_current: pd.DataFrame,
    qb: pd.DataFrame,
    defense: pd.DataFrame,
    history_context: pd.DataFrame,
) -> pd.DataFrame:
    advanced_drop = [
        "player_name", "team", "position", "position_group", "current_team",
        "current_position", "current_position_group", *IDENTITY_COLUMNS,
    ]
    out = master.merge(
        advanced_current.drop(columns=advanced_drop, errors="ignore"),
        on="player_id", how="left", validate="one_to_one",
    )
    # The rebuilt context owns these participation fields. Remove any older
    # copies from the advanced view before the one-to-one merge to prevent
    # pandas _x/_y suffixes and silent downstream defaults.
    context_columns = [column for column in history_context.columns if column != "player_id"]
    out = out.drop(columns=context_columns, errors="ignore")
    out = out.merge(history_context, on="player_id", how="left", validate="one_to_one")

    qb_prefix = qb.copy()
    rename_qb = {
        "qb_multi_year_score": "qb_rbsdm_multi_year_raw",
        "qb_recent_score": "qb_rbsdm_recent_raw",
        "qb_multi_year_rank": "qb_rbsdm_multi_year_rank",
        "qb_recent_rank": "qb_rbsdm_recent_rank",
        "qb_multi_year_percentile": "qb_rbsdm_multi_year_percentile",
        "qb_recent_percentile": "qb_rbsdm_recent_percentile",
        "qb_multi_year_available": "qb_rbsdm_multi_year_available",
        "qb_recent_available": "qb_rbsdm_recent_available",
    }
    qb_prefix = qb_prefix.rename(columns=rename_qb)
    qb_keep = ["player_id", *rename_qb.values(), "qb_rating_override", "qb_plays_4yr", "qb_plays_l1"]
    for column in qb_keep:
        if column not in qb_prefix.columns:
            qb_prefix[column] = np.nan
    out = out.merge(qb_prefix[qb_keep], on="player_id", how="left", validate="one_to_one")

    # Advanced history owns defense_snaps. The dedicated defensive model also
    # publishes a field with that name, so exclude the duplicate rather than
    # allowing a suffix collision that would erase the canonical snap column.
    defense_merge_numeric = [column for column in DEFENSE_NUMERIC_COLUMNS if column != "defense_snaps"]
    defense_keep = ["player_id", *defense_merge_numeric, "defensive_metrics_version"]
    for column in defense_keep:
        if column not in defense.columns:
            defense[column] = np.nan if column != "defensive_metrics_version" else None
    out = out.merge(defense[defense_keep], on="player_id", how="left", validate="one_to_one")

    for column in ADVANCED_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in DEFENSE_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_recent_scores(
    history: pd.DataFrame,
    master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = history[pd.to_numeric(history["season"], errors="coerce").eq(RECENT_SEASON)].copy()
    if recent.empty:
        raise RuntimeError(f"{ADVANCED_HISTORY_TABLE} contains no {RECENT_SEASON} rows.")

    # Score all 2025 participants against their 2025 position peers, then keep
    # current-roster canonical IDs. This avoids current-roster survivorship bias.
    recent_scored = score_position_frame(recent)
    recent_scored["season_confidence"] = 0.0
    half = recent_scored["position_group"].map(RECENT_HALF_VOLUME).fillna(250.0)
    recent_scored["season_confidence"] = (
        0.75 * reliability(recent_scored["sample_volume"], half)
        + 0.25 * reliability(numeric_series(recent_scored, "games", 0.0), 8.0)
    ).clip(0.0, 1.0)
    threshold = recent_scored["position_group"].map(RECENT_QUALIFY_VOLUME).fillna(100.0)
    recent_scored["qualified_season"] = recent_scored["sample_volume"].ge(threshold).astype(int)
    recent_scored["performance_version"] = PERFORMANCE_VERSION
    recent_scored["date_imported"] = pd.to_datetime(dt.date.today())

    seasonal = history.copy()
    seasonal_parts: list[pd.DataFrame] = []
    for season in HISTORICAL_SEASONS:
        part = seasonal[pd.to_numeric(seasonal["season"], errors="coerce").eq(season)].copy()
        if part.empty:
            continue
        scored = score_position_frame(part)
        half_part = scored["position_group"].map(RECENT_HALF_VOLUME).fillna(250.0)
        scored["season_confidence"] = (
            0.75 * reliability(scored["sample_volume"], half_part)
            + 0.25 * reliability(numeric_series(scored, "games", 0.0), 8.0)
        ).clip(0.0, 1.0)
        threshold_part = scored["position_group"].map(RECENT_QUALIFY_VOLUME).fillna(100.0)
        scored["qualified_season"] = scored["sample_volume"].ge(threshold_part).astype(int)
        scored["season_volume"] = scored["sample_volume"]
        scored["performance_version"] = PERFORMANCE_VERSION
        scored["date_imported"] = pd.to_datetime(dt.date.today())
        for column in SEASONAL_OUTPUT_COLUMNS:
            if column not in scored.columns:
                scored[column] = np.nan
        seasonal_parts.append(scored[SEASONAL_OUTPUT_COLUMNS])

    seasonal_output = pd.concat(seasonal_parts, ignore_index=True) if seasonal_parts else pd.DataFrame(columns=SEASONAL_OUTPUT_COLUMNS)

    current_ids = set(master["player_id"].dropna().astype(str))
    recent_scored = recent_scored[recent_scored["player_id"].astype(str).isin(current_ids)].copy()
    recent_scored = recent_scored.sort_values(
        ["player_id", "total_snaps"], ascending=[True, False]
    ).drop_duplicates("player_id", keep="first")
    recent_out = recent_scored[[
        "player_id", "position_specific_score", "games", "total_snaps",
        "sample_volume", "season_confidence", "qualified_season",
    ]].rename(columns={
        "position_specific_score": "recent_position_score",
        "games": "recent_games",
        "total_snaps": "recent_total_snaps",
        "sample_volume": "recent_volume",
        "season_confidence": "recent_score_confidence_from_history",
        "qualified_season": "qualified_recent_from_history",
    })
    return recent_out, seasonal_output


def add_final_scores(
    frame: pd.DataFrame,
    recent_scores: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    out = score_position_frame(out)
    out = out.rename(columns={"position_specific_score": "multi_year_position_score"})
    out = out.merge(recent_scores, on="player_id", how="left", validate="one_to_one")

    pff_numeric = [
        "ol_pff_history_available", "ol_pff_talent_grade", "ol_pff_recent_grade",
        "ol_pff_latest_season", "ol_pff_seasons_observed", "ol_pff_total_snaps",
        "ol_pff_effective_sample", "ol_pff_confidence",
    ]
    for column in pff_numeric:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["ol_pff_history_available"] = out["ol_pff_history_available"].fillna(0).gt(0).astype(int)
    for column in ["ol_pff_player_id", "ol_pff_position", "ol_pff_match_method", "ol_pff_source"]:
        if column not in out.columns:
            out[column] = None

    # Dedicated QB scores already arrive on a 0-100 scale. Use raw score fields
    # as the model grades and retain ranks/percentiles for audit.
    out["qb_rbsdm_multi_year_score"] = pd.to_numeric(
        out.get("qb_rbsdm_multi_year_raw"), errors="coerce"
    )
    out["qb_rbsdm_recent_score"] = pd.to_numeric(
        out.get("qb_rbsdm_recent_raw"), errors="coerce"
    )
    out["qb_rbsdm_multi_year_available"] = pd.to_numeric(
        out.get("qb_rbsdm_multi_year_available"), errors="coerce"
    ).fillna(0).gt(0).astype(int)
    out["qb_rbsdm_recent_available"] = pd.to_numeric(
        out.get("qb_rbsdm_recent_available"), errors="coerce"
    ).fillna(0).gt(0).astype(int)

    out["multi_year_performance_score"] = out["multi_year_position_score"]
    out["recent_season_performance_score"] = out["recent_position_score"]

    qb_mask = out["position_group"].eq("QB")
    qb_multi_available = qb_mask & out["qb_rbsdm_multi_year_available"].eq(1) & out["qb_rbsdm_multi_year_score"].notna()
    qb_recent_available = qb_mask & out["qb_rbsdm_recent_available"].eq(1) & out["qb_rbsdm_recent_score"].notna()
    # RBSDM efficiency is authoritative for supported QBs. Do not blend the
    # same EPA/success/CPOE information back in through the generic QB score.
    out.loc[qb_multi_available, "multi_year_performance_score"] = out.loc[
        qb_multi_available, "qb_rbsdm_multi_year_score"
    ]
    out.loc[qb_recent_available, "recent_season_performance_score"] = out.loc[
        qb_recent_available, "qb_rbsdm_recent_score"
    ]
    # If the dedicated multi-year estimate is available but the 2025 sample is
    # not, hold the multi-year grade and give the recent leg zero weight. This
    # prevents a low-sample generic QB score from re-entering the model.
    qb_recent_unavailable = qb_multi_available & ~qb_recent_available
    out.loc[qb_recent_unavailable, "recent_season_performance_score"] = out.loc[
        qb_recent_unavailable, "multi_year_performance_score"
    ]

    defensive_mask = out["position_group"].isin(DEFENSIVE_GROUPS)
    dedicated_defense = (
        defensive_mask
        & pd.to_numeric(out.get("defensive_history_available"), errors="coerce").fillna(0).gt(0)
        & pd.to_numeric(out.get("defensive_advanced_score"), errors="coerce").notna()
    )
    out.loc[dedicated_defense, "multi_year_performance_score"] = (
        DEFENSE_DEDICATED_WEIGHT * out.loc[dedicated_defense, "defensive_advanced_score"]
        + (1.0 - DEFENSE_DEDICATED_WEIGHT) * out.loc[dedicated_defense, "multi_year_position_score"].fillna(50.0)
    )

    # Dedicated-source history is also valid history. This primarily protects
    # locally supplied QB records or defensive records that are usable even if
    # an upstream advanced row is temporarily absent.
    out["history_available"] = np.maximum(
        pd.to_numeric(out.get("history_available"), errors="coerce").fillna(0).astype(int),
        (qb_multi_available | dedicated_defense).astype(int),
    )

    # OL is intentionally different from every other position. The prior
    # advanced-data OL score measured participation, so it is never allowed to
    # masquerade as blocking talent. PFF blocking history replaces it outright.
    ol_mask = out["position_group"].eq("OL")
    pff_ol = (
        ol_mask
        & out["ol_pff_history_available"].eq(1)
        & out["ol_pff_talent_grade"].notna()
    )
    out.loc[pff_ol, "multi_year_performance_score"] = out.loc[pff_ol, "ol_pff_talent_grade"]
    out.loc[pff_ol, "recent_season_performance_score"] = out.loc[pff_ol, "ol_pff_recent_grade"].fillna(
        out.loc[pff_ol, "ol_pff_talent_grade"]
    )
    out.loc[pff_ol, "history_available"] = 1
    out.loc[pff_ol, "usable_performance_grade"] = 1

    out["recent_score_available"] = (
        out["recent_position_score"].notna()
        | qb_recent_available
    ).astype(int)
    out.loc[qb_multi_available, "recent_score_available"] = qb_recent_available.loc[
        qb_multi_available
    ].astype(int)
    out.loc[pff_ol, "recent_score_available"] = (
        out.loc[pff_ol, "ol_pff_latest_season"].fillna(0).astype(int).eq(RECENT_SEASON)
    ).astype(int)
    out["recent_position_score"] = out["recent_position_score"].where(
        out["recent_position_score"].notna(), out["multi_year_position_score"]
    )
    out["recent_season_performance_score"] = out["recent_season_performance_score"].where(
        out["recent_season_performance_score"].notna(), out["multi_year_performance_score"]
    )

    out["recent_confidence_score"] = pd.to_numeric(
        out.get("recent_confidence_score"), errors="coerce"
    ).fillna(pd.to_numeric(out.get("recent_score_confidence_from_history"), errors="coerce")).fillna(0.0).clip(0.0, 1.0)
    out["qualified_recent"] = np.maximum(
        pd.to_numeric(out.get("qualified_recent"), errors="coerce").fillna(0),
        pd.to_numeric(out.get("qualified_recent_from_history"), errors="coerce").fillna(0),
    ).astype(int)

    base_recent = out["position_group"].map(RECENT_BASE_WEIGHT).fillna(0.25)
    out["effective_recent_weight"] = (
        base_recent * out["recent_confidence_score"] * out["recent_score_available"]
    ).clip(0.0, 0.55)
    # ol_pff_talent_grade already contains the explicit recency/sample blend.
    out.loc[pff_ol, "effective_recent_weight"] = 0.0

    # A player may have historical participation without any scoreable sample
    # for the current position (for example, a WR with only special-teams
    # snaps).  Preserve that participation in history_available, but do not
    # allow a missing position grade to flow into the final score.
    out["multi_year_performance_score"] = out["multi_year_performance_score"].where(
        out["multi_year_performance_score"].notna(),
        out["recent_season_performance_score"],
    )
    out["multi_year_performance_score"] = clip_0_100(out["multi_year_performance_score"])
    out["recent_season_performance_score"] = out["recent_season_performance_score"].where(
        out["recent_season_performance_score"].notna(),
        out["multi_year_performance_score"],
    )
    out["recent_season_performance_score"] = clip_0_100(out["recent_season_performance_score"])
    out["usable_performance_grade"] = out["multi_year_performance_score"].notna().astype(int)
    out.loc[ol_mask & ~pff_ol, "usable_performance_grade"] = 0

    out["replacement_baseline"] = out["position_group"].map(REPLACEMENT_BASELINE).fillna(18.0)
    out.loc[ol_mask, "replacement_baseline"] = out.loc[ol_mask, "position"].map(ol_replacement_baseline)
    history_score = (
        (1.0 - out["effective_recent_weight"]) * out["multi_year_performance_score"]
        + out["effective_recent_weight"] * out["recent_season_performance_score"]
    )
    history_score = history_score.where(
        out["usable_performance_grade"].eq(1),
        out["replacement_baseline"],
    )

    out["durability_score"] = (
        pd.to_numeric(out.get("games_played_3yr"), errors="coerce").fillna(0.0)
        / 51.0 * 100.0
    ).clip(0.0, 100.0)
    out["positional_scarcity_score"] = out["position_group"].map(POSITION_SCARCITY).fillna(10.0)
    out["age_curve_score_placeholder"] = out.apply(
        lambda row: age_curve_score(row.get("age"), row.get("position_group", "OTHER")), axis=1
    )
    out["age_score"] = out["age_curve_score_placeholder"]

    # Context is deliberately small. Positional scarcity is not mixed into the
    # quality grade because unit construction already owns positional value.
    out["performance_input_score_raw"] = (
        0.88 * history_score
        + 0.08 * out["durability_score"]
        + 0.04 * out["age_curve_score_placeholder"]
    )
    # QB durability and age remain visible audit metadata, but QB talent is the
    # dedicated efficiency history blend alone. Confidence is applied once
    # below against the QB replacement baseline.
    out.loc[qb_multi_available, "performance_input_score_raw"] = history_score.loc[
        qb_multi_available
    ]
    # Talent is the complete OL quality input. Durability stays available as a
    # separate depth/availability field and is not counted again in quality.
    out.loc[pff_ol, "performance_input_score_raw"] = out.loc[pff_ol, "ol_pff_talent_grade"]
    out["pre_confidence_normalized_score"] = clip_0_100(out["performance_input_score_raw"])
    out["position_confidence_score"] = pd.to_numeric(
        out.get("position_confidence_score"), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)

    qb_source_confidence = (
        0.75 * reliability(pd.to_numeric(out.get("qb_plays_4yr"), errors="coerce"), 600.0)
        + 0.25 * reliability(pd.to_numeric(out.get("qb_plays_l1"), errors="coerce"), 300.0)
    )
    defense_source_confidence = (
        0.60 * reliability(pd.to_numeric(out.get("defensive_event_games"), errors="coerce"), 17.0)
        + 0.40 * reliability(pd.to_numeric(out.get("defense_snaps"), errors="coerce"), 700.0)
    )
    out.loc[qb_multi_available, "position_confidence_score"] = qb_source_confidence.loc[
        qb_multi_available
    ]
    out.loc[dedicated_defense, "position_confidence_score"] = np.maximum(
        out.loc[dedicated_defense, "position_confidence_score"],
        defense_source_confidence.loc[dedicated_defense],
    )
    out.loc[pff_ol, "position_confidence_score"] = out.loc[pff_ol, "ol_pff_confidence"].fillna(0.0)
    out["position_confidence_score"] = out["position_confidence_score"].clip(0.0, 1.0)
    out.loc[out["usable_performance_grade"].eq(0), "position_confidence_score"] = 0.0

    out["position_normalized_score"] = (
        out["replacement_baseline"]
        + out["position_confidence_score"]
        * (out["pre_confidence_normalized_score"] - out["replacement_baseline"])
    ).clip(0.0, 100.0)
    out["performance_input_score"] = out["position_normalized_score"]

    # No-history and history-but-unscoreable rows remain exactly at
    # replacement, regardless of age or durability context.
    no_history = pd.to_numeric(out.get("history_available"), errors="coerce").fillna(0).eq(0)
    unusable_grade = out["usable_performance_grade"].eq(0)
    replacement_only = no_history | unusable_grade
    out.loc[replacement_only, "position_confidence_score"] = 0.0
    out.loc[replacement_only, "performance_input_score"] = out.loc[replacement_only, "replacement_baseline"]
    out.loc[replacement_only, "position_normalized_score"] = out.loc[replacement_only, "replacement_baseline"]
    out.loc[replacement_only, "performance_input_score_raw"] = out.loc[replacement_only, "replacement_baseline"]
    out.loc[replacement_only, "pre_confidence_normalized_score"] = out.loc[replacement_only, "replacement_baseline"]

    out["qb_starter_pool"] = 0
    qb_candidates = out[qb_mask].copy()
    if not qb_candidates.empty:
        qb_candidates["qb_sort_score"] = pd.to_numeric(
            qb_candidates["performance_input_score"], errors="coerce"
        ).fillna(-1.0)
        qb_candidates["qb_recent_plays"] = pd.to_numeric(
            qb_candidates.get("qb_plays_l1"), errors="coerce"
        ).fillna(0.0)
        starter_indices = (
            qb_candidates.sort_values(
                ["team", "qb_sort_score", "qb_recent_plays", "player_name"],
                ascending=[True, False, False, True],
            ).groupby("team", dropna=False).head(1).index
        )
        out.loc[starter_indices, "qb_starter_pool"] = 1

    # Compatibility position scores.
    for column in ["qb_score", "rb_score", "wr_te_score", "ol_score", "edge_score", "dl_score", "lb_score", "db_score", "st_score"]:
        out[column] = 0.0
    mapping = {
        "QB": "qb_score", "RB": "rb_score", "WR_TE": "wr_te_score", "OL": "ol_score",
        "EDGE": "edge_score", "DL": "dl_score", "LB_EDGE": "lb_score", "LB": "lb_score",
        "DB": "db_score", "ST": "st_score",
    }
    for pg, column in mapping.items():
        mask = out["position_group"].eq(pg)
        out.loc[mask, column] = out.loc[mask, "multi_year_performance_score"].fillna(0.0)

    out["position_specific_score"] = out["multi_year_position_score"]
    out["qb_rbsdm_score"] = out["qb_rbsdm_multi_year_score"]
    out["qb_advanced_score"] = np.where(qb_mask, out["multi_year_position_score"], 0.0)
    out["advanced_efficiency_score"] = out["multi_year_position_score"]

    # Volume and playmaking audit scores are position-relative but are never
    # fed back into the final grade a second time.
    out["advanced_volume_score"] = np.nan
    out["advanced_playmaking_score"] = np.nan
    for pg, idx in out.groupby("position_group").groups.items():
        volume = pd.to_numeric(out.loc[idx, "volume_3yr"], errors="coerce")
        eligible = volume.gt(0)
        if eligible.any():
            vals = volume[eligible]
            out.loc[vals.index, "advanced_volume_score"] = (
                vals.rank(method="average", pct=True) * 100.0
            )
        playmaking = (
            numeric_series(out.loc[idx], "defensive_playmaking_score", 0.0)
            + numeric_series(out.loc[idx], "receiving_epa", 0.0)
            + numeric_series(out.loc[idx], "rush_epa", 0.0)
        )
        if playmaking.nunique(dropna=True) > 1:
            out.loc[idx, "advanced_playmaking_score"] = playmaking.rank(method="average", pct=True) * 100.0
    out["advanced_score_used"] = out["multi_year_position_score"]

    out["has_dedicated_defensive_metric"] = dedicated_defense.astype(int)
    out["master_matched"] = True
    out["recent_season_used"] = RECENT_SEASON
    out["historical_seasons_used"] = ",".join(str(x) for x in HISTORICAL_SEASONS)
    out["performance_version"] = PERFORMANCE_VERSION
    out["date_imported"] = pd.to_datetime(dt.date.today())

    # Current advanced metrics remain available for downstream diagnostics.
    for column in ADVANCED_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    for column in DEFENSE_NUMERIC_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out["recent_games"] = pd.to_numeric(out.get("recent_games"), errors="coerce").fillna(
        pd.to_numeric(out.get("games_played_l1"), errors="coerce")
    ).fillna(0.0)
    out["recent_total_snaps"] = pd.to_numeric(out.get("recent_total_snaps"), errors="coerce").fillna(
        pd.to_numeric(out.get("total_snaps_l1"), errors="coerce")
    ).fillna(0.0)
    out["recent_volume"] = pd.to_numeric(out.get("recent_volume"), errors="coerce").fillna(
        pd.to_numeric(out.get("volume_l1"), errors="coerce")
    ).fillna(0.0)

    # Old downstream names expected by team-unit code.
    out["games"] = pd.to_numeric(out.get("games"), errors="coerce").fillna(0.0)
    out["total_snaps"] = pd.to_numeric(out.get("total_snaps"), errors="coerce").fillna(0.0)
    out["offense_snaps"] = pd.to_numeric(out.get("offense_snaps"), errors="coerce").fillna(0.0)
    out["defense_snaps"] = pd.to_numeric(out.get("defense_snaps"), errors="coerce").fillna(0.0)
    out["st_snaps"] = pd.to_numeric(out.get("st_snaps"), errors="coerce").fillna(0.0)

    for column in FINAL_COLUMNS:
        if column not in out.columns:
            out[column] = None
    final = out[FINAL_COLUMNS].copy()
    return final


# =============================================================================
# AUDITS AND VALIDATION
# =============================================================================


def build_position_summary(performance: pd.DataFrame) -> pd.DataFrame:
    summary = (
        performance.groupby("position_group", dropna=False)
        .agg(
            players=("player_id", "count"),
            with_history=("history_available", "sum"),
            with_usable_grade=("usable_performance_grade", "sum"),
            recent_available=("recent_score_available", "sum"),
            avg_confidence=("position_confidence_score", "mean"),
            avg_final_score=("performance_input_score", "mean"),
            min_final_score=("performance_input_score", "min"),
            max_final_score=("performance_input_score", "max"),
            avg_multi_year_score=("multi_year_performance_score", "mean"),
            avg_recent_score=("recent_season_performance_score", "mean"),
        )
        .reset_index()
    )
    summary["history_rate"] = np.where(summary["players"] > 0, summary["with_history"] / summary["players"], 0.0)
    return summary.sort_values("position_group").reset_index(drop=True)


def build_component_audit(performance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_id", "player_name", "team", "position", "position_group",
        "history_available", "usable_performance_grade", "seasons_observed", "games_played_3yr", "volume_3yr",
        "position_confidence_score", "recent_score_available", "recent_volume",
        "recent_confidence_score", "effective_recent_weight",
        "multi_year_position_score", "recent_position_score",
        "qb_rbsdm_multi_year_score", "qb_rbsdm_recent_score",
        "defensive_advanced_score", "multi_year_performance_score",
        "recent_season_performance_score", "durability_score", "age_score",
        "ol_pff_history_available", "ol_pff_player_id", "ol_pff_position",
        "ol_pff_match_method", "ol_pff_talent_grade", "ol_pff_recent_grade",
        "ol_pff_latest_season", "ol_pff_seasons_observed", "ol_pff_total_snaps",
        "ol_pff_effective_sample", "ol_pff_confidence", "ol_pff_source",
        "replacement_baseline", "performance_input_score_raw",
        "performance_input_score", "qb_starter_pool", "performance_version",
    ]
    return performance[columns].sort_values(
        ["position_group", "performance_input_score"], ascending=[True, False]
    ).reset_index(drop=True)


def validate_outputs(
    performance: pd.DataFrame,
    seasonal: pd.DataFrame,
    master: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    validate_unique_ids(performance, OUTPUT_TABLE)
    expected_ids = set(master["player_id"].dropna().astype(str))
    actual_ids = set(performance["player_id"].dropna().astype(str))
    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise RuntimeError(f"Performance/master ID reconciliation failed. missing={missing}, extra={extra}")

    for column in ["performance_input_score", "position_normalized_score", "position_confidence_score"]:
        values = pd.to_numeric(performance[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(f"{column} contains {int(values.isna().sum())} missing values.")

    usable = pd.to_numeric(performance["usable_performance_grade"], errors="coerce").fillna(0).astype(int)
    if (~usable.isin([0, 1])).any():
        raise RuntimeError("usable_performance_grade must contain only 0 or 1.")
    unusable = usable.eq(0)
    if unusable.any():
        baseline = pd.to_numeric(performance.loc[unusable, "replacement_baseline"], errors="coerce")
        final_score = pd.to_numeric(performance.loc[unusable, "performance_input_score"], errors="coerce")
        confidence = pd.to_numeric(performance.loc[unusable, "position_confidence_score"], errors="coerce")
        if not np.allclose(final_score.to_numpy(), baseline.to_numpy(), atol=1e-9, equal_nan=False):
            raise RuntimeError("Players without a usable performance grade are not exactly at replacement.")
        if not np.allclose(confidence.to_numpy(), 0.0, atol=1e-12, equal_nan=False):
            raise RuntimeError("Players without a usable performance grade must have zero confidence.")
    for column in ["performance_input_score", "position_normalized_score"]:
        values = pd.to_numeric(performance[column], errors="coerce")
        if (~values.between(0.0, 100.0)).any():
            raise RuntimeError(f"{column} contains out-of-range values.")
    if (~pd.to_numeric(performance["position_confidence_score"], errors="coerce").between(0.0, 1.0)).any():
        raise RuntimeError("position_confidence_score contains out-of-range values.")

    history_ids = set(history["player_id"].dropna().astype(str)) & expected_ids
    retained_history = set(
        performance.loc[pd.to_numeric(performance["history_available"], errors="coerce").fillna(0).gt(0), "player_id"].astype(str)
    )
    if not history_ids.issubset(retained_history):
        examples = sorted(history_ids - retained_history)[:10]
        raise RuntimeError(f"History rollup lost current players: {examples}")

    seasons = set(pd.to_numeric(seasonal["season"], errors="coerce").dropna().astype(int))
    if set(HISTORICAL_SEASONS) - seasons:
        raise RuntimeError(f"Seasonal output missing seasons: {sorted(set(HISTORICAL_SEASONS) - seasons)}")

    # Every sufficiently populated position with usable grades must retain
    # real score variation.  Participation-only players are intentionally tied
    # at replacement and are excluded from this diagnostic.
    for pg, group in performance.groupby("position_group"):
        graded = group[pd.to_numeric(group["usable_performance_grade"], errors="coerce").fillna(0).gt(0)]
        if len(graded) >= 10 and graded["performance_input_score"].nunique() < 3:
            raise RuntimeError(f"{pg} final grades have insufficient variation.")

    # Dedicated QB rows must reconcile to the authoritative efficiency scores,
    # the confidence-adjusted recent blend, and exactly one final shrink toward
    # replacement. These checks prevent the removed generic/age/durability
    # terms from silently re-entering the production QB grade.
    qb = performance[
        performance["position_group"].eq("QB")
        & pd.to_numeric(
            performance["qb_rbsdm_multi_year_available"], errors="coerce"
        ).fillna(0).eq(1)
    ].copy()
    if not qb.empty:
        if not np.allclose(
            pd.to_numeric(qb["multi_year_performance_score"], errors="coerce"),
            pd.to_numeric(qb["qb_rbsdm_multi_year_score"], errors="coerce"),
            atol=1e-9,
        ):
            raise RuntimeError("QB multi-year grade is not the dedicated efficiency score.")

        qb_recent = pd.to_numeric(
            qb["qb_rbsdm_recent_available"], errors="coerce"
        ).fillna(0).eq(1)
        if qb_recent.any() and not np.allclose(
            pd.to_numeric(qb.loc[qb_recent, "recent_season_performance_score"], errors="coerce"),
            pd.to_numeric(qb.loc[qb_recent, "qb_rbsdm_recent_score"], errors="coerce"),
            atol=1e-9,
        ):
            raise RuntimeError("QB recent grade is not the dedicated efficiency score.")
        if (~qb_recent).any() and not np.allclose(
            pd.to_numeric(qb.loc[~qb_recent, "effective_recent_weight"], errors="coerce"),
            0.0,
            atol=1e-12,
        ):
            raise RuntimeError("QB without a dedicated recent sample received recent weight.")

        expected_qb_raw = (
            (1.0 - pd.to_numeric(qb["effective_recent_weight"], errors="coerce"))
            * pd.to_numeric(qb["multi_year_performance_score"], errors="coerce")
            + pd.to_numeric(qb["effective_recent_weight"], errors="coerce")
            * pd.to_numeric(qb["recent_season_performance_score"], errors="coerce")
        )
        if not np.allclose(
            pd.to_numeric(qb["performance_input_score_raw"], errors="coerce"),
            expected_qb_raw,
            atol=1e-9,
        ):
            raise RuntimeError("QB raw talent includes a non-efficiency component.")

        expected_qb_final = (
            pd.to_numeric(qb["replacement_baseline"], errors="coerce")
            + pd.to_numeric(qb["position_confidence_score"], errors="coerce")
            * (
                pd.to_numeric(qb["performance_input_score_raw"], errors="coerce")
                - pd.to_numeric(qb["replacement_baseline"], errors="coerce")
            )
        )
        if not np.allclose(
            pd.to_numeric(qb["performance_input_score"], errors="coerce"),
            expected_qb_final,
            atol=1e-9,
        ):
            raise RuntimeError("QB final grade does not reconcile to one confidence shrink.")

    ol = performance[performance["position_group"].eq("OL")].copy()
    pff_ol = ol[pd.to_numeric(ol["ol_pff_history_available"], errors="coerce").fillna(0).eq(1)]
    if pff_ol.empty:
        raise RuntimeError("No current-roster OL players matched the required PFF archive.")
    expected = (
        pd.to_numeric(pff_ol["replacement_baseline"], errors="coerce")
        + pd.to_numeric(pff_ol["ol_pff_confidence"], errors="coerce")
        * (
            pd.to_numeric(pff_ol["ol_pff_talent_grade"], errors="coerce")
            - pd.to_numeric(pff_ol["replacement_baseline"], errors="coerce")
        )
    )
    if not np.allclose(
        pd.to_numeric(pff_ol["performance_input_score"], errors="coerce"),
        expected,
        atol=1e-9,
    ):
        raise RuntimeError("PFF OL talent does not reconcile to the single confidence shrink.")
    no_pff = ol[pd.to_numeric(ol["ol_pff_history_available"], errors="coerce").fillna(0).eq(0)]
    if not no_pff.empty and not np.allclose(
        pd.to_numeric(no_pff["performance_input_score"], errors="coerce"),
        pd.to_numeric(no_pff["replacement_baseline"], errors="coerce"),
        atol=1e-9,
    ):
        raise RuntimeError("OL players without PFF history are not exactly at replacement.")


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
    logger = configure_logging(log_dir / "build_nfl_player_performance.log")

    logger.info("[PERF] Building canonical NFL player performance inputs")
    logger.info("[PERF] Build ID: %s", BUILD_ID)
    logger.info("[PERF] Version: %s", PERFORMANCE_VERSION)
    logger.info("[PERF] Database: %s", db_path)

    engine = get_engine(db_path)
    required = [MASTER_TABLE, ADVANCED_CURRENT_TABLE, ADVANCED_HISTORY_TABLE, QB_TABLE, DEFENSE_TABLE]
    missing = [table for table in required if not table_exists(engine, table)]
    if missing:
        raise RuntimeError(f"Missing required input tables: {', '.join(missing)}")

    master_raw = read_table(engine, MASTER_TABLE)
    advanced_current_raw = read_table(engine, ADVANCED_CURRENT_TABLE)
    history_raw = read_table(engine, ADVANCED_HISTORY_TABLE)
    qb_raw = read_table(engine, QB_TABLE)
    defense_raw = read_table(engine, DEFENSE_TABLE)

    pff_paths = resolve_pff_paths(project_root, args)
    for season, path in sorted(pff_paths.items()):
        logger.info("[PERF] PFF OL %s source: %s", season, path)
    pff_history_raw = load_pff_ol_history(pff_paths)

    master, unresolved = prepare_master(master_raw)
    advanced_current = prepare_advanced(advanced_current_raw, ADVANCED_CURRENT_TABLE, require_unique=True)
    history = prepare_advanced(history_raw, ADVANCED_HISTORY_TABLE, require_unique=False)
    qb = prepare_qb(qb_raw)
    defense = prepare_defense(defense_raw)
    pff_history, pff_summary, pff_identity_audit = match_pff_ol_to_master(
        pff_history_raw,
        master,
        history,
    )

    seasons_found = sorted(set(pd.to_numeric(history["season"], errors="coerce").dropna().astype(int)))
    if not set(HISTORICAL_SEASONS).issubset(seasons_found):
        raise RuntimeError(f"Advanced history missing required seasons. Found: {seasons_found}")

    logger.info("[PERF] Canonical master players: %s", f"{len(master):,}")
    logger.info("[PERF] Unresolved master rows excluded: %s", f"{len(unresolved):,}")
    logger.info("[PERF] Advanced current rows: %s", f"{len(advanced_current):,}")
    logger.info("[PERF] Advanced historical rows: %s", f"{len(history):,}")
    logger.info("[PERF] QB rows: %s", f"{len(qb):,}")
    logger.info("[PERF] Defensive rows: %s", f"{len(defense):,}")
    logger.info("[PERF] Historical seasons: %s", seasons_found)

    history_context = build_history_context(history, master)
    merged = merge_sources(master, advanced_current, qb, defense, history_context)
    if not pff_summary.empty:
        merged = merged.merge(pff_summary, on="player_id", how="left", validate="one_to_one")
    recent_scores, seasonal_output = build_recent_scores(history, master)
    performance = add_final_scores(merged, recent_scores)

    summary = build_position_summary(performance)
    component_audit = build_component_audit(performance)
    validate_outputs(performance, seasonal_output, master, history)

    csv = None if args.no_csv else output_dir / "nfl_player_performance_inputs_2026.csv"
    seasonal_csv = None if args.no_csv else output_dir / "nfl_player_performance_seasonal_2022_2025.csv"
    summary_csv = None if args.no_csv else output_dir / "nfl_player_performance_position_summary_2026.csv"
    component_csv = None if args.no_csv else output_dir / "nfl_player_performance_component_audit_2026.csv"
    unresolved_csv = None if args.no_csv else output_dir / "nfl_player_performance_unresolved_master_audit_2026.csv"
    pff_history_csv = None if args.no_csv else output_dir / "nfl_ol_pff_player_season_2022_2025.csv"
    pff_identity_csv = None if args.no_csv else output_dir / "nfl_ol_pff_identity_audit_2026.csv"

    save_frame(performance, engine, OUTPUT_TABLE, csv)
    save_frame(seasonal_output, engine, SEASONAL_OUTPUT_TABLE, seasonal_csv)
    save_frame(summary, engine, POSITION_SUMMARY_TABLE, summary_csv)
    save_frame(component_audit, engine, COMPONENT_AUDIT_TABLE, component_csv)
    save_frame(unresolved, engine, UNRESOLVED_AUDIT_TABLE, unresolved_csv)
    save_frame(pff_history, engine, PFF_OL_HISTORY_TABLE, pff_history_csv)
    save_frame(pff_identity_audit, engine, PFF_OL_IDENTITY_AUDIT_TABLE, pff_identity_csv)

    history_count = int(pd.to_numeric(performance["history_available"], errors="coerce").fillna(0).gt(0).sum())
    usable_count = int(pd.to_numeric(performance["usable_performance_grade"], errors="coerce").fillna(0).gt(0).sum())
    recent_count = int(pd.to_numeric(performance["recent_score_available"], errors="coerce").fillna(0).gt(0).sum())
    qb_match_count = int(pd.to_numeric(performance["qb_rbsdm_multi_year_available"], errors="coerce").fillna(0).gt(0).sum())
    defense_match_count = int(pd.to_numeric(performance["has_dedicated_defensive_metric"], errors="coerce").fillna(0).gt(0).sum())
    pff_ol_match_count = int(pd.to_numeric(performance["ol_pff_history_available"], errors="coerce").fillna(0).gt(0).sum())

    logger.info("[PERF] Saved %s rows to %s", f"{len(performance):,}", OUTPUT_TABLE)
    logger.info("[PERF] Saved %s player-season rows to %s", f"{len(seasonal_output):,}", SEASONAL_OUTPUT_TABLE)
    logger.info("[PERF] Players with history: %s/%s (%.2f%%)", f"{history_count:,}", f"{len(performance):,}", 100.0 * history_count / max(len(performance), 1))
    logger.info("[PERF] Players with usable performance grade: %s", f"{usable_count:,}")
    logger.info("[PERF] Players with 2025 score: %s", f"{recent_count:,}")
    logger.info("[PERF] QB dedicated multi-year matches: %s", f"{qb_match_count:,}")
    logger.info("[PERF] Dedicated defensive matches: %s", f"{defense_match_count:,}")
    logger.info("[PERF] Current-roster OL players matched to PFF: %s", f"{pff_ol_match_count:,}")
    logger.info("[PERF] PFF OL player-season rows saved: %s", f"{len(pff_history):,}")
    expected_history_ids = set(history["player_id"].dropna().astype(str)) & set(master["player_id"].astype(str))
    retained_history_ids = set(
        performance.loc[
            pd.to_numeric(performance["history_available"], errors="coerce").fillna(0).gt(0),
            "player_id",
        ].astype(str)
    ) & expected_history_ids
    logger.info(
        "[PERF] History reconciliation: %s/%s current historical players retained",
        f"{len(retained_history_ids):,}",
        f"{len(expected_history_ids):,}",
    )

    print("\n[PERF] Position summary:")
    print(summary.to_string(index=False))

    display_columns = [
        "player_name", "team", "position", "position_group",
        "multi_year_performance_score", "recent_season_performance_score",
        "position_confidence_score", "performance_input_score", "qb_starter_pool",
    ]
    print("\n[PERF] Top 30 final player grades:")
    print(
        performance.sort_values("performance_input_score", ascending=False)
        .head(30)[display_columns].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PERF] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
