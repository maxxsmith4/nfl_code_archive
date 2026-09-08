#!/usr/bin/env python
"""
Build prior-only weekly NFL matchup and personnel ratings, validate a compact
market-residual model across four rolling seasons, and produce a guarded
implementation bundle.

Architecture
------------
    actual_home_margin = market_home_margin + predicted_market_residual

The market remains the anchor. The model is permitted to adjust it only when
prior-only matchup and personnel features demonstrate stable residual value.

Historical information policy
-----------------------------
For prediction week W:
- matchup ratings use current-season games through W-1 plus a low-weight
  prior-season sample;
- personnel ratings use snap participation only through W-1;
- quarterback identity and recent quarterback performance use only games
  completed through W-1;
- Week 1 is never graded.

QB identity policy
------------------
Snap-count QB names are resolved to nflverse GSIS passer identifiers through
an audited first-initial/surname crosswalk.  Ambiguous mappings are fatal, and
recent QB performance follows the player across team changes.  Model fitting
is blocked unless QB identity coverage, EPA/CPOE coverage, nonzero counts, and
feature variance all clear explicit integrity thresholds.

Evaluation policy
-----------------
- 2018-2019: initial development.
- 2020-2023: rolling validation, one season at a time.
- 2024-2025: fixed benchmark safety check only; never used to select the
  feature set, model alpha, rating alpha, or betting threshold.
- 2026: true forward evaluation.

A model is marked implementation-ready only when:
1. residual correlation is positive in at least three of four validation years;
2. average validation residual correlation is positive;
3. coefficient signs are stable across rolling folds;
4. the selected threshold has sufficient bets in every validation year;
5. at least three validation years are profitable and no year breaches the
   fixed ROI floor;
6. the locked 2024-2025 benchmark clears a fixed safety floor without any
   retuning.

Output database
---------------
    backtests/nfl_weekly_matchup_residual.sqlite

Deployment bundle
-----------------
    models/nfl_matchup_residual_model_v2.joblib
    models/nfl_matchup_residual_model_v2_metadata.json

No production or previously built historical database is modified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUILD_ID = "NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V3"
VERSION = "v3_qb_gsis_identity_crosswalk_dead_feature_guard"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)

PBP_SOURCE_SEASONS = tuple(range(2017, 2026))
SNAP_SOURCE_SEASONS = tuple(range(2018, 2026))
PREDICTION_SEASONS = tuple(range(2018, 2026))
ROLLING_VALIDATION_SEASONS = (2020, 2021, 2022, 2023)
BENCHMARK_SEASONS = (2024, 2025)

DEFAULT_RATING_ALPHAS = (10.0, 25.0, 50.0)
DEFAULT_MODEL_ALPHAS = (25.0, 100.0, 500.0)
DEFAULT_THRESHOLDS = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
)
DEFAULT_ATS_PRICE = -110.0
DEFAULT_PRIOR_SEASON_WEIGHT = 0.20
DEFAULT_CURRENT_DECAY = 0.92
DEFAULT_PRIOR_DECAY = 0.97
DEFAULT_MINIMUM_TOTAL_VALIDATION_BETS = 160
DEFAULT_MINIMUM_SEASON_VALIDATION_BETS = 30
DEFAULT_MINIMUM_PROFITABLE_VALIDATION_SEASONS = 3
DEFAULT_VALIDATION_ROI_FLOOR = -0.10
DEFAULT_BENCHMARK_COMBINED_ROI_FLOOR = -0.02
DEFAULT_BENCHMARK_SEASON_ROI_FLOOR = -0.08
DEFAULT_BENCHMARK_CORRELATION_FLOOR = -0.03
DEFAULT_MINIMUM_SIGN_STABILITY = 0.75
DEFAULT_MINIMUM_BENCHMARK_BETS = 100

CACHE_DATABASE_NAME = "nfl_weekly_matchup_source_cache_v2.sqlite"
OUTPUT_DATABASE_NAME = "nfl_weekly_matchup_residual.sqlite"
MODEL_FILENAME = "nfl_matchup_residual_model_v2.joblib"
METADATA_FILENAME = "nfl_matchup_residual_model_v2_metadata.json"

TEAM_GAME_TABLE = "nfl_matchup_team_game_features"
TEAM_WEEK_TABLE = "nfl_matchup_team_week_ratings"
PERSONNEL_WEEK_TABLE = "nfl_matchup_weekly_personnel_features"
GAME_MATRIX_TABLE = "nfl_matchup_game_matrix"
DIRECTION_AUDIT_TABLE = "nfl_matchup_feature_direction_audit"
CANDIDATE_VALIDATION_TABLE = "nfl_matchup_candidate_validation"
COEFFICIENT_STABILITY_TABLE = "nfl_matchup_coefficient_stability"
THRESHOLD_VALIDATION_TABLE = "nfl_matchup_threshold_validation"
BENCHMARK_PREDICTION_TABLE = "nfl_matchup_benchmark_predictions"
BENCHMARK_SUMMARY_TABLE = "nfl_matchup_benchmark_summary"
MODEL_COEFFICIENT_TABLE = "nfl_matchup_model_coefficients"
MODEL_COMPARISON_TABLE = "nfl_matchup_model_comparison"
QB_IDENTITY_AUDIT_TABLE = "nfl_matchup_qb_identity_crosswalk_audit"
QB_FEATURE_AUDIT_TABLE = "nfl_matchup_qb_feature_integrity_audit"
RUN_AUDIT_TABLE = "nfl_matchup_run_audit"

MINIMUM_QB_IDENTITY_MATCH_RATE = 0.99
MINIMUM_QB_RECENT_PERFORMANCE_COVERAGE = 0.99
MINIMUM_QB_ADVANTAGE_STANDARD_DEVIATION = 1e-6

TEAM_ALIASES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "GNB": "GB",
    "HST": "HOU",
    "JAC": "JAX",
    "KAN": "KC",
    "KCC": "KC",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "SDG": "LAC",
    "LVR": "LV",
    "OAK": "LV",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "WSH": "WAS",
    "WFT": "WAS",
}

POSITION_GROUPS = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR_TE",
    "TE": "WR_TE",
    "C": "OL",
    "G": "OL",
    "OG": "OL",
    "T": "OL",
    "OT": "OL",
    "LT": "OL",
    "RT": "OL",
    "LG": "OL",
    "RG": "OL",
    "OL": "OL",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "OLB": "LB_EDGE",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
}

PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "down",
    "qtr",
    "epa",
    "success",
    "cpoe",
    "yards_gained",
    "pass",
    "rush",
    "qb_dropback",
    "pass_attempt",
    "rush_attempt",
    "sack",
    "interception",
    "fumble_lost",
    "no_play",
    "qb_kneel",
    "qb_spike",
    "special_teams_play",
    "score_differential",
    "wp",
    "passer_player_id",
    "passer_player_name",
]

METRIC_SPECS = {
    "pass_epa": ("pass_epa_per_dropback", True),
    "rush_epa": ("rush_epa_per_attempt", True),
    "early_down_epa": ("early_down_epa_per_play", True),
    "success_rate": ("success_rate", True),
    "explosive_rate": ("explosive_rate", True),
    "sack_rate": ("sack_rate", False),
    "turnover_rate": ("turnover_rate", False),
    "cpoe": ("cpoe", True),
}

MARKET_CONTROL_FEATURES = (
    "market_home_margin",
    "absolute_market_home_margin",
)

FEATURE_SETS = {
    "MARKET_CONTROL": MARKET_CONTROL_FEATURES,
    "PASS_COMPACT": (
        "pass_epa_advantage",
        "sack_advantage",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
        "qb_week1_stability_advantage",
        "qb_snap_share_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "RUSH_TRENCH": (
        "rush_epa_advantage",
        "early_down_epa_advantage",
        "ol_continuity_advantage",
        "ol_stability_advantage",
        "core_ol_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "PERSONNEL_QB": (
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
        "qb_week1_stability_advantage",
        "qb_last_game_stability_advantage",
        "qb_snap_share_advantage",
        "offense_continuity_advantage",
        "ol_continuity_advantage",
        "core_offense_health_advantage",
        "core_ol_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "FULL_COMPACT": (
        "pass_epa_advantage",
        "rush_epa_advantage",
        "sack_advantage",
        "turnover_advantage",
        "explosive_rate_advantage",
        "special_teams_advantage",
        "qb_recent_epa_advantage",
        "qb_week1_stability_advantage",
        "ol_continuity_advantage",
        "core_offense_health_advantage",
        "core_defense_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
}

DIRECTIONAL_FEATURES = tuple(
    sorted(
        {
            feature
            for feature_set, features in FEATURE_SETS.items()
            if feature_set != "MARKET_CONTROL"
            for feature in features
            if feature not in MARKET_CONTROL_FEATURES
        }
    )
)


# =============================================================================
# ARGUMENTS AND GENERIC HELPERS
# =============================================================================


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {
                float(piece.strip())
                for piece in value.split(",")
                if piece.strip()
            }
        )
    )
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--pbp-path", type=Path, default=None)
    parser.add_argument("--schedule-path", type=Path, default=None)
    parser.add_argument("--snap-counts-path", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--rating-alphas",
        type=parse_float_list,
        default=DEFAULT_RATING_ALPHAS,
    )
    parser.add_argument(
        "--model-alphas",
        type=parse_float_list,
        default=DEFAULT_MODEL_ALPHAS,
    )
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--ats-price", type=float, default=DEFAULT_ATS_PRICE)
    parser.add_argument(
        "--prior-season-weight",
        type=float,
        default=DEFAULT_PRIOR_SEASON_WEIGHT,
    )
    parser.add_argument(
        "--current-decay",
        type=float,
        default=DEFAULT_CURRENT_DECAY,
    )
    parser.add_argument(
        "--prior-decay",
        type=float,
        default=DEFAULT_PRIOR_DECAY,
    )
    parser.add_argument(
        "--minimum-total-validation-bets",
        type=int,
        default=DEFAULT_MINIMUM_TOTAL_VALIDATION_BETS,
    )
    parser.add_argument(
        "--minimum-season-validation-bets",
        type=int,
        default=DEFAULT_MINIMUM_SEASON_VALIDATION_BETS,
    )
    parser.add_argument(
        "--minimum-profitable-validation-seasons",
        type=int,
        default=DEFAULT_MINIMUM_PROFITABLE_VALIDATION_SEASONS,
    )
    parser.add_argument(
        "--validation-roi-floor",
        type=float,
        default=DEFAULT_VALIDATION_ROI_FLOOR,
    )
    parser.add_argument(
        "--benchmark-combined-roi-floor",
        type=float,
        default=DEFAULT_BENCHMARK_COMBINED_ROI_FLOOR,
    )
    parser.add_argument(
        "--benchmark-season-roi-floor",
        type=float,
        default=DEFAULT_BENCHMARK_SEASON_ROI_FLOOR,
    )
    parser.add_argument(
        "--benchmark-correlation-floor",
        type=float,
        default=DEFAULT_BENCHMARK_CORRELATION_FLOOR,
    )
    parser.add_argument(
        "--minimum-sign-stability",
        type=float,
        default=DEFAULT_MINIMUM_SIGN_STABILITY,
    )
    parser.add_argument(
        "--minimum-benchmark-bets",
        type=int,
        default=DEFAULT_MINIMUM_BENCHMARK_BETS,
    )
    parser.add_argument(
        "--minimum-rolling-train-rows",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--minimum-rolling-validation-rows",
        type=int,
        default=200,
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    for path_argument in ("pbp_path", "schedule_path", "snap_counts_path"):
        value = getattr(args, path_argument)
        if value is not None:
            setattr(args, path_argument, value.resolve())

    if not 0 < args.prior_season_weight <= 1:
        parser.error("--prior-season-weight must be in (0, 1].")
    if not 0 < args.current_decay <= 1:
        parser.error("--current-decay must be in (0, 1].")
    if not 0 < args.prior_decay <= 1:
        parser.error("--prior-decay must be in (0, 1].")
    if args.minimum_total_validation_bets < 16:
        parser.error("--minimum-total-validation-bets must be at least 16.")
    if args.minimum_season_validation_bets < 4:
        parser.error("--minimum-season-validation-bets must be at least 4.")
    if not 1 <= args.minimum_profitable_validation_seasons <= 4:
        parser.error("--minimum-profitable-validation-seasons must be 1 through 4.")
    if not 0.50 <= args.minimum_sign_stability <= 1.0:
        parser.error("--minimum-sign-stability must be between 0.50 and 1.00.")
    if args.minimum_benchmark_bets < 20:
        parser.error("--minimum-benchmark-bets must be at least 20.")
    if args.minimum_rolling_train_rows < 50:
        parser.error("--minimum-rolling-train-rows must be at least 50.")
    if args.minimum_rolling_validation_rows < 40:
        parser.error("--minimum-rolling-validation-rows must be at least 40.")

    return args


def now_string() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def normalize_team(value: Any) -> str:
    text = "" if value is None else str(value).upper().strip()
    return TEAM_ALIASES.get(text, text)


def normalize_position(value: Any) -> str:
    text = "" if value is None else re.sub(r"[^A-Z]", "", str(value).upper())
    return text or "OTHER"


def position_group(value: Any) -> str:
    return POSITION_GROUPS.get(normalize_position(value), "OTHER")


def normalize_name(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def qb_name_match_key(value: Any) -> str:
    """Return a stable first-initial plus surname QB identity key.

    nflverse PBP commonly supplies abbreviated names such as ``M.Ryan`` while
    snap counts supply ``Matt Ryan``.  The prior implementation compared a
    GSIS passer id to a normalized full name, which made every recent-QB
    performance lookup miss.  This key is used only to resolve the GSIS id;
    all performance joins use that resolved id.
    """
    text = "" if value is None else str(value).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    pieces = re.findall(r"[a-z0-9]+", text)
    if not pieces:
        return ""
    if len(pieces) == 1:
        return pieces[0]
    return pieces[0][0] + pieces[-1]


def first_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {str(column).lower().strip(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def numeric(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def frame_from_any(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def safe_correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3:
        return np.nan
    if pair["left"].std(ddof=0) <= 0 or pair["right"].std(ddof=0) <= 0:
        return np.nan
    return float(pair["left"].corr(pair["right"]))


# =============================================================================
# SOURCE LOADING AND CACHE
# =============================================================================


def load_package_pbp(seasons: list[int]) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        import nflreadpy  # type: ignore

        for function_name in ("load_pbp", "load_pbp_data"):
            loader = getattr(nflreadpy, function_name, None)
            if loader is None:
                continue
            for attempt in (
                lambda l=loader: l(seasons),
                lambda l=loader: l(seasons=seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, f"nflreadpy.{function_name}"
                except Exception as exc:
                    errors.append(f"nflreadpy.{function_name}: {exc}")
    except Exception as exc:
        errors.append(f"nflreadpy import: {exc}")

    try:
        import nfl_data_py  # type: ignore

        loader = getattr(nfl_data_py, "import_pbp_data", None)
        if loader is not None:
            for attempt in (
                lambda: loader(
                    seasons,
                    columns=PBP_COLUMNS,
                    downcast=True,
                    cache=False,
                ),
                lambda: loader(seasons, columns=PBP_COLUMNS, downcast=True),
                lambda: loader(seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, "nfl_data_py.import_pbp_data"
                except Exception as exc:
                    errors.append(f"nfl_data_py.import_pbp_data: {exc}")
    except Exception as exc:
        errors.append(f"nfl_data_py import: {exc}")

    raise RuntimeError("Unable to load play-by-play: " + " | ".join(errors))


def load_package_schedule(seasons: list[int]) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        import nflreadpy  # type: ignore

        loader = getattr(nflreadpy, "load_schedules", None)
        if loader is not None:
            for attempt in (
                lambda: loader(seasons),
                lambda: loader(seasons=seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, "nflreadpy.load_schedules"
                except Exception as exc:
                    errors.append(f"nflreadpy.load_schedules: {exc}")
    except Exception as exc:
        errors.append(f"nflreadpy import: {exc}")

    try:
        import nfl_data_py  # type: ignore

        loader = getattr(nfl_data_py, "import_schedules", None)
        if loader is not None:
            for attempt in (
                lambda: loader(seasons),
                lambda: loader(seasons=seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, "nfl_data_py.import_schedules"
                except Exception as exc:
                    errors.append(f"nfl_data_py.import_schedules: {exc}")
    except Exception as exc:
        errors.append(f"nfl_data_py import: {exc}")

    raise RuntimeError("Unable to load schedules: " + " | ".join(errors))


def load_package_snaps(seasons: list[int]) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        import nflreadpy  # type: ignore

        loader = getattr(nflreadpy, "load_snap_counts", None)
        if loader is not None:
            for attempt in (
                lambda: loader(seasons),
                lambda: loader(seasons=seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, "nflreadpy.load_snap_counts"
                except Exception as exc:
                    errors.append(f"nflreadpy.load_snap_counts: {exc}")
    except Exception as exc:
        errors.append(f"nflreadpy import: {exc}")

    try:
        import nfl_data_py  # type: ignore

        for function_name in ("import_snap_counts", "import_snap_count_data"):
            loader = getattr(nfl_data_py, function_name, None)
            if loader is None:
                continue
            for attempt in (
                lambda l=loader: l(seasons),
                lambda l=loader: l(seasons=seasons),
            ):
                try:
                    result = frame_from_any(attempt())
                    if not result.empty:
                        return result, f"nfl_data_py.{function_name}"
                except Exception as exc:
                    errors.append(f"nfl_data_py.{function_name}: {exc}")
    except Exception as exc:
        errors.append(f"nfl_data_py import: {exc}")

    raise RuntimeError("Unable to load snap counts: " + " | ".join(errors))


def standardize_pbp(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    aliases = {
        "passer_player_id": (
            "passer_player_id",
            "passer_id",
        ),
        "passer_player_name": (
            "passer_player_name",
            "passer_name",
            "passer",
        ),
    }

    output = pd.DataFrame(index=frame.index)
    for column in PBP_COLUMNS:
        if column in aliases:
            source = first_existing(frame.columns, aliases[column])
            output[column] = frame[source] if source is not None else ""
        else:
            output[column] = frame[column] if column in frame.columns else np.nan

    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output["week"] = pd.to_numeric(output["week"], errors="coerce")
    for column in ("posteam", "defteam", "home_team", "away_team"):
        output[column] = output[column].map(normalize_team)

    numeric_columns = [
        "down",
        "qtr",
        "epa",
        "success",
        "cpoe",
        "yards_gained",
        "pass",
        "rush",
        "qb_dropback",
        "pass_attempt",
        "rush_attempt",
        "sack",
        "interception",
        "fumble_lost",
        "no_play",
        "qb_kneel",
        "qb_spike",
        "special_teams_play",
        "score_differential",
        "wp",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output["passer_player_id"] = output["passer_player_id"].fillna("").astype(str)
    output["passer_player_name"] = (
        output["passer_player_name"].fillna("").astype(str).str.strip()
    )
    output["passer_key"] = np.where(
        output["passer_player_id"].ne(""),
        output["passer_player_id"],
        output["passer_player_name"].map(normalize_name),
    )

    output = output[
        output["season"].isin(PBP_SOURCE_SEASONS)
        & output["week"].between(1, 18)
        & output["game_id"].notna()
    ].copy()
    output[["season", "week"]] = output[["season", "week"]].astype(int)

    for flag in (
        "pass",
        "rush",
        "qb_dropback",
        "pass_attempt",
        "rush_attempt",
        "sack",
        "interception",
        "fumble_lost",
        "no_play",
        "qb_kneel",
        "qb_spike",
        "special_teams_play",
    ):
        output[flag] = output[flag].fillna(0.0)

    return output.reset_index(drop=True)


def standardize_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    aliases = {
        "season": ("season",),
        "week": ("week",),
        "game_id": ("game_id",),
        "game_type": ("game_type", "season_type"),
        "home_team": ("home_team",),
        "away_team": ("away_team",),
        "home_score": ("home_score",),
        "away_score": ("away_score",),
        "spread_line": ("spread_line", "market_home_margin"),
    }

    output = pd.DataFrame(index=frame.index)
    for target, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        output[target] = frame[source] if source is not None else np.nan

    for column in ("season", "week", "home_score", "away_score", "spread_line"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["home_team"] = output["home_team"].map(normalize_team)
    output["away_team"] = output["away_team"].map(normalize_team)

    regular = (
        output["game_type"]
        .fillna("REG")
        .astype(str)
        .str.upper()
        .isin({"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"})
    )

    output = output[
        regular
        & output["season"].isin(PREDICTION_SEASONS)
        & output["week"].between(1, 18)
        & output["game_id"].notna()
        & output["home_score"].notna()
        & output["away_score"].notna()
        & output["spread_line"].notna()
    ].copy()

    output[["season", "week"]] = output[["season", "week"]].astype(int)
    output["game_id"] = output["game_id"].astype(str)
    output["actual_home_margin"] = output["home_score"] - output["away_score"]
    output["actual_market_residual"] = (
        output["actual_home_margin"] - output["spread_line"]
    )

    return (
        output.sort_values(["season", "week", "game_id"])
        .drop_duplicates("game_id", keep="last")
        .reset_index(drop=True)
    )


def standardize_snaps(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    aliases = {
        "season": ("season",),
        "week": ("week", "game_week"),
        "team": ("team", "recent_team"),
        "player_name": ("player_name", "player", "player_display_name"),
        "position": ("position", "pos"),
        "offense_snaps": ("offense_snaps", "off_snaps"),
        "defense_snaps": ("defense_snaps", "def_snaps"),
        "st_snaps": ("st_snaps", "special_teams_snaps"),
        "game_type": ("game_type", "season_type"),
    }

    output = pd.DataFrame(index=frame.index)
    for target, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        output[target] = frame[source] if source is not None else np.nan

    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output["week"] = pd.to_numeric(output["week"], errors="coerce")
    output["team"] = output["team"].map(normalize_team)
    output["player_name"] = output["player_name"].fillna("").astype(str).str.strip()
    output["position"] = output["position"].map(normalize_position)
    output["position_group"] = output["position"].map(position_group)

    for column in ("offense_snaps", "defense_snaps", "st_snaps"):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    if output["game_type"].notna().any():
        regular = (
            output["game_type"]
            .fillna("REG")
            .astype(str)
            .str.upper()
            .isin({"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"})
        )
        output = output[regular].copy()

    output = output[
        output["season"].isin(SNAP_SOURCE_SEASONS)
        & output["week"].between(1, 18)
        & output["team"].ne("")
        & output["player_name"].ne("")
    ].copy()
    output[["season", "week"]] = output[["season", "week"]].astype(int)
    output["player_key"] = (
        output["player_name"].map(normalize_name) + "|" + output["position"].astype(str)
    )

    return (
        output.groupby(
            [
                "season",
                "week",
                "team",
                "player_key",
                "player_name",
                "position",
                "position_group",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            offense_snaps=("offense_snaps", "sum"),
            defense_snaps=("defense_snaps", "sum"),
            st_snaps=("st_snaps", "sum"),
        )
        .sort_values(["season", "week", "team", "player_key"])
        .reset_index(drop=True)
    )


def load_or_build_cache(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, Path]:
    cache_path = args.project_root / "backtests" / CACHE_DATABASE_NAME
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    use_cache = (
        cache_path.exists()
        and not args.rebuild_cache
        and args.pbp_path is None
        and args.schedule_path is None
        and args.snap_counts_path is None
    )
    if use_cache:
        with sqlite3.connect(cache_path) as connection:
            pbp = read_table(connection, "pbp_source")
            schedule = read_table(connection, "schedule_source")
            snaps = read_table(connection, "snap_source")
            audit = read_table(connection, "cache_audit").iloc[0]
        return pbp, schedule, snaps, audit, cache_path

    if args.pbp_path is not None:
        raw_pbp = pd.read_csv(args.pbp_path, low_memory=False)
        pbp_source = str(args.pbp_path)
    else:
        raw_pbp, pbp_source = load_package_pbp(list(PBP_SOURCE_SEASONS))

    if args.schedule_path is not None:
        raw_schedule = pd.read_csv(args.schedule_path, low_memory=False)
        schedule_source = str(args.schedule_path)
    else:
        raw_schedule, schedule_source = load_package_schedule(
            list(PREDICTION_SEASONS)
        )

    if args.snap_counts_path is not None:
        raw_snaps = pd.read_csv(args.snap_counts_path, low_memory=False)
        snap_source = str(args.snap_counts_path)
    else:
        raw_snaps, snap_source = load_package_snaps(list(SNAP_SOURCE_SEASONS))

    pbp = standardize_pbp(raw_pbp)
    schedule = standardize_schedule(raw_schedule)
    snaps = standardize_snaps(raw_snaps)

    if cache_path.exists():
        cache_path.unlink()

    audit_frame = pd.DataFrame(
        [
            {
                "build_id": BUILD_ID,
                "version": VERSION,
                "pbp_source": pbp_source,
                "schedule_source": schedule_source,
                "snap_source": snap_source,
                "pbp_rows": len(pbp),
                "schedule_rows": len(schedule),
                "snap_rows": len(snaps),
                "created_at": now_string(),
            }
        ]
    )

    with sqlite3.connect(cache_path) as connection:
        pbp.to_sql("pbp_source", connection, if_exists="replace", index=False)
        schedule.to_sql("schedule_source", connection, if_exists="replace", index=False)
        snaps.to_sql("snap_source", connection, if_exists="replace", index=False)
        audit_frame.to_sql("cache_audit", connection, if_exists="replace", index=False)

    return pbp, schedule, snaps, audit_frame.iloc[0], cache_path


# =============================================================================
# TEAM-GAME PROCESS FEATURES
# =============================================================================


def safe_mean(values: pd.Series) -> float:
    usable = pd.to_numeric(values, errors="coerce").dropna()
    return float(usable.mean()) if not usable.empty else np.nan


def build_team_game_features(pbp: pd.DataFrame) -> pd.DataFrame:
    valid_scrimmage = (
        pbp["posteam"].ne("")
        & pbp["defteam"].ne("")
        & pbp["no_play"].eq(0)
        & pbp["qb_kneel"].eq(0)
        & pbp["qb_spike"].eq(0)
        & pbp["special_teams_play"].eq(0)
        & (pbp["pass"].eq(1) | pbp["rush"].eq(1) | pbp["qb_dropback"].eq(1))
    )
    plays = pbp[valid_scrimmage].copy()
    plays["is_pass_play"] = (
        plays["qb_dropback"].eq(1) | plays["pass"].eq(1)
    ).astype(int)
    plays["is_rush_play"] = (
        plays["rush"].eq(1) & plays["qb_dropback"].ne(1)
    ).astype(int)
    plays["is_early_down"] = plays["down"].isin([1, 2]).astype(int)
    plays["is_explosive"] = (
        (
            plays["is_pass_play"].eq(1)
            & plays["yards_gained"].ge(15)
        )
        | (
            plays["is_rush_play"].eq(1)
            & plays["yards_gained"].ge(10)
        )
    ).astype(int)
    plays["is_turnover"] = (
        plays["interception"].eq(1) | plays["fumble_lost"].eq(1)
    ).astype(int)

    rows: list[dict[str, Any]] = []
    group_columns = ["season", "week", "game_id", "posteam", "defteam"]

    for keys, group in plays.groupby(group_columns, sort=False, dropna=False):
        season, week, game_id, offense, defense = keys
        pass_plays = group[group["is_pass_play"].eq(1)].copy()
        rush_plays = group[group["is_rush_play"].eq(1)].copy()
        early_down = group[group["is_early_down"].eq(1)].copy()

        primary_qb_key = ""
        primary_qb_name = ""
        primary_qb_share = np.nan
        primary_qb_epa = np.nan
        primary_qb_cpoe = np.nan
        if not pass_plays.empty:
            pass_plays["resolved_qb_key"] = np.where(
                pass_plays["passer_key"].ne(""),
                pass_plays["passer_key"],
                "unknown",
            )
            qb_summary = (
                pass_plays.groupby("resolved_qb_key", as_index=False)
                .agg(
                    dropbacks=("is_pass_play", "sum"),
                    qb_epa=("epa", "mean"),
                    qb_cpoe=("cpoe", "mean"),
                )
                .sort_values(["dropbacks", "resolved_qb_key"], ascending=[False, True])
            )
            primary = qb_summary.iloc[0]
            primary_qb_key = str(primary["resolved_qb_key"])
            primary_qb_share = float(primary["dropbacks"]) / float(len(pass_plays))
            primary_qb_epa = float(primary["qb_epa"])
            primary_qb_cpoe = (
                float(primary["qb_cpoe"])
                if pd.notna(primary["qb_cpoe"])
                else np.nan
            )
            names = pass_plays[
                pass_plays["resolved_qb_key"].eq(primary_qb_key)
            ]["passer_player_name"]
            nonempty_names = names[names.astype(str).str.strip().ne("")]
            primary_qb_name = (
                str(nonempty_names.iloc[0]) if not nonempty_names.empty else primary_qb_key
            )

        rows.append(
            {
                "season": int(season),
                "week": int(week),
                "game_id": str(game_id),
                "offense_team": str(offense),
                "defense_team": str(defense),
                "plays": int(len(group)),
                "dropbacks": int(len(pass_plays)),
                "rush_attempts": int(len(rush_plays)),
                "pass_epa_per_dropback": safe_mean(pass_plays["epa"]),
                "rush_epa_per_attempt": safe_mean(rush_plays["epa"]),
                "early_down_epa_per_play": safe_mean(early_down["epa"]),
                "success_rate": safe_mean(group["success"]),
                "explosive_rate": safe_mean(group["is_explosive"]),
                "sack_rate": safe_mean(pass_plays["sack"]),
                "turnover_rate": safe_mean(group["is_turnover"]),
                "cpoe": safe_mean(pass_plays["cpoe"]),
                "primary_qb_key": primary_qb_key,
                "primary_qb_name": primary_qb_name,
                "primary_qb_share": primary_qb_share,
                "primary_qb_epa": primary_qb_epa,
                "primary_qb_cpoe": primary_qb_cpoe,
            }
        )

    output = pd.DataFrame(rows)

    special = pbp[
        pbp["special_teams_play"].eq(1)
        & pbp["posteam"].ne("")
        & pbp["epa"].notna()
    ].copy()
    if not special.empty:
        special_summary = (
            special.groupby(["season", "week", "game_id", "posteam"], as_index=False)
            .agg(special_teams_epa=("epa", "sum"))
            .rename(columns={"posteam": "offense_team"})
        )
        output = output.merge(
            special_summary,
            on=["season", "week", "game_id", "offense_team"],
            how="left",
            validate="one_to_one",
        )
    else:
        output["special_teams_epa"] = 0.0

    output["special_teams_epa"] = output["special_teams_epa"].fillna(0.0)
    if output.empty:
        raise RuntimeError("Team-game feature build produced zero rows.")

    return output.sort_values(
        ["season", "week", "game_id", "offense_team"]
    ).reset_index(drop=True)


# =============================================================================
# OPPONENT-ADJUSTED WEEKLY MATCHUP RATINGS
# =============================================================================


def observation_weights(
    history: pd.DataFrame,
    target_season: int,
    target_week: int,
    args: argparse.Namespace,
) -> np.ndarray:
    weights = np.zeros(len(history), dtype=float)

    current_mask = history["season"].eq(target_season).to_numpy()
    if current_mask.any():
        age = target_week - history.loc[current_mask, "week"].to_numpy(float) - 1.0
        weights[current_mask] = np.power(args.current_decay, np.maximum(age, 0.0))

    prior_mask = history["season"].eq(target_season - 1).to_numpy()
    if prior_mask.any():
        age = 18.0 - history.loc[prior_mask, "week"].to_numpy(float)
        weights[prior_mask] = args.prior_season_weight * np.power(
            args.prior_decay,
            np.maximum(age, 0.0),
        )

    return weights


def fit_metric_model(
    history: pd.DataFrame,
    metric_column: str,
    teams: list[str],
    rating_alpha: float,
    sample_weights: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    usable = history[["offense_team", "defense_team", metric_column]].copy()
    usable["sample_weight"] = sample_weights
    usable[metric_column] = pd.to_numeric(usable[metric_column], errors="coerce")
    usable = usable[
        usable[metric_column].notna() & usable["sample_weight"].gt(0)
    ].copy()

    if len(usable) < 20:
        return (
            0.0,
            {team: 0.0 for team in teams},
            {team: 0.0 for team in teams},
        )

    team_index = {team: index for index, team in enumerate(teams)}
    matrix = np.zeros((len(usable), 2 * len(teams)), dtype=float)
    for row_index, row in enumerate(usable.itertuples(index=False)):
        offense = str(row.offense_team)
        defense = str(row.defense_team)
        if offense in team_index:
            matrix[row_index, team_index[offense]] = 1.0
        if defense in team_index:
            matrix[row_index, len(teams) + team_index[defense]] = 1.0

    target = usable[metric_column].to_numpy(float)
    weights = usable["sample_weight"].to_numpy(float)
    league_mean = float(np.average(target, weights=weights))

    model = Ridge(alpha=float(rating_alpha), fit_intercept=False)
    model.fit(matrix, target - league_mean, sample_weight=weights)

    offense_effect = {
        team: float(model.coef_[team_index[team]]) for team in teams
    }
    defense_allowance = {
        team: float(model.coef_[len(teams) + team_index[team]]) for team in teams
    }

    offense_center = float(np.mean(list(offense_effect.values())))
    defense_center = float(np.mean(list(defense_allowance.values())))
    offense_effect = {
        team: value - offense_center for team, value in offense_effect.items()
    }
    defense_allowance = {
        team: value - defense_center for team, value in defense_allowance.items()
    }

    return league_mean, offense_effect, defense_allowance


def weighted_special_teams_ratings(
    history: pd.DataFrame,
    teams: list[str],
    sample_weights: np.ndarray,
) -> dict[str, float]:
    usable = history[["offense_team", "special_teams_epa"]].copy()
    usable["sample_weight"] = sample_weights
    usable["special_teams_epa"] = pd.to_numeric(
        usable["special_teams_epa"], errors="coerce"
    )
    usable = usable[
        usable["sample_weight"].gt(0) & usable["special_teams_epa"].notna()
    ]

    ratings: dict[str, float] = {}
    for team in teams:
        sample = usable[usable["offense_team"].eq(team)]
        ratings[team] = (
            float(
                np.average(
                    sample["special_teams_epa"],
                    weights=sample["sample_weight"],
                )
            )
            if not sample.empty
            else 0.0
        )

    center = float(np.mean(list(ratings.values())))
    return {team: value - center for team, value in ratings.items()}


def build_team_week_ratings(
    team_games: pd.DataFrame,
    schedule: pd.DataFrame,
    rating_alpha: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for season in PREDICTION_SEASONS:
        teams = sorted(
            set(schedule.loc[schedule["season"].eq(season), "home_team"])
            | set(schedule.loc[schedule["season"].eq(season), "away_team"])
        )
        for prediction_week in range(2, 19):
            history = team_games[
                (
                    team_games["season"].eq(season)
                    & team_games["week"].lt(prediction_week)
                )
                | team_games["season"].eq(season - 1)
            ].copy()
            sample_weights = observation_weights(
                history,
                season,
                prediction_week,
                args,
            )
            current_games = (
                history[history["season"].eq(season)]
                .groupby("offense_team")["game_id"]
                .nunique()
                .to_dict()
            )

            models = {
                metric_name: fit_metric_model(
                    history,
                    metric_column,
                    teams,
                    rating_alpha,
                    sample_weights,
                )
                for metric_name, (metric_column, _) in METRIC_SPECS.items()
            }
            special_teams = weighted_special_teams_ratings(
                history,
                teams,
                sample_weights,
            )

            for team in teams:
                row: dict[str, Any] = {
                    "season": season,
                    "prediction_week": prediction_week,
                    "team": team,
                    "rating_alpha": float(rating_alpha),
                    "current_games": int(current_games.get(team, 0)),
                    "special_teams_rating": special_teams.get(team, 0.0),
                }
                for metric_name, (league_mean, offense_effect, defense_allowance) in (
                    models.items()
                ):
                    row[f"{metric_name}_league_mean"] = league_mean
                    row[f"{metric_name}_offense_rating"] = offense_effect.get(
                        team, 0.0
                    )
                    row[f"{metric_name}_defense_allow_rating"] = (
                        defense_allowance.get(team, 0.0)
                    )
                rows.append(row)

    ratings = pd.DataFrame(rows)
    if ratings.empty:
        raise RuntimeError("Weekly matchup rating build produced zero rows.")
    return ratings.sort_values(
        ["season", "prediction_week", "team", "rating_alpha"]
    ).reset_index(drop=True)


# =============================================================================
# WEEKLY PERSONNEL AND QUARTERBACK FEATURES
# =============================================================================


def build_qb_identity_crosswalk(
    team_games: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    required = {"primary_qb_key", "primary_qb_name", "offense_team"}
    missing = sorted(required - set(team_games.columns))
    if missing:
        raise RuntimeError(
            f"Team-game data cannot build the QB identity crosswalk: {missing}"
        )

    source = team_games[
        ["primary_qb_key", "primary_qb_name", "offense_team"]
    ].copy()
    source["primary_qb_key"] = (
        source["primary_qb_key"].fillna("").astype(str).str.strip()
    )
    source["primary_qb_name"] = (
        source["primary_qb_name"].fillna("").astype(str).str.strip()
    )
    source["qb_name_match_key"] = source["primary_qb_name"].map(
        qb_name_match_key
    )
    source = source[
        source["primary_qb_key"].ne("")
        & source["qb_name_match_key"].ne("")
    ].copy()
    if source.empty:
        raise RuntimeError("QB identity crosswalk source is empty.")
    invalid_ids = source[
        ~source["primary_qb_key"].str.match(r"^00-\d+$", na=False)
    ]
    if not invalid_ids.empty:
        raise RuntimeError(
            "QB identity crosswalk contains non-GSIS passer identifiers:\n"
            + invalid_ids[
                ["primary_qb_key", "primary_qb_name", "offense_team"]
            ].drop_duplicates().head(25).to_string(index=False)
        )

    ambiguity = (
        source.groupby("qb_name_match_key")["primary_qb_key"]
        .nunique()
        .loc[lambda values: values.gt(1)]
    )
    if not ambiguity.empty:
        conflicts = source[
            source["qb_name_match_key"].isin(ambiguity.index)
        ].drop_duplicates(
            ["qb_name_match_key", "primary_qb_key", "primary_qb_name"]
        )
        raise RuntimeError(
            "Ambiguous QB name-to-GSIS mappings detected; refusing to guess:\n"
            + conflicts.sort_values(
                ["qb_name_match_key", "primary_qb_key"]
            ).to_string(index=False)
        )

    audit = (
        source.groupby(
            ["qb_name_match_key", "primary_qb_key"], as_index=False
        )
        .agg(
            pbp_name_variants=(
                "primary_qb_name",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            teams_observed=(
                "offense_team",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            team_game_rows=("primary_qb_name", "size"),
        )
        .sort_values(["qb_name_match_key", "primary_qb_key"])
        .reset_index(drop=True)
    )
    crosswalk = dict(
        zip(audit["qb_name_match_key"], audit["primary_qb_key"])
    )
    return crosswalk, audit


def share_dictionary(snapshot: pd.DataFrame, snap_column: str) -> dict[str, float]:
    if snapshot.empty:
        return {}
    maximum = float(pd.to_numeric(snapshot[snap_column], errors="coerce").max())
    if not np.isfinite(maximum) or maximum <= 0:
        return {}
    shares = (
        snapshot.assign(
            share=pd.to_numeric(snapshot[snap_column], errors="coerce").fillna(0.0)
            / maximum
        )
        .groupby("player_key")["share"]
        .max()
    )
    return {
        str(player_key): float(share)
        for player_key, share in shares.items()
        if share > 0
    }


def weighted_overlap(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    if not left or not right:
        return np.nan
    keys = set(left) | set(right)
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator > 0 else np.nan


def average_share_dictionary(
    dictionaries: list[dict[str, float]],
) -> dict[str, float]:
    usable = [dictionary for dictionary in dictionaries if dictionary]
    if not usable:
        return {}
    keys = set().union(*[set(dictionary) for dictionary in usable])
    return {
        key: float(np.mean([dictionary.get(key, 0.0) for dictionary in usable]))
        for key in keys
    }


def missing_core_share(
    baseline: dict[str, float],
    current: dict[str, float],
    core_threshold: float = 0.50,
    absent_threshold: float = 0.05,
) -> float:
    core = {key: share for key, share in baseline.items() if share >= core_threshold}
    denominator = sum(core.values())
    if denominator <= 0:
        return np.nan
    missing = sum(
        share
        for key, share in core.items()
        if current.get(key, 0.0) < absent_threshold
    )
    return missing / denominator


def build_snapshots(
    snaps: pd.DataFrame,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    snapshots: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for (season, week, team), group in snaps.groupby(
        ["season", "week", "team"], sort=True
    ):
        offense = group[group["offense_snaps"].gt(0)].copy()
        defense = group[group["defense_snaps"].gt(0)].copy()
        offensive_line = offense[offense["position_group"].eq("OL")].copy()
        quarterbacks = offense[offense["position_group"].eq("QB")].copy()

        qb_name = ""
        qb_key = ""
        qb_snap_share = np.nan
        if not quarterbacks.empty:
            quarterback = quarterbacks.sort_values(
                ["offense_snaps", "player_name"], ascending=[False, True]
            ).iloc[0]
            qb_name = str(quarterback["player_name"])
            qb_key = normalize_name(qb_name)
            offense_maximum = (
                float(offense["offense_snaps"].max()) if not offense.empty else np.nan
            )
            qb_snap_share = (
                float(quarterback["offense_snaps"]) / offense_maximum
                if np.isfinite(offense_maximum) and offense_maximum > 0
                else np.nan
            )

        record = {
            "season": int(season),
            "week": int(week),
            "team": str(team),
            "offense": share_dictionary(offense, "offense_snaps"),
            "defense": share_dictionary(defense, "defense_snaps"),
            "ol": share_dictionary(offensive_line, "offense_snaps"),
            "qb_name": qb_name,
            "qb_key": qb_key,
            "qb_snap_share": qb_snap_share,
        }
        snapshots.setdefault((int(season), str(team)), []).append(record)

    for key in snapshots:
        snapshots[key] = sorted(snapshots[key], key=lambda row: row["week"])
    return snapshots


def weighted_qb_performance(
    team_games: pd.DataFrame,
    target_season: int,
    prediction_week: int,
    qb_pbp_key: str,
    args: argparse.Namespace,
) -> tuple[float, float, int]:
    if not qb_pbp_key:
        return np.nan, np.nan, 0

    sample = team_games[
        team_games["primary_qb_key"].eq(qb_pbp_key)
        & (
            (
                team_games["season"].eq(target_season)
                & team_games["week"].lt(prediction_week)
            )
            | team_games["season"].eq(target_season - 1)
        )
    ].copy()
    if sample.empty:
        return np.nan, np.nan, 0

    weights = observation_weights(sample, target_season, prediction_week, args)
    usable = weights > 0
    if not usable.any():
        return np.nan, np.nan, 0

    sample = sample.loc[usable].copy()
    weights = weights[usable]

    epa = pd.to_numeric(sample["primary_qb_epa"], errors="coerce")
    cpoe = pd.to_numeric(sample["primary_qb_cpoe"], errors="coerce")

    epa_valid = epa.notna().to_numpy()
    cpoe_valid = cpoe.notna().to_numpy()
    weighted_epa = (
        float(np.average(epa[epa_valid], weights=weights[epa_valid]))
        if epa_valid.any()
        else np.nan
    )
    weighted_cpoe = (
        float(np.average(cpoe[cpoe_valid], weights=weights[cpoe_valid]))
        if cpoe_valid.any()
        else np.nan
    )
    return weighted_epa, weighted_cpoe, int(len(sample))


def build_weekly_personnel_features(
    snaps: pd.DataFrame,
    team_games: pd.DataFrame,
    schedule: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = build_snapshots(snaps)
    qb_crosswalk, qb_crosswalk_audit = build_qb_identity_crosswalk(team_games)
    rows: list[dict[str, Any]] = []

    teams_by_season = {
        season: sorted(
            set(schedule.loc[schedule["season"].eq(season), "home_team"])
            | set(schedule.loc[schedule["season"].eq(season), "away_team"])
        )
        for season in PREDICTION_SEASONS
    }

    for season in PREDICTION_SEASONS:
        for team in teams_by_season[season]:
            team_snapshots = snapshots.get((season, team), [])
            if not team_snapshots:
                continue
            week_one = next(
                (snapshot for snapshot in team_snapshots if snapshot["week"] == 1),
                team_snapshots[0],
            )

            for prediction_week in range(2, 19):
                prior_games = [
                    snapshot
                    for snapshot in team_snapshots
                    if snapshot["week"] < prediction_week
                ]
                if not prior_games:
                    continue

                current = prior_games[-1]
                previous = prior_games[-2] if len(prior_games) >= 2 else None
                rolling_prior = prior_games[max(0, len(prior_games) - 4) : -1]

                current_qb_identity_key = qb_name_match_key(
                    current["qb_name"]
                )
                current_qb_pbp_key = qb_crosswalk.get(
                    current_qb_identity_key,
                    "",
                )

                current_qb_epa, current_qb_cpoe, current_qb_games = (
                    weighted_qb_performance(
                        team_games,
                        season,
                        prediction_week,
                        current_qb_pbp_key,
                        args,
                    )
                )

                row: dict[str, Any] = {
                    "season": season,
                    "prediction_week": prediction_week,
                    "team": team,
                    "last_completed_game_week": current["week"],
                    "weeks_since_last_game": prediction_week - current["week"],
                    "history_games": len(prior_games),
                    "current_qb_name": current["qb_name"],
                    "current_qb_key": current["qb_key"],
                    "current_qb_identity_key": current_qb_identity_key,
                    "current_qb_pbp_key": current_qb_pbp_key,
                    "qb_identity_matched": int(bool(current_qb_pbp_key)),
                    "week_one_qb_name": week_one["qb_name"],
                    "week_one_qb_key": week_one["qb_key"],
                    "qb_snap_share": current["qb_snap_share"],
                    "qb_recent_epa": current_qb_epa,
                    "qb_recent_cpoe": current_qb_cpoe,
                    "qb_recent_games": current_qb_games,
                    "qb_changed_from_week_one": int(
                        bool(week_one["qb_key"])
                        and bool(current["qb_key"])
                        and current["qb_key"] != week_one["qb_key"]
                    ),
                    "qb_changed_last_game": int(
                        previous is not None
                        and bool(previous["qb_key"])
                        and bool(current["qb_key"])
                        and previous["qb_key"] != current["qb_key"]
                    ),
                }

                for label in ("offense", "defense", "ol"):
                    current_dictionary = current[label]
                    previous_dictionary = previous[label] if previous is not None else {}
                    rolling_dictionary = average_share_dictionary(
                        [snapshot[label] for snapshot in rolling_prior]
                    )
                    row[f"{label}_continuity_last2"] = (
                        weighted_overlap(current_dictionary, previous_dictionary)
                        if previous is not None
                        else 1.0
                    )
                    row[f"{label}_rolling_stability"] = (
                        weighted_overlap(current_dictionary, rolling_dictionary)
                        if rolling_dictionary
                        else 1.0
                    )
                    row[f"missing_core_{label}_share"] = (
                        missing_core_share(rolling_dictionary, current_dictionary)
                        if rolling_dictionary
                        else 0.0
                    )

                rows.append(row)

    personnel = pd.DataFrame(rows)
    if personnel.empty:
        raise RuntimeError("Weekly personnel reconstruction produced zero rows.")

    personnel = personnel.sort_values(
        ["season", "prediction_week", "team"]
    ).reset_index(drop=True)
    return personnel, qb_crosswalk_audit


# =============================================================================
# GAME-LEVEL MATCHUP MATRIX
# =============================================================================


def side_ratings(ratings: pd.DataFrame, side: str) -> pd.DataFrame:
    team_key = "home_team" if side == "home" else "away_team"
    output = ratings.rename(
        columns={"prediction_week": "week", "team": team_key}
    ).copy()
    return output.rename(
        columns={
            column: f"{side}_{column}"
            for column in output.columns
            if column not in {"season", "week", team_key, "rating_alpha"}
        }
    )


def side_personnel(personnel: pd.DataFrame, side: str) -> pd.DataFrame:
    team_key = "home_team" if side == "home" else "away_team"
    output = personnel.rename(
        columns={"prediction_week": "week", "team": team_key}
    ).copy()
    return output.rename(
        columns={
            column: f"{side}_{column}"
            for column in output.columns
            if column not in {"season", "week", team_key}
        }
    )


def expected_metric(
    frame: pd.DataFrame,
    offense_side: str,
    defense_side: str,
    metric_name: str,
) -> pd.Series:
    return (
        numeric(frame, f"{offense_side}_{metric_name}_league_mean", 0.0).fillna(0.0)
        + numeric(
            frame,
            f"{offense_side}_{metric_name}_offense_rating",
            0.0,
        ).fillna(0.0)
        + numeric(
            frame,
            f"{defense_side}_{metric_name}_defense_allow_rating",
            0.0,
        ).fillna(0.0)
    )


def build_game_matrix(
    schedule: pd.DataFrame,
    ratings: pd.DataFrame,
    personnel: pd.DataFrame,
) -> pd.DataFrame:
    frame = schedule.merge(
        side_ratings(ratings, "home"),
        on=["season", "week", "home_team"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        side_ratings(ratings, "away"),
        on=["season", "week", "away_team"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        side_personnel(personnel, "home"),
        on=["season", "week", "home_team"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        side_personnel(personnel, "away"),
        on=["season", "week", "away_team"],
        how="left",
        validate="many_to_one",
    )

    for metric_name, (_, higher_is_good) in METRIC_SPECS.items():
        home_expected = expected_metric(frame, "home", "away", metric_name)
        away_expected = expected_metric(frame, "away", "home", metric_name)
        frame[f"home_expected_{metric_name}"] = home_expected
        frame[f"away_expected_{metric_name}"] = away_expected

        if higher_is_good:
            frame[f"{metric_name}_advantage"] = home_expected - away_expected
        elif metric_name == "sack_rate":
            frame["sack_advantage"] = away_expected - home_expected
        elif metric_name == "turnover_rate":
            frame["turnover_advantage"] = away_expected - home_expected

    frame["rating_alpha"] = float(ratings["rating_alpha"].iloc[0])

    frame["special_teams_advantage"] = (
        numeric(frame, "home_special_teams_rating", 0.0).fillna(0.0)
        - numeric(frame, "away_special_teams_rating", 0.0).fillna(0.0)
    )
    frame["market_home_margin"] = frame["spread_line"]
    frame["absolute_market_home_margin"] = frame["market_home_margin"].abs()

    frame["qb_recent_epa_advantage"] = (
        numeric(frame, "home_qb_recent_epa", 0.0).fillna(0.0)
        - numeric(frame, "away_qb_recent_epa", 0.0).fillna(0.0)
    )
    frame["qb_recent_cpoe_advantage"] = (
        numeric(frame, "home_qb_recent_cpoe", 0.0).fillna(0.0)
        - numeric(frame, "away_qb_recent_cpoe", 0.0).fillna(0.0)
    )
    frame["qb_week1_stability_advantage"] = (
        numeric(frame, "away_qb_changed_from_week_one", 0.0).fillna(0.0)
        - numeric(frame, "home_qb_changed_from_week_one", 0.0).fillna(0.0)
    )
    frame["qb_last_game_stability_advantage"] = (
        numeric(frame, "away_qb_changed_last_game", 0.0).fillna(0.0)
        - numeric(frame, "home_qb_changed_last_game", 0.0).fillna(0.0)
    )
    frame["qb_snap_share_advantage"] = (
        numeric(frame, "home_qb_snap_share", 0.0).fillna(0.0)
        - numeric(frame, "away_qb_snap_share", 0.0).fillna(0.0)
    )

    for label in ("offense", "defense", "ol"):
        frame[f"{label}_continuity_advantage"] = (
            numeric(frame, f"home_{label}_continuity_last2", 0.0).fillna(0.0)
            - numeric(frame, f"away_{label}_continuity_last2", 0.0).fillna(0.0)
        )
        frame[f"{label}_stability_advantage"] = (
            numeric(frame, f"home_{label}_rolling_stability", 0.0).fillna(0.0)
            - numeric(frame, f"away_{label}_rolling_stability", 0.0).fillna(0.0)
        )
        frame[f"core_{label}_health_advantage"] = (
            numeric(frame, f"away_missing_core_{label}_share", 0.0).fillna(0.0)
            - numeric(frame, f"home_missing_core_{label}_share", 0.0).fillna(0.0)
        )

    required_personnel = [
        "home_last_completed_game_week",
        "away_last_completed_game_week",
    ]
    frame["personnel_complete"] = frame[required_personnel].notna().all(axis=1).astype(int)

    return frame.sort_values(
        ["rating_alpha", "season", "week", "game_id"]
    ).reset_index(drop=True)


def build_qb_feature_integrity_audit(
    personnel: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Profile and enforce QB identity/performance integrity before fitting."""
    required_personnel = {
        "season",
        "prediction_week",
        "current_qb_name",
        "qb_identity_matched",
        "qb_recent_epa",
        "qb_recent_cpoe",
        "qb_recent_games",
    }
    required_matrix = {
        "season",
        "week",
        "rating_alpha",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
    }
    missing = sorted(required_personnel - set(personnel.columns))
    if missing:
        raise RuntimeError(f"QB personnel integrity fields are missing: {missing}")
    missing = sorted(required_matrix - set(matrix.columns))
    if missing:
        raise RuntimeError(f"QB matrix integrity fields are missing: {missing}")

    personnel_scope = personnel[
        personnel["season"].isin(PREDICTION_SEASONS)
        & personnel["prediction_week"].between(2, 17)
        & personnel["current_qb_name"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if personnel_scope.empty:
        raise RuntimeError("QB integrity audit has no personnel rows.")

    first_alpha = float(
        pd.to_numeric(matrix["rating_alpha"], errors="coerce").dropna().min()
    )
    matrix_scope = matrix[
        matrix["season"].isin(PREDICTION_SEASONS)
        & matrix["week"].between(2, 17)
        & pd.to_numeric(matrix["rating_alpha"], errors="coerce").eq(first_alpha)
    ].copy()
    if matrix_scope.empty:
        raise RuntimeError("QB integrity audit has no game-matrix rows.")

    rows: list[dict[str, Any]] = []
    scope_definitions = (
        ("ALL_2018_2025", tuple(PREDICTION_SEASONS)),
        ("VALIDATION_2020_2023", tuple(ROLLING_VALIDATION_SEASONS)),
        ("BENCHMARK_2024_2025", tuple(BENCHMARK_SEASONS)),
    )
    for scope, seasons in scope_definitions:
        p = personnel_scope[personnel_scope["season"].isin(seasons)].copy()
        m = matrix_scope[matrix_scope["season"].isin(seasons)].copy()
        epa = pd.to_numeric(p["qb_recent_epa"], errors="coerce")
        cpoe = pd.to_numeric(p["qb_recent_cpoe"], errors="coerce")
        games = pd.to_numeric(p["qb_recent_games"], errors="coerce")
        epa_advantage = pd.to_numeric(
            m["qb_recent_epa_advantage"], errors="coerce"
        )
        cpoe_advantage = pd.to_numeric(
            m["qb_recent_cpoe_advantage"], errors="coerce"
        )
        rows.append(
            {
                "scope": scope,
                "seasons": "|".join(map(str, seasons)),
                "personnel_rows": int(len(p)),
                "qb_identity_matched_rows": int(
                    pd.to_numeric(
                        p["qb_identity_matched"], errors="coerce"
                    ).fillna(0).eq(1).sum()
                ),
                "qb_identity_match_rate": float(
                    pd.to_numeric(
                        p["qb_identity_matched"], errors="coerce"
                    ).fillna(0).eq(1).mean()
                ),
                "qb_recent_epa_rows": int(epa.notna().sum()),
                "qb_recent_epa_coverage": float(epa.notna().mean()),
                "qb_recent_cpoe_rows": int(cpoe.notna().sum()),
                "qb_recent_cpoe_coverage": float(cpoe.notna().mean()),
                "qb_recent_games_positive_rows": int(games.fillna(0).gt(0).sum()),
                "game_rows": int(len(m)),
                "qb_epa_advantage_nonzero_rows": int(
                    epa_advantage.fillna(0).abs().gt(1e-12).sum()
                ),
                "qb_epa_advantage_std": float(epa_advantage.std(ddof=0)),
                "qb_cpoe_advantage_nonzero_rows": int(
                    cpoe_advantage.fillna(0).abs().gt(1e-12).sum()
                ),
                "qb_cpoe_advantage_std": float(cpoe_advantage.std(ddof=0)),
                "feature_integrity_passed": 0,
            }
        )

    audit = pd.DataFrame(rows)
    audit["feature_integrity_passed"] = (
        audit["qb_identity_match_rate"].ge(MINIMUM_QB_IDENTITY_MATCH_RATE)
        & audit["qb_recent_epa_coverage"].ge(
            MINIMUM_QB_RECENT_PERFORMANCE_COVERAGE
        )
        & audit["qb_recent_cpoe_coverage"].ge(
            MINIMUM_QB_RECENT_PERFORMANCE_COVERAGE
        )
        & audit["qb_epa_advantage_nonzero_rows"].gt(0)
        & audit["qb_cpoe_advantage_nonzero_rows"].gt(0)
        & audit["qb_epa_advantage_std"].gt(
            MINIMUM_QB_ADVANTAGE_STANDARD_DEVIATION
        )
        & audit["qb_cpoe_advantage_std"].gt(
            MINIMUM_QB_ADVANTAGE_STANDARD_DEVIATION
        )
    ).astype(int)
    failed = audit[audit["feature_integrity_passed"].ne(1)]
    if not failed.empty:
        raise RuntimeError(
            "QB feature integrity gate failed; model fitting is blocked:\n"
            + failed.to_string(index=False)
        )
    return audit


# =============================================================================
# DIRECTION AND COEFFICIENT AUDITS
# =============================================================================


def build_direction_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation = matrix[
        matrix["season"].isin(ROLLING_VALIDATION_SEASONS)
        & matrix["week"].between(2, 17)
    ].copy()

    for rating_alpha in sorted(validation["rating_alpha"].dropna().unique()):
        alpha_frame = validation[validation["rating_alpha"].eq(rating_alpha)]
        for feature in DIRECTIONAL_FEATURES:
            if feature not in alpha_frame.columns:
                continue
            year_correlations = {
                int(season): safe_correlation(
                    season_frame[feature],
                    season_frame["actual_home_margin"],
                )
                for season, season_frame in alpha_frame.groupby("season", sort=True)
            }
            usable_correlations = [
                value for value in year_correlations.values() if np.isfinite(value)
            ]
            rows.append(
                {
                    "rating_alpha": float(rating_alpha),
                    "feature": feature,
                    "combined_actual_margin_correlation": safe_correlation(
                        alpha_frame[feature],
                        alpha_frame["actual_home_margin"],
                    ),
                    "positive_direction_years": sum(
                        value >= 0 for value in usable_correlations
                    ),
                    "observed_years": len(usable_correlations),
                    "minimum_year_correlation": (
                        min(usable_correlations) if usable_correlations else np.nan
                    ),
                    "maximum_year_correlation": (
                        max(usable_correlations) if usable_correlations else np.nan
                    ),
                    **{
                        f"correlation_{season}": year_correlations.get(season, np.nan)
                        for season in ROLLING_VALIDATION_SEASONS
                    },
                }
            )

    return pd.DataFrame(rows)


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def prediction_quality(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_values = pd.to_numeric(actual, errors="coerce")
    predicted_values = pd.Series(predicted, index=actual.index, dtype=float)
    valid = actual_values.notna() & predicted_values.notna()
    if valid.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "correlation": np.nan}
    error = actual_values[valid] - predicted_values[valid]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(error.abs().mean()),
        "correlation": safe_correlation(actual_values[valid], predicted_values[valid]),
    }


def extract_base_coefficients(
    model: Pipeline,
    features: tuple[str, ...],
) -> dict[str, float]:
    imputer = model.named_steps["imputer"]
    ridge = model.named_steps["ridge"]
    names = list(imputer.get_feature_names_out(list(features)))
    coefficient_lookup = {
        str(name): float(coefficient)
        for name, coefficient in zip(names, ridge.coef_)
    }
    return {feature: coefficient_lookup.get(feature, np.nan) for feature in features}


def coefficient_stability(
    fold_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["rating_alpha", "feature_set", "model_alpha", "feature"]
    for keys, group in fold_coefficients.groupby(group_columns, sort=True):
        rating_alpha, feature_set, model_alpha, feature = keys
        coefficients = pd.to_numeric(group["coefficient"], errors="coerce").dropna()
        nonzero = coefficients[coefficients.abs().gt(1e-10)]
        if nonzero.empty:
            stability = np.nan
            dominant_sign = 0
        else:
            positive_share = float(nonzero.gt(0).mean())
            negative_share = float(nonzero.lt(0).mean())
            stability = max(positive_share, negative_share)
            dominant_sign = 1 if positive_share >= negative_share else -1
        rows.append(
            {
                "rating_alpha": float(rating_alpha),
                "feature_set": str(feature_set),
                "model_alpha": float(model_alpha),
                "feature": str(feature),
                "folds_observed": int(len(coefficients)),
                "average_coefficient": (
                    float(coefficients.mean()) if not coefficients.empty else np.nan
                ),
                "average_absolute_coefficient": (
                    float(coefficients.abs().mean()) if not coefficients.empty else np.nan
                ),
                "dominant_sign": dominant_sign,
                "sign_stability": stability,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# ROLLING MODEL VALIDATION
# =============================================================================


def rolling_validation(
    matrix: pd.DataFrame,
    features: tuple[str, ...],
    model_alpha: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for validation_season in ROLLING_VALIDATION_SEASONS:
        training = matrix[
            matrix["season"].lt(validation_season)
            & matrix["season"].ge(PREDICTION_SEASONS[0])
            & matrix["week"].between(2, 17)
        ].copy()
        validation = matrix[
            matrix["season"].eq(validation_season)
            & matrix["week"].between(2, 17)
        ].copy()

        if (
            len(training) < args.minimum_rolling_train_rows
            or len(validation) < args.minimum_rolling_validation_rows
        ):
            raise RuntimeError(
                "Insufficient rolling validation rows for "
                f"{validation_season}: train={len(training)}, validation={len(validation)}"
            )

        model = build_pipeline(model_alpha)
        model.fit(training[list(features)], training["actual_market_residual"])
        predicted = model.predict(validation[list(features)])
        quality = prediction_quality(validation["actual_market_residual"], predicted)

        validation["predicted_market_residual"] = predicted
        validation["validation_season"] = validation_season
        predictions.append(validation)
        quality_rows.append(
            {
                "validation_season": validation_season,
                "train_rows": len(training),
                "validation_rows": len(validation),
                **quality,
            }
        )

        coefficients = extract_base_coefficients(model, features)
        for feature, coefficient in coefficients.items():
            coefficient_rows.append(
                {
                    "validation_season": validation_season,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

    return (
        pd.concat(predictions, ignore_index=True),
        quality_rows,
        coefficient_rows,
    )


def evaluate_candidates(
    matrix: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    candidate_detail_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_lookup: dict[tuple[float, str, float], pd.DataFrame] = {}

    for rating_alpha in args.rating_alphas:
        alpha_matrix = matrix[matrix["rating_alpha"].eq(float(rating_alpha))].copy()
        for feature_set, features in FEATURE_SETS.items():
            for model_alpha in args.model_alphas:
                predictions, quality_rows, fold_coefficients = rolling_validation(
                    alpha_matrix,
                    features,
                    model_alpha,
                    args,
                )
                prediction_lookup[(float(rating_alpha), feature_set, float(model_alpha))] = (
                    predictions
                )
                for quality in quality_rows:
                    candidate_detail_rows.append(
                        {
                            "rating_alpha": float(rating_alpha),
                            "feature_set": feature_set,
                            "model_alpha": float(model_alpha),
                            "features": "|".join(features),
                            **quality,
                        }
                    )
                for coefficient in fold_coefficients:
                    coefficient_rows.append(
                        {
                            "rating_alpha": float(rating_alpha),
                            "feature_set": feature_set,
                            "model_alpha": float(model_alpha),
                            **coefficient,
                        }
                    )

    candidate_detail = pd.DataFrame(candidate_detail_rows)
    fold_coefficient_frame = pd.DataFrame(coefficient_rows)
    stability = coefficient_stability(fold_coefficient_frame)

    stability_summary_rows: list[dict[str, Any]] = []
    for keys, group in stability.groupby(
        ["rating_alpha", "feature_set", "model_alpha"],
        sort=True,
    ):
        rating_alpha, feature_set, model_alpha = keys
        usable = group[
            group["sign_stability"].notna()
            & group["average_absolute_coefficient"].notna()
        ].copy()
        if usable.empty:
            weighted_stability = np.nan
            material_minimum = np.nan
            average_stability = np.nan
            minimum_stability = np.nan
        else:
            weights = usable["average_absolute_coefficient"].clip(lower=1e-12)
            weighted_stability = float(
                np.average(usable["sign_stability"], weights=weights)
            )
            maximum_weight = float(weights.max())
            material = usable[
                usable["average_absolute_coefficient"].ge(0.25 * maximum_weight)
            ]
            material_minimum = float(material["sign_stability"].min())
            average_stability = float(usable["sign_stability"].mean())
            minimum_stability = float(usable["sign_stability"].min())
        stability_summary_rows.append(
            {
                "rating_alpha": float(rating_alpha),
                "feature_set": str(feature_set),
                "model_alpha": float(model_alpha),
                "average_sign_stability": average_stability,
                "minimum_sign_stability": minimum_stability,
                "weighted_sign_stability": weighted_stability,
                "material_minimum_sign_stability": material_minimum,
            }
        )
    stability_summary = pd.DataFrame(stability_summary_rows)

    aggregate = (
        candidate_detail.groupby(
            ["rating_alpha", "feature_set", "model_alpha", "features"],
            as_index=False,
        )
        .agg(
            positive_correlation_years=("correlation", lambda values: int((values > 0).sum())),
            minimum_validation_correlation=("correlation", "min"),
            average_validation_correlation=("correlation", "mean"),
            average_validation_rmse=("rmse", "mean"),
            average_validation_mae=("mae", "mean"),
        )
        .merge(
            stability_summary,
            on=["rating_alpha", "feature_set", "model_alpha"],
            how="left",
            validate="one_to_one",
        )
    )

    control = aggregate[aggregate["feature_set"].eq("MARKET_CONTROL")][
        [
            "rating_alpha",
            "model_alpha",
            "average_validation_correlation",
            "average_validation_rmse",
        ]
    ].rename(
        columns={
            "average_validation_correlation": "control_average_correlation",
            "average_validation_rmse": "control_average_rmse",
        }
    )
    aggregate = aggregate.merge(
        control,
        on=["rating_alpha", "model_alpha"],
        how="left",
        validate="many_to_one",
    )
    aggregate["correlation_improvement_vs_control"] = (
        aggregate["average_validation_correlation"]
        - aggregate["control_average_correlation"]
    )
    aggregate["rmse_improvement_vs_control"] = (
        aggregate["control_average_rmse"] - aggregate["average_validation_rmse"]
    )
    aggregate["signal_eligible"] = (
        aggregate["feature_set"].ne("MARKET_CONTROL")
        & aggregate["positive_correlation_years"].ge(3)
        & aggregate["average_validation_correlation"].gt(0)
        & aggregate["minimum_validation_correlation"].gt(-0.05)
        & aggregate["weighted_sign_stability"].ge(args.minimum_sign_stability)
        & aggregate["material_minimum_sign_stability"].ge(
            args.minimum_sign_stability
        )
        & aggregate["correlation_improvement_vs_control"].gt(0)
    ).astype(int)

    eligible = aggregate[aggregate["signal_eligible"].eq(1)].copy()
    if eligible.empty:
        diagnostic_pool = aggregate[aggregate["feature_set"].ne("MARKET_CONTROL")].copy()
        selected_row = diagnostic_pool.sort_values(
            [
                "positive_correlation_years",
                "average_validation_correlation",
                "average_validation_rmse",
                "average_sign_stability",
            ],
            ascending=[False, False, True, False],
        ).iloc[0]
        signal_validated = False
    else:
        selected_row = eligible.sort_values(
            [
                "positive_correlation_years",
                "average_validation_correlation",
                "average_validation_rmse",
                "average_sign_stability",
            ],
            ascending=[False, False, True, False],
        ).iloc[0]
        signal_validated = True

    selected_key = (
        float(selected_row["rating_alpha"]),
        str(selected_row["feature_set"]),
        float(selected_row["model_alpha"]),
    )
    selected = {
        "rating_alpha": selected_key[0],
        "feature_set": selected_key[1],
        "model_alpha": selected_key[2],
        "features": FEATURE_SETS[selected_key[1]],
        "signal_validated": signal_validated,
        "predictions": prediction_lookup[selected_key],
    }

    candidate_output = candidate_detail.merge(
        aggregate,
        on=["rating_alpha", "feature_set", "model_alpha", "features"],
        how="left",
        validate="many_to_one",
    )
    return selected, candidate_output, stability, fold_coefficient_frame


# =============================================================================
# ATS THRESHOLD VALIDATION
# =============================================================================


def win_profit(price: float) -> float:
    return 100.0 / abs(price) if price < 0 else price / 100.0


def grade_predictions(
    frame: pd.DataFrame,
    predicted_residual: np.ndarray,
    threshold: float,
    price: float,
) -> pd.DataFrame:
    output = frame.copy()
    output["predicted_market_residual"] = predicted_residual
    output["residual_absolute_edge"] = output["predicted_market_residual"].abs()
    output["selected_side"] = np.where(
        output["predicted_market_residual"].gt(0),
        "HOME",
        np.where(output["predicted_market_residual"].lt(0), "AWAY", "NONE"),
    )
    output["selected_cover_margin"] = np.where(
        output["selected_side"].eq("HOME"),
        output["actual_market_residual"],
        np.where(
            output["selected_side"].eq("AWAY"),
            -output["actual_market_residual"],
            np.nan,
        ),
    )
    output["ats_result"] = np.where(
        output["selected_cover_margin"].gt(0),
        "W",
        np.where(
            output["selected_cover_margin"].lt(0),
            "L",
            np.where(output["selected_cover_margin"].eq(0), "P", "N"),
        ),
    )
    output["qualifying_bet"] = (
        output["week"].lt(18)
        & output["residual_absolute_edge"].ge(float(threshold))
        & output["ats_result"].isin(["W", "L", "P"])
    ).astype(int)
    output["profit_units"] = np.where(
        output["qualifying_bet"].eq(1),
        np.where(
            output["ats_result"].eq("W"),
            win_profit(price),
            np.where(output["ats_result"].eq("L"), -1.0, 0.0),
        ),
        0.0,
    )
    return output


def maximum_drawdown(profits: pd.Series) -> float:
    cumulative = pd.to_numeric(profits, errors="coerce").fillna(0.0).cumsum()
    if cumulative.empty:
        return 0.0
    return float((cumulative.cummax() - cumulative).max())


def betting_summary(
    graded: pd.DataFrame,
    scope: str,
    threshold: float,
) -> dict[str, Any]:
    sample = graded[graded["qualifying_bet"].eq(1)].copy()
    wins = int(sample["ats_result"].eq("W").sum())
    losses = int(sample["ats_result"].eq("L").sum())
    pushes = int(sample["ats_result"].eq("P").sum())
    bets = int(len(sample))
    profit = float(sample["profit_units"].sum())
    return {
        "scope": scope,
        "threshold": float(threshold),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_win_rate": wins / (wins + losses) if wins + losses else np.nan,
        "profit_units": profit,
        "roi": profit / bets if bets else np.nan,
        "maximum_drawdown_units": maximum_drawdown(sample["profit_units"]),
    }


def validate_thresholds(
    selected_predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[Optional[float], pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []

    for threshold in args.thresholds:
        graded = grade_predictions(
            selected_predictions,
            selected_predictions["predicted_market_residual"].to_numpy(),
            threshold,
            args.ats_price,
        )
        combined = betting_summary(
            graded,
            "VALIDATION_2020_2023",
            threshold,
        )
        season_summaries = {
            int(season): betting_summary(
                season_frame,
                f"VALIDATION_{int(season)}",
                threshold,
            )
            for season, season_frame in graded.groupby("season", sort=True)
        }

        row = dict(combined)
        season_rois: list[float] = []
        profitable_seasons = 0
        for season in ROLLING_VALIDATION_SEASONS:
            season_summary = season_summaries[season]
            row[f"bets_{season}"] = season_summary["bets"]
            row[f"roi_{season}"] = season_summary["roi"]
            if np.isfinite(season_summary["roi"]):
                season_rois.append(float(season_summary["roi"]))
                profitable_seasons += int(season_summary["roi"] >= 0)

        row["profitable_validation_seasons"] = profitable_seasons
        row["minimum_validation_season_roi"] = (
            min(season_rois) if season_rois else np.nan
        )
        row["sample_eligible"] = int(
            row["bets"] >= args.minimum_total_validation_bets
            and all(
                row[f"bets_{season}"] >= args.minimum_season_validation_bets
                for season in ROLLING_VALIDATION_SEASONS
            )
        )
        row["roi_stability_eligible"] = int(
            row["roi"] > 0
            and profitable_seasons >= args.minimum_profitable_validation_seasons
            and np.isfinite(row["minimum_validation_season_roi"])
            and row["minimum_validation_season_roi"] >= args.validation_roi_floor
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    robust = table[
        table["sample_eligible"].eq(1) & table["roi_stability_eligible"].eq(1)
    ].copy()
    if robust.empty:
        return None, table, False

    selected = robust.sort_values(
        [
            "minimum_validation_season_roi",
            "profitable_validation_seasons",
            "roi",
            "bets",
        ],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"]), table, True


# =============================================================================
# BENCHMARK, DEPLOYMENT BUNDLE, AND REPORTING
# =============================================================================


def build_benchmark(
    matrix: pd.DataFrame,
    selected: dict[str, Any],
    threshold: Optional[float],
    args: argparse.Namespace,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_matrix = matrix[
        matrix["rating_alpha"].eq(selected["rating_alpha"])
    ].copy()
    training = selected_matrix[
        selected_matrix["season"].le(2023)
        & selected_matrix["week"].between(2, 17)
    ].copy()
    benchmark = selected_matrix[
        selected_matrix["season"].isin(BENCHMARK_SEASONS)
        & selected_matrix["week"].between(2, 17)
    ].copy()

    model = build_pipeline(selected["model_alpha"])
    model.fit(training[list(selected["features"])], training["actual_market_residual"])
    predicted = model.predict(benchmark[list(selected["features"])])
    applied_threshold = float(threshold) if threshold is not None else np.inf
    graded = grade_predictions(
        benchmark,
        predicted,
        applied_threshold,
        args.ats_price,
    )

    summary_rows: list[dict[str, Any]] = []
    combined_quality = prediction_quality(
        benchmark["actual_market_residual"],
        predicted,
    )
    combined_summary = betting_summary(
        graded,
        "BENCHMARK_2024_2025",
        applied_threshold,
    )
    summary_rows.append({**combined_summary, **combined_quality})

    for season, season_frame in graded.groupby("season", sort=True):
        season_prediction = season_frame["predicted_market_residual"].to_numpy()
        season_quality = prediction_quality(
            season_frame["actual_market_residual"],
            season_prediction,
        )
        summary_rows.append(
            {
                **betting_summary(
                    season_frame,
                    f"BENCHMARK_{int(season)}",
                    applied_threshold,
                ),
                **season_quality,
            }
        )

    coefficient_lookup = extract_base_coefficients(model, selected["features"])
    coefficients = pd.DataFrame(
        [
            {
                "feature": feature,
                "standardized_coefficient": coefficient,
                "rating_alpha": selected["rating_alpha"],
                "model_alpha": selected["model_alpha"],
            }
            for feature, coefficient in coefficient_lookup.items()
        ]
    ).sort_values(
        "standardized_coefficient",
        key=lambda values: values.abs(),
        ascending=False,
    )

    return model, graded, pd.DataFrame(summary_rows), coefficients


def benchmark_safety_check(
    benchmark_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, Any]]:
    combined = benchmark_summary[
        benchmark_summary["scope"].eq("BENCHMARK_2024_2025")
    ].iloc[0]
    season_rows = benchmark_summary[
        benchmark_summary["scope"].isin(["BENCHMARK_2024", "BENCHMARK_2025"])
    ]

    combined_roi_pass = bool(
        pd.notna(combined["roi"])
        and float(combined["roi"]) >= args.benchmark_combined_roi_floor
    )
    season_roi_pass = bool(
        not season_rows.empty
        and season_rows["roi"].notna().all()
        and season_rows["roi"].min() >= args.benchmark_season_roi_floor
    )
    benchmark_bet_count_pass = bool(
        pd.notna(combined["bets"])
        and int(combined["bets"]) >= args.minimum_benchmark_bets
    )
    correlation_pass = bool(
        pd.notna(combined["correlation"])
        and float(combined["correlation"]) >= args.benchmark_correlation_floor
    )
    passed = (
        benchmark_bet_count_pass
        and combined_roi_pass
        and season_roi_pass
        and correlation_pass
    )

    return passed, {
        "benchmark_bet_count_pass": int(benchmark_bet_count_pass),
        "benchmark_combined_roi_pass": int(combined_roi_pass),
        "benchmark_season_roi_pass": int(season_roi_pass),
        "benchmark_correlation_pass": int(correlation_pass),
        "benchmark_safety_passed": int(passed),
    }


def build_comparison(benchmark_summary: pd.DataFrame) -> pd.DataFrame:
    matchup = benchmark_summary[
        benchmark_summary["scope"].eq("BENCHMARK_2024_2025")
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "model": "OLD_FULL_MODEL",
                "scope": "2024_2025",
                "bets": 147,
                "wins": 61,
                "losses": 86,
                "pushes": 0,
                "profit_units": -30.545455,
                "roi": -0.207792,
            },
            {
                "model": "PERSONNEL_RESIDUAL_V1",
                "scope": "2024_2025",
                "bets": 239,
                "wins": 115,
                "losses": 123,
                "pushes": 1,
                "profit_units": -18.454545,
                "roi": -0.077216,
            },
            {
                "model": "MATCHUP_RESIDUAL_V1",
                "scope": "2024_2025",
                "bets": 229,
                "wins": 117,
                "losses": 111,
                "pushes": 1,
                "profit_units": -4.636364,
                "roi": -0.020246,
            },
            {
                "model": "MATCHUP_RESIDUAL_V2",
                "scope": "2024_2025",
                "bets": int(matchup["bets"]),
                "wins": int(matchup["wins"]),
                "losses": int(matchup["losses"]),
                "pushes": int(matchup["pushes"]),
                "profit_units": float(matchup["profit_units"]),
                "roi": float(matchup["roi"]) if pd.notna(matchup["roi"]) else np.nan,
            },
        ]
    )


def save_deployment_bundle(
    args: argparse.Namespace,
    matrix: pd.DataFrame,
    selected: dict[str, Any],
    threshold: Optional[float],
    implementation_ready: bool,
    validation_passed: bool,
    benchmark_passed: bool,
    qb_integrity: pd.Series,
) -> tuple[Path, Path]:
    selected_matrix = matrix[
        matrix["rating_alpha"].eq(selected["rating_alpha"])
        & matrix["season"].isin(PREDICTION_SEASONS)
        & matrix["week"].between(2, 17)
    ].copy()
    deployment_model = build_pipeline(selected["model_alpha"])
    deployment_model.fit(
        selected_matrix[list(selected["features"])],
        selected_matrix["actual_market_residual"],
    )

    model_directory = args.project_root / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / MODEL_FILENAME
    metadata_path = model_directory / METADATA_FILENAME

    bundle = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "model": deployment_model,
        "feature_order": list(selected["features"]),
        "rating_alpha": selected["rating_alpha"],
        "model_alpha": selected["model_alpha"],
        "threshold": threshold,
        "market_sign_convention": "positive_is_expected_home_margin",
        "implementation_ready": implementation_ready,
        "qb_feature_integrity_passed": int(
            qb_integrity["feature_integrity_passed"]
        ),
        "qb_identity_match_rate": float(
            qb_integrity["qb_identity_match_rate"]
        ),
        "qb_side_epa_coverage": float(
            qb_integrity["qb_recent_epa_coverage"]
        ),
        "qb_side_cpoe_coverage": float(
            qb_integrity["qb_recent_cpoe_coverage"]
        ),
        "qb_epa_advantage_nonzero_rows": int(
            qb_integrity["qb_epa_advantage_nonzero_rows"]
        ),
        "qb_epa_advantage_std": float(
            qb_integrity["qb_epa_advantage_std"]
        ),
        "qb_cpoe_advantage_nonzero_rows": int(
            qb_integrity["qb_cpoe_advantage_nonzero_rows"]
        ),
        "qb_cpoe_advantage_std": float(
            qb_integrity["qb_cpoe_advantage_std"]
        ),
        "created_at": now_string(),
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "feature_set": selected["feature_set"],
        "feature_order": list(selected["features"]),
        "rating_alpha": selected["rating_alpha"],
        "model_alpha": selected["model_alpha"],
        "threshold": threshold,
        "validation_passed": validation_passed,
        "benchmark_safety_passed": benchmark_passed,
        "implementation_ready": implementation_ready,
        "qb_feature_integrity_passed": int(
            qb_integrity["feature_integrity_passed"]
        ),
        "qb_identity_match_rate": float(
            qb_integrity["qb_identity_match_rate"]
        ),
        "qb_side_epa_coverage": float(
            qb_integrity["qb_recent_epa_coverage"]
        ),
        "qb_side_cpoe_coverage": float(
            qb_integrity["qb_recent_cpoe_coverage"]
        ),
        "qb_epa_advantage_nonzero_rows": int(
            qb_integrity["qb_epa_advantage_nonzero_rows"]
        ),
        "qb_epa_advantage_std": float(
            qb_integrity["qb_epa_advantage_std"]
        ),
        "qb_cpoe_advantage_nonzero_rows": int(
            qb_integrity["qb_cpoe_advantage_nonzero_rows"]
        ),
        "qb_cpoe_advantage_std": float(
            qb_integrity["qb_cpoe_advantage_std"]
        ),
        "production_usage": (
            "enabled"
            if implementation_ready
            else "disabled_until_validation_and_benchmark_gates_pass"
        ),
        "created_at": now_string(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    print("[MATCHUP_RESIDUAL] Building weekly matchup residual model")
    print(f"[MATCHUP_RESIDUAL] Build ID: {BUILD_ID}")
    print(f"[MATCHUP_RESIDUAL] Version: {VERSION}")
    print(
        "[MATCHUP_RESIDUAL] Development=2018-2019 | "
        "Rolling validation=2020-2023 | Benchmark safety=2024-2025"
    )

    pbp, schedule, snaps, cache_audit, cache_path = load_or_build_cache(args)
    print(f"[MATCHUP_RESIDUAL] PBP source: {cache_audit['pbp_source']}")
    print(
        f"[MATCHUP_RESIDUAL] PBP rows: {len(pbp):,} | "
        f"seasons={sorted(pbp['season'].unique().tolist())}"
    )
    print(f"[MATCHUP_RESIDUAL] Schedule source: {cache_audit['schedule_source']}")
    print(
        f"[MATCHUP_RESIDUAL] Schedule rows: {len(schedule):,} | "
        f"seasons={sorted(schedule['season'].unique().tolist())}"
    )
    print(f"[MATCHUP_RESIDUAL] Snap source: {cache_audit['snap_source']}")
    print(
        f"[MATCHUP_RESIDUAL] Snap rows: {len(snaps):,} | "
        f"seasons={sorted(snaps['season'].unique().tolist())}"
    )

    missing_pbp = sorted(set(PBP_SOURCE_SEASONS) - set(pbp["season"].unique()))
    missing_schedule = sorted(
        set(PREDICTION_SEASONS) - set(schedule["season"].unique())
    )
    missing_snaps = sorted(set(SNAP_SOURCE_SEASONS) - set(snaps["season"].unique()))
    if missing_pbp:
        raise RuntimeError(f"Missing PBP seasons: {missing_pbp}")
    if missing_schedule:
        raise RuntimeError(f"Missing schedule seasons: {missing_schedule}")
    if missing_snaps:
        raise RuntimeError(f"Missing snap-count seasons: {missing_snaps}")

    team_games = build_team_game_features(pbp)
    print(f"[MATCHUP_RESIDUAL] Team-game rows: {len(team_games):,}")

    personnel, qb_identity_audit = build_weekly_personnel_features(
        snaps,
        team_games,
        schedule,
        args,
    )
    print(f"[MATCHUP_RESIDUAL] Weekly personnel rows: {len(personnel):,}")

    rating_frames: list[pd.DataFrame] = []
    matrix_frames: list[pd.DataFrame] = []
    for rating_alpha in args.rating_alphas:
        print(
            f"[MATCHUP_RESIDUAL] Building weekly matchup ratings | "
            f"rating_alpha={rating_alpha:g}"
        )
        ratings = build_team_week_ratings(
            team_games,
            schedule,
            rating_alpha,
            args,
        )
        matrix = build_game_matrix(schedule, ratings, personnel)
        rating_frames.append(ratings)
        matrix_frames.append(matrix)

    ratings_all = pd.concat(rating_frames, ignore_index=True)
    matrix_all = pd.concat(matrix_frames, ignore_index=True)
    qb_feature_audit = build_qb_feature_integrity_audit(
        personnel,
        matrix_all,
    )
    all_qb_audit = qb_feature_audit[
        qb_feature_audit["scope"].eq("ALL_2018_2025")
    ].iloc[0]
    print(
        "[MATCHUP_RESIDUAL] QB identity/performance integrity: "
        f"match={all_qb_audit['qb_identity_match_rate']:.2%} | "
        f"EPA coverage={all_qb_audit['qb_recent_epa_coverage']:.2%} | "
        f"CPOE coverage={all_qb_audit['qb_recent_cpoe_coverage']:.2%} | "
        f"EPA advantage std={all_qb_audit['qb_epa_advantage_std']:.6f} | "
        f"CPOE advantage std={all_qb_audit['qb_cpoe_advantage_std']:.6f}"
    )

    validation_matrix = matrix_all[
        matrix_all["season"].isin(ROLLING_VALIDATION_SEASONS)
        & matrix_all["week"].between(2, 17)
    ]
    personnel_completeness = float(validation_matrix["personnel_complete"].mean())
    if personnel_completeness < 0.98:
        raise RuntimeError(
            "Validation personnel completeness is only "
            f"{personnel_completeness:.2%}; expected at least 98%."
        )

    if args.preflight_only:
        print(
            "[MATCHUP_RESIDUAL] Preflight passed: source coverage, QB identity "
            "crosswalk, QB EPA/CPOE population, and feature variance are valid."
        )
        print(
            "[MATCHUP_RESIDUAL] Preflight-only run complete; no output database written."
        )
        return 0

    direction_audit = build_direction_audit(matrix_all)
    selected, candidate_validation, coefficient_stability_table, _ = (
        evaluate_candidates(matrix_all, args)
    )
    selected_threshold, threshold_validation, threshold_passed = validate_thresholds(
        selected["predictions"],
        args,
    )
    validation_passed = bool(selected["signal_validated"] and threshold_passed)

    _, benchmark_predictions, benchmark_summary, coefficients = build_benchmark(
        matrix_all,
        selected,
        selected_threshold if validation_passed else None,
        args,
    )
    benchmark_passed, benchmark_checks = benchmark_safety_check(
        benchmark_summary,
        args,
    )
    implementation_ready = bool(validation_passed and benchmark_passed)

    comparison = build_comparison(benchmark_summary)
    model_path, metadata_path = save_deployment_bundle(
        args,
        matrix_all,
        selected,
        selected_threshold if validation_passed else None,
        implementation_ready,
        validation_passed,
        benchmark_passed,
        all_qb_audit,
    )

    output_database = args.project_root / "backtests" / OUTPUT_DATABASE_NAME
    output_database.parent.mkdir(parents=True, exist_ok=True)
    if output_database.exists():
        output_database.unlink()

    with sqlite3.connect(output_database) as connection:
        team_games.to_sql(TEAM_GAME_TABLE, connection, if_exists="replace", index=False)
        ratings_all.to_sql(TEAM_WEEK_TABLE, connection, if_exists="replace", index=False)
        personnel.to_sql(
            PERSONNEL_WEEK_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        matrix_all.to_sql(GAME_MATRIX_TABLE, connection, if_exists="replace", index=False)
        direction_audit.to_sql(
            DIRECTION_AUDIT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        candidate_validation.to_sql(
            CANDIDATE_VALIDATION_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        coefficient_stability_table.to_sql(
            COEFFICIENT_STABILITY_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        threshold_validation.to_sql(
            THRESHOLD_VALIDATION_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        benchmark_predictions.to_sql(
            BENCHMARK_PREDICTION_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        benchmark_summary.to_sql(
            BENCHMARK_SUMMARY_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        coefficients.to_sql(
            MODEL_COEFFICIENT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        comparison.to_sql(
            MODEL_COMPARISON_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        qb_identity_audit.to_sql(
            QB_IDENTITY_AUDIT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        qb_feature_audit.to_sql(
            QB_FEATURE_AUDIT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "build_id": BUILD_ID,
                    "version": VERSION,
                    "cache_path": str(cache_path),
                    "team_game_rows": len(team_games),
                    "team_week_rows": len(ratings_all),
                    "personnel_week_rows": len(personnel),
                    "game_matrix_rows": len(matrix_all),
                    "validation_personnel_completeness": personnel_completeness,
                    "qb_identity_crosswalk_rows": len(qb_identity_audit),
                    "qb_identity_match_rate": float(
                        all_qb_audit["qb_identity_match_rate"]
                    ),
                    "qb_recent_epa_coverage": float(
                        all_qb_audit["qb_recent_epa_coverage"]
                    ),
                    "qb_recent_cpoe_coverage": float(
                        all_qb_audit["qb_recent_cpoe_coverage"]
                    ),
                    "qb_epa_advantage_nonzero_rows": int(
                        all_qb_audit["qb_epa_advantage_nonzero_rows"]
                    ),
                    "qb_epa_advantage_std": float(
                        all_qb_audit["qb_epa_advantage_std"]
                    ),
                    "qb_cpoe_advantage_nonzero_rows": int(
                        all_qb_audit["qb_cpoe_advantage_nonzero_rows"]
                    ),
                    "qb_cpoe_advantage_std": float(
                        all_qb_audit["qb_cpoe_advantage_std"]
                    ),
                    "qb_feature_integrity_passed": int(
                        all_qb_audit["feature_integrity_passed"]
                    ),
                    "selected_rating_alpha": selected["rating_alpha"],
                    "selected_feature_set": selected["feature_set"],
                    "selected_model_alpha": selected["model_alpha"],
                    "selected_threshold": selected_threshold,
                    "residual_signal_validated": int(selected["signal_validated"]),
                    "threshold_validation_passed": int(threshold_passed),
                    "validation_passed": int(validation_passed),
                    **benchmark_checks,
                    "implementation_ready": int(implementation_ready),
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                    "created_at": now_string(),
                }
            ]
        ).to_sql(RUN_AUDIT_TABLE, connection, if_exists="replace", index=False)

    if not args.no_csv:
        output_directory = (
            args.project_root
            / "outputs"
            / "historical_weekly_replay"
            / "matchup_residual_v2"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        direction_audit.to_csv(
            output_directory / "nfl_matchup_feature_direction_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        candidate_validation.to_csv(
            output_directory / "nfl_matchup_candidate_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        coefficient_stability_table.to_csv(
            output_directory / "nfl_matchup_coefficient_stability.csv",
            index=False,
            encoding="utf-8-sig",
        )
        qb_identity_audit.to_csv(
            output_directory / "nfl_matchup_qb_identity_crosswalk_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        qb_feature_audit.to_csv(
            output_directory / "nfl_matchup_qb_feature_integrity_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        threshold_validation.to_csv(
            output_directory / "nfl_matchup_threshold_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        benchmark_predictions.to_csv(
            output_directory / "nfl_matchup_benchmark_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        benchmark_summary.to_csv(
            output_directory / "nfl_matchup_benchmark_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        coefficients.to_csv(
            output_directory / "nfl_matchup_model_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )
        comparison.to_csv(
            output_directory / "nfl_matchup_model_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    selected_candidate_rows = candidate_validation[
        candidate_validation["rating_alpha"].eq(selected["rating_alpha"])
        & candidate_validation["feature_set"].eq(selected["feature_set"])
        & candidate_validation["model_alpha"].eq(selected["model_alpha"])
    ]

    print("\n" + "=" * 124)
    print("[MATCHUP_RESIDUAL] SELECTED ROLLING VALIDATION MODEL")
    print("=" * 124)
    print(selected_candidate_rows.to_string(index=False))
    print(f"\n[MATCHUP_RESIDUAL] Selected rating alpha: {selected['rating_alpha']}")
    print(f"[MATCHUP_RESIDUAL] Selected feature set: {selected['feature_set']}")
    print(f"[MATCHUP_RESIDUAL] Selected model alpha: {selected['model_alpha']}")
    print(
        f"[MATCHUP_RESIDUAL] Residual signal validation passed: "
        f"{selected['signal_validated']}"
    )
    print(f"[MATCHUP_RESIDUAL] Selected threshold: {selected_threshold}")
    print(f"[MATCHUP_RESIDUAL] Threshold validation passed: {threshold_passed}")

    print("\n" + "=" * 124)
    print("[MATCHUP_RESIDUAL] 2020-2023 THRESHOLD VALIDATION")
    print("=" * 124)
    print(threshold_validation.to_string(index=False))

    print("\n" + "=" * 124)
    print("[MATCHUP_RESIDUAL] 2024-2025 FIXED BENCHMARK SAFETY CHECK")
    print("=" * 124)
    print(benchmark_summary.to_string(index=False))

    print("\n[MATCHUP_RESIDUAL] Model progression:")
    print(comparison.to_string(index=False))
    print("\n[MATCHUP_RESIDUAL] Final coefficients:")
    print(coefficients.to_string(index=False))

    print("\n" + "=" * 124)
    print("[MATCHUP_RESIDUAL] IMPLEMENTATION DECISION")
    print("=" * 124)
    print(f"Validation passed:         {validation_passed}")
    print(f"Benchmark safety passed:   {benchmark_passed}")
    print(f"IMPLEMENTATION READY:      {implementation_ready}")
    if implementation_ready:
        print(
            "[MATCHUP_RESIDUAL] The deployment bundle is enabled for building "
            "the 2026 weekly predictor."
        )
    else:
        print(
            "[MATCHUP_RESIDUAL] The deployment bundle was saved for diagnostics "
            "but production usage remains disabled."
        )

    print(f"[MATCHUP_RESIDUAL] Output database: {output_database}")
    print(f"[MATCHUP_RESIDUAL] Model bundle: {model_path}")
    print(f"[MATCHUP_RESIDUAL] Metadata: {metadata_path}")
    print("[MATCHUP_RESIDUAL] Production/existing historical databases modified: NO")
    print(
        "[MATCHUP_RESIDUAL] Completed successfully in "
        f"{(dt.datetime.now() - started).total_seconds():.2f} seconds"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[MATCHUP_RESIDUAL] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[MATCHUP_RESIDUAL] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
