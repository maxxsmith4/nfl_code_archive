#!/usr/bin/env python
"""
Build an opponent-adjusted 2026 NFL form rating and blend it incrementally into
the current roster/depth-chart structural power rating.

Purpose
-------
This script creates the in-season layer that sits on top of
nfl_power_ratings_2026.

The immutable preseason snapshot remains an audit comparator. The current
market-independent structural rating is the active prior, so authoritative
roster, quarterback, depth-chart, and offensive-line refreshes reach the live
weekly model. Completed 2026 games enter gradually using:

    current_season_weight = games_played / (games_played + 5)

The weight is capped at 75 percent. Therefore:
    0 games  ->  0.0% 2026 form
    1 game   -> 16.7% 2026 form
    2 games  -> 28.6% 2026 form
    4 games  -> 44.4% 2026 form
    8 games  -> 61.5% 2026 form
    15+ games -> 75.0% maximum

The 2026 form rating is built from two independent signals:

1. Process rating
   Predictive play-by-play efficiency:
       EPA/play
       success rate
       passing EPA
       rushing EPA
       early-down EPA
       explosive-play rate
       turnover rate
       sack rate
       CPOE
       special-teams EPA

   Historical 2018-2025 play-by-play is used to estimate how these game-level
   process metrics translate to points.

2. Result rating
   Capped actual scoring margin.

Both signals are opponent-adjusted through a ridge SRS system:

    observed_home_margin
    = home_team_rating - away_team_rating + home_field

The final form rating is:

    season_form = 70% process + 30% results
    recent_form = 70% process + 30% results

    form_rating = 70% season_form + 30% recent_form

The live internal rating is:

    live_power_rating
    = structural_prior * (1 - current_season_weight)
    + form_rating * current_season_weight

Before any 2026 regular-season games are completed, this script safely writes
a 32-team output where live_power_rating equals the current structural rating
exactly. Preseason drift remains explicit in separate audit columns.

Inputs
------
SQLite:
    nfl_power_ratings_2026

External data:
    nflreadpy.load_schedules
    nfl_data_py.import_schedules fallback
    nfl_data_py.import_pbp_data
    nflreadpy play-by-play fallback

Outputs
-------
SQLite and CSV:
    nfl_2026_form_ratings
    nfl_2026_form_game_audit
    nfl_2026_form_team_game_audit
    nfl_2026_form_process_model

Primary point-spread field
--------------------------
Use:

    nfl_2026_form_ratings.power_rating_points

This field is a neutral-field points-above-average rating. Once games begin it
is the incrementally blended live internal rating. Before Week 1 it equals the
current structural power_rating_points value.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sql


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2026
BUILD_ID = "NFL_2026_FORM_RATING_CANONICAL_V3"
VERSION = "v3_current_structural_prior_asof_opponent_adjusted_form"
IMPLEMENTATION_VERSION = "v3_1_explicit_zero_week_cutoff"
SOURCE_RECOVERY_VERSION = "v1_local_current_inputs_and_cached_process_model"
FEATURE_VERSION = "v2_0_pbp_process_features"
PROCESS_MODEL_VERSION = "v2_0_2018_2025_process_to_points"
EXPECTED_STRUCTURAL_POWER_BUILD_ID = (
    "NFL_POWER_RATINGS_2026_CANONICAL_V2"
)

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

# Runtime globals retain compatibility with helper scripts that import this module.
PROJECT_ROOT = DEFAULT_PROJECT_ROOT
DB_PATH = DEFAULT_DB_PATH
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

PRESEASON_POWER_TABLE = "nfl_power_ratings_2026"
PRESEASON_SNAPSHOT_TABLE = "nfl_2026_preseason_power_snapshot"

OUTPUT_TABLE = "nfl_2026_form_ratings"
GAME_AUDIT_TABLE = "nfl_2026_form_game_audit"
TEAM_GAME_AUDIT_TABLE = "nfl_2026_form_team_game_audit"
PROCESS_MODEL_TABLE = "nfl_2026_form_process_model"
HISTORICAL_FEATURE_CACHE_TABLE = (
    "nfl_form_historical_team_game_features_2018_2025"
)

CALIBRATION_SEASONS = list(range(2018, 2026))
GAME_TYPE_FILTER = {"REG"}

# Incremental current-structural-prior-to-form transition. The current roster
# and depth-chart structural rating is never fully discarded; at least 25%
# remains after the form weight reaches its cap. The immutable preseason
# snapshot remains in the output only for drift and no-lookahead auditing.
EFFECTIVE_PRIOR_GAMES = 5.0
MAX_CURRENT_SEASON_WEIGHT = 0.75

# Form construction.
PROCESS_WEIGHT = 0.70
RESULT_WEIGHT = 0.30
SEASON_FORM_WEIGHT = 0.70
RECENT_FORM_WEIGHT = 0.30
RECENT_WEEK_DECAY = 0.55

# Robustness and scale.
RESULT_MARGIN_CAP = 24.0
PROCESS_MARGIN_CAP = 28.0
FORM_RATING_CAP = 8.0
LIVE_POWER_CAP = 12.0

# Ridge regularization.
PROCESS_RIDGE_ALPHA = 25.0
SEASON_SRS_RIDGE_ALPHA = 6.0
RECENT_SRS_RIDGE_ALPHA = 10.0

# Historical process-model weighting.
CALIBRATION_SEASON_DECAY = 0.90
MIN_HISTORICAL_GAMES = 800
MIN_CURRENT_PROCESS_GAMES = 1

PROCESS_FEATURES = [
    "net_epa_per_play",
    "net_success_rate",
    "net_pass_epa_per_play",
    "net_rush_epa_per_play",
    "net_early_down_epa_per_play",
    "net_explosive_rate",
    "turnover_margin_rate",
    "net_sack_rate",
    "net_cpoe",
    "st_epa_diff",
]

TEAM_ALIASES = {
    "ARI": "ARI", "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL", "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE", "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB", "GNB": "GB",
    "HOU": "HOU", "HST": "HOU",
    "IND": "IND",
    "JAC": "JAX", "JAX": "JAX",
    "KC": "KC", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "LAR": "LAR", "STL": "LAR",
    "LAC": "LAC", "SD": "LAC", "SDG": "LAC",
    "LV": "LV", "LVR": "LV", "OAK": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE", "NWE": "NE",
    "NO": "NO", "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF", "SFO": "SF",
    "TB": "TB", "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS", "WSH": "WAS", "WFT": "WAS",
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
]

# =============================================================================
# ARGUMENTS AND LOGGING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical opponent-adjusted 2026 NFL form layer."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--db-path",
        "--database",
        dest="db_path",
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help=(
            "Include only completed games on or before YYYY-MM-DD. "
            "Defaults to today."
        ),
    )
    parser.add_argument(
        "--through-week",
        type=int,
        default=None,
        help=(
            "Optional regular-season week cutoff; 0 uses only the current "
            "structural prior."
        ),
    )
    parser.add_argument(
        "--refresh-preseason-snapshot",
        action="store_true",
        help=(
            "Replace the immutable preseason snapshot. Allowed only when "
            "no 2026 regular-season games have completed."
        ),
    )
    parser.add_argument(
        "--rebuild-process-cache",
        action="store_true",
        help="Delete and rebuild the historical process cache and model.",
    )
    parser.add_argument(
        "--prepare-process-model",
        action="store_true",
        help=(
            "Build the 2018-2025 process cache/model even before a 2026 "
            "regular-season game is complete."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write SQLite tables only.",
    )
    parser.add_argument(
        "--current-pbp-path",
        type=Path,
        default=None,
        help="Use an existing 2026 raw play-by-play CSV instead of downloading it.",
    )
    parser.add_argument(
        "--schedule-path",
        type=Path,
        default=None,
        help="Use an existing current-season schedule CSV instead of downloading it.",
    )
    args = parser.parse_args()
    if args.through_week is not None and args.through_week < 0:
        parser.error("--through-week must be at least 0.")
    for name in ("current_pbp_path", "schedule_path"):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f"--{name.replace('_', '-')} must name an existing CSV file: {value}")
            setattr(args, name, value)
    return args


def configure_runtime(
    project_root: Path,
    db_path: Path,
) -> None:
    global PROJECT_ROOT, DB_PATH, OUTPUT_DIR, LOG_DIR
    PROJECT_ROOT = project_root.resolve()
    DB_PATH = db_path.resolve()
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    LOG_DIR = PROJECT_ROOT / "logs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"build_nfl_2026_form_rating_{timestamp}.log"

    logger = logging.getLogger("nfl_2026_form_rating")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger, path


# Import-safe defaults. main() reconfigures these using command-line paths.
configure_runtime(DEFAULT_PROJECT_ROOT, DEFAULT_DB_PATH)
LOGGER, LOG_PATH = configure_logging()

# =============================================================================
# GENERIC HELPERS
# =============================================================================

def get_engine():
    return sql.create_engine(
        f"sqlite:///{DB_PATH}",
        pool_pre_ping=True,
    )


def table_exists(engine, table_name: str) -> bool:
    query = """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = :table_name
        LIMIT 1
    """
    with engine.connect() as connection:
        return connection.execute(
            sql.text(query),
            {"table_name": table_name},
        ).fetchone() is not None


def table_columns(engine, table_name: str) -> list[str]:
    if not table_exists(engine, table_name):
        return []

    escaped = table_name.replace('"', '""')
    with engine.connect() as connection:
        rows = connection.execute(
            sql.text(f'PRAGMA table_info("{escaped}")')
        ).fetchall()

    return [
        str(row[1]).strip().lower()
        for row in rows
    ]


def read_table(engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")

    escaped = table_name.replace('"', '""')
    with engine.connect() as connection:
        frame = pd.read_sql(
            sql.text(f'SELECT * FROM "{escaped}"'),
            connection,
        )

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]
    return frame



def drop_table_if_exists(engine, table_name: str) -> None:
    escaped = table_name.replace('"', '""')
    with engine.begin() as connection:
        connection.execute(
            sql.text(f'DROP TABLE IF EXISTS "{escaped}"')
        )


def parse_as_of_date(value: Optional[str]) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(dt.date.today())
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise RuntimeError(
            f"Invalid --as-of-date value: {value}. Use YYYY-MM-DD."
        )
    return pd.Timestamp(parsed).normalize()


def stable_rating_hash(frame: pd.DataFrame) -> str:
    required = ["team", "preseason_power_rating_points"]
    canonical = (
        frame[required]
        .copy()
        .sort_values("team")
        .reset_index(drop=True)
    )
    payload = canonical.to_csv(index=False, float_format="%.12f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_current_structural_hash(frame: pd.DataFrame) -> str:
    canonical = (
        frame[["team", "preseason_power_rating_points"]]
        .copy()
        .rename(
            columns={
                "preseason_power_rating_points": "power_rating_points"
            }
        )
        .sort_values("team")
        .reset_index(drop=True)
    )
    payload = canonical.to_csv(index=False, float_format="%.12f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def empty_process_model() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "term",
            "coefficient",
            "feature_mean",
            "feature_std",
            "clip_lower",
            "clip_upper",
            "historical_team_game_rows",
            "historical_unique_games",
            "weighted_training_rmse",
            "weighted_training_r_squared",
            "ridge_alpha",
            "calibration_season_decay",
            "calibration_seasons",
            "feature_version",
            "model_version",
            "date_imported",
        ]
    )


def normalize_team(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip().upper().replace(".", "")
    if not text:
        return None

    return TEAM_ALIASES.get(text, text)


def first_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def numeric_series(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).fillna(default)


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


def frame_from_any(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if hasattr(value, "to_pandas"):
        return value.to_pandas()

    return pd.DataFrame(value)


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )
    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = numerator / denominator.replace(0, np.nan)
    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def center_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    if values.empty:
        return values

    return values - values.mean()


def clip_and_recenter(
    series: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    values = values.clip(lower, upper)
    values = values - values.mean()
    values = values.clip(lower, upper)
    values = values - values.mean()

    return values


def empty_game_audit() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season",
            "week",
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "actual_home_margin",
            "capped_actual_home_margin",
            "process_home_margin",
            "process_model_residual",
            "recent_game_weight",
            "neutral_site",
            "schedule_source",
            "pbp_source",
            "form_version",
            "date_imported",
        ]
    )


def empty_team_game_audit() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "season",
            "week",
            "game_id",
            "game_date",
            "team",
            "opponent",
            "is_home",
            "home_indicator",
            "points_for",
            "points_against",
            "margin",
            "capped_margin",
            "off_plays",
            "off_epa_total",
            "off_epa_per_play",
            "off_success_rate",
            "off_pass_epa_per_play",
            "off_rush_epa_per_play",
            "off_early_down_epa_per_play",
            "off_explosive_rate",
            "off_turnover_rate",
            "off_sack_rate",
            "off_cpoe",
            "st_epa_total",
            "opp_off_epa_per_play",
            "opp_off_success_rate",
            "opp_off_pass_epa_per_play",
            "opp_off_rush_epa_per_play",
            "opp_off_early_down_epa_per_play",
            "opp_off_explosive_rate",
            "opp_off_turnover_rate",
            "opp_off_sack_rate",
            "opp_off_cpoe",
            "opp_st_epa_total",
            *PROCESS_FEATURES,
            "predicted_process_margin",
            "feature_version",
            "form_version",
            "date_imported",
        ]
    )


# =============================================================================
# PRESEASON PRIOR
# =============================================================================

def prepare_power_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["season", "team", "power_rating_points"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{PRESEASON_POWER_TABLE} is missing columns: {missing}"
        )

    frame = frame.copy()
    frame["season"] = numeric_series(frame, "season", SEASON).astype(int)
    frame = frame[frame["season"].eq(SEASON)].copy()
    frame["team"] = frame["team"].map(normalize_team)
    frame["preseason_power_rating_points"] = numeric_series(
        frame, "power_rating_points", 0.0
    )

    optional_columns = [
        "player_name",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_quality_adjusted_completeness",
        "median_historical_home_field",
        "historical_target_team_std",
        "build_id",
        "calibration_version",
        "selected_point_slope",
        "date_imported",
    ]
    for column in optional_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    keep = [
        "season",
        "team",
        "preseason_power_rating_points",
        *optional_columns,
    ]
    frame = frame[keep].drop_duplicates(subset=["team"], keep="first")

    if len(frame) != 32 or frame["team"].nunique() != 32:
        raise RuntimeError(
            f"{PRESEASON_POWER_TABLE} must contain 32 unique 2026 teams."
        )

    frame["preseason_power_rating_points"] = center_series(
        frame["preseason_power_rating_points"]
    )
    return frame.sort_values("team").reset_index(drop=True)


def load_current_structural_power(engine) -> pd.DataFrame:
    return prepare_power_frame(read_table(engine, PRESEASON_POWER_TABLE))


def load_or_create_preseason_snapshot(
    engine,
    refresh: bool = False,
    allow_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    current = load_current_structural_power(engine)

    if refresh and not allow_refresh:
        raise RuntimeError(
            "The preseason snapshot cannot be refreshed after any "
            "2026 regular-season game has completed."
        )

    snapshot_exists = table_exists(engine, PRESEASON_SNAPSHOT_TABLE)
    if snapshot_exists and not refresh:
        snapshot = read_table(engine, PRESEASON_SNAPSHOT_TABLE)
        required = {
            "season",
            "team",
            "preseason_power_rating_points",
            "snapshot_hash",
            "snapshot_created_at",
        }
        if not required.issubset(snapshot.columns):
            raise RuntimeError(
                f"{PRESEASON_SNAPSHOT_TABLE} exists but has an invalid schema."
            )
        snapshot["season"] = pd.to_numeric(
            snapshot["season"], errors="coerce"
        ).astype("Int64")
        snapshot = snapshot[snapshot["season"].eq(SEASON)].copy()
        snapshot["team"] = snapshot["team"].map(normalize_team)
        if len(snapshot) != 32 or snapshot["team"].nunique() != 32:
            raise RuntimeError(
                f"{PRESEASON_SNAPSHOT_TABLE} must contain 32 unique teams."
            )
        snapshot["preseason_power_rating_points"] = pd.to_numeric(
            snapshot["preseason_power_rating_points"], errors="coerce"
        )
        if snapshot["preseason_power_rating_points"].isna().any():
            raise RuntimeError("Preseason snapshot contains missing ratings.")
        expected_hash = stable_rating_hash(snapshot)
        stored_hashes = set(snapshot["snapshot_hash"].dropna().astype(str))
        if stored_hashes != {expected_hash}:
            raise RuntimeError(
                "Preseason snapshot hash validation failed; the immutable "
                "snapshot appears to have been altered."
            )
        source = "sqlite_immutable_snapshot"
    else:
        snapshot = current.copy()
        snapshot_hash = stable_rating_hash(snapshot)
        snapshot_created_at = dt.datetime.now().isoformat(timespec="seconds")
        snapshot["snapshot_hash"] = snapshot_hash
        snapshot["snapshot_created_at"] = snapshot_created_at
        snapshot["snapshot_build_id"] = BUILD_ID
        snapshot["snapshot_source_table"] = PRESEASON_POWER_TABLE
        snapshot.to_sql(
            PRESEASON_SNAPSHOT_TABLE,
            con=engine,
            if_exists="replace",
            index=False,
        )
        source = "created_from_current_power"

    current_build_ids = set(current["build_id"].dropna().astype(str))
    if current_build_ids != {EXPECTED_STRUCTURAL_POWER_BUILD_ID}:
        raise RuntimeError(
            "Current structural power build ID mismatch: "
            f"expected={EXPECTED_STRUCTURAL_POWER_BUILD_ID!r}, "
            f"found={sorted(current_build_ids)}."
        )
    current_structural_hash = stable_current_structural_hash(current)

    # Carry the current roster/depth-chart structural prior and its exact
    # lineage separately from the immutable preseason comparison snapshot.
    current_audit = current[
        [
            "team",
            "preseason_power_rating_points",
            "build_id",
            "calibration_version",
            "date_imported",
        ]
    ].rename(
        columns={
            "preseason_power_rating_points": (
                "current_structural_power_rating_points"
            ),
            "build_id": "current_structural_build_id",
            "calibration_version": "current_structural_version",
            "date_imported": "current_structural_date_imported",
        }
    )
    current_audit["current_structural_snapshot_hash"] = (
        current_structural_hash
    )
    snapshot = snapshot.merge(
        current_audit,
        on="team",
        how="left",
        validate="one_to_one",
    )
    snapshot["structural_drift_from_preseason"] = (
        snapshot["current_structural_power_rating_points"]
        - snapshot["preseason_power_rating_points"]
    )
    return snapshot.sort_values("team").reset_index(drop=True), source


# =============================================================================
# SCHEDULE LOADING
# =============================================================================

def load_schedules_external(
    seasons: list[int],
) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    try:
        import nflreadpy as nfl  # type: ignore

        if hasattr(nfl, "load_schedules"):
            loader = getattr(nfl, "load_schedules")
            attempts = [
                lambda: loader(seasons),
                lambda: loader(seasons=seasons),
            ]

            for attempt in attempts:
                try:
                    frame = frame_from_any(attempt())
                    if not frame.empty:
                        return (
                            frame,
                            "nflreadpy.load_schedules",
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"nflreadpy.load_schedules: {exc}"
                    )

    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nflreadpy: {exc}")

    try:
        import nfl_data_py as nfl  # type: ignore

        if hasattr(nfl, "import_schedules"):
            try:
                frame = frame_from_any(
                    nfl.import_schedules(seasons)
                )
                if not frame.empty:
                    return (
                        frame,
                        "nfl_data_py.import_schedules",
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"nfl_data_py.import_schedules: {exc}"
                )

    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nfl_data_py: {exc}")

    raise RuntimeError(
        "Unable to load NFL schedules externally. "
        + " | ".join(errors)
    )


def load_schedules_database(
    engine,
    seasons: list[int],
) -> tuple[pd.DataFrame, str]:
    candidates = [
        "nfl_schedule_2026",
        "nfl_schedule",
        "nfl_schedules",
        "nfl_games_2026",
        "nfl_games",
    ]

    for table_name in candidates:
        columns = table_columns(
            engine,
            table_name,
        )
        if not columns:
            continue

        if "season" not in columns:
            continue

        escaped = table_name.replace('"', '""')
        placeholders = ",".join(
            f":season_{index}"
            for index, _ in enumerate(seasons)
        )
        params = {
            f"season_{index}": int(season)
            for index, season in enumerate(seasons)
        }

        query = (
            f'SELECT * FROM "{escaped}" '
            f"WHERE CAST(season AS INTEGER) IN ({placeholders})"
        )

        with engine.connect() as connection:
            frame = pd.read_sql(
                sql.text(query),
                connection,
                params=params,
            )

        if not frame.empty:
            return frame, f"sqlite.{table_name}"

    raise RuntimeError(
        "No compatible NFL schedule table was found in SQLite."
    )


def load_schedules(
    engine,
    seasons: list[int],
) -> tuple[pd.DataFrame, str]:
    external_error: Optional[Exception] = None

    try:
        return load_schedules_external(seasons)
    except Exception as exc:  # noqa: BLE001
        external_error = exc

    try:
        return load_schedules_database(
            engine,
            seasons,
        )
    except Exception as database_exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to load schedules. "
            f"External error: {external_error}. "
            f"Database error: {database_exc}."
        ) from database_exc


def read_current_source_csv(path: Path, source_kind: str) -> tuple[pd.DataFrame, str]:
    """Read an explicit current-season input without inventing its data date.

    These are the same raw CSV inputs accepted by the weekly predictor. The
    existing standardizers and completed-game cutoff filters remain authoritative.
    An invalid explicit file fails; it never silently falls back to a download.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"The supplied {source_kind} CSV is empty: {path}")
    columns = {str(column).lower().strip(): column for column in frame.columns}
    season_column = columns.get("season")
    if season_column is None or not pd.to_numeric(
        frame[season_column], errors="coerce"
    ).eq(SEASON).any():
        raise RuntimeError(f"The supplied {source_kind} CSV has no {SEASON} season rows: {path}")
    return frame, f"csv:{path}"


def load_current_schedule(engine, path: Optional[Path] = None) -> tuple[pd.DataFrame, str]:
    if path is not None:
        return read_current_source_csv(path, "schedule")
    return load_schedules(engine, [SEASON])


def load_current_pbp(path: Optional[Path] = None) -> tuple[pd.DataFrame, str]:
    if path is not None:
        return read_current_source_csv(path, "play-by-play")
    return load_pbp_for_season(SEASON)


def standardize_schedules(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    season_col = first_existing(
        frame.columns,
        ["season"],
    )
    week_col = first_existing(
        frame.columns,
        ["week", "game_week"],
    )
    game_type_col = first_existing(
        frame.columns,
        ["game_type", "season_type"],
    )
    game_id_col = first_existing(
        frame.columns,
        ["game_id", "id", "gsis_id"],
    )
    game_date_col = first_existing(
        frame.columns,
        ["gameday", "game_date", "date"],
    )
    home_team_col = first_existing(
        frame.columns,
        ["home_team", "home"],
    )
    away_team_col = first_existing(
        frame.columns,
        ["away_team", "away"],
    )
    home_score_col = first_existing(
        frame.columns,
        ["home_score", "score_home"],
    )
    away_score_col = first_existing(
        frame.columns,
        ["away_score", "score_away"],
    )
    neutral_col = first_existing(
        frame.columns,
        ["neutral", "neutral_site", "location"],
    )

    required = {
        "season": season_col,
        "week": week_col,
        "home_team": home_team_col,
        "away_team": away_team_col,
        "home_score": home_score_col,
        "away_score": away_score_col,
    }
    missing = [
        name
        for name, column in required.items()
        if column is None
    ]
    if missing:
        raise RuntimeError(
            f"Schedule data is missing required fields: {missing}"
        )

    out = pd.DataFrame(index=frame.index)
    out["season"] = pd.to_numeric(
        frame[season_col],
        errors="coerce",
    )
    out["week"] = pd.to_numeric(
        frame[week_col],
        errors="coerce",
    )
    out["game_type"] = (
        frame[game_type_col]
        .astype(str)
        .str.upper()
        .str.strip()
        if game_type_col is not None
        else "REG"
    )
    out["home_team"] = frame[
        home_team_col
    ].map(normalize_team)
    out["away_team"] = frame[
        away_team_col
    ].map(normalize_team)
    out["home_score"] = pd.to_numeric(
        frame[home_score_col],
        errors="coerce",
    )
    out["away_score"] = pd.to_numeric(
        frame[away_score_col],
        errors="coerce",
    )

    if game_id_col is not None:
        out["game_id"] = (
            frame[game_id_col]
            .astype(str)
            .str.strip()
        )
    else:
        out["game_id"] = (
            out["season"].astype("Int64").astype(str)
            + "_"
            + out["week"].astype("Int64").astype(str)
            + "_"
            + out["away_team"].astype(str)
            + "_"
            + out["home_team"].astype(str)
        )

    if game_date_col is not None:
        out["game_date"] = pd.to_datetime(
            frame[game_date_col],
            errors="coerce",
        )
    else:
        out["game_date"] = pd.NaT

    out["neutral_site"] = 0
    if neutral_col is not None:
        raw_neutral = frame[neutral_col]

        if raw_neutral.dtype == bool:
            out["neutral_site"] = raw_neutral.astype(int)
        else:
            text = (
                raw_neutral
                .astype(str)
                .str.upper()
                .str.strip()
            )
            out["neutral_site"] = text.isin(
                {
                    "1",
                    "TRUE",
                    "YES",
                    "Y",
                    "NEUTRAL",
                }
            ).astype(int)

    out = out[
        out["game_type"].isin(GAME_TYPE_FILTER)
        & out["season"].notna()
        & out["week"].notna()
        & out["home_team"].notna()
        & out["away_team"].notna()
    ].copy()

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)

    out["actual_home_margin"] = (
        out["home_score"]
        - out["away_score"]
    )
    out["completed"] = (
        out["home_score"].notna()
        & out["away_score"].notna()
    ).astype(int)

    out = out.drop_duplicates(
        subset=["season", "game_id"],
        keep="last",
    )

    return out.reset_index(drop=True)



def filter_schedule_as_of(
    schedules: pd.DataFrame,
    as_of_date: pd.Timestamp,
    through_week: Optional[int],
) -> pd.DataFrame:
    frame = schedules.copy()
    mask = frame["completed"].eq(1)

    if through_week is not None:
        if through_week < 0:
            raise RuntimeError("--through-week must be at least 0.")
        mask &= frame["week"].le(int(through_week))

    # Completed games with a valid date must be no later than the cutoff.
    dated = frame["game_date"].notna()
    mask &= (~dated) | frame["game_date"].dt.normalize().le(as_of_date)

    filtered = frame[mask].copy()
    return filtered.sort_values(["week", "game_id"]).reset_index(drop=True)


def filter_pbp_to_completed_games(
    pbp: pd.DataFrame,
    completed_games: pd.DataFrame,
) -> pd.DataFrame:
    game_ids = set(completed_games["game_id"].dropna().astype(str))
    if not game_ids:
        return pbp.iloc[0:0].copy()
    frame = pbp[pbp["game_id"].astype(str).isin(game_ids)].copy()
    if "week" in frame.columns and completed_games["week"].notna().any():
        maximum_week = int(completed_games["week"].max())
        frame = frame[
            pd.to_numeric(frame["week"], errors="coerce").le(maximum_week)
        ].copy()
    return frame.reset_index(drop=True)


# =============================================================================
# PLAY-BY-PLAY LOADING
# =============================================================================

def load_pbp_for_season(
    season: int,
) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    try:
        import nfl_data_py as nfl  # type: ignore

        try:
            frame = frame_from_any(
                nfl.import_pbp_data(
                    [int(season)],
                    columns=PBP_COLUMNS,
                )
            )
            if not frame.empty:
                return (
                    frame,
                    "nfl_data_py.import_pbp_data",
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"nfl_data_py filtered load: {exc}"
            )

        try:
            frame = frame_from_any(
                nfl.import_pbp_data([int(season)])
            )
            if not frame.empty:
                return (
                    frame,
                    "nfl_data_py.import_pbp_data_full",
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"nfl_data_py full load: {exc}"
            )

    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nfl_data_py: {exc}")

    try:
        import nflreadpy as nfl  # type: ignore

        for function_name in [
            "load_pbp",
            "load_play_by_play",
            "load_pbp_data",
        ]:
            if not hasattr(nfl, function_name):
                continue

            loader = getattr(nfl, function_name)
            attempts = [
                lambda: loader([int(season)]),
                lambda: loader(seasons=[int(season)]),
                lambda: loader(int(season)),
                lambda: loader(seasons=int(season)),
            ]

            for attempt in attempts:
                try:
                    frame = frame_from_any(attempt())
                    if not frame.empty:
                        return (
                            frame,
                            f"nflreadpy.{function_name}",
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"nflreadpy.{function_name}: {exc}"
                    )

    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nflreadpy: {exc}")

    raise RuntimeError(
        f"Unable to load {season} play-by-play. "
        + " | ".join(errors)
    )


def standardize_pbp(
    raw: pd.DataFrame,
    requested_season: int,
) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    rename_map = {
        "posteam": "posteam",
        "defteam": "defteam",
        "pass_play": "pass",
        "rush_play": "rush",
        "fumble_lost_player_id": "fumble_lost",
    }
    frame = frame.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in frame.columns
            and new not in frame.columns
        }
    )

    for column in PBP_COLUMNS:
        ensure_column(
            frame,
            column,
            np.nan,
        )

    frame["season"] = pd.to_numeric(
        frame["season"],
        errors="coerce",
    ).fillna(requested_season)

    frame = frame[
        frame["season"].eq(requested_season)
    ].copy()

    frame["week"] = pd.to_numeric(
        frame["week"],
        errors="coerce",
    )
    frame["game_id"] = (
        frame["game_id"]
        .astype(str)
        .str.strip()
    )
    frame["posteam"] = frame[
        "posteam"
    ].map(normalize_team)
    frame["defteam"] = frame[
        "defteam"
    ].map(normalize_team)
    frame["home_team"] = frame[
        "home_team"
    ].map(normalize_team)
    frame["away_team"] = frame[
        "away_team"
    ].map(normalize_team)

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
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["play_type"] = (
        frame["play_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return frame


# =============================================================================
# TEAM-GAME FEATURE BUILDING
# =============================================================================

def aggregate_offense(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    frame = pbp.copy()

    regular_play_mask = (
        frame["posteam"].notna()
        & frame["defteam"].notna()
        & frame["game_id"].notna()
        & frame["epa"].notna()
        & numeric_series(frame, "no_play", 0.0).eq(0)
        & numeric_series(frame, "qb_kneel", 0.0).eq(0)
        & numeric_series(frame, "qb_spike", 0.0).eq(0)
    )

    special_mask = (
        numeric_series(
            frame,
            "special_teams_play",
            0.0,
        ).eq(1)
        | frame["play_type"].isin(
            {
                "punt",
                "kickoff",
                "field_goal",
                "extra_point",
            }
        )
    )

    offense = frame[
        regular_play_mask
        & ~special_mask
    ].copy()

    if offense.empty:
        return pd.DataFrame()

    offense["epa"] = offense["epa"].clip(-5.0, 5.0)
    offense["success_value"] = (
        offense["success"]
        .where(offense["success"].notna())
        .fillna((offense["epa"] > 0).astype(float))
    )

    offense["is_pass"] = (
        numeric_series(offense, "qb_dropback", 0.0).eq(1)
        | numeric_series(offense, "pass", 0.0).eq(1)
        | numeric_series(offense, "pass_attempt", 0.0).eq(1)
    ).astype(int)

    offense["is_rush"] = (
        numeric_series(offense, "rush", 0.0).eq(1)
        | numeric_series(offense, "rush_attempt", 0.0).eq(1)
    ).astype(int)

    offense["is_early_down"] = (
        numeric_series(offense, "down", np.nan).le(2)
    ).astype(int)

    yards = numeric_series(
        offense,
        "yards_gained",
        0.0,
    )
    offense["is_explosive"] = np.where(
        offense["is_pass"].eq(1),
        yards.ge(20),
        np.where(
            offense["is_rush"].eq(1),
            yards.ge(10),
            False,
        ),
    ).astype(int)

    offense["is_turnover"] = (
        numeric_series(
            offense,
            "interception",
            0.0,
        ).eq(1)
        | numeric_series(
            offense,
            "fumble_lost",
            0.0,
        ).eq(1)
    ).astype(int)

    offense["is_sack"] = numeric_series(
        offense,
        "sack",
        0.0,
    ).eq(1).astype(int)

    group_columns = [
        "season",
        "week",
        "game_id",
        "posteam",
        "defteam",
    ]

    rows: list[dict[str, Any]] = []

    for keys, group in offense.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        season, week, game_id, team, opponent = keys

        pass_group = group[
            group["is_pass"].eq(1)
        ]
        rush_group = group[
            group["is_rush"].eq(1)
        ]
        early_group = group[
            group["is_early_down"].eq(1)
        ]
        cpoe_values = pd.to_numeric(
            pass_group["cpoe"],
            errors="coerce",
        ).dropna()

        dropbacks = int(
            group["is_pass"].sum()
        )
        sacks = int(
            group["is_sack"].sum()
        )

        rows.append(
            {
                "season": int(season),
                "week": int(week)
                if pd.notna(week)
                else np.nan,
                "game_id": str(game_id),
                "team": team,
                "opponent": opponent,
                "off_plays": int(len(group)),
                "off_epa_total": float(
                    group["epa"].sum()
                ),
                "off_epa_per_play": float(
                    group["epa"].mean()
                ),
                "off_success_rate": float(
                    group["success_value"].mean()
                ),
                "off_dropbacks": dropbacks,
                "off_pass_epa_per_play": (
                    float(pass_group["epa"].mean())
                    if not pass_group.empty
                    else 0.0
                ),
                "off_rushes": int(
                    group["is_rush"].sum()
                ),
                "off_rush_epa_per_play": (
                    float(rush_group["epa"].mean())
                    if not rush_group.empty
                    else 0.0
                ),
                "off_early_down_epa_per_play": (
                    float(early_group["epa"].mean())
                    if not early_group.empty
                    else 0.0
                ),
                "off_explosive_rate": float(
                    group["is_explosive"].mean()
                ),
                "off_turnover_rate": float(
                    group["is_turnover"].mean()
                ),
                "off_sack_rate": (
                    float(sacks / dropbacks)
                    if dropbacks > 0
                    else 0.0
                ),
                "off_cpoe": (
                    float(cpoe_values.mean())
                    if not cpoe_values.empty
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_special_teams(
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    frame = pbp.copy()

    special_mask = (
        frame["posteam"].notna()
        & frame["game_id"].notna()
        & frame["epa"].notna()
        & (
            numeric_series(
                frame,
                "special_teams_play",
                0.0,
            ).eq(1)
            | frame["play_type"].isin(
                {
                    "punt",
                    "kickoff",
                    "field_goal",
                    "extra_point",
                }
            )
        )
        & numeric_series(
            frame,
            "no_play",
            0.0,
        ).eq(0)
    )

    special = frame[
        special_mask
    ].copy()

    if special.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "game_id",
                "team",
                "st_plays",
                "st_epa_total",
            ]
        )

    special["epa"] = special["epa"].clip(
        -6.0,
        6.0,
    )

    return (
        special.groupby(
            [
                "season",
                "week",
                "game_id",
                "posteam",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            st_plays=("epa", "size"),
            st_epa_total=("epa", "sum"),
        )
        .rename(columns={"posteam": "team"})
    )


def build_schedule_team_rows(
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    completed = schedules[
        schedules["completed"].eq(1)
    ].copy()

    if completed.empty:
        return pd.DataFrame()

    home = pd.DataFrame(
        {
            "season": completed["season"],
            "week": completed["week"],
            "game_id": completed["game_id"],
            "game_date": completed["game_date"],
            "team": completed["home_team"],
            "opponent": completed["away_team"],
            "is_home": 1,
            "home_indicator": np.where(
                completed["neutral_site"].eq(1),
                0.0,
                1.0,
            ),
            "points_for": completed["home_score"],
            "points_against": completed["away_score"],
            "neutral_site": completed["neutral_site"],
        }
    )

    away = pd.DataFrame(
        {
            "season": completed["season"],
            "week": completed["week"],
            "game_id": completed["game_id"],
            "game_date": completed["game_date"],
            "team": completed["away_team"],
            "opponent": completed["home_team"],
            "is_home": 0,
            "home_indicator": np.where(
                completed["neutral_site"].eq(1),
                0.0,
                -1.0,
            ),
            "points_for": completed["away_score"],
            "points_against": completed["home_score"],
            "neutral_site": completed["neutral_site"],
        }
    )

    rows = pd.concat(
        [home, away],
        ignore_index=True,
    )
    rows["margin"] = (
        rows["points_for"]
        - rows["points_against"]
    )
    rows["capped_margin"] = rows[
        "margin"
    ].clip(
        -RESULT_MARGIN_CAP,
        RESULT_MARGIN_CAP,
    )

    return rows


def build_team_game_features(
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    schedule_rows = build_schedule_team_rows(
        schedules
    )
    if schedule_rows.empty:
        return empty_team_game_audit()

    offense = aggregate_offense(pbp)
    special = aggregate_special_teams(pbp)

    if offense.empty:
        return empty_team_game_audit()

    offense = offense.merge(
        special,
        on=[
            "season",
            "week",
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )
    offense["st_plays"] = numeric_series(
        offense,
        "st_plays",
        0.0,
    )
    offense["st_epa_total"] = numeric_series(
        offense,
        "st_epa_total",
        0.0,
    )

    opponent_columns = {
        column: f"opp_{column}"
        for column in offense.columns
        if column.startswith("off_")
        or column.startswith("st_")
    }

    opponent = offense.rename(
        columns={
            "team": "opponent",
            "opponent": "team",
            **opponent_columns,
        }
    )

    opponent_keep = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        *opponent_columns.values(),
    ]

    combined = offense.merge(
        opponent[opponent_keep],
        on=[
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
        ],
        how="inner",
        validate="one_to_one",
    )

    rows = schedule_rows.merge(
        combined,
        on=[
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
        ],
        how="left",
        validate="one_to_one",
    )

    rows["net_epa_per_play"] = (
        rows["off_epa_per_play"]
        - rows["opp_off_epa_per_play"]
    )
    rows["net_success_rate"] = (
        rows["off_success_rate"]
        - rows["opp_off_success_rate"]
    )
    rows["net_pass_epa_per_play"] = (
        rows["off_pass_epa_per_play"]
        - rows["opp_off_pass_epa_per_play"]
    )
    rows["net_rush_epa_per_play"] = (
        rows["off_rush_epa_per_play"]
        - rows["opp_off_rush_epa_per_play"]
    )
    rows["net_early_down_epa_per_play"] = (
        rows["off_early_down_epa_per_play"]
        - rows["opp_off_early_down_epa_per_play"]
    )
    rows["net_explosive_rate"] = (
        rows["off_explosive_rate"]
        - rows["opp_off_explosive_rate"]
    )
    rows["turnover_margin_rate"] = (
        rows["opp_off_turnover_rate"]
        - rows["off_turnover_rate"]
    )
    rows["net_sack_rate"] = (
        rows["opp_off_sack_rate"]
        - rows["off_sack_rate"]
    )
    rows["net_cpoe"] = (
        rows["off_cpoe"]
        - rows["opp_off_cpoe"]
    )
    rows["st_epa_diff"] = (
        rows["st_epa_total"]
        - rows["opp_st_epa_total"]
    )

    for column in PROCESS_FEATURES:
        rows[column] = pd.to_numeric(
            rows[column],
            errors="coerce",
        )

    rows["feature_version"] = FEATURE_VERSION

    return rows


# =============================================================================
# HISTORICAL PROCESS MODEL
# =============================================================================

def historical_cache_is_valid(
    frame: pd.DataFrame,
) -> bool:
    if frame.empty:
        return False

    required = {
        "season",
        "game_id",
        "team",
        "margin",
        "feature_version",
        *PROCESS_FEATURES,
    }
    if not required.issubset(frame.columns):
        return False

    seasons = set(
        pd.to_numeric(
            frame["season"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    if not set(CALIBRATION_SEASONS).issubset(seasons):
        return False

    versions = set(
        frame["feature_version"]
        .dropna()
        .astype(str)
    )
    return versions == {FEATURE_VERSION}


def load_or_build_historical_features(
    engine,
    schedules: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if table_exists(
        engine,
        HISTORICAL_FEATURE_CACHE_TABLE,
    ):
        cached = read_table(
            engine,
            HISTORICAL_FEATURE_CACHE_TABLE,
        )
        if historical_cache_is_valid(cached):
            LOGGER.info(
                "[FORM] Loaded historical process-feature cache: "
                "%s rows",
                f"{len(cached):,}",
            )
            return cached, "sqlite_cache"

    frames: list[pd.DataFrame] = []
    source_names: list[str] = []

    for season in CALIBRATION_SEASONS:
        season_schedule = schedules[
            schedules["season"].eq(season)
            & schedules["completed"].eq(1)
        ].copy()

        if season_schedule.empty:
            raise RuntimeError(
                f"No completed schedule rows for calibration season {season}."
            )

        LOGGER.info(
            "[FORM] Loading historical PBP season %s...",
            season,
        )

        raw_pbp, source = load_pbp_for_season(
            season
        )
        pbp = standardize_pbp(
            raw_pbp,
            season,
        )
        features = build_team_game_features(
            pbp,
            season_schedule,
        )

        if features.empty:
            raise RuntimeError(
                f"No process features were built for {season}."
            )

        features["pbp_source"] = source
        frames.append(features)
        source_names.append(source)

        LOGGER.info(
            "[FORM] Historical %s team-game rows: %s",
            season,
            f"{len(features):,}",
        )

        del raw_pbp
        del pbp

    historical = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    historical["feature_version"] = FEATURE_VERSION
    historical.to_sql(
        HISTORICAL_FEATURE_CACHE_TABLE,
        con=engine,
        if_exists="replace",
        index=False,
    )

    LOGGER.info(
        "[FORM] Saved historical process-feature cache: %s rows",
        f"{len(historical):,}",
    )

    return (
        historical,
        ",".join(sorted(set(source_names))),
    )


def load_available_process_model(engine) -> tuple[pd.DataFrame, str, str]:
    """Reuse a validated fitted model before requesting unused historical data."""
    if table_exists(engine, PROCESS_MODEL_TABLE):
        cached = read_table(engine, PROCESS_MODEL_TABLE)
        if model_cache_is_valid(cached):
            LOGGER.info("[FORM] Loaded cached process model; historical schedule not required.")
            return cached, "sqlite_cache", "not_required_cached_process_model"
    raw_schedule, schedule_source = load_schedules(engine, CALIBRATION_SEASONS)
    model, model_source = load_or_fit_process_model(
        engine, standardize_schedules(raw_schedule)
    )
    return model, model_source, schedule_source


def model_cache_is_valid(
    model: pd.DataFrame,
) -> bool:
    if model.empty:
        return False

    required = {
        "term",
        "coefficient",
        "model_version",
        "feature_version",
        "calibration_seasons",
    }
    if not required.issubset(model.columns):
        return False

    if set(
        model["model_version"]
        .dropna()
        .astype(str)
    ) != {PROCESS_MODEL_VERSION}:
        return False

    if set(
        model["feature_version"]
        .dropna()
        .astype(str)
    ) != {FEATURE_VERSION}:
        return False

    expected_seasons = ",".join(
        map(str, CALIBRATION_SEASONS)
    )
    if set(
        model["calibration_seasons"]
        .dropna()
        .astype(str)
    ) != {expected_seasons}:
        return False

    terms = set(
        model["term"]
        .dropna()
        .astype(str)
    )
    needed = {
        "intercept",
        "home_field",
        *PROCESS_FEATURES,
    }

    return needed.issubset(terms)


def weighted_ridge_fit(
    design: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
    unpenalized_columns: set[int],
) -> np.ndarray:
    sqrt_weight = np.sqrt(
        np.clip(
            sample_weight.astype(float),
            1e-9,
            None,
        )
    )

    weighted_design = (
        design
        * sqrt_weight[:, None]
    )
    weighted_target = (
        target
        * sqrt_weight
    )

    penalty = np.eye(
        design.shape[1],
        dtype=float,
    ) * float(alpha)

    for column in unpenalized_columns:
        penalty[column, column] = 0.0

    system = (
        weighted_design.T @ weighted_design
        + penalty
    )
    rhs = (
        weighted_design.T @ weighted_target
    )

    return np.linalg.solve(system, rhs)


def fit_process_model(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    frame = historical.copy()

    frame = frame[
        frame["margin"].notna()
    ].copy()

    if len(frame) < MIN_HISTORICAL_GAMES * 2:
        raise RuntimeError(
            "Insufficient historical team-game rows for process calibration: "
            f"{len(frame):,}"
        )

    model_rows: list[dict[str, Any]] = []

    transformed = pd.DataFrame(
        index=frame.index,
    )

    for feature in PROCESS_FEATURES:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

        median = float(values.median())
        values = values.fillna(median)

        lower = float(
            values.quantile(0.01)
        )
        upper = float(
            values.quantile(0.99)
        )
        clipped = values.clip(
            lower,
            upper,
        )

        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-9:
            std = 1.0

        transformed[feature] = (
            clipped - mean
        ) / std

        model_rows.append(
            {
                "term": feature,
                "feature_mean": mean,
                "feature_std": std,
                "clip_lower": lower,
                "clip_upper": upper,
            }
        )

    design = np.column_stack(
        [
            np.ones(len(frame)),
            pd.to_numeric(
                frame["home_indicator"],
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float),
            *[
                transformed[feature].to_numpy(dtype=float)
                for feature in PROCESS_FEATURES
            ],
        ]
    )

    target = pd.to_numeric(
        frame["capped_margin"],
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    season_values = pd.to_numeric(
        frame["season"],
        errors="coerce",
    ).fillna(
        min(CALIBRATION_SEASONS)
    ).to_numpy(dtype=float)

    latest_season = max(
        CALIBRATION_SEASONS
    )
    sample_weight = np.power(
        CALIBRATION_SEASON_DECAY,
        latest_season - season_values,
    )

    coefficients = weighted_ridge_fit(
        design=design,
        target=target,
        sample_weight=sample_weight,
        alpha=PROCESS_RIDGE_ALPHA,
        unpenalized_columns={0, 1},
    )

    fitted = design @ coefficients
    residual = target - fitted
    rmse = float(
        np.sqrt(
            np.average(
                np.square(residual),
                weights=sample_weight,
            )
        )
    )

    weighted_mean_target = float(
        np.average(
            target,
            weights=sample_weight,
        )
    )
    total_variance = float(
        np.average(
            np.square(
                target - weighted_mean_target
            ),
            weights=sample_weight,
        )
    )
    residual_variance = float(
        np.average(
            np.square(residual),
            weights=sample_weight,
        )
    )
    r_squared = (
        1.0 - residual_variance / total_variance
        if total_variance > 1e-12
        else np.nan
    )

    now = dt.datetime.now().isoformat(
        timespec="seconds"
    )
    calibration_seasons = ",".join(
        map(str, CALIBRATION_SEASONS)
    )

    output_rows: list[dict[str, Any]] = [
        {
            "term": "intercept",
            "coefficient": float(
                coefficients[0]
            ),
            "feature_mean": 0.0,
            "feature_std": 1.0,
            "clip_lower": np.nan,
            "clip_upper": np.nan,
        },
        {
            "term": "home_field",
            "coefficient": float(
                coefficients[1]
            ),
            "feature_mean": 0.0,
            "feature_std": 1.0,
            "clip_lower": np.nan,
            "clip_upper": np.nan,
        },
    ]

    for index, row in enumerate(
        model_rows,
        start=2,
    ):
        output_rows.append(
            {
                **row,
                "coefficient": float(
                    coefficients[index]
                ),
            }
        )

    model = pd.DataFrame(output_rows)
    model["historical_team_game_rows"] = len(frame)
    model["historical_unique_games"] = frame[
        "game_id"
    ].nunique()
    model["weighted_training_rmse"] = rmse
    model["weighted_training_r_squared"] = r_squared
    model["ridge_alpha"] = PROCESS_RIDGE_ALPHA
    model["calibration_season_decay"] = (
        CALIBRATION_SEASON_DECAY
    )
    model["calibration_seasons"] = (
        calibration_seasons
    )
    model["feature_version"] = FEATURE_VERSION
    model["model_version"] = (
        PROCESS_MODEL_VERSION
    )
    model["date_imported"] = now

    return model


def load_or_fit_process_model(
    engine,
    schedules: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if table_exists(
        engine,
        PROCESS_MODEL_TABLE,
    ):
        cached = read_table(
            engine,
            PROCESS_MODEL_TABLE,
        )
        if model_cache_is_valid(cached):
            LOGGER.info(
                "[FORM] Loaded cached process model."
            )
            return cached, "sqlite_cache"

    historical, historical_source = (
        load_or_build_historical_features(
            engine,
            schedules,
        )
    )
    model = fit_process_model(
        historical
    )

    model.to_sql(
        PROCESS_MODEL_TABLE,
        con=engine,
        if_exists="replace",
        index=False,
    )

    LOGGER.info(
        "[FORM] Fitted process model. "
        "Rows=%s RMSE=%.3f R2=%.3f",
        f"{int(model['historical_team_game_rows'].iloc[0]):,}",
        float(
            model["weighted_training_rmse"].iloc[0]
        ),
        float(
            model["weighted_training_r_squared"].iloc[0]
        ),
    )

    return model, historical_source


def apply_process_model(
    team_games: pd.DataFrame,
    model: pd.DataFrame,
) -> pd.DataFrame:
    frame = team_games.copy()

    coefficient_lookup = (
        model.set_index("term")[
            "coefficient"
        ].to_dict()
    )
    mean_lookup = (
        model.set_index("term")[
            "feature_mean"
        ].to_dict()
    )
    std_lookup = (
        model.set_index("term")[
            "feature_std"
        ].to_dict()
    )
    lower_lookup = (
        model.set_index("term")[
            "clip_lower"
        ].to_dict()
    )
    upper_lookup = (
        model.set_index("term")[
            "clip_upper"
        ].to_dict()
    )

    prediction = pd.Series(
        float(
            coefficient_lookup.get(
                "intercept",
                0.0,
            )
        ),
        index=frame.index,
        dtype=float,
    )

    prediction = prediction + (
        pd.to_numeric(
            frame["home_indicator"],
            errors="coerce",
        ).fillna(0.0)
        * float(
            coefficient_lookup.get(
                "home_field",
                0.0,
            )
        )
    )

    for feature in PROCESS_FEATURES:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

        mean = float(
            mean_lookup.get(
                feature,
                0.0,
            )
        )
        std = float(
            std_lookup.get(
                feature,
                1.0,
            )
        )
        if not np.isfinite(std) or std <= 1e-9:
            std = 1.0

        lower = lower_lookup.get(
            feature,
            np.nan,
        )
        upper = upper_lookup.get(
            feature,
            np.nan,
        )

        values = values.fillna(mean)

        if pd.notna(lower):
            values = values.clip(
                lower=float(lower),
            )
        if pd.notna(upper):
            values = values.clip(
                upper=float(upper),
            )

        standardized = (
            values - mean
        ) / std

        prediction = prediction + (
            standardized
            * float(
                coefficient_lookup.get(
                    feature,
                    0.0,
                )
            )
        )

    frame["predicted_process_margin"] = (
        prediction.clip(
            -PROCESS_MARGIN_CAP,
            PROCESS_MARGIN_CAP,
        )
    )

    return frame


# =============================================================================
# OPPONENT-ADJUSTED SRS
# =============================================================================

def solve_srs(
    games: pd.DataFrame,
    teams: list[str],
    target_column: str,
    sample_weight: Optional[pd.Series],
    ridge_alpha: float,
) -> tuple[pd.Series, float]:
    if games.empty:
        return (
            pd.Series(
                0.0,
                index=teams,
                dtype=float,
            ),
            0.0,
        )

    frame = games[
        [
            "home_team",
            "away_team",
            "neutral_site",
            target_column,
        ]
    ].copy()

    frame[target_column] = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )

    frame = frame[
        frame[target_column].notna()
        & frame["home_team"].isin(teams)
        & frame["away_team"].isin(teams)
    ].copy()

    if frame.empty:
        return (
            pd.Series(
                0.0,
                index=teams,
                dtype=float,
            ),
            0.0,
        )

    team_index = {
        team: index
        for index, team in enumerate(teams)
    }

    design = np.zeros(
        (len(frame), len(teams) + 1),
        dtype=float,
    )

    for row_number, row in enumerate(
        frame.itertuples(index=False)
    ):
        design[
            row_number,
            team_index[row.home_team],
        ] = 1.0
        design[
            row_number,
            team_index[row.away_team],
        ] = -1.0
        design[
            row_number,
            len(teams),
        ] = (
            0.0
            if int(row.neutral_site) == 1
            else 1.0
        )

    target = frame[
        target_column
    ].to_numpy(dtype=float)

    if sample_weight is None:
        weights = np.ones(
            len(frame),
            dtype=float,
        )
    else:
        weights = pd.to_numeric(
            sample_weight.loc[frame.index],
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float)

    coefficients = weighted_ridge_fit(
        design=design,
        target=target,
        sample_weight=weights,
        alpha=ridge_alpha,
        unpenalized_columns={len(teams)},
    )

    ratings = pd.Series(
        coefficients[:len(teams)],
        index=teams,
        dtype=float,
    )
    ratings = ratings - ratings.mean()

    home_field = float(
        coefficients[len(teams)]
    )

    return ratings, home_field


# =============================================================================
# CURRENT-SEASON FORM
# =============================================================================

def build_game_audit(
    schedules: pd.DataFrame,
    team_games: pd.DataFrame,
    schedule_source: str,
    pbp_source: str,
) -> pd.DataFrame:
    completed = schedules[
        schedules["completed"].eq(1)
    ].copy()

    if completed.empty:
        return empty_game_audit()

    home_process = team_games[
        team_games["is_home"].eq(1)
    ][
        [
            "season",
            "week",
            "game_id",
            "predicted_process_margin",
        ]
    ].rename(
        columns={
            "predicted_process_margin": (
                "process_home_margin"
            )
        }
    )

    audit = completed.merge(
        home_process,
        on=[
            "season",
            "week",
            "game_id",
        ],
        how="left",
        validate="one_to_one",
    )

    audit["capped_actual_home_margin"] = (
        audit["actual_home_margin"]
        .clip(
            -RESULT_MARGIN_CAP,
            RESULT_MARGIN_CAP,
        )
    )
    audit["process_model_residual"] = (
        audit["capped_actual_home_margin"]
        - audit["process_home_margin"]
    )

    max_week = int(
        audit["week"].max()
    )
    audit["recent_game_weight"] = np.power(
        RECENT_WEEK_DECAY,
        max_week - audit["week"],
    )

    audit["schedule_source"] = (
        schedule_source
    )
    audit["pbp_source"] = pbp_source
    audit["form_version"] = VERSION
    audit["date_imported"] = (
        dt.datetime.now().isoformat(
            timespec="seconds"
        )
    )

    columns = [
        "season",
        "week",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_home_margin",
        "capped_actual_home_margin",
        "process_home_margin",
        "process_model_residual",
        "recent_game_weight",
        "neutral_site",
        "schedule_source",
        "pbp_source",
        "form_version",
        "date_imported",
    ]

    return audit[columns].sort_values(
        ["week", "game_id"]
    ).reset_index(drop=True)


def build_no_games_output(
    preseason: pd.DataFrame,
    schedule_source: str,
    snapshot_source: str,
    as_of_date: pd.Timestamp,
    through_week: Optional[int] = None,
) -> pd.DataFrame:
    frame = preseason.copy()

    # Report the requested cutoff even when it contains no completed games.
    frame["through_week"] = 0 if through_week is None else int(through_week)
    frame["games_played"] = 0
    frame["process_games_played"] = 0
    frame["result_games_played"] = 0
    frame["process_game_coverage"] = 0.0
    frame["latest_game_date"] = pd.NaT

    frame["current_season_weight"] = 0.0
    frame["structural_prior_weight"] = 1.0

    for column in [
        "season_process_rating",
        "season_result_rating",
        "season_form_rating",
        "recent_process_rating",
        "recent_result_rating",
        "recent_form_rating",
        "form_rating_points",
        "form_adjustment_points",
    ]:
        frame[column] = 0.0

    frame["power_rating_points"] = frame[
        "current_structural_power_rating_points"
    ]
    frame["points_above_average"] = frame["power_rating_points"]
    frame["neutral_field_rating"] = frame["power_rating_points"]
    frame["power_rating_rank"] = (
        frame["power_rating_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    for column in [
        "season_process_hfa",
        "season_result_hfa",
        "recent_process_hfa",
        "recent_result_hfa",
    ]:
        frame[column] = 0.0

    frame["process_model_available"] = 0
    frame["completed_regular_season_games"] = 0
    frame["schedule_source"] = schedule_source
    frame["pbp_source"] = None
    frame["process_model_source"] = None
    frame["preseason_snapshot_source"] = snapshot_source
    frame["as_of_date"] = as_of_date.date().isoformat()
    frame["no_lookahead_filter_applied_flag"] = 1
    frame["incremental_weight_formula"] = (
        "games_played/(games_played+5), capped at 0.75"
    )
    frame["form_formula"] = (
        "week0=current structural prior; after games, "
        "structural_prior_weight*current structural power + "
        "current_season_weight*(70% season form + 30% recent form); "
        "each form component 70% process + 30% capped results"
    )
    frame["build_id"] = BUILD_ID
    frame["form_version"] = VERSION
    frame["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")
    return finalize_output_columns(frame)


def build_live_form_output(
    preseason: pd.DataFrame,
    schedules: pd.DataFrame,
    team_games: pd.DataFrame,
    game_audit: pd.DataFrame,
    teams: list[str],
    schedule_source: str,
    pbp_source: str,
    process_model_source: str,
    snapshot_source: str,
    as_of_date: pd.Timestamp,
    through_week: Optional[int] = None,
) -> pd.DataFrame:
    season_process, season_process_hfa = solve_srs(
        games=game_audit,
        teams=teams,
        target_column="process_home_margin",
        sample_weight=None,
        ridge_alpha=SEASON_SRS_RIDGE_ALPHA,
    )
    season_result, season_result_hfa = solve_srs(
        games=game_audit,
        teams=teams,
        target_column="capped_actual_home_margin",
        sample_weight=None,
        ridge_alpha=SEASON_SRS_RIDGE_ALPHA,
    )
    recent_process, recent_process_hfa = solve_srs(
        games=game_audit,
        teams=teams,
        target_column="process_home_margin",
        sample_weight=game_audit["recent_game_weight"],
        ridge_alpha=RECENT_SRS_RIDGE_ALPHA,
    )
    recent_result, recent_result_hfa = solve_srs(
        games=game_audit,
        teams=teams,
        target_column="capped_actual_home_margin",
        sample_weight=game_audit["recent_game_weight"],
        ridge_alpha=RECENT_SRS_RIDGE_ALPHA,
    )

    season_form = PROCESS_WEIGHT * season_process + RESULT_WEIGHT * season_result
    recent_form = PROCESS_WEIGHT * recent_process + RESULT_WEIGHT * recent_result
    form_rating = (
        SEASON_FORM_WEIGHT * season_form
        + RECENT_FORM_WEIGHT * recent_form
    )
    form_rating = clip_and_recenter(
        form_rating, -FORM_RATING_CAP, FORM_RATING_CAP
    )

    completed = schedules[schedules["completed"].eq(1)].copy()
    home_counts = completed.groupby("home_team").size().rename("home_games")
    away_counts = completed.groupby("away_team").size().rename("away_games")
    games_played = (
        home_counts.add(away_counts, fill_value=0)
        .reindex(teams, fill_value=0)
        .astype(int)
    )

    process_games = game_audit[game_audit["process_home_margin"].notna()].copy()
    process_home = process_games.groupby("home_team").size()
    process_away = process_games.groupby("away_team").size()
    process_games_played = (
        process_home.add(process_away, fill_value=0)
        .reindex(teams, fill_value=0)
        .astype(int)
    )

    latest_home = completed.groupby("home_team")["game_date"].max()
    latest_away = completed.groupby("away_team")["game_date"].max()
    latest_game_date = pd.concat([latest_home, latest_away], axis=1).max(axis=1)
    latest_game_date = latest_game_date.reindex(teams)

    current_weight = (
        games_played.astype(float)
        / (games_played.astype(float) + EFFECTIVE_PRIOR_GAMES)
    ).clip(upper=MAX_CURRENT_SEASON_WEIGHT)

    frame = preseason.copy().set_index("team")
    # A cutoff can include a week with no games; it remains the run's cutoff.
    frame["through_week"] = (
        int(completed["week"].max()) if through_week is None else int(through_week)
    )
    frame["games_played"] = games_played
    frame["result_games_played"] = games_played
    frame["process_games_played"] = process_games_played
    frame["process_game_coverage"] = safe_divide(
        process_games_played.astype(float),
        games_played.astype(float),
    ).fillna(0.0)
    frame["latest_game_date"] = latest_game_date
    frame["current_season_weight"] = current_weight
    frame["structural_prior_weight"] = 1.0 - current_weight

    frame["season_process_rating"] = season_process.reindex(teams)
    frame["season_result_rating"] = season_result.reindex(teams)
    frame["season_form_rating"] = season_form.reindex(teams)
    frame["recent_process_rating"] = recent_process.reindex(teams)
    frame["recent_result_rating"] = recent_result.reindex(teams)
    frame["recent_form_rating"] = recent_form.reindex(teams)
    frame["form_rating_points"] = form_rating.reindex(teams)

    live_raw = (
        frame["structural_prior_weight"]
        * frame["current_structural_power_rating_points"]
        + frame["current_season_weight"]
        * frame["form_rating_points"]
    )
    live = clip_and_recenter(live_raw, -LIVE_POWER_CAP, LIVE_POWER_CAP)

    frame["power_rating_points"] = live
    frame["points_above_average"] = live
    frame["neutral_field_rating"] = live
    frame["form_adjustment_points"] = (
        frame["power_rating_points"]
        - frame["current_structural_power_rating_points"]
    )
    frame["power_rating_rank"] = (
        frame["power_rating_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    frame["season_process_hfa"] = season_process_hfa
    frame["season_result_hfa"] = season_result_hfa
    frame["recent_process_hfa"] = recent_process_hfa
    frame["recent_result_hfa"] = recent_result_hfa
    frame["process_model_available"] = 1
    frame["completed_regular_season_games"] = completed["game_id"].nunique()
    frame["schedule_source"] = schedule_source
    frame["pbp_source"] = pbp_source
    frame["process_model_source"] = process_model_source
    frame["preseason_snapshot_source"] = snapshot_source
    frame["as_of_date"] = as_of_date.date().isoformat()
    frame["no_lookahead_filter_applied_flag"] = 1
    frame["incremental_weight_formula"] = (
        "games_played/(games_played+5), capped at 0.75"
    )
    frame["form_formula"] = (
        "structural_prior_weight*current structural power + "
        "current_season_weight*(70% season form + 30% recent form); "
        "each form component 70% process + 30% capped results"
    )
    frame["build_id"] = BUILD_ID
    frame["form_version"] = VERSION
    frame["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")
    return finalize_output_columns(frame.reset_index())


def finalize_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "season",
        "through_week",
        "team",
        "power_rating_rank",
        "power_rating_points",
        "points_above_average",
        "neutral_field_rating",
        "preseason_power_rating_points",
        "current_structural_power_rating_points",
        "structural_drift_from_preseason",
        "current_structural_build_id",
        "current_structural_version",
        "current_structural_date_imported",
        "current_structural_snapshot_hash",
        "form_rating_points",
        "form_adjustment_points",
        "games_played",
        "process_games_played",
        "result_games_played",
        "process_game_coverage",
        "current_season_weight",
        "structural_prior_weight",
        "season_process_rating",
        "season_result_rating",
        "season_form_rating",
        "recent_process_rating",
        "recent_result_rating",
        "recent_form_rating",
        "latest_game_date",
        "player_name",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_quality_adjusted_completeness",
        "median_historical_home_field",
        "historical_target_team_std",
        "season_process_hfa",
        "season_result_hfa",
        "recent_process_hfa",
        "recent_result_hfa",
        "process_model_available",
        "completed_regular_season_games",
        "schedule_source",
        "pbp_source",
        "process_model_source",
        "preseason_snapshot_source",
        "snapshot_hash",
        "snapshot_created_at",
        "as_of_date",
        "no_lookahead_filter_applied_flag",
        "incremental_weight_formula",
        "form_formula",
        "build_id",
        "form_version",
        "date_imported",
    ]
    for column in output_columns:
        ensure_column(frame, column, np.nan)
    return (
        frame[output_columns]
        .sort_values("power_rating_rank")
        .reset_index(drop=True)
    )


# =============================================================================
# VALIDATION, OUTPUT AND REPORTING
# =============================================================================

def validate_output(
    form: pd.DataFrame,
    game_audit: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> None:
    if len(form) != 32 or form["team"].nunique() != 32:
        raise RuntimeError("Form output must contain 32 unique teams.")

    required_numeric = [
        "power_rating_points",
        "preseason_power_rating_points",
        "current_structural_power_rating_points",
        "current_season_weight",
        "structural_prior_weight",
        "games_played",
        "process_games_played",
        "result_games_played",
    ]
    for column in required_numeric:
        values = pd.to_numeric(form[column], errors="coerce")
        if values.isna().any():
            raise RuntimeError(
                f"{column} contains {int(values.isna().sum())} missing values."
            )

    if abs(float(form["power_rating_points"].mean())) > 1e-8:
        raise RuntimeError("Live power ratings are not centered at zero.")
    if form["current_season_weight"].min() < -1e-9:
        raise RuntimeError("Current-season weight is negative.")
    if form["current_season_weight"].max() > MAX_CURRENT_SEASON_WEIGHT + 1e-9:
        raise RuntimeError("Current-season weight exceeds its configured cap.")
    if not np.allclose(
        form["structural_prior_weight"],
        1.0 - form["current_season_weight"],
        atol=1e-10,
    ):
        raise RuntimeError("Prior and current-season weights do not reconcile.")
    if form["power_rating_points"].abs().max() > LIVE_POWER_CAP + 1e-8:
        raise RuntimeError("Live power rating exceeds configured cap.")
    if not form["no_lookahead_filter_applied_flag"].eq(1).all():
        raise RuntimeError("No-lookahead filter flag is not set.")
    if set(form["build_id"].astype(str)) != {BUILD_ID}:
        raise RuntimeError("Form output build ID mismatch.")
    if set(form["form_version"].astype(str)) != {VERSION}:
        raise RuntimeError("Form output version mismatch.")
    if set(form["current_structural_build_id"].astype(str)) != {
        EXPECTED_STRUCTURAL_POWER_BUILD_ID
    }:
        raise RuntimeError("Current structural power lineage mismatch.")
    if form["current_structural_snapshot_hash"].nunique(dropna=True) != 1:
        raise RuntimeError(
            "Form output does not carry one current structural snapshot hash."
        )
    structural_for_hash = form[
        ["team", "current_structural_power_rating_points"]
    ].rename(
        columns={
            "current_structural_power_rating_points": (
                "preseason_power_rating_points"
            )
        }
    )
    expected_structural_hash = stable_current_structural_hash(
        structural_for_hash
    )
    if set(form["current_structural_snapshot_hash"].astype(str)) != {
        expected_structural_hash
    }:
        raise RuntimeError("Current structural snapshot hash mismatch.")
    structural_imported = pd.to_datetime(
        form["current_structural_date_imported"], errors="coerce"
    )
    form_imported = pd.to_datetime(form["date_imported"], errors="coerce")
    if structural_imported.isna().any() or form_imported.isna().any():
        raise RuntimeError("Structural/form lineage timestamps are missing.")
    if (structural_imported > form_imported).any():
        raise RuntimeError(
            "Form output predates its current structural power source."
        )

    expected_live = (
        pd.to_numeric(form["structural_prior_weight"], errors="coerce")
        * pd.to_numeric(
            form["current_structural_power_rating_points"],
            errors="coerce",
        )
        + pd.to_numeric(form["current_season_weight"], errors="coerce")
        * pd.to_numeric(form["form_rating_points"], errors="coerce")
    )
    expected_live = clip_and_recenter(
        expected_live, -LIVE_POWER_CAP, LIVE_POWER_CAP
    )
    if not np.allclose(
        form["power_rating_points"], expected_live, atol=1e-8
    ):
        raise RuntimeError(
            "Live form ratings do not reconcile to the current structural "
            "prior plus current-season form."
        )

    if not game_audit.empty:
        if game_audit["game_date"].notna().any():
            maximum_date = pd.to_datetime(
                game_audit["game_date"], errors="coerce"
            ).max()
            if pd.notna(maximum_date) and maximum_date.normalize() > as_of_date:
                raise RuntimeError("Game audit contains a game after as-of date.")
        if game_audit["process_home_margin"].isna().any():
            missing = int(game_audit["process_home_margin"].isna().sum())
            raise RuntimeError(
                f"{missing} completed games lack a process margin."
            )
    else:
        if not np.allclose(
            form["power_rating_points"],
            form["current_structural_power_rating_points"],
            atol=1e-9,
        ):
            raise RuntimeError(
                "No-game output does not equal the current structural prior."
            )
        if not form["current_season_weight"].eq(0.0).all():
            raise RuntimeError("No-game output has nonzero form weight.")

    if form["snapshot_hash"].nunique(dropna=True) != 1:
        raise RuntimeError("Form output does not carry one snapshot hash.")


def save_outputs(
    engine,
    form: pd.DataFrame,
    game_audit: pd.DataFrame,
    team_game_audit: pd.DataFrame,
    process_model: pd.DataFrame,
    write_csv: bool,
) -> None:
    outputs = {
        OUTPUT_TABLE: form,
        GAME_AUDIT_TABLE: game_audit,
        TEAM_GAME_AUDIT_TABLE: team_game_audit,
        PROCESS_MODEL_TABLE: process_model,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for table_name, frame in outputs.items():
        frame.to_sql(
            table_name,
            con=engine,
            if_exists="replace",
            index=False,
        )
        LOGGER.info(
            "[FORM] Saved %s rows to %s",
            f"{len(frame):,}",
            table_name,
        )
        if write_csv:
            output_path = OUTPUT_DIR / f"{table_name}.csv"
            frame.to_csv(output_path, index=False, encoding="utf-8-sig")
            LOGGER.info("[FORM] CSV saved: %s", output_path)

    with engine.begin() as connection:
        connection.execute(
            sql.text(
                f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_team "
                f"ON {OUTPUT_TABLE}(team)"
            )
        )
        if not game_audit.empty:
            connection.execute(
                sql.text(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"idx_{GAME_AUDIT_TABLE}_week_game "
                    f"ON {GAME_AUDIT_TABLE}(week, game_id)"
                )
            )
        if not team_game_audit.empty:
            connection.execute(
                sql.text(
                    f"CREATE INDEX IF NOT EXISTS "
                    f"idx_{TEAM_GAME_AUDIT_TABLE}_team_week "
                    f"ON {TEAM_GAME_AUDIT_TABLE}(team, week)"
                )
            )


def print_report(
    form: pd.DataFrame,
    game_audit: pd.DataFrame,
    process_model: pd.DataFrame,
) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 110)
    LOGGER.info("[FORM] 2026 NFL CANONICAL INCREMENTAL FORM RATINGS")
    LOGGER.info("=" * 110)
    LOGGER.info("[FORM] Build ID: %s", BUILD_ID)
    LOGGER.info("[FORM] Version: %s", VERSION)
    LOGGER.info("[FORM] Implementation: %s", IMPLEMENTATION_VERSION)
    LOGGER.info("[FORM] Through week: %s", int(form["through_week"].max()))
    LOGGER.info(
        "[FORM] Completed regular-season games: %s",
        int(form["completed_regular_season_games"].max()),
    )
    LOGGER.info("[FORM] As-of date: %s", form["as_of_date"].iloc[0])
    LOGGER.info(
        "[FORM] Immutable preseason snapshot: %s",
        form["preseason_snapshot_source"].iloc[0],
    )
    LOGGER.info(
        "[FORM] Snapshot hash: %s",
        str(form["snapshot_hash"].iloc[0]),
    )
    LOGGER.info(
        "[FORM] Rating mean/std: %.6f / %.3f",
        form["power_rating_points"].mean(),
        form["power_rating_points"].std(ddof=1),
    )
    LOGGER.info(
        "[FORM] Maximum 2026 form weight: %.1f%%",
        100.0 * form["current_season_weight"].max(),
    )
    LOGGER.info(
        "[FORM] No-lookahead filter applied: YES"
    )

    if process_model.empty:
        LOGGER.info("[FORM] Process model: not required/prepared yet.")
    else:
        LOGGER.info(
            "[FORM] Process model RMSE/R2: %.3f / %.3f",
            float(process_model["weighted_training_rmse"].iloc[0]),
            float(process_model["weighted_training_r_squared"].iloc[0]),
        )

    display_columns = [
        "power_rating_rank",
        "team",
        "player_name",
        "power_rating_points",
        "preseason_power_rating_points",
        "form_rating_points",
        "form_adjustment_points",
        "games_played",
        "current_season_weight",
    ]
    LOGGER.info("")
    LOGGER.info(
        "[FORM] Live internal ratings:\n%s",
        form[display_columns].to_string(
            index=False,
            formatters={
                "current_season_weight": lambda value: f"{value:.1%}"
            },
        ),
    )

    if not game_audit.empty:
        LOGGER.info("")
        LOGGER.info(
            "[FORM] Most recent game process audit:\n%s",
            game_audit.sort_values(
                ["week", "game_id"], ascending=[False, True]
            )[
                [
                    "week",
                    "away_team",
                    "home_team",
                    "actual_home_margin",
                    "process_home_margin",
                    "process_model_residual",
                    "recent_game_weight",
                ]
            ].head(16).to_string(index=False),
        )
    LOGGER.info("=" * 110)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    global LOGGER, LOG_PATH

    args = parse_args()
    configure_runtime(args.project_root, args.db_path)
    LOGGER, LOG_PATH = configure_logging()
    started = dt.datetime.now()
    as_of_date = parse_as_of_date(args.as_of_date)

    LOGGER.info("[FORM] Building canonical 2026 NFL form rating")
    LOGGER.info("[FORM] Source recovery: %s", SOURCE_RECOVERY_VERSION)
    LOGGER.info("[FORM] Build ID: %s", BUILD_ID)
    LOGGER.info("[FORM] Version: %s", VERSION)
    LOGGER.info("[FORM] Database: %s", DB_PATH)
    LOGGER.info("[FORM] As-of date: %s", as_of_date.date().isoformat())
    LOGGER.info("[FORM] Through-week cutoff: %s", args.through_week)

    engine = get_engine()

    if args.rebuild_process_cache:
        drop_table_if_exists(engine, HISTORICAL_FEATURE_CACHE_TABLE)
        drop_table_if_exists(engine, PROCESS_MODEL_TABLE)
        LOGGER.info("[FORM] Historical process cache/model cleared.")

    raw_current_schedule, schedule_source = load_current_schedule(engine, args.schedule_path)
    current_schedule = standardize_schedules(raw_current_schedule)
    current_schedule = current_schedule[current_schedule["season"].eq(SEASON)].copy()
    completed_current = filter_schedule_as_of(
        current_schedule,
        as_of_date=as_of_date,
        through_week=args.through_week,
    )

    preseason, snapshot_source = load_or_create_preseason_snapshot(
        engine,
        refresh=args.refresh_preseason_snapshot,
        # A historical date or week cutoff must not reopen snapshot refreshes.
        allow_refresh=not current_schedule["completed"].eq(1).any(),
    )
    teams = sorted(preseason["team"].unique().tolist())

    if completed_current.empty:
        LOGGER.info(
            "[FORM] No completed 2026 regular-season games through cutoff. "
            "Writing the current structural prior; retaining the immutable "
            "preseason snapshot only as an audit comparator."
        )
        form = build_no_games_output(
            preseason=preseason,
            schedule_source=schedule_source,
            snapshot_source=snapshot_source,
            as_of_date=as_of_date,
            through_week=args.through_week,
        )
        game_audit = empty_game_audit()
        team_game_audit = empty_team_game_audit()

        if args.prepare_process_model:
            process_model, process_model_source, historical_schedule_source = (
                load_available_process_model(engine)
            )
            LOGGER.info(
                "[FORM] Preseason process model prepared via %s; schedules=%s",
                process_model_source,
                historical_schedule_source,
            )
        elif table_exists(engine, PROCESS_MODEL_TABLE):
            cached_model = read_table(engine, PROCESS_MODEL_TABLE)
            process_model = (
                cached_model if model_cache_is_valid(cached_model)
                else empty_process_model()
            )
        else:
            process_model = empty_process_model()

        validate_output(form, game_audit, as_of_date)
        save_outputs(
            engine,
            form,
            game_audit,
            team_game_audit,
            process_model,
            write_csv=not args.no_csv,
        )
        print_report(form, game_audit, process_model)
        LOGGER.info(
            "[FORM] Completed successfully in %.2f seconds",
            (dt.datetime.now() - started).total_seconds(),
        )
        LOGGER.info("[FORM] Log saved: %s", LOG_PATH)
        return 0

    LOGGER.info(
        "[FORM] Completed regular-season games through cutoff: %s",
        completed_current["game_id"].nunique(),
    )

    raw_current_pbp, pbp_source = load_current_pbp(args.current_pbp_path)
    current_pbp = standardize_pbp(raw_current_pbp, SEASON)
    current_pbp = filter_pbp_to_completed_games(current_pbp, completed_current)
    if current_pbp.empty:
        raise RuntimeError(
            "Completed games were found, but no matching play-by-play rows "
            "were available through the selected cutoff."
        )

    team_games = build_team_game_features(current_pbp, completed_current)

    process_model, process_model_source, historical_schedule_source = (
        load_available_process_model(engine)
    )
    team_games = apply_process_model(team_games, process_model)

    now = dt.datetime.now().isoformat(timespec="seconds")
    team_games["form_version"] = VERSION
    team_games["date_imported"] = now

    game_audit = build_game_audit(
        completed_current,
        team_games,
        schedule_source=(
            f"{schedule_source}; historical={historical_schedule_source}"
        ),
        pbp_source=pbp_source,
    )

    usable_process_games = int(game_audit["process_home_margin"].notna().sum())
    if usable_process_games < MIN_CURRENT_PROCESS_GAMES:
        raise RuntimeError(
            "Completed games were found, but no usable process margins were created."
        )
    if usable_process_games != len(game_audit):
        raise RuntimeError(
            f"Process coverage is incomplete: {usable_process_games}/{len(game_audit)} games."
        )

    form = build_live_form_output(
        preseason=preseason,
        schedules=completed_current,
        team_games=team_games,
        game_audit=game_audit,
        teams=teams,
        schedule_source=schedule_source,
        pbp_source=pbp_source,
        process_model_source=process_model_source,
        snapshot_source=snapshot_source,
        as_of_date=as_of_date,
        through_week=args.through_week,
    )

    team_game_columns = list(empty_team_game_audit().columns)
    for column in team_game_columns:
        ensure_column(team_games, column, np.nan)
    team_game_audit = (
        team_games[team_game_columns]
        .sort_values(["week", "game_id", "team"])
        .reset_index(drop=True)
    )

    validate_output(form, game_audit, as_of_date)
    save_outputs(
        engine,
        form,
        game_audit,
        team_game_audit,
        process_model,
        write_csv=not args.no_csv,
    )
    print_report(form, game_audit, process_model)

    LOGGER.info(
        "[FORM] Completed successfully in %.2f seconds",
        (dt.datetime.now() - started).total_seconds(),
    )
    LOGGER.info("[FORM] Log saved: %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[FORM] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        try:
            LOGGER.error("[FORM] FAILED: %s", exc)
            LOGGER.error(traceback.format_exc())
            LOGGER.error("[FORM] Log saved: %s", LOG_PATH)
        except Exception:
            print(f"[FORM] FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
        raise SystemExit(1)
