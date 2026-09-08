#!/usr/bin/env python
"""
Build canonical weekly 2026 NFL market-independent spread projections.

Independent projection
----------------------
    projected_home_margin
    =
    home live neutral-field form rating
    - away live neutral-field form rating
    + venue-adjusted historical home-field advantage
    + bounded rest adjustment
    + bounded travel/schedule adjustment
    + explicit manual availability adjustment

Market information is attached only after the independent projection is frozen.
It is never used to create the projected spread.

The manual availability layer is intentionally explicit. It supports temporary
QB/injury information without silently rebuilding or double-counting the
structural ratings. The authoritative structural QB remains embedded in the
live form rating.

Required input
--------------
SQLite:
    nfl_2026_form_ratings

Preferred supporting inputs
---------------------------
SQLite:
    nfl_power_ratings_2026
    nfl_power_rating_calibration_season_audit
    nfl_weekly_manual_adjustments_2026

Schedule sources, in priority order:
    --schedule-path
    recognized SQLite schedule tables
    recognized project CSV/XLSX files
    nflreadpy.load_schedules([2026])

Optional schedule-factor source:
    --schedule-factors-path
    SQLite table nfl_schedule_factors_2026
    outputs/nfl_schedule_factors_2026.csv
    project-root/nfl_schedule_factors_2026.csv

Optional market source:
    --market-path

Outputs
-------
SQLite and CSV:
    nfl_weekly_power_spread_predictions_2026
    nfl_weekly_power_spread_prediction_history_2026
    nfl_weekly_power_spread_run_audit_2026
    nfl_weekly_power_spread_run_audit_history_2026

Important interpretation
------------------------
The probability layer uses historical NFL game-noise dispersion from the market
calibration audit when available. It is a transparent provisional probability
mapping, not a claim that the new structural/form model has already been
historically calibrated. The forthcoming leakage-controlled backtest should
replace this provisional mapping with model-specific residual calibration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import sqlite3
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2026
BUILD_ID = "NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4"
VERSION = "v4_structural_form_hfa_fail_closed_lineage"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

FORM_RATING_TABLE = "nfl_2026_form_ratings"
POWER_RATING_TABLE = "nfl_power_ratings_2026"
POWER_CALIBRATION_TABLE = "nfl_power_rating_calibration_season_audit"
MANUAL_ADJUSTMENT_TABLE = "nfl_weekly_manual_adjustments_2026"
SCHEDULE_FACTOR_TABLE = "nfl_schedule_factors_2026"

OUTPUT_TABLE = "nfl_weekly_power_spread_predictions_2026"
OUTPUT_HISTORY_TABLE = "nfl_weekly_power_spread_prediction_history_2026"
RUN_AUDIT_TABLE = "nfl_weekly_power_spread_run_audit_2026"
RUN_AUDIT_HISTORY_TABLE = "nfl_weekly_power_spread_run_audit_history_2026"

MODEL_VARIANT = "STRUCTURAL_FORM_HFA"
EXPECTED_FORM_BUILD_ID = "NFL_2026_FORM_RATING_CANONICAL_V3"
EXPECTED_FORM_VERSION = (
    "v3_current_structural_prior_asof_opponent_adjusted_form"
)
EXPECTED_STRUCTURAL_POWER_BUILD_ID = (
    "NFL_POWER_RATINGS_2026_CANONICAL_V2"
)

DEFAULT_MINIMUM_SPREAD_DIFFERENCE = 2.5
DEFAULT_MAXIMUM_MARKET_DISAGREEMENT = 5.0
DEFAULT_HOME_FIELD_POINTS = 1.65
DEFAULT_FLAT_STAKE = 500.0
DEFAULT_BANKROLL = 50_000.0
DEFAULT_QUARTER_KELLY_MULTIPLIER = 0.25
DEFAULT_MAX_KELLY_BET_FRACTION = 0.05
DEFAULT_SPREAD_PRICE = -110
DEFAULT_MARKET_LINE_PREFERENCE = "current"

# Existing schedule-factor values retained only for directly interpretable
# rest/travel effects. Turf, primetime, division, Super Bowl, and team-tier
# heuristics are intentionally excluded.
HOME_BYE_ADVANTAGE_POINTS = 0.50
AWAY_BYE_ADVANTAGE_POINTS = 0.60
REST_ADVANTAGE_3PLUS_POINTS = 0.40
TRAVEL_2000_PLUS_POINTS = 0.20
BACK_TO_BACK_WEST_EAST_POINTS = 0.40
BACK_TO_BACK_WEST_CENTRAL_POINTS = 0.40
BACK_TO_BACK_AWAY_POINTS = 0.20
THREE_OF_FOUR_AWAY_POINTS = 0.20

REST_ADJUSTMENT_CAP = 0.60
TRAVEL_SCHEDULE_ADJUSTMENT_CAP = 0.75
TOTAL_SCHEDULE_ADJUSTMENT_CAP = 1.25
MANUAL_TEAM_ADJUSTMENT_CAP = 4.00
MANUAL_GAME_ADJUSTMENT_CAP = 6.00

FALLBACK_RESIDUAL_MEAN = 0.0
FALLBACK_RESIDUAL_SIGMA = 13.5
MIN_RESIDUAL_SIGMA = 10.0
MAX_RESIDUAL_SIGMA = 17.0
PROBABILITY_CLIP = 0.01

REGULAR_SEASON_TYPES = {
    "REG",
    "R",
    "REGULAR",
    "REGULAR SEASON",
    "REGULAR_SEASON",
}

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
    "JAX": "JAX", "JAC": "JAX",
    "KC": "KC", "KAN": "KC", "KCC": "KC",
    "LAC": "LAC", "SD": "LAC", "SDG": "LAC",
    "LAR": "LAR", "LA": "LAR", "STL": "LAR",
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

    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",

    "OAKLAND RAIDERS": "LV",
    "SAN DIEGO CHARGERS": "LAC",
    "ST LOUIS RAMS": "LAR",
    "ST. LOUIS RAMS": "LAR",
    "WASHINGTON REDSKINS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
}

SCHEDULE_TABLE_CANDIDATES = (
    "nfl_schedule_2026",
    "nfl_2026_schedule",
    "nfl_schedule_2026_complete",
    "nfl_schedule_2026_full",
    "nfl_schedule_2026_final",
    "nfl_games_2026",
    "nfl_2026_games",
    "nfl_game_schedule_2026",
    "nfl_schedule_master",
    "nfl_game_schedule",
    "nfl_schedule",
    "nfl_schedules",
    "nfl_games",
)

SCHEDULE_ALIASES = {
    "season": (
        "season", "season_year", "schedule_season", "year",
    ),
    "week": (
        "week", "game_week", "week_num", "week_number",
        "week_no", "nfl_week",
    ),
    "game_type": (
        "game_type", "season_type", "type", "game_type_abbr",
    ),
    "game_id": (
        "game_id", "id", "gsis_id", "gameid", "event_id",
    ),
    "game_date": (
        "game_datetime_et", "game_datetime_ct", "gameday",
        "game_date", "date", "start_time", "kickoff",
        "game_datetime", "game_datetime_utc",
    ),
    "home_team": (
        "home_team_abbr", "home_team", "home",
        "home_abbr", "home_team_code", "team_home",
    ),
    "away_team": (
        "away_team_abbr", "away_team", "away",
        "away_abbr", "away_team_code", "team_away",
    ),
    "home_score": (
        "home_score", "score_home", "home_points", "home_pts",
    ),
    "away_score": (
        "away_score", "score_away", "away_points", "away_pts",
    ),
    "neutral": (
        "neutral", "neutral_site", "is_neutral", "neutral_site_flag",
    ),
    "international": (
        "international_game", "is_international",
    ),
    "opening_spread": (
        "opening_spread", "open_spread", "spread_line_open",
        "spread_open", "opening_market_home_margin",
        "open_line",
    ),
    "opening_home_spread": (
        "home_opening_spread", "opening_home_spread",
    ),
    "current_spread": (
        "spread_line", "closing_spread", "close_spread",
        "market_spread", "spread", "current_spread",
        "current_market_home_margin",
    ),
    "current_home_spread": (
        "home_spread", "current_home_spread",
    ),
    "home_spread_price": (
        "home_spread_price", "spread_price_home",
        "home_spread_odds", "spread_odds_home",
        "current_home_spread_price",
    ),
    "away_spread_price": (
        "away_spread_price", "spread_price_away",
        "away_spread_odds", "spread_odds_away",
        "current_away_spread_price",
    ),
    "generic_spread_price": (
        "spread_price", "spread_odds", "ats_price",
    ),
    "sportsbook": (
        "sportsbook", "sportsbook_name", "bookmaker", "bookmaker_title",
    ),
    "bookmaker_key": (
        "bookmaker_key", "sportsbook_key", "book_key",
    ),
    "market_source": (
        "market_source", "odds_source", "line_source",
    ),
    "market_retrieved_at_utc": (
        "market_retrieved_at_utc", "retrieved_at_utc", "odds_retrieved_at",
    ),
    "market_last_update_utc": (
        "market_last_update_utc", "last_update_utc", "bookmaker_last_update",
    ),
    "market_line_age_minutes": (
        "market_line_age_minutes", "line_age_minutes", "odds_age_minutes",
    ),
    "market_stale_flag": (
        "market_stale_flag", "stale_flag", "odds_stale_flag",
    ),
    "source_priority_rank": (
        "source_priority_rank", "bookmaker_priority_rank",
    ),
    "circa_line_flag": (
        "circa_line_flag", "is_circa_line",
    ),
}

LOGGER = logging.getLogger("nfl_weekly_power_spreads_2026")


# =============================================================================
# ARGUMENTS AND LOGGING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--schedule-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--schedule-factors-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--market-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Prediction-date label and automatic-week reference.",
    )
    parser.add_argument(
        "--home-field-points",
        type=float,
        default=None,
        help="Override historical HFA. Neutral/international games remain zero.",
    )
    parser.add_argument(
        "--market-line-preference",
        choices=("current", "opening"),
        default=DEFAULT_MARKET_LINE_PREFERENCE,
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=DEFAULT_BANKROLL,
    )
    parser.add_argument(
        "--flat-stake",
        type=float,
        default=DEFAULT_FLAT_STAKE,
    )
    parser.add_argument(
        "--minimum-spread-difference",
        type=float,
        default=DEFAULT_MINIMUM_SPREAD_DIFFERENCE,
    )
    parser.add_argument(
        "--maximum-market-disagreement",
        type=float,
        default=DEFAULT_MAXIMUM_MARKET_DISAGREEMENT,
    )
    parser.add_argument(
        "--quarter-kelly-multiplier",
        type=float,
        default=DEFAULT_QUARTER_KELLY_MULTIPLIER,
    )
    parser.add_argument(
        "--max-kelly-bet-fraction",
        type=float,
        default=DEFAULT_MAX_KELLY_BET_FRACTION,
    )
    parser.add_argument(
        "--default-spread-price",
        type=int,
        default=DEFAULT_SPREAD_PRICE,
    )
    parser.add_argument(
        "--include-week18",
        action="store_true",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
    )
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    for name in (
        "schedule_path",
        "schedule_factors_path",
        "market_path",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if args.week is not None and not (1 <= args.week <= 18):
        parser.error("--week must be between 1 and 18.")
    if args.bankroll <= 0:
        parser.error("--bankroll must be positive.")
    if args.flat_stake < 0:
        parser.error("--flat-stake cannot be negative.")
    if args.minimum_spread_difference < 0:
        parser.error("--minimum-spread-difference cannot be negative.")
    if (
        args.maximum_market_disagreement
        < args.minimum_spread_difference
    ):
        parser.error(
            "--maximum-market-disagreement must be at least the minimum."
        )
    if not (0 <= args.quarter_kelly_multiplier <= 1):
        parser.error("--quarter-kelly-multiplier must be in [0, 1].")
    if not (0 <= args.max_kelly_bet_fraction <= 1):
        parser.error("--max-kelly-bet-fraction must be in [0, 1].")
    return args


def configure_logging(project_root: Path) -> Path:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / (
        "predict_nfl_weekly_power_spreads_2026_"
        + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".log"
    )

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(stream)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(file_handler)
    return path


def close_logger() -> None:
    for handler in list(LOGGER.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            LOGGER.removeHandler(handler)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def now_string() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def normalize_team(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).upper().strip().replace(".", "")
    if text in {"", "NAN", "NONE", "NULL", "<NA>"}:
        return ""
    return TEAM_ALIASES.get(text, text)


def first_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    available = {
        str(column).lower().strip(): str(column)
        for column in columns
    }
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in available:
            return available[key]
    return None


def bool_series(
    frame: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:
    if column is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.upper().str.strip()
    return (
        numeric.fillna(0).ne(0)
        | text.isin({"TRUE", "T", "YES", "Y", "1"})
    )


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def implied_probability_from_american(price: int) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 100.0 / (price + 100.0)


def net_profit_per_unit(price: int) -> float:
    if price < 0:
        return 100.0 / abs(price)
    return price / 100.0


def sanitize_american_price(
    value: Any,
    fallback: int,
) -> tuple[int, int]:
    try:
        price = int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback), 1
    if price == 0 or abs(price) < 100 or abs(price) > 10000:
        return int(fallback), 1
    return price, 0


def spread_label(
    home_team: str,
    away_team: str,
    home_margin: Any,
) -> str:
    if pd.isna(home_margin):
        return "UNAVAILABLE"
    margin = float(home_margin)
    if abs(margin) < 0.025:
        return "PICK"
    if margin > 0:
        return f"{home_team} -{abs(margin):.2f}"
    return f"{away_team} -{abs(margin):.2f}"


def selected_market_line_label(
    selected_team: str,
    home_team: str,
    away_team: str,
    market_home_margin: Any,
) -> str:
    if pd.isna(market_home_margin) or not selected_team:
        return ""
    margin = float(market_home_margin)
    if selected_team == home_team:
        line = -margin
    elif selected_team == away_team:
        line = margin
    else:
        return ""
    if abs(line) < 0.025:
        return f"{selected_team} PK"
    return f"{selected_team} {line:+.2f}"


def independent_projection_hash(row: dict[str, Any]) -> str:
    keys = [
        "game_id",
        "home_power_rating_points",
        "away_power_rating_points",
        "neutral_rating_difference",
        "home_field_points",
        "rest_adjustment_points",
        "travel_schedule_adjustment_points",
        "manual_availability_adjustment_points",
        "projected_home_margin",
    ]
    payload = {
        key: (
            round(float(row[key]), 8)
            if isinstance(row.get(key), (int, float, np.integer, np.floating))
            and not pd.isna(row[key])
            else row.get(key)
        )
        for key in keys
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# =============================================================================
# SQLITE AND FILE HELPERS
# =============================================================================

def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute("PRAGMA busy_timeout = 60000")
    return connection


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    return connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone() is not None


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        return pd.DataFrame()
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(
        f'SELECT * FROM "{escaped}"',
        connection,
    )
    frame.columns = [
        str(column).lower().strip()
        for column in frame.columns
    ]
    return frame


def load_tabular_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise RuntimeError(f"Unsupported file type: {path}")


def append_with_schema_evolution(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return

    if not table_exists(connection, table_name):
        frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )
        return

    escaped_table = table_name.replace('"', '""')
    existing_info = connection.execute(
        f'PRAGMA table_info("{escaped_table}")'
    ).fetchall()
    existing_columns = [str(row[1]) for row in existing_info]

    for column in frame.columns:
        if column in existing_columns:
            continue
        series = frame[column]
        if pd.api.types.is_integer_dtype(series.dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(series.dtype):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        escaped_column = column.replace('"', '""')
        connection.execute(
            f'ALTER TABLE "{escaped_table}" '
            f'ADD COLUMN "{escaped_column}" {sql_type}'
        )
        existing_columns.append(column)

    aligned = frame.copy()
    for column in existing_columns:
        if column not in aligned.columns:
            aligned[column] = None
    aligned = aligned[existing_columns]
    aligned.to_sql(
        table_name,
        connection,
        if_exists="append",
        index=False,
    )


# =============================================================================
# RATINGS, HFA, AND PROBABILITY CALIBRATION
# =============================================================================

def load_live_ratings(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, str]:
    frame = read_table(connection, FORM_RATING_TABLE)
    if frame.empty:
        raise RuntimeError(
            f"Missing or empty required table: {FORM_RATING_TABLE}"
        )

    required = {
        "season",
        "team",
        "power_rating_points",
        "current_structural_power_rating_points",
        "current_structural_build_id",
        "current_structural_version",
        "current_structural_date_imported",
        "current_structural_snapshot_hash",
        "as_of_date",
        "through_week",
        "snapshot_hash",
        "build_id",
        "form_version",
        "date_imported",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{FORM_RATING_TABLE} is missing columns: {missing}"
        )

    frame["team"] = frame["team"].map(normalize_team)
    frame["power_rating_points"] = pd.to_numeric(
        frame["power_rating_points"],
        errors="coerce",
    )

    if len(frame) != 32 or frame["team"].nunique() != 32:
        raise RuntimeError(
            f"{FORM_RATING_TABLE} must contain 32 unique teams."
        )
    if frame["power_rating_points"].isna().any():
        raise RuntimeError("Live ratings contain missing values.")
    if abs(float(frame["power_rating_points"].mean())) > 1e-6:
        raise RuntimeError("Live ratings are not zero-centered.")

    keep = [
        "season",
        "team",
        "power_rating_points",
        "current_structural_power_rating_points",
        "current_structural_build_id",
        "current_structural_version",
        "current_structural_date_imported",
        "current_structural_snapshot_hash",
        "player_name",
        "preseason_power_rating_points",
        "form_rating_points",
        "form_adjustment_points",
        "games_played",
        "current_season_weight",
        "as_of_date",
        "through_week",
        "snapshot_hash",
        "build_id",
        "form_version",
        "date_imported",
    ]
    for column in keep:
        if column not in frame.columns:
            frame[column] = np.nan

    rating_source = FORM_RATING_TABLE
    return frame[keep].copy(), rating_source


def stable_current_power_hash(frame: pd.DataFrame) -> str:
    canonical = (
        frame[["team", "power_rating_points"]]
        .copy()
        .sort_values("team")
        .reset_index(drop=True)
    )
    canonical["power_rating_points"] = pd.to_numeric(
        canonical["power_rating_points"], errors="coerce"
    )
    payload = canonical.to_csv(index=False, float_format="%.12f")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def one_text(frame: pd.DataFrame, column: str, label: str) -> str:
    values = {
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    }
    if len(values) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one nonempty {column}; "
            f"found={sorted(values)}."
        )
    return next(iter(values))


def prediction_date(value: Optional[str]) -> pd.Timestamp:
    parsed = pd.to_datetime(
        value if value is not None else dt.date.today(),
        errors="coerce",
    )
    if pd.isna(parsed):
        raise RuntimeError(
            f"Invalid --as-of-date value: {value!r}; use YYYY-MM-DD."
        )
    return pd.Timestamp(parsed).normalize()


def validate_live_rating_contract(
    connection: sqlite3.Connection,
    ratings: pd.DataFrame,
    week: int,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    season_values = pd.to_numeric(
        ratings["season"], errors="coerce"
    ).dropna().unique()
    if len(season_values) != 1 or int(season_values[0]) != SEASON:
        raise RuntimeError(
            f"{FORM_RATING_TABLE} season mismatch: {season_values}."
        )
    form_build_id = one_text(
        ratings, "build_id", FORM_RATING_TABLE
    )
    form_version = one_text(
        ratings, "form_version", FORM_RATING_TABLE
    )
    if form_build_id != EXPECTED_FORM_BUILD_ID:
        raise RuntimeError(
            "Stale or incompatible form build: "
            f"expected={EXPECTED_FORM_BUILD_ID!r}, found={form_build_id!r}."
        )
    if form_version != EXPECTED_FORM_VERSION:
        raise RuntimeError(
            "Stale or incompatible form version: "
            f"expected={EXPECTED_FORM_VERSION!r}, found={form_version!r}."
        )

    through = pd.to_numeric(
        ratings["through_week"], errors="coerce"
    ).dropna().unique()
    expected_through_week = max(0, int(week) - 1)
    if len(through) != 1 or int(through[0]) != expected_through_week:
        raise RuntimeError(
            "Form through-week is stale or ahead of the prediction: "
            f"prediction_week={week}, expected_through_week="
            f"{expected_through_week}, found={through}."
        )

    form_dates = pd.to_datetime(
        ratings["as_of_date"], errors="coerce"
    ).dt.normalize()
    if form_dates.isna().any() or set(form_dates) != {as_of_date}:
        found = sorted(
            pd.Timestamp(value).date().isoformat()
            for value in form_dates.dropna().unique()
        )
        raise RuntimeError(
            "Form as-of date does not match the prediction as-of date: "
            f"expected={as_of_date.date().isoformat()}, found={found}."
        )

    structural_build_id = one_text(
        ratings,
        "current_structural_build_id",
        FORM_RATING_TABLE,
    )
    if structural_build_id != EXPECTED_STRUCTURAL_POWER_BUILD_ID:
        raise RuntimeError(
            "Current structural power lineage mismatch: "
            f"expected={EXPECTED_STRUCTURAL_POWER_BUILD_ID!r}, "
            f"found={structural_build_id!r}."
        )
    structural_version = one_text(
        ratings,
        "current_structural_version",
        FORM_RATING_TABLE,
    )
    structural_hash = one_text(
        ratings,
        "current_structural_snapshot_hash",
        FORM_RATING_TABLE,
    )
    preseason_snapshot_hash = one_text(
        ratings, "snapshot_hash", FORM_RATING_TABLE
    )

    current = read_table(connection, POWER_RATING_TABLE)
    required = {
        "season", "team", "power_rating_points", "build_id",
        "calibration_version", "date_imported",
    }
    missing = sorted(required - set(current.columns))
    if missing:
        raise RuntimeError(
            f"{POWER_RATING_TABLE} is missing lineage columns: {missing}."
        )
    current["season"] = pd.to_numeric(
        current["season"], errors="coerce"
    )
    current = current[current["season"].eq(SEASON)].copy()
    current["team"] = current["team"].map(normalize_team)
    current["power_rating_points"] = pd.to_numeric(
        current["power_rating_points"], errors="coerce"
    )
    if (
        len(current) != 32
        or current["team"].nunique() != 32
        or current["power_rating_points"].isna().any()
    ):
        raise RuntimeError(
            f"{POWER_RATING_TABLE} must contain 32 valid 2026 teams."
        )
    current_build_id = one_text(
        current, "build_id", POWER_RATING_TABLE
    )
    current_version = one_text(
        current, "calibration_version", POWER_RATING_TABLE
    )
    if current_build_id != EXPECTED_STRUCTURAL_POWER_BUILD_ID:
        raise RuntimeError(
            f"{POWER_RATING_TABLE} build ID mismatch: {current_build_id!r}."
        )
    if current_version != structural_version:
        raise RuntimeError(
            "Form structural version does not match current power ratings: "
            f"form={structural_version!r}, current={current_version!r}."
        )
    current_hash = stable_current_power_hash(current)
    if current_hash != structural_hash:
        raise RuntimeError(
            "Form structural snapshot hash does not match current power "
            "ratings. Rebuild the form table after the structural refresh."
        )

    comparison = ratings[
        ["team", "current_structural_power_rating_points"]
    ].merge(
        current[["team", "power_rating_points"]],
        on="team",
        how="outer",
        validate="one_to_one",
    )
    if comparison.isna().any().any() or not np.allclose(
        pd.to_numeric(
            comparison["current_structural_power_rating_points"],
            errors="coerce",
        ),
        comparison["power_rating_points"],
        atol=1e-9,
    ):
        raise RuntimeError(
            "Form structural ratings do not match current power ratings. "
            "Rebuild the form table after the structural refresh."
        )

    form_imported = pd.to_datetime(
        ratings["date_imported"], errors="coerce"
    )
    structural_imported = pd.to_datetime(
        current["date_imported"], errors="coerce"
    )
    if form_imported.isna().any() or structural_imported.isna().any():
        raise RuntimeError("Form/structural timestamps are missing.")
    if form_imported.min() < structural_imported.max():
        raise RuntimeError(
            "Form ratings predate the current structural power refresh. "
            "Rebuild the form table before generating weekly spreads."
        )

    return {
        "form_build_id": form_build_id,
        "form_version": form_version,
        "form_as_of_date": as_of_date.date().isoformat(),
        "form_through_week": expected_through_week,
        "form_date_imported": form_imported.max().isoformat(),
        "preseason_snapshot_hash": preseason_snapshot_hash,
        "structural_power_build_id": current_build_id,
        "structural_power_version": current_version,
        "structural_power_date_imported": (
            structural_imported.max().isoformat()
        ),
        "structural_power_snapshot_hash": current_hash,
    }


def load_home_field(
    connection: sqlite3.Connection,
    override: Optional[float],
) -> tuple[float, str]:
    if override is not None:
        if not np.isfinite(override) or override < 0 or override > 5:
            raise RuntimeError(
                f"Invalid HFA override: {override}"
            )
        return float(override), "command_line_override"

    power = read_table(connection, POWER_RATING_TABLE)
    if (
        not power.empty
        and "median_historical_home_field" in power.columns
    ):
        values = pd.to_numeric(
            power["median_historical_home_field"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            return float(values.median()), POWER_RATING_TABLE

    audit = read_table(connection, POWER_CALIBRATION_TABLE)
    if not audit.empty and "home_field_points" in audit.columns:
        values = pd.to_numeric(
            audit["home_field_points"],
            errors="coerce",
        ).dropna()
        if not values.empty:
            return float(values.median()), POWER_CALIBRATION_TABLE

    return DEFAULT_HOME_FIELD_POINTS, "fallback_default"


def load_probability_noise(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    audit = read_table(connection, POWER_CALIBRATION_TABLE)
    if not audit.empty:
        required = {"games", "actual_margin_rmse"}
        if required.issubset(audit.columns):
            games = pd.to_numeric(
                audit["games"], errors="coerce"
            )
            rmse = pd.to_numeric(
                audit["actual_margin_rmse"], errors="coerce"
            )
            valid = games.gt(0) & rmse.gt(0)
            if valid.any():
                sigma = math.sqrt(
                    float(
                        np.average(
                            np.square(rmse.loc[valid]),
                            weights=games.loc[valid],
                        )
                    )
                )
                sigma = float(
                    np.clip(
                        sigma,
                        MIN_RESIDUAL_SIGMA,
                        MAX_RESIDUAL_SIGMA,
                    )
                )
                return {
                    "probability_method": (
                        "HISTORICAL_MARKET_GAME_NOISE_NORMAL"
                    ),
                    "historical_residual_rows": int(
                        games.loc[valid].sum()
                    ),
                    "historical_residual_mean": 0.0,
                    "historical_residual_sigma": sigma,
                    "probability_is_model_specific_calibration": 0,
                    "probability_source": POWER_CALIBRATION_TABLE,
                }

    return {
        "probability_method": "FALLBACK_GAME_NOISE_NORMAL",
        "historical_residual_rows": 0,
        "historical_residual_mean": FALLBACK_RESIDUAL_MEAN,
        "historical_residual_sigma": FALLBACK_RESIDUAL_SIGMA,
        "probability_is_model_specific_calibration": 0,
        "probability_source": "fallback_default",
    }


# =============================================================================
# SCHEDULE AND MARKET LOADING
# =============================================================================

def market_margin_from_column(
    frame: pd.DataFrame,
    column: Optional[str],
    home_spread_convention: bool,
) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return -values if home_spread_convention else values


def standardize_schedule(
    raw: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    frame.columns = [
        str(column).lower().strip()
        for column in frame.columns
    ]

    found = {
        key: first_existing(frame.columns, aliases)
        for key, aliases in SCHEDULE_ALIASES.items()
    }

    required = ["week", "home_team", "away_team"]
    missing = [key for key in required if found[key] is None]
    if missing:
        raise RuntimeError(
            f"Schedule source {source} is missing fields: {missing}"
        )

    out = pd.DataFrame(index=frame.index)
    out["season"] = (
        pd.to_numeric(frame[found["season"]], errors="coerce")
        if found["season"] is not None
        else SEASON
    )
    out["week"] = pd.to_numeric(
        frame[found["week"]], errors="coerce"
    )
    out["game_type"] = (
        frame[found["game_type"]]
        .astype(str)
        .str.upper()
        .str.strip()
        if found["game_type"] is not None
        else "REG"
    )
    out["home_team"] = frame[found["home_team"]].map(
        normalize_team
    )
    out["away_team"] = frame[found["away_team"]].map(
        normalize_team
    )
    out["game_date"] = (
        pd.to_datetime(
            frame[found["game_date"]],
            errors="coerce",
        )
        if found["game_date"] is not None
        else pd.NaT
    )
    out["home_score"] = (
        pd.to_numeric(
            frame[found["home_score"]],
            errors="coerce",
        )
        if found["home_score"] is not None
        else np.nan
    )
    out["away_score"] = (
        pd.to_numeric(
            frame[found["away_score"]],
            errors="coerce",
        )
        if found["away_score"] is not None
        else np.nan
    )

    explicit_neutral = bool_series(frame, found["neutral"])
    international = bool_series(frame, found["international"])
    out["international_game"] = international.astype(int)
    out["explicit_neutral_site"] = explicit_neutral.astype(int)
    out["neutral_site"] = (
        explicit_neutral | international
    ).astype(int)
    out["international_treated_as_neutral_flag"] = (
        international & ~explicit_neutral
    ).astype(int)

    out["opening_market_home_margin"] = (
        market_margin_from_column(
            frame,
            found["opening_spread"],
            home_spread_convention=False,
        )
    )
    opening_home = market_margin_from_column(
        frame,
        found["opening_home_spread"],
        home_spread_convention=True,
    )
    out["opening_market_home_margin"] = (
        out["opening_market_home_margin"]
        .where(
            out["opening_market_home_margin"].notna(),
            opening_home,
        )
    )

    out["current_market_home_margin"] = (
        market_margin_from_column(
            frame,
            found["current_spread"],
            home_spread_convention=False,
        )
    )
    current_home = market_margin_from_column(
        frame,
        found["current_home_spread"],
        home_spread_convention=True,
    )
    out["current_market_home_margin"] = (
        out["current_market_home_margin"]
        .where(
            out["current_market_home_margin"].notna(),
            current_home,
        )
    )

    out["market_home_price"] = (
        pd.to_numeric(
            frame[found["home_spread_price"]],
            errors="coerce",
        )
        if found["home_spread_price"] is not None
        else np.nan
    )
    out["market_away_price"] = (
        pd.to_numeric(
            frame[found["away_spread_price"]],
            errors="coerce",
        )
        if found["away_spread_price"] is not None
        else np.nan
    )
    if found["generic_spread_price"] is not None:
        generic = pd.to_numeric(
            frame[found["generic_spread_price"]],
            errors="coerce",
        )
        out["market_home_price"] = out["market_home_price"].where(
            out["market_home_price"].notna(), generic
        )
        out["market_away_price"] = out["market_away_price"].where(
            out["market_away_price"].notna(), generic
        )

    for output_column, source_key in (
        ("sportsbook", "sportsbook"),
        ("bookmaker_key", "bookmaker_key"),
        ("market_source", "market_source"),
        ("market_retrieved_at_utc", "market_retrieved_at_utc"),
        ("market_last_update_utc", "market_last_update_utc"),
    ):
        if found[source_key] is None:
            out[output_column] = np.nan
        else:
            out[output_column] = frame[found[source_key]].map(
                lambda value: (
                    np.nan
                    if pd.isna(value) or not str(value).strip()
                    else str(value).strip()
                )
            )

    for output_column, source_key in (
        ("market_line_age_minutes", "market_line_age_minutes"),
        ("market_stale_flag", "market_stale_flag"),
        ("source_priority_rank", "source_priority_rank"),
        ("circa_line_flag", "circa_line_flag"),
    ):
        out[output_column] = (
            pd.to_numeric(frame[found[source_key]], errors="coerce")
            if found[source_key] is not None
            else np.nan
        )

    if found["game_id"] is not None:
        out["game_id"] = (
            frame[found["game_id"]].astype(str).str.strip()
        )
    else:
        out["game_id"] = (
            out["season"].fillna(SEASON).astype("Int64").astype(str)
            + "_"
            + out["week"].astype("Int64").astype(str)
            + "_"
            + out["away_team"]
            + "_"
            + out["home_team"]
        )

    out = out[
        out["week"].notna()
        & out["home_team"].ne("")
        & out["away_team"].ne("")
        & out["game_type"].isin(REGULAR_SEASON_TYPES)
    ].copy()
    if out.empty:
        return out

    out["season"] = out["season"].fillna(SEASON).astype(int)
    out = out[out["season"].eq(SEASON)].copy()
    out["week"] = out["week"].astype(int)
    out["completed"] = (
        out["home_score"].notna()
        & out["away_score"].notna()
    ).astype(int)
    out["schedule_source"] = source

    return (
        out.sort_values(["week", "game_date", "game_id"])
        .drop_duplicates(["week", "home_team", "away_team"], keep="last")
        .reset_index(drop=True)
    )


def load_external_schedule() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        import nflreadpy as nfl  # type: ignore

        loader = getattr(nfl, "load_schedules", None)
        if loader is not None:
            for attempt in (
                lambda: loader([SEASON]),
                lambda: loader(seasons=[SEASON]),
            ):
                try:
                    frame = attempt()
                    if hasattr(frame, "to_pandas"):
                        frame = frame.to_pandas()
                    frame = pd.DataFrame(frame)
                    if not frame.empty:
                        return frame, "nflreadpy.load_schedules"
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    raise RuntimeError(
        "Unable to load external schedule: " + " | ".join(errors)
    )


def schedule_file_candidates(
    project_root: Path,
) -> list[Path]:
    names = (
        "nfl_schedule_2026.csv",
        "nfl_2026_schedule.csv",
        "2026_nfl_schedule.csv",
        "nfl_schedule_factors_2026.csv",
        "nfl_schedule_2026.xlsx",
        "nfl_schedule_factors_2026.xlsx",
    )
    paths: list[Path] = []
    for folder in (
        project_root,
        project_root / "outputs",
        project_root / "data",
    ):
        for name in names:
            paths.append(folder / name)
    return paths


def load_primary_schedule(
    connection: sqlite3.Connection,
    project_root: Path,
    explicit_path: Optional[Path],
) -> pd.DataFrame:
    errors: list[str] = []

    if explicit_path is not None:
        frame = standardize_schedule(
            load_tabular_file(explicit_path),
            f"file.{explicit_path}",
        )
        if not frame.empty:
            return frame

    for table_name in SCHEDULE_TABLE_CANDIDATES:
        raw = read_table(connection, table_name)
        if raw.empty:
            continue
        try:
            frame = standardize_schedule(
                raw, f"sqlite.{table_name}"
            )
            if not frame.empty:
                return frame
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{table_name}: {exc}")

    for path in schedule_file_candidates(project_root):
        if not path.exists():
            continue
        try:
            frame = standardize_schedule(
                load_tabular_file(path),
                f"file.{path}",
            )
            if not frame.empty:
                return frame
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")

    try:
        raw, source = load_external_schedule()
        frame = standardize_schedule(raw, source)
        if not frame.empty:
            return frame
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    raise RuntimeError(
        "Unable to locate a usable 2026 schedule. "
        + " | ".join(errors)
    )


FACTOR_COLUMNS = (
    "travel_miles",
    "home_days_rest",
    "away_days_rest",
    "rest_differential",
    "home_rest_advantage_3plus",
    "away_rest_advantage_3plus",
    "travel_dist_2000_plus",
    "back_to_back_west_east",
    "back_to_back_west_central",
    "back_to_back_away",
    "three_of_four_away",
    "bye_before_away",
    "bye_before_home",
    "international_game",
    "away_consecutive_road_games",
    "away_road_games_last4",
    "location",
)


def standardize_factors(
    raw: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    frame.columns = [
        str(column).lower().strip()
        for column in frame.columns
    ]
    week_col = first_existing(
        frame.columns, SCHEDULE_ALIASES["week"]
    )
    home_col = first_existing(
        frame.columns, SCHEDULE_ALIASES["home_team"]
    )
    away_col = first_existing(
        frame.columns, SCHEDULE_ALIASES["away_team"]
    )
    if week_col is None or home_col is None or away_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(index=frame.index)
    out["week"] = pd.to_numeric(
        frame[week_col], errors="coerce"
    )
    out["home_team"] = frame[home_col].map(normalize_team)
    out["away_team"] = frame[away_col].map(normalize_team)

    boolean_columns = {
        "home_rest_advantage_3plus",
        "away_rest_advantage_3plus",
        "travel_dist_2000_plus",
        "back_to_back_west_east",
        "back_to_back_west_central",
        "back_to_back_away",
        "three_of_four_away",
        "bye_before_away",
        "bye_before_home",
        "international_game",
    }
    numeric_columns = {
        "travel_miles",
        "home_days_rest",
        "away_days_rest",
        "rest_differential",
        "away_consecutive_road_games",
        "away_road_games_last4",
    }

    for column in FACTOR_COLUMNS:
        source_col = first_existing(frame.columns, (column,))
        if column in boolean_columns:
            out[column] = bool_series(frame, source_col).astype(int)
        elif column in numeric_columns:
            out[column] = (
                pd.to_numeric(
                    frame[source_col], errors="coerce"
                )
                if source_col is not None
                else np.nan
            )
        else:
            out[column] = (
                frame[source_col].astype(str)
                if source_col is not None
                else ""
            )

    out["factor_source"] = source
    out = out[
        out["week"].notna()
        & out["home_team"].ne("")
        & out["away_team"].ne("")
    ].copy()
    out["week"] = out["week"].astype(int)
    return (
        out.drop_duplicates(
            ["week", "home_team", "away_team"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def load_schedule_factors(
    connection: sqlite3.Connection,
    project_root: Path,
    explicit_path: Optional[Path],
) -> pd.DataFrame:
    candidates: list[tuple[pd.DataFrame, str]] = []

    if explicit_path is not None:
        candidates.append(
            (
                load_tabular_file(explicit_path),
                f"file.{explicit_path}",
            )
        )

    raw_table = read_table(connection, SCHEDULE_FACTOR_TABLE)
    if not raw_table.empty:
        candidates.append(
            (raw_table, f"sqlite.{SCHEDULE_FACTOR_TABLE}")
        )

    for path in (
        project_root / "outputs" / "nfl_schedule_factors_2026.csv",
        project_root / "nfl_schedule_factors_2026.csv",
        project_root / "outputs" / "nfl_schedule_factors_2026.xlsx",
        project_root / "nfl_schedule_factors_2026.xlsx",
    ):
        if path.exists():
            candidates.append(
                (load_tabular_file(path), f"file.{path}")
            )

    for raw, source in candidates:
        factors = standardize_factors(raw, source)
        if not factors.empty:
            return factors
    return pd.DataFrame()


def merge_market_source(
    schedule: pd.DataFrame,
    market_path: Optional[Path],
) -> pd.DataFrame:
    if market_path is None:
        return schedule

    market = standardize_schedule(
        load_tabular_file(market_path),
        f"market_file.{market_path}",
    )
    if market.empty:
        return schedule

    columns = [
        "week",
        "home_team",
        "away_team",
        "opening_market_home_margin",
        "current_market_home_margin",
        "market_home_price",
        "market_away_price",
        "sportsbook",
        "bookmaker_key",
        "market_source",
        "market_retrieved_at_utc",
        "market_last_update_utc",
        "market_line_age_minutes",
        "market_stale_flag",
        "source_priority_rank",
        "circa_line_flag",
        "schedule_source",
    ]
    market = market[columns].rename(
        columns={"schedule_source": "market_file_source"}
    )

    out = schedule.merge(
        market,
        on=["week", "home_team", "away_team"],
        how="left",
        suffixes=("", "_external"),
        validate="one_to_one",
    )
    for column in (
        "opening_market_home_margin",
        "current_market_home_margin",
        "market_home_price",
        "market_away_price",
        "sportsbook",
        "bookmaker_key",
        "market_source",
        "market_retrieved_at_utc",
        "market_last_update_utc",
        "market_line_age_minutes",
        "market_stale_flag",
        "source_priority_rank",
        "circa_line_flag",
    ):
        external = f"{column}_external"
        out[column] = out[external].where(
            out[external].notna(), out[column]
        )
        out = out.drop(columns=[external])
    out["market_source"] = (
        out["market_source"]
        .fillna(out["market_file_source"])
        .fillna(out["schedule_source"])
    )
    out = out.drop(columns=["market_file_source"])
    return out


def attach_factors(
    schedule: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    out = schedule.copy()
    numeric_optional = {
        "travel_miles",
        "home_days_rest",
        "away_days_rest",
        "rest_differential",
        "away_consecutive_road_games",
        "away_road_games_last4",
    }
    boolean_columns = {
        "home_rest_advantage_3plus",
        "away_rest_advantage_3plus",
        "travel_dist_2000_plus",
        "back_to_back_west_east",
        "back_to_back_west_central",
        "back_to_back_away",
        "three_of_four_away",
        "bye_before_away",
        "bye_before_home",
        "international_game",
    }

    if factors.empty:
        for column in FACTOR_COLUMNS:
            if column in out.columns:
                continue
            out[column] = np.nan if column in numeric_optional else 0
        out["factor_source"] = "not_available"
    else:
        rename_map = {
            column: f"factor__{column}"
            for column in FACTOR_COLUMNS
            if column in factors.columns
        }
        factor_payload = factors.rename(columns=rename_map)
        out = out.merge(
            factor_payload,
            on=["week", "home_team", "away_team"],
            how="left",
            validate="one_to_one",
        )

        for column in FACTOR_COLUMNS:
            factor_column = f"factor__{column}"
            if factor_column not in out.columns:
                if column not in out.columns:
                    out[column] = (
                        np.nan if column in numeric_optional else 0
                    )
                continue

            factor_values = out[factor_column]
            if column in boolean_columns:
                base_values = (
                    pd.to_numeric(out[column], errors="coerce")
                    if column in out.columns
                    else pd.Series(0, index=out.index, dtype=float)
                )
                out[column] = (
                    base_values.fillna(0).ne(0)
                    | pd.to_numeric(
                        factor_values, errors="coerce"
                    ).fillna(0).ne(0)
                ).astype(int)
            else:
                if column in out.columns:
                    out[column] = factor_values.where(
                        factor_values.notna(), out[column]
                    )
                else:
                    out[column] = factor_values
            out = out.drop(columns=[factor_column])

        out["factor_source"] = out["factor_source"].fillna(
            "not_matched"
        )

    for column in boolean_columns:
        out[column] = (
            pd.to_numeric(out[column], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    factor_international = out["international_game"].eq(1)
    out["neutral_site"] = (
        out["neutral_site"].eq(1) | factor_international
    ).astype(int)
    out["international_treated_as_neutral_flag"] = (
        out["international_treated_as_neutral_flag"].eq(1)
        | (
            factor_international
            & out["explicit_neutral_site"].eq(0)
        )
    ).astype(int)
    return out


def select_prediction_week(
    schedule: pd.DataFrame,
    requested_week: Optional[int],
    as_of_date: Optional[str],
) -> int:
    if requested_week is not None:
        if schedule["week"].eq(requested_week).sum() == 0:
            raise RuntimeError(
                f"No games found for 2026 Week {requested_week}."
            )
        return int(requested_week)

    incomplete = schedule[schedule["completed"].eq(0)].copy()
    if incomplete.empty:
        raise RuntimeError("No incomplete 2026 regular-season games found.")

    reference = (
        pd.Timestamp(as_of_date).normalize()
        if as_of_date is not None
        else pd.Timestamp.now().normalize()
    )
    dated = incomplete[
        incomplete["game_date"].notna()
        & incomplete["game_date"].ge(reference)
    ]
    if not dated.empty:
        return int(dated["week"].min())
    return int(incomplete["week"].min())


def choose_market_line(
    frame: pd.DataFrame,
    preference: str,
) -> pd.DataFrame:
    out = frame.copy()
    if preference == "opening":
        first = "opening_market_home_margin"
        second = "current_market_home_margin"
        first_label = "OPENING"
        second_label = "CURRENT_OR_CLOSING"
    else:
        first = "current_market_home_margin"
        second = "opening_market_home_margin"
        first_label = "CURRENT_OR_CLOSING"
        second_label = "OPENING"

    out["available_market_home_margin"] = out[first].where(
        out[first].notna(), out[second]
    )
    out["available_market_line_source"] = np.where(
        out[first].notna(),
        first_label,
        np.where(
            out[second].notna(), second_label, "NONE"
        ),
    )
    out["market_line_preference"] = preference
    return out


# =============================================================================
# MANUAL ADJUSTMENTS AND SCHEDULE ADJUSTMENTS
# =============================================================================

def ensure_manual_adjustment_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MANUAL_ADJUSTMENT_TABLE} (
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team TEXT NOT NULL,
            adjustment_points REAL NOT NULL,
            reason TEXT,
            active_flag INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT,
            PRIMARY KEY (season, week, team, reason)
        )
        """
    )
    connection.commit()


def load_manual_adjustments(
    connection: sqlite3.Connection,
    week: int,
) -> pd.DataFrame:
    ensure_manual_adjustment_table(connection)
    frame = read_table(connection, MANUAL_ADJUSTMENT_TABLE)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "manual_team_adjustment",
                "manual_adjustment_reason",
                "manual_adjustment_rows",
            ]
        )

    required = {
        "season", "week", "team",
        "adjustment_points", "active_flag",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"{MANUAL_ADJUSTMENT_TABLE} has an invalid schema."
        )

    frame["season"] = pd.to_numeric(
        frame["season"], errors="coerce"
    )
    frame["week"] = pd.to_numeric(
        frame["week"], errors="coerce"
    )
    frame["active_flag"] = pd.to_numeric(
        frame["active_flag"], errors="coerce"
    ).fillna(0)
    frame["adjustment_points"] = pd.to_numeric(
        frame["adjustment_points"], errors="coerce"
    )
    frame["team"] = frame["team"].map(normalize_team)
    if "reason" not in frame.columns:
        frame["reason"] = ""

    frame = frame[
        frame["season"].eq(SEASON)
        & frame["week"].eq(week)
        & frame["active_flag"].eq(1)
        & frame["team"].ne("")
        & frame["adjustment_points"].notna()
    ].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "manual_team_adjustment",
                "manual_adjustment_reason",
                "manual_adjustment_rows",
            ]
        )

    grouped = (
        frame.groupby("team", as_index=False)
        .agg(
            manual_team_adjustment=(
                "adjustment_points", "sum"
            ),
            manual_adjustment_reason=(
                "reason",
                lambda values: " | ".join(
                    sorted(
                        {
                            str(value).strip()
                            for value in values
                            if str(value).strip()
                        }
                    )
                ),
            ),
            manual_adjustment_rows=(
                "adjustment_points", "size"
            ),
        )
    )
    grouped["manual_team_adjustment"] = grouped[
        "manual_team_adjustment"
    ].clip(
        -MANUAL_TEAM_ADJUSTMENT_CAP,
        MANUAL_TEAM_ADJUSTMENT_CAP,
    )
    return grouped


def compute_rest_adjustment(game: pd.Series) -> tuple[float, str]:
    home_bye = int(game.get("bye_before_home", 0) or 0) == 1
    away_bye = int(game.get("bye_before_away", 0) or 0) == 1

    if home_bye and not away_bye:
        return HOME_BYE_ADVANTAGE_POINTS, "HOME_OFF_BYE"
    if away_bye and not home_bye:
        return -AWAY_BYE_ADVANTAGE_POINTS, "AWAY_OFF_BYE"

    home_adv = (
        int(game.get("home_rest_advantage_3plus", 0) or 0)
        == 1
    )
    away_adv = (
        int(game.get("away_rest_advantage_3plus", 0) or 0)
        == 1
    )
    differential = pd.to_numeric(
        pd.Series([game.get("rest_differential")]),
        errors="coerce",
    ).iloc[0]

    if home_adv or (
        pd.notna(differential) and float(differential) >= 3
    ):
        return REST_ADVANTAGE_3PLUS_POINTS, "HOME_REST_3PLUS"
    if away_adv or (
        pd.notna(differential) and float(differential) <= -3
    ):
        return -REST_ADVANTAGE_3PLUS_POINTS, "AWAY_REST_3PLUS"
    return 0.0, "NONE"


def compute_travel_adjustment(
    game: pd.Series,
) -> tuple[float, str]:
    adjustment = 0.0
    reasons: list[str] = []

    factor_values = (
        (
            "travel_dist_2000_plus",
            TRAVEL_2000_PLUS_POINTS,
            "AWAY_TRAVEL_2000_PLUS",
        ),
        (
            "back_to_back_west_east",
            BACK_TO_BACK_WEST_EAST_POINTS,
            "AWAY_B2B_WEST_EAST",
        ),
        (
            "back_to_back_west_central",
            BACK_TO_BACK_WEST_CENTRAL_POINTS,
            "AWAY_B2B_WEST_CENTRAL",
        ),
        (
            "back_to_back_away",
            BACK_TO_BACK_AWAY_POINTS,
            "AWAY_B2B_ROAD",
        ),
        (
            "three_of_four_away",
            THREE_OF_FOUR_AWAY_POINTS,
            "AWAY_3_OF_4_ROAD",
        ),
    )
    for column, points, label in factor_values:
        if int(game.get(column, 0) or 0) == 1:
            adjustment += points
            reasons.append(label)

    adjustment = float(
        np.clip(
            adjustment,
            -TRAVEL_SCHEDULE_ADJUSTMENT_CAP,
            TRAVEL_SCHEDULE_ADJUSTMENT_CAP,
        )
    )
    return adjustment, " | ".join(reasons) if reasons else "NONE"


# =============================================================================
# PREDICTIONS AND STAKING
# =============================================================================

def build_predictions(
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    week: int,
    rating_source: str,
    hfa: float,
    hfa_source: str,
    residual_info: dict[str, Any],
    manual: pd.DataFrame,
    args: argparse.Namespace,
    run_id: str,
    rating_lineage: dict[str, Any],
) -> pd.DataFrame:
    rating_index = ratings.set_index("team")
    manual_index = (
        manual.set_index("team")
        if not manual.empty
        else pd.DataFrame().set_index(
            pd.Index([], name="team")
        )
    )

    rows: list[dict[str, Any]] = []
    prediction_timestamp = now_string()
    as_of_date = str(rating_lineage["form_as_of_date"])

    residual_mean = float(
        residual_info["historical_residual_mean"]
    )
    residual_sigma = float(
        residual_info["historical_residual_sigma"]
    )

    for _, game in games.iterrows():
        home = str(game["home_team"])
        away = str(game["away_team"])
        if home not in rating_index.index:
            raise RuntimeError(f"Missing live rating for {home}.")
        if away not in rating_index.index:
            raise RuntimeError(f"Missing live rating for {away}.")

        home_rating = float(
            rating_index.loc[home, "power_rating_points"]
        )
        away_rating = float(
            rating_index.loc[away, "power_rating_points"]
        )
        neutral_diff = home_rating - away_rating

        neutral_site = int(game["neutral_site"])
        home_field = 0.0 if neutral_site else float(hfa)

        rest_adjustment, rest_reason = compute_rest_adjustment(game)
        travel_adjustment, travel_reason = (
            compute_travel_adjustment(game)
        )
        schedule_adjustment = float(
            np.clip(
                rest_adjustment + travel_adjustment,
                -TOTAL_SCHEDULE_ADJUSTMENT_CAP,
                TOTAL_SCHEDULE_ADJUSTMENT_CAP,
            )
        )

        def manual_value(team: str) -> tuple[float, str, int]:
            if manual.empty or team not in manual_index.index:
                return 0.0, "", 0
            row = manual_index.loc[team]
            return (
                float(row["manual_team_adjustment"]),
                str(row["manual_adjustment_reason"]),
                int(row["manual_adjustment_rows"]),
            )

        home_manual, home_manual_reason, home_manual_rows = (
            manual_value(home)
        )
        away_manual, away_manual_reason, away_manual_rows = (
            manual_value(away)
        )
        manual_game_adjustment = float(
            np.clip(
                home_manual - away_manual,
                -MANUAL_GAME_ADJUSTMENT_CAP,
                MANUAL_GAME_ADJUSTMENT_CAP,
            )
        )

        projected_home_margin = (
            neutral_diff
            + home_field
            + schedule_adjustment
            + manual_game_adjustment
        )

        frozen = {
            "game_id": str(game["game_id"]),
            "home_power_rating_points": home_rating,
            "away_power_rating_points": away_rating,
            "neutral_rating_difference": neutral_diff,
            "home_field_points": home_field,
            "rest_adjustment_points": rest_adjustment,
            "travel_schedule_adjustment_points": travel_adjustment,
            "manual_availability_adjustment_points": (
                manual_game_adjustment
            ),
            "projected_home_margin": projected_home_margin,
        }
        projection_hash = independent_projection_hash(frozen)

        market_home_margin = game[
            "available_market_home_margin"
        ]
        market_attached = int(pd.notna(market_home_margin))
        edge_home = (
            projected_home_margin - float(market_home_margin)
            if market_attached
            else np.nan
        )
        absolute_difference = (
            abs(float(edge_home))
            if pd.notna(edge_home)
            else np.nan
        )

        selected_side = ""
        selected_team = ""
        if pd.notna(edge_home):
            if float(edge_home) > 0:
                selected_side = "HOME"
                selected_team = home
            elif float(edge_home) < 0:
                selected_side = "AWAY"
                selected_team = away
            else:
                selected_side = "NONE"

        home_cover_probability = np.nan
        away_cover_probability = np.nan
        selected_probability = np.nan
        selected_price = int(args.default_spread_price)
        price_fallback_used = 1
        break_even = np.nan
        probability_edge = np.nan
        full_kelly_fraction = 0.0

        if pd.notna(edge_home):
            home_cover_probability = normal_cdf(
                (
                    float(edge_home)
                    + residual_mean
                )
                / residual_sigma
            )
            home_cover_probability = float(
                np.clip(
                    home_cover_probability,
                    PROBABILITY_CLIP,
                    1.0 - PROBABILITY_CLIP,
                )
            )
            away_cover_probability = 1.0 - home_cover_probability

            if selected_side == "HOME":
                selected_probability = home_cover_probability
                raw_price = game["market_home_price"]
            elif selected_side == "AWAY":
                selected_probability = away_cover_probability
                raw_price = game["market_away_price"]
            else:
                raw_price = np.nan

            selected_price, price_fallback_used = (
                sanitize_american_price(
                    raw_price,
                    fallback=args.default_spread_price,
                )
            )
            break_even = implied_probability_from_american(
                selected_price
            )
            probability_edge = (
                selected_probability - break_even
                if pd.notna(selected_probability)
                else np.nan
            )

            if pd.notna(selected_probability):
                net_win = net_profit_per_unit(selected_price)
                full_kelly_fraction = (
                    net_win * float(selected_probability)
                    - (1.0 - float(selected_probability))
                ) / net_win
                full_kelly_fraction = float(
                    np.clip(full_kelly_fraction, 0.0, 1.0)
                )

        completed = int(game["completed"])
        if completed:
            decision = "COMPLETED_GAME"
            qualifies = 0
        elif not market_attached:
            decision = "MARKET_UNAVAILABLE"
            qualifies = 0
        elif week == 18 and not args.include_week18:
            decision = "WEEK18_EXCLUDED"
            qualifies = 0
        elif (
            absolute_difference
            < args.minimum_spread_difference
        ):
            decision = "NO_BET_BELOW_THRESHOLD"
            qualifies = 0
        elif (
            absolute_difference
            > args.maximum_market_disagreement
        ):
            decision = "REVIEW_EXTREME_DISAGREEMENT"
            qualifies = 0
        elif pd.isna(probability_edge) or probability_edge <= 0:
            decision = "NO_BET_NONPOSITIVE_PRICE_EDGE"
            qualifies = 0
        else:
            decision = "BET"
            qualifies = 1

        quarter_kelly_fraction = min(
            max(
                0.0,
                args.quarter_kelly_multiplier
                * full_kelly_fraction,
            ),
            args.max_kelly_bet_fraction,
        )

        flat_stake = float(args.flat_stake) if qualifies else 0.0
        kelly_stake_raw = (
            args.bankroll * quarter_kelly_fraction
            if qualifies
            else 0.0
        )
        kelly_stake = (
            float(round(kelly_stake_raw, 0))
            if qualifies
            else 0.0
        )

        projected_spread = spread_label(
            home, away, projected_home_margin
        )
        market_spread = spread_label(
            home, away, market_home_margin
        )
        selected_market_line = selected_market_line_label(
            selected_team,
            home,
            away,
            market_home_margin,
        )
        recommendation = (
            f"BET {selected_market_line}"
            if qualifies
            else decision
        )

        home_rating_row = rating_index.loc[home]
        away_rating_row = rating_index.loc[away]

        row = {
            "run_id": run_id,
            "season": SEASON,
            "week": int(week),
            "game_id": str(game["game_id"]),
            "game_date": game["game_date"],
            "prediction_as_of_date": as_of_date,
            "prediction_timestamp": prediction_timestamp,
            "away_team": away,
            "home_team": home,
            "neutral_site": neutral_site,
            "explicit_neutral_site": int(
                game["explicit_neutral_site"]
            ),
            "international_game": int(
                game.get("international_game", 0)
            ),
            "international_treated_as_neutral_flag": int(
                game["international_treated_as_neutral_flag"]
            ),
            "completed": completed,

            "build_id": BUILD_ID,
            "version": VERSION,
            "model_variant": MODEL_VARIANT,
            "rating_source": rating_source,
            **rating_lineage,
            "home_rating_as_of_date": home_rating_row.get(
                "as_of_date", np.nan
            ),
            "away_rating_as_of_date": away_rating_row.get(
                "as_of_date", np.nan
            ),
            "home_rating_through_week": home_rating_row.get(
                "through_week", np.nan
            ),
            "away_rating_through_week": away_rating_row.get(
                "through_week", np.nan
            ),
            "home_power_rating_points": home_rating,
            "away_power_rating_points": away_rating,
            "neutral_rating_difference": neutral_diff,
            "home_field_points": home_field,
            "base_home_field_points": float(hfa),
            "home_field_source": hfa_source,

            "home_days_rest": game.get("home_days_rest", np.nan),
            "away_days_rest": game.get("away_days_rest", np.nan),
            "rest_differential": game.get(
                "rest_differential", np.nan
            ),
            "rest_adjustment_points": rest_adjustment,
            "rest_adjustment_reason": rest_reason,
            "travel_miles": game.get("travel_miles", np.nan),
            "travel_dist_2000_plus": int(
                game.get("travel_dist_2000_plus", 0) or 0
            ),
            "back_to_back_west_east": int(
                game.get("back_to_back_west_east", 0) or 0
            ),
            "back_to_back_west_central": int(
                game.get("back_to_back_west_central", 0) or 0
            ),
            "back_to_back_away": int(
                game.get("back_to_back_away", 0) or 0
            ),
            "three_of_four_away": int(
                game.get("three_of_four_away", 0) or 0
            ),
            "travel_schedule_adjustment_points": travel_adjustment,
            "travel_schedule_adjustment_reason": travel_reason,
            "total_schedule_adjustment_points": schedule_adjustment,
            "schedule_adjustment_cap": (
                TOTAL_SCHEDULE_ADJUSTMENT_CAP
            ),
            "factor_source": game.get(
                "factor_source", "not_available"
            ),

            "home_manual_adjustment_points": home_manual,
            "away_manual_adjustment_points": away_manual,
            "manual_availability_adjustment_points": (
                manual_game_adjustment
            ),
            "home_manual_adjustment_reason": home_manual_reason,
            "away_manual_adjustment_reason": away_manual_reason,
            "home_manual_adjustment_rows": home_manual_rows,
            "away_manual_adjustment_rows": away_manual_rows,
            "manual_adjustment_table": MANUAL_ADJUSTMENT_TABLE,

            "projected_home_margin": projected_home_margin,
            "projected_spread": projected_spread,
            "independent_projection_hash": projection_hash,
            "prediction_uses_market_inputs": 0,
            "market_attached_after_prediction_freeze": (
                market_attached
            ),

            "opening_market_home_margin": game[
                "opening_market_home_margin"
            ],
            "current_market_home_margin": game[
                "current_market_home_margin"
            ],
            "available_market_home_margin": market_home_margin,
            "available_market_spread": market_spread,
            "available_market_line_source": game[
                "available_market_line_source"
            ],
            "market_line_preference": game[
                "market_line_preference"
            ],
            "market_home_price": game["market_home_price"],
            "market_away_price": game["market_away_price"],
            "schedule_source": game["schedule_source"],
            "market_source": game.get(
                "market_source", game["schedule_source"]
            ),
            "sportsbook": game.get("sportsbook", np.nan),
            "bookmaker_key": game.get("bookmaker_key", np.nan),
            "market_retrieved_at_utc": game.get(
                "market_retrieved_at_utc", np.nan
            ),
            "market_last_update_utc": game.get(
                "market_last_update_utc", np.nan
            ),
            "market_line_age_minutes": game.get(
                "market_line_age_minutes", np.nan
            ),
            "market_stale_flag": game.get(
                "market_stale_flag", np.nan
            ),
            "source_priority_rank": game.get(
                "source_priority_rank", np.nan
            ),
            "circa_line_flag": game.get("circa_line_flag", np.nan),

            "model_market_edge_home_points": edge_home,
            "absolute_spread_difference_points": (
                absolute_difference
            ),
            "minimum_spread_difference_points": (
                args.minimum_spread_difference
            ),
            "maximum_market_disagreement_points": (
                args.maximum_market_disagreement
            ),
            "selected_side": selected_side,
            "selected_team": selected_team,
            "selected_market_line": selected_market_line,
            "qualifies_for_stake": qualifies,
            "decision": decision,
            "recommendation": recommendation,

            "probability_method": residual_info[
                "probability_method"
            ],
            "probability_source": residual_info[
                "probability_source"
            ],
            "probability_is_model_specific_calibration": (
                residual_info[
                    "probability_is_model_specific_calibration"
                ]
            ),
            "historical_residual_rows": residual_info[
                "historical_residual_rows"
            ],
            "historical_residual_mean": residual_mean,
            "historical_residual_sigma": residual_sigma,
            "home_cover_probability": home_cover_probability,
            "away_cover_probability": away_cover_probability,
            "model_selected_cover_probability": (
                selected_probability
            ),
            "selected_spread_price": selected_price,
            "price_fallback_used": price_fallback_used,
            "market_break_even_probability": break_even,
            "probability_edge": probability_edge,

            "starting_bankroll": float(args.bankroll),
            "flat_stake_setting": float(args.flat_stake),
            "flat_recommended_stake": flat_stake,
            "full_kelly_fraction": full_kelly_fraction,
            "fractional_kelly_multiplier": (
                args.quarter_kelly_multiplier
            ),
            "uncapped_fractional_kelly_fraction": (
                args.quarter_kelly_multiplier
                * full_kelly_fraction
            ),
            "max_kelly_bet_fraction": (
                args.max_kelly_bet_fraction
            ),
            "actual_fractional_kelly_fraction": (
                quarter_kelly_fraction
            ),
            "fractional_kelly_stake_raw": kelly_stake_raw,
            "fractional_kelly_recommended_stake": kelly_stake,
            "date_imported": prediction_timestamp,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# VALIDATION, SAVE, AND REPORT
# =============================================================================

def validate_predictions(
    predictions: pd.DataFrame,
    games: pd.DataFrame,
) -> None:
    if predictions.empty:
        raise RuntimeError("Prediction output is empty.")
    if len(predictions) != len(games):
        raise RuntimeError(
            "Prediction output does not reconcile to scheduled games."
        )
    if predictions["game_id"].nunique() != len(predictions):
        raise RuntimeError("Prediction output has duplicate game IDs.")
    if predictions["prediction_uses_market_inputs"].ne(0).any():
        raise RuntimeError(
            "Market information entered the independent projection."
        )
    for column, expected in (
        ("build_id", BUILD_ID),
        ("version", VERSION),
        ("model_variant", MODEL_VARIANT),
        ("form_build_id", EXPECTED_FORM_BUILD_ID),
        ("form_version", EXPECTED_FORM_VERSION),
        (
            "structural_power_build_id",
            EXPECTED_STRUCTURAL_POWER_BUILD_ID,
        ),
    ):
        if set(predictions[column].astype(str)) != {expected}:
            raise RuntimeError(
                f"Prediction lineage mismatch for {column}."
            )
    for column in (
        "run_id",
        "preseason_snapshot_hash",
        "structural_power_snapshot_hash",
        "structural_power_date_imported",
        "form_date_imported",
    ):
        if predictions[column].isna().any() or predictions[column].astype(
            str
        ).str.strip().eq("").any():
            raise RuntimeError(
                f"Prediction lineage field {column} is missing."
            )
    if predictions["run_id"].nunique() != 1:
        raise RuntimeError("Prediction output contains multiple run IDs.")
    if predictions["projected_home_margin"].isna().any():
        raise RuntimeError("Projected margins contain missing values.")
    if (
        predictions["total_schedule_adjustment_points"].abs().max()
        > TOTAL_SCHEDULE_ADJUSTMENT_CAP + 1e-9
    ):
        raise RuntimeError("Schedule adjustment cap was exceeded.")
    if (
        predictions["manual_availability_adjustment_points"]
        .abs()
        .max()
        > MANUAL_GAME_ADJUSTMENT_CAP + 1e-9
    ):
        raise RuntimeError("Manual adjustment cap was exceeded.")
    if (
        predictions.loc[
            predictions["neutral_site"].eq(1),
            "home_field_points",
        ]
        .abs()
        .max()
        > 1e-9
    ):
        raise RuntimeError(
            "Neutral/international game received home-field points."
        )
    if (
        predictions.loc[
            predictions["qualifies_for_stake"].eq(0),
            [
                "flat_recommended_stake",
                "fractional_kelly_recommended_stake",
            ],
        ]
        .abs()
        .to_numpy()
        .max(initial=0)
        > 1e-9
    ):
        raise RuntimeError(
            "A non-qualifying game received a stake."
        )
    if (
        predictions.loc[
            predictions["qualifies_for_stake"].eq(1),
            "probability_edge",
        ]
        .le(0)
        .any()
    ):
        raise RuntimeError(
            "A qualifying bet has nonpositive price-adjusted edge."
        )

    recalculated = (
        predictions["neutral_rating_difference"]
        + predictions["home_field_points"]
        + predictions["total_schedule_adjustment_points"]
        + predictions[
            "manual_availability_adjustment_points"
        ]
    )
    if not np.allclose(
        recalculated,
        predictions["projected_home_margin"],
        atol=1e-9,
    ):
        raise RuntimeError(
            "Projection components do not reconcile."
        )


def save_outputs(
    connection: sqlite3.Connection,
    predictions: pd.DataFrame,
    run_audit: pd.DataFrame,
    project_root: Path,
    no_csv: bool,
) -> tuple[Optional[Path], Optional[Path]]:
    predictions.to_sql(
        OUTPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )
    append_with_schema_evolution(
        connection,
        OUTPUT_HISTORY_TABLE,
        predictions,
    )

    run_audit.to_sql(
        RUN_AUDIT_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )
    append_with_schema_evolution(
        connection,
        RUN_AUDIT_HISTORY_TABLE,
        run_audit,
    )
    connection.commit()

    if no_csv:
        return None, None

    output_dir = project_root / "outputs"
    snapshot_dir = output_dir / "weekly_snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    current_path = output_dir / f"{OUTPUT_TABLE}.csv"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = snapshot_dir / (
        f"{OUTPUT_TABLE}_{stamp}.csv"
    )
    predictions.to_csv(
        current_path,
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        snapshot_path,
        index=False,
        encoding="utf-8-sig",
    )
    return current_path, snapshot_path


def print_report(
    predictions: pd.DataFrame,
    week: int,
    hfa: float,
    hfa_source: str,
    residual_info: dict[str, Any],
) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 132)
    LOGGER.info(
        "[PREDICT] 2026 NFL CANONICAL WEEKLY POWER-SPREAD PROJECTIONS"
    )
    LOGGER.info("=" * 132)
    LOGGER.info("[PREDICT] Build ID: %s", BUILD_ID)
    LOGGER.info("[PREDICT] Version: %s", VERSION)
    LOGGER.info("[PREDICT] Week: %s", week)
    LOGGER.info(
        "[PREDICT] Games: %s | markets attached: %s | bets: %s",
        len(predictions),
        int(
            predictions[
                "market_attached_after_prediction_freeze"
            ].sum()
        ),
        int(predictions["qualifies_for_stake"].sum()),
    )
    LOGGER.info(
        "[PREDICT] Base historical HFA: %.3f via %s",
        hfa,
        hfa_source,
    )
    LOGGER.info(
        "[PREDICT] Market used in independent projection: NO"
    )
    LOGGER.info(
        "[PREDICT] Schedule adjustment range: %.3f to %.3f",
        predictions[
            "total_schedule_adjustment_points"
        ].min(),
        predictions[
            "total_schedule_adjustment_points"
        ].max(),
    )
    LOGGER.info(
        "[PREDICT] Manual availability adjustment range: "
        "%.3f to %.3f",
        predictions[
            "manual_availability_adjustment_points"
        ].min(),
        predictions[
            "manual_availability_adjustment_points"
        ].max(),
    )
    LOGGER.info(
        "[PREDICT] Probability method: %s | sigma=%.3f | "
        "model-specific calibration=%s",
        residual_info["probability_method"],
        residual_info["historical_residual_sigma"],
        "YES"
        if residual_info[
            "probability_is_model_specific_calibration"
        ]
        else "NO",
    )

    display = predictions[
        [
            "game_date",
            "away_team",
            "home_team",
            "home_power_rating_points",
            "away_power_rating_points",
            "home_field_points",
            "total_schedule_adjustment_points",
            "manual_availability_adjustment_points",
            "projected_home_margin",
            "projected_spread",
            "sportsbook",
            "available_market_home_margin",
            "available_market_spread",
            "market_line_age_minutes",
            "absolute_spread_difference_points",
            "selected_market_line",
            "model_selected_cover_probability",
            "probability_edge",
            "decision",
            "flat_recommended_stake",
            "fractional_kelly_recommended_stake",
        ]
    ].copy()

    for column in (
        "model_selected_cover_probability",
        "probability_edge",
    ):
        display[column] = display[column].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"{float(value):.2%}"
            )
        )

    LOGGER.info("")
    LOGGER.info(
        "[PREDICT] Weekly projections:\n%s",
        display.to_string(index=False),
    )
    LOGGER.info("=" * 132)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.project_root)
    run_id = str(uuid.uuid4())
    started = dt.datetime.now()

    LOGGER.info("[PREDICT] Building canonical weekly NFL spreads")
    LOGGER.info("[PREDICT] Build ID: %s", BUILD_ID)
    LOGGER.info("[PREDICT] Version: %s", VERSION)
    LOGGER.info("[PREDICT] Database: %s", args.db_path)

    connection = connect_database(args.db_path)

    try:
        ratings, rating_source = load_live_ratings(connection)
        hfa, hfa_source = load_home_field(
            connection, args.home_field_points
        )
        residual_info = load_probability_noise(connection)

        schedule = load_primary_schedule(
            connection,
            args.project_root,
            args.schedule_path,
        )
        schedule = merge_market_source(
            schedule,
            args.market_path,
        )
        factors = load_schedule_factors(
            connection,
            args.project_root,
            args.schedule_factors_path,
        )
        schedule = attach_factors(schedule, factors)
        schedule = choose_market_line(
            schedule,
            args.market_line_preference,
        )

        week = select_prediction_week(
            schedule,
            args.week,
            args.as_of_date,
        )
        as_of_date = prediction_date(args.as_of_date)
        rating_lineage = validate_live_rating_contract(
            connection,
            ratings,
            week,
            as_of_date,
        )
        games = schedule[
            schedule["week"].eq(week)
        ].copy()
        if games.empty:
            raise RuntimeError(
                f"No games found for 2026 Week {week}."
            )

        manual = load_manual_adjustments(
            connection, week
        )
        predictions = build_predictions(
            games=games,
            ratings=ratings,
            week=week,
            rating_source=rating_source,
            hfa=hfa,
            hfa_source=hfa_source,
            residual_info=residual_info,
            manual=manual,
            args=args,
            run_id=run_id,
            rating_lineage=rating_lineage,
        )
        validate_predictions(predictions, games)

        sportsbook_counts = (
            predictions["sportsbook"]
            .dropna()
            .astype(str)
            .loc[lambda series: series.str.strip().ne("")]
            .value_counts()
            .to_dict()
        )
        market_ages = pd.to_numeric(
            predictions["market_line_age_minutes"],
            errors="coerce",
        ).dropna()
        maximum_market_line_age = (
            float(market_ages.max()) if not market_ages.empty else None
        )

        completed = dt.datetime.now()
        run_audit = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "season": SEASON,
                    "week": week,
                    "build_id": BUILD_ID,
                    "version": VERSION,
                    "model_variant": MODEL_VARIANT,
                    "status": "SUCCESS",
                    "games": int(len(predictions)),
                    "markets_attached": int(
                        predictions[
                            "market_attached_after_prediction_freeze"
                        ].sum()
                    ),
                    "qualifying_bets": int(
                        predictions[
                            "qualifies_for_stake"
                        ].sum()
                    ),
                    "sportsbooks_json": json.dumps(
                        sportsbook_counts,
                        sort_keys=True,
                    ),
                    "maximum_market_line_age_minutes": (
                        maximum_market_line_age
                    ),
                    "circa_lines_attached": int(
                        pd.to_numeric(
                            predictions["circa_line_flag"],
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                    "rating_source": rating_source,
                    **rating_lineage,
                    "schedule_source": " | ".join(
                        sorted(
                            predictions[
                                "schedule_source"
                            ].astype(str).unique()
                        )
                    ),
                    "factor_source": " | ".join(
                        sorted(
                            predictions[
                                "factor_source"
                            ].astype(str).unique()
                        )
                    ),
                    "hfa_points": hfa,
                    "hfa_source": hfa_source,
                    "probability_method": residual_info[
                        "probability_method"
                    ],
                    "probability_source": residual_info[
                        "probability_source"
                    ],
                    "probability_is_model_specific_calibration": (
                        residual_info[
                            "probability_is_model_specific_calibration"
                        ]
                    ),
                    "historical_residual_rows": residual_info[
                        "historical_residual_rows"
                    ],
                    "historical_residual_sigma": residual_info[
                        "historical_residual_sigma"
                    ],
                    "prediction_uses_market_inputs": 0,
                    "market_attached_after_prediction_freeze": int(
                        predictions[
                            "market_attached_after_prediction_freeze"
                        ].sum() > 0
                    ),
                    "started_at": started.isoformat(
                        timespec="seconds"
                    ),
                    "completed_at": completed.isoformat(
                        timespec="seconds"
                    ),
                    "elapsed_seconds": (
                        completed - started
                    ).total_seconds(),
                    "date_imported": now_string(),
                }
            ]
        )

        current_csv, snapshot_csv = save_outputs(
            connection,
            predictions,
            run_audit,
            args.project_root,
            args.no_csv,
        )

        LOGGER.info(
            "[PREDICT] Saved %s rows to %s",
            len(predictions),
            OUTPUT_TABLE,
        )
        LOGGER.info(
            "[PREDICT] Appended %s rows to %s",
            len(predictions),
            OUTPUT_HISTORY_TABLE,
        )
        if current_csv is not None:
            LOGGER.info("[PREDICT] CSV saved: %s", current_csv)
        if snapshot_csv is not None:
            LOGGER.info(
                "[PREDICT] Snapshot saved: %s",
                snapshot_csv,
            )

        print_report(
            predictions,
            week,
            hfa,
            hfa_source,
            residual_info,
        )
        LOGGER.info(
            "[PREDICT] Completed successfully in %.2f seconds",
            (dt.datetime.now() - started).total_seconds(),
        )
        LOGGER.info("[PREDICT] Log saved: %s", log_path)
        return 0

    finally:
        connection.close()
        close_logger()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[PREDICT] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[PREDICT] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
