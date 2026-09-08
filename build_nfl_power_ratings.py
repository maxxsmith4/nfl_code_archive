#!/usr/bin/env python
"""
Build canonical 2026 NFL neutral-field power ratings.

Purpose
-------
Convert the validated replacement-anchored team-strength index into neutral-
field point ratings without forcing the 2026 distribution to match historical
market-rating variance.

Core principles
---------------
1. `nfl_team_strength_2026.point_ready_index` is the native structural point
   scale and remains the default point rating.
2. Historical closing spreads are used for:
   - home-field diagnostics;
   - market-rating dispersion diagnostics;
   - reproducible audit tables.
3. Historical market standard deviation is NOT used to rescale 2026 ratings.
4. An empirical slope may be learned only from an explicit paired calibration
   table containing historical structural differences and market-neutral target
   differences. If that table is absent or insufficient, the mapping is exactly
   identity: 1 structural point = 1 neutral-field point.
5. Home field, current injuries, rest, travel, weather, and matchup adjustments
   remain outside this preseason neutral-field rating.

Inputs
------
SQLite:
    nfl_team_strength_2026

Optional SQLite paired calibration table:
    nfl_power_rating_calibration_observations

The optional table must provide either:
    structural_diff
    market_neutral_diff

or recognizable aliases documented in `prepare_paired_calibration`.

External historical schedule source:
    nflreadpy.load_schedules
    nfl_data_py.import_schedules fallback

Outputs
-------
SQLite and CSV:
    nfl_power_ratings_2026
    nfl_power_rating_calibration_season_audit
    nfl_power_rating_calibration_team_audit
    nfl_power_rating_point_mapping_audit

Interpretation
--------------
A `power_rating_points` value of +5.0 means the team is rated five points
better than an average NFL team on a neutral field.

Neutral-field matchup:
    team_a_power_rating - team_b_power_rating

Home-field matchup:
    home_power - away_power + current_home_field_adjustment
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sql


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON = 2026
BUILD_ID = "NFL_POWER_RATINGS_2026_CANONICAL_V2"
VERSION = "v2_native_point_scale_optional_paired_calibration"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

TEAM_STRENGTH_TABLE = "nfl_team_strength_2026"
DEFAULT_PAIRED_CALIBRATION_TABLE = (
    "nfl_power_rating_calibration_observations"
)

OUTPUT_TABLE = "nfl_power_ratings_2026"
SEASON_AUDIT_TABLE = "nfl_power_rating_calibration_season_audit"
TEAM_AUDIT_TABLE = "nfl_power_rating_calibration_team_audit"
MAPPING_AUDIT_TABLE = "nfl_power_rating_point_mapping_audit"

CALIBRATION_SEASONS = list(range(2018, 2026))
GAME_TYPE_FILTER = {"REG"}

MARKET_RIDGE_ALPHA = 6.0
MIN_GAMES_PER_SEASON = 200
MIN_TEAMS_PER_SEASON = 30

# Optional paired calibration requirements.
MIN_PAIRED_ROWS = 250
MIN_PAIRED_SEASONS = 3
PAIRED_RIDGE_ALPHA = 25.0
PAIRED_SLOPE_MIN = 0.50
PAIRED_SLOPE_MAX = 1.50
RECENCY_DECAY = 0.90

# The native point-ready index is already centered in team strength.
DEFAULT_NATIVE_POINT_SLOPE = 1.0

# Safety boundary only; current ratings should remain well inside this.
POWER_RATING_CAP = 12.0

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

SEASON_AUDIT_COLUMNS = [
    "season",
    "games",
    "teams",
    "ridge_alpha",
    "home_field_points",
    "team_rating_mean",
    "team_rating_std",
    "team_rating_min",
    "team_rating_max",
    "market_fit_rmse",
    "market_fit_mae",
    "actual_margin_rmse",
    "spread_sign_multiplier",
    "spread_actual_margin_correlation",
    "historical_market_std_median",
    "historical_market_std_mean",
    "market_dispersion_used_to_rescale_flag",
    "schedule_source",
    "calibration_version",
    "date_imported",
]

TEAM_AUDIT_COLUMNS = [
    "season",
    "team",
    "market_power_rating",
    "market_power_rank",
    "games_in_calibration",
    "calibration_version",
    "date_imported",
]

MAPPING_AUDIT_COLUMNS = [
    "calibration_method",
    "paired_table_name",
    "paired_table_available",
    "paired_rows_raw",
    "paired_rows_usable",
    "paired_seasons",
    "paired_start_season",
    "paired_end_season",
    "native_point_slope_requested",
    "fitted_slope_raw",
    "selected_point_slope",
    "slope_lower_bound",
    "slope_upper_bound",
    "slope_clipped_flag",
    "paired_fit_rmse",
    "paired_fit_mae",
    "paired_fit_correlation",
    "identity_fallback_flag",
    "historical_market_std_median",
    "historical_market_std_mean",
    "market_dispersion_used_to_rescale_flag",
    "mapping_formula",
    "calibration_version",
    "date_imported",
]


# =============================================================================
# ARGUMENTS AND LOGGING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical 2026 NFL neutral-field power ratings from the "
            "native structural point-ready index."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root containing outputs and logs directories.",
    )
    parser.add_argument(
        "--db-path",
        "--database",
        dest="db_path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database path.",
    )
    parser.add_argument(
        "--paired-calibration-table",
        default=DEFAULT_PAIRED_CALIBRATION_TABLE,
        help=(
            "Optional SQLite table containing paired historical structural "
            "and market-neutral differences."
        ),
    )
    parser.add_argument(
        "--native-point-slope",
        type=float,
        default=DEFAULT_NATIVE_POINT_SLOPE,
        help=(
            "Identity/fallback multiplier for point_ready_index. Default 1.0."
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write SQLite outputs only.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"build_nfl_power_ratings_{timestamp}.log"

    logger = logging.getLogger("nfl_power_ratings")
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


# =============================================================================
# DATABASE AND GENERIC HELPERS
# =============================================================================

def get_engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sql.create_engine(
        f"sqlite:///{db_path}",
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
    with engine.connect() as conn:
        return conn.execute(
            sql.text(query),
            {"table_name": table_name},
        ).fetchone() is not None


def read_table(engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")

    escaped = table_name.replace('"', '""')
    with engine.connect() as conn:
        frame = pd.read_sql(
            sql.text(f'SELECT * FROM "{escaped}"'),
            conn,
        )

    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]
    return frame


def read_optional_table(engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        return pd.DataFrame()
    return read_table(engine, table_name)


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


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def first_existing(
    columns: list[str],
    candidates: list[str],
) -> Optional[str]:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def frame_from_any(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def clip_and_recenter(
    values: pd.Series,
    lower: float,
    upper: float,
) -> tuple[pd.Series, int]:
    raw = pd.to_numeric(values, errors="coerce")
    if raw.isna().any():
        raise RuntimeError(
            f"Cannot clip/recenter series with {int(raw.isna().sum())} nulls."
        )

    clipped = raw.clip(lower=lower, upper=upper)
    cap_count = int((~np.isclose(raw, clipped, atol=1e-12)).sum())

    # Iteratively recenter while respecting the hard bounds.
    centered = clipped.copy()
    for _ in range(20):
        centered = (centered - centered.mean()).clip(lower, upper)
        if abs(float(centered.mean())) <= 1e-12:
            break

    if abs(float(centered.mean())) > 1e-8:
        raise RuntimeError("Unable to recenter capped power ratings.")
    return centered, cap_count


# =============================================================================
# HISTORICAL SCHEDULE LOADING
# =============================================================================

def load_historical_schedules() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    try:
        import nflreadpy as nfl  # type: ignore

        if hasattr(nfl, "load_schedules"):
            attempts = [
                lambda: nfl.load_schedules(CALIBRATION_SEASONS),
                lambda: nfl.load_schedules(
                    seasons=CALIBRATION_SEASONS
                ),
            ]
            for attempt in attempts:
                try:
                    frame = frame_from_any(attempt())
                    if not frame.empty:
                        return frame, "nflreadpy.load_schedules"
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"nflreadpy.load_schedules: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nflreadpy: {exc}")

    try:
        import nfl_data_py as nfl  # type: ignore

        if hasattr(nfl, "import_schedules"):
            try:
                frame = frame_from_any(
                    nfl.import_schedules(CALIBRATION_SEASONS)
                )
                if not frame.empty:
                    return frame, "nfl_data_py.import_schedules"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"nfl_data_py.import_schedules: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nfl_data_py: {exc}")

    raise RuntimeError(
        "Unable to load historical schedules. " + " | ".join(errors)
    )


def standardize_schedules(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    season_col = first_existing(list(frame.columns), ["season"])
    game_type_col = first_existing(
        list(frame.columns),
        ["game_type", "season_type"],
    )
    home_team_col = first_existing(
        list(frame.columns),
        ["home_team", "home"],
    )
    away_team_col = first_existing(
        list(frame.columns),
        ["away_team", "away"],
    )
    spread_col = first_existing(
        list(frame.columns),
        [
            "spread_line",
            "closing_spread",
            "home_spread",
            "spread",
        ],
    )
    home_score_col = first_existing(
        list(frame.columns),
        ["home_score", "score_home"],
    )
    away_score_col = first_existing(
        list(frame.columns),
        ["away_score", "score_away"],
    )
    neutral_col = first_existing(
        list(frame.columns),
        ["location", "neutral", "neutral_site"],
    )

    required = {
        "season": season_col,
        "home_team": home_team_col,
        "away_team": away_team_col,
        "spread": spread_col,
    }
    missing = [
        name for name, column in required.items()
        if column is None
    ]
    if missing:
        raise RuntimeError(
            "Schedule data is missing required fields "
            f"{missing}. Columns: {list(frame.columns)}"
        )

    out = pd.DataFrame(index=frame.index)
    out["season"] = pd.to_numeric(
        frame[season_col], errors="coerce"
    )
    out["game_type"] = (
        frame[game_type_col].astype(str).str.upper()
        if game_type_col
        else "REG"
    )
    out["home_team"] = frame[home_team_col].map(normalize_team)
    out["away_team"] = frame[away_team_col].map(normalize_team)
    out["spread_line"] = pd.to_numeric(
        frame[spread_col], errors="coerce"
    )
    out["home_score"] = (
        pd.to_numeric(frame[home_score_col], errors="coerce")
        if home_score_col
        else np.nan
    )
    out["away_score"] = (
        pd.to_numeric(frame[away_score_col], errors="coerce")
        if away_score_col
        else np.nan
    )

    if neutral_col:
        neutral_raw = frame[neutral_col]
        if neutral_raw.dtype == bool:
            out["neutral_site_flag"] = neutral_raw.astype(int)
        else:
            neutral_text = (
                neutral_raw.astype(str).str.upper().str.strip()
            )
            out["neutral_site_flag"] = neutral_text.isin(
                {"NEUTRAL", "TRUE", "1", "YES"}
            ).astype(int)
    else:
        out["neutral_site_flag"] = 0

    out = out[
        out["season"].isin(CALIBRATION_SEASONS)
        & out["game_type"].isin(GAME_TYPE_FILTER)
        & out["home_team"].notna()
        & out["away_team"].notna()
        & out["spread_line"].notna()
    ].copy()

    out["season"] = out["season"].astype(int)
    out["actual_home_margin"] = (
        out["home_score"] - out["away_score"]
    )
    return out.reset_index(drop=True)


# =============================================================================
# HISTORICAL MARKET DIAGNOSTICS
# =============================================================================

def infer_spread_sign(frame: pd.DataFrame) -> tuple[float, float]:
    completed = frame[
        frame["actual_home_margin"].notna()
        & frame["spread_line"].notna()
    ].copy()

    if len(completed) < 100:
        raise RuntimeError(
            "Not enough completed games to infer spread convention."
        )

    corr_direct = completed[
        ["spread_line", "actual_home_margin"]
    ].corr().iloc[0, 1]
    corr_inverted = completed.assign(
        inverted=-completed["spread_line"]
    )[["inverted", "actual_home_margin"]].corr().iloc[0, 1]

    # Direct and inverted correlations have identical absolute magnitude by
    # construction. Select the orientation with the positive relationship;
    # do not compare absolute values, which can retain the negative sign.
    candidates = [
        (1.0, float(corr_direct)),
        (-1.0, float(corr_inverted)),
    ]
    positive = [
        (multiplier, correlation)
        for multiplier, correlation in candidates
        if np.isfinite(correlation) and correlation > 0
    ]
    if positive:
        multiplier, chosen_corr = max(
            positive,
            key=lambda item: item[1],
        )
    else:
        multiplier, chosen_corr = (1.0, float(corr_direct))

    if not np.isfinite(chosen_corr) or chosen_corr <= 0:
        raise RuntimeError(
            "Unable to infer a positive spread/margin relationship. "
            f"direct_corr={corr_direct}, inverted_corr={corr_inverted}"
        )
    return multiplier, float(chosen_corr)


def fit_market_ratings_for_season(
    season_games: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    games = season_games.reset_index(drop=True).copy()
    teams = sorted(
        set(games["home_team"]) | set(games["away_team"])
    )

    if len(teams) < MIN_TEAMS_PER_SEASON:
        raise RuntimeError(
            f"Season {season} has only {len(teams)} teams."
        )
    if len(games) < MIN_GAMES_PER_SEASON:
        raise RuntimeError(
            f"Season {season} has only {len(games)} games."
        )

    team_to_index = {
        team: index for index, team in enumerate(teams)
    }
    n_games = len(games)
    n_teams = len(teams)

    design = np.zeros((n_games, n_teams + 1), dtype=float)
    target = games[
        "market_expected_home_margin"
    ].to_numpy(dtype=float)

    for row_index, row in enumerate(games.itertuples(index=False)):
        design[row_index, team_to_index[row.home_team]] = 1.0
        design[row_index, team_to_index[row.away_team]] = -1.0
        design[row_index, n_teams] = (
            0.0 if int(row.neutral_site_flag) == 1 else 1.0
        )

    penalty = np.eye(n_teams + 1, dtype=float) * MARKET_RIDGE_ALPHA
    penalty[n_teams, n_teams] = 0.0

    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    team_ratings = coefficients[:n_teams]
    team_ratings = team_ratings - team_ratings.mean()
    home_field = float(coefficients[n_teams])

    fitted_market = (
        design[:, :n_teams] @ team_ratings
        + design[:, n_teams] * home_field
    )
    market_residual = target - fitted_market

    team_frame = pd.DataFrame(
        {
            "season": season,
            "team": teams,
            "market_power_rating": team_ratings,
        }
    )
    team_frame["market_power_rank"] = (
        team_frame["market_power_rating"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    team_counts = pd.concat(
        [games["home_team"], games["away_team"]],
        ignore_index=True,
    ).value_counts()
    team_frame["games_in_calibration"] = (
        team_frame["team"].map(team_counts).fillna(0).astype(int)
    )

    completed_mask = games["actual_home_margin"].notna().to_numpy()
    if completed_mask.any():
        actual_residual = (
            games.loc[
                completed_mask, "actual_home_margin"
            ].to_numpy(dtype=float)
            - fitted_market[completed_mask]
        )
        actual_margin_rmse = float(
            np.sqrt(np.mean(np.square(actual_residual)))
        )
    else:
        actual_margin_rmse = np.nan

    summary = {
        "season": season,
        "games": int(n_games),
        "teams": int(n_teams),
        "ridge_alpha": MARKET_RIDGE_ALPHA,
        "home_field_points": home_field,
        "team_rating_mean": float(team_ratings.mean()),
        "team_rating_std": float(team_ratings.std(ddof=1)),
        "team_rating_min": float(team_ratings.min()),
        "team_rating_max": float(team_ratings.max()),
        "market_fit_rmse": float(
            np.sqrt(np.mean(np.square(market_residual)))
        ),
        "market_fit_mae": float(
            np.mean(np.abs(market_residual))
        ),
        "actual_margin_rmse": actual_margin_rmse,
    }
    return team_frame, summary


def build_historical_market_diagnostics(
    schedules: pd.DataFrame,
    schedule_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    spread_multiplier, sign_corr = infer_spread_sign(schedules)

    frame = schedules.copy()
    frame["market_expected_home_margin"] = (
        spread_multiplier * frame["spread_line"]
    )

    team_frames: list[pd.DataFrame] = []
    season_rows: list[dict[str, Any]] = []

    for season in CALIBRATION_SEASONS:
        games = frame[frame["season"].eq(season)].copy()
        if len(games) < MIN_GAMES_PER_SEASON:
            continue
        team_frame, summary = fit_market_ratings_for_season(
            games, season
        )
        team_frames.append(team_frame)
        season_rows.append(summary)

    if not team_frames or not season_rows:
        raise RuntimeError(
            "No historical seasons were available for market diagnostics."
        )

    team_audit = pd.concat(team_frames, ignore_index=True)
    season_audit = pd.DataFrame(season_rows)

    market_std_median = float(
        season_audit["team_rating_std"].median()
    )
    market_std_mean = float(
        season_audit["team_rating_std"].mean()
    )
    now = dt.datetime.now().isoformat(timespec="seconds")

    season_audit["spread_sign_multiplier"] = spread_multiplier
    season_audit["spread_actual_margin_correlation"] = sign_corr
    season_audit["historical_market_std_median"] = (
        market_std_median
    )
    season_audit["historical_market_std_mean"] = market_std_mean
    season_audit["market_dispersion_used_to_rescale_flag"] = 0
    season_audit["schedule_source"] = schedule_source
    season_audit["calibration_version"] = VERSION
    season_audit["date_imported"] = now

    team_audit["calibration_version"] = VERSION
    team_audit["date_imported"] = now

    return (
        season_audit[SEASON_AUDIT_COLUMNS],
        team_audit[TEAM_AUDIT_COLUMNS],
        spread_multiplier,
        sign_corr,
    )


# =============================================================================
# OPTIONAL PAIRED POINT-SLOPE CALIBRATION
# =============================================================================

def prepare_paired_calibration(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(
            columns=["season", "structural_diff", "market_neutral_diff"]
        )

    frame = raw.copy()
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    season_col = first_existing(
        list(frame.columns),
        ["season", "year"],
    )
    structural_col = first_existing(
        list(frame.columns),
        [
            "structural_diff",
            "structural_neutral_diff",
            "point_ready_diff",
            "model_neutral_diff",
            "neutral_model_diff",
        ],
    )
    target_col = first_existing(
        list(frame.columns),
        [
            "market_neutral_diff",
            "market_expected_neutral_margin",
            "closing_neutral_diff",
            "neutral_market_diff",
        ],
    )

    if structural_col is None or target_col is None:
        return pd.DataFrame(
            columns=["season", "structural_diff", "market_neutral_diff"]
        )

    out = pd.DataFrame(index=frame.index)
    out["season"] = (
        pd.to_numeric(frame[season_col], errors="coerce")
        if season_col
        else np.nan
    )
    out["structural_diff"] = pd.to_numeric(
        frame[structural_col], errors="coerce"
    )
    out["market_neutral_diff"] = pd.to_numeric(
        frame[target_col], errors="coerce"
    )

    out = out[
        out["structural_diff"].notna()
        & out["market_neutral_diff"].notna()
        & out["structural_diff"].abs().le(30)
        & out["market_neutral_diff"].abs().le(30)
    ].copy()

    # A zero structural difference contributes no slope information.
    out = out[out["structural_diff"].abs().gt(1e-9)].copy()

    if out["season"].notna().any():
        out["season"] = out["season"].astype("Int64")
    return out.reset_index(drop=True)


def select_point_slope(
    paired_raw: pd.DataFrame,
    paired_table_name: str,
    native_point_slope: float,
    market_std_median: float,
    market_std_mean: float,
) -> tuple[float, pd.DataFrame]:
    if not np.isfinite(native_point_slope) or native_point_slope <= 0:
        raise RuntimeError(
            f"Invalid native point slope: {native_point_slope}"
        )

    paired = prepare_paired_calibration(paired_raw)
    raw_rows = int(len(paired_raw))
    usable_rows = int(len(paired))

    season_values = (
        sorted(
            paired["season"].dropna().astype(int).unique().tolist()
        )
        if not paired.empty
        else []
    )
    paired_seasons = len(season_values)

    sufficient = (
        usable_rows >= MIN_PAIRED_ROWS
        and (
            paired_seasons >= MIN_PAIRED_SEASONS
            or paired["season"].isna().all()
        )
    )

    fitted_slope_raw = np.nan
    selected_slope = float(native_point_slope)
    slope_clipped_flag = 0
    rmse = np.nan
    mae = np.nan
    corr = np.nan
    identity_fallback_flag = 1
    method = "native_point_ready_identity"

    if sufficient:
        x = paired["structural_diff"].to_numpy(dtype=float)
        y = paired["market_neutral_diff"].to_numpy(dtype=float)

        if paired["season"].notna().any():
            max_season = int(paired["season"].dropna().max())
            age = (
                max_season
                - paired["season"].fillna(max_season).astype(int)
            ).to_numpy(dtype=float)
            weights = np.power(RECENCY_DECAY, age)
        else:
            weights = np.ones(len(paired), dtype=float)

        numerator = float(np.sum(weights * x * y))
        denominator = float(
            np.sum(weights * x * x) + PAIRED_RIDGE_ALPHA
        )

        if denominator > 1e-12:
            fitted_slope_raw = numerator / denominator
            selected_slope = float(
                np.clip(
                    fitted_slope_raw,
                    PAIRED_SLOPE_MIN,
                    PAIRED_SLOPE_MAX,
                )
            )
            slope_clipped_flag = int(
                not np.isclose(
                    selected_slope,
                    fitted_slope_raw,
                    atol=1e-12,
                )
            )
            predictions = selected_slope * x
            residual = y - predictions
            rmse = float(
                np.sqrt(np.average(np.square(residual), weights=weights))
            )
            mae = float(
                np.average(np.abs(residual), weights=weights)
            )
            if len(paired) >= 3:
                corr = float(
                    np.corrcoef(x, y)[0, 1]
                )
            identity_fallback_flag = 0
            method = "paired_structural_to_market_zero_intercept_ridge"

    now = dt.datetime.now().isoformat(timespec="seconds")
    row = {
        "calibration_method": method,
        "paired_table_name": paired_table_name,
        "paired_table_available": int(not paired_raw.empty),
        "paired_rows_raw": raw_rows,
        "paired_rows_usable": usable_rows,
        "paired_seasons": paired_seasons,
        "paired_start_season": (
            min(season_values) if season_values else np.nan
        ),
        "paired_end_season": (
            max(season_values) if season_values else np.nan
        ),
        "native_point_slope_requested": float(native_point_slope),
        "fitted_slope_raw": fitted_slope_raw,
        "selected_point_slope": float(selected_slope),
        "slope_lower_bound": PAIRED_SLOPE_MIN,
        "slope_upper_bound": PAIRED_SLOPE_MAX,
        "slope_clipped_flag": slope_clipped_flag,
        "paired_fit_rmse": rmse,
        "paired_fit_mae": mae,
        "paired_fit_correlation": corr,
        "identity_fallback_flag": identity_fallback_flag,
        "historical_market_std_median": market_std_median,
        "historical_market_std_mean": market_std_mean,
        "market_dispersion_used_to_rescale_flag": 0,
        "mapping_formula": (
            "power_rating_points = centered(point_ready_index) "
            "* selected_point_slope"
        ),
        "calibration_version": VERSION,
        "date_imported": now,
    }
    audit = pd.DataFrame([row], columns=MAPPING_AUDIT_COLUMNS)
    return selected_slope, audit


# =============================================================================
# 2026 TEAM STRENGTH AND POWER RATINGS
# =============================================================================

def prepare_team_strength(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [
        str(column).strip().lower()
        for column in frame.columns
    ]

    required = [
        "season",
        "team",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_strength_z",
        "point_ready_index",
        "overall_strength_rank",
        "player_name",
        "qb_context_adjustment",
        "ol_continuity_strength_adjustment",
        "overall_quality_adjusted_completeness",
        "prior_snap_season",
        "stale_history_flag",
    ]
    missing = [
        column for column in required
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"{TEAM_STRENGTH_TABLE} is missing columns: {missing}"
        )

    frame["season"] = numeric_column(
        frame, "season", SEASON
    ).astype(int)
    frame = frame[frame["season"].eq(SEASON)].copy()
    frame["team"] = frame["team"].map(normalize_team)

    numeric_fields = [
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_strength_z",
        "point_ready_index",
        "overall_strength_rank",
        "qb_context_adjustment",
        "ol_continuity_strength_adjustment",
        "overall_quality_adjusted_completeness",
        "prior_snap_season",
        "stale_history_flag",
    ]
    for column in numeric_fields:
        frame[column] = pd.to_numeric(
            frame[column], errors="coerce"
        )

    if len(frame) != 32 or frame["team"].nunique() != 32:
        raise RuntimeError(
            f"{TEAM_STRENGTH_TABLE} must contain 32 unique teams."
        )
    if frame[required].isna().any().any():
        missing_counts = (
            frame[required].isna().sum()
            .loc[lambda s: s.gt(0)]
            .to_dict()
        )
        raise RuntimeError(
            f"Team strength contains missing required values: "
            f"{missing_counts}"
        )
    if set(frame["prior_snap_season"].astype(int)) != {2025}:
        raise RuntimeError(
            "Team strength does not use 2025 OL continuity."
        )
    if set(frame["stale_history_flag"].astype(int)) != {0}:
        raise RuntimeError(
            "Team strength contains stale OL history."
        )
    if not np.allclose(
        frame["point_ready_index"],
        frame["overall_strength"] - frame["overall_strength"].mean(),
        atol=1e-8,
    ):
        raise RuntimeError(
            "point_ready_index is not the centered raw overall-strength index."
        )
    if not np.allclose(
        frame["qb_context_adjustment"],
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Separate QB context adjustment is nonzero."
        )
    return frame.sort_values("team").reset_index(drop=True)


def build_power_ratings(
    strength: pd.DataFrame,
    selected_slope: float,
    season_audit: pd.DataFrame,
    mapping_audit: pd.DataFrame,
) -> pd.DataFrame:
    frame = strength.copy()

    native_index = (
        frame["point_ready_index"].astype(float)
        - frame["point_ready_index"].astype(float).mean()
    )
    frame["native_structural_point_index"] = native_index
    frame["selected_point_slope"] = float(selected_slope)
    frame["power_rating_uncapped"] = (
        native_index * selected_slope
    )

    (
        frame["power_rating_points"],
        cap_count,
    ) = clip_and_recenter(
        frame["power_rating_uncapped"],
        -POWER_RATING_CAP,
        POWER_RATING_CAP,
    )

    frame["power_rating_rank"] = (
        frame["power_rating_points"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    frame["points_above_average"] = frame["power_rating_points"]
    frame["neutral_field_rating"] = frame["power_rating_points"]

    market_std_median = float(
        season_audit["team_rating_std"].median()
    )
    market_std_mean = float(
        season_audit["team_rating_std"].mean()
    )

    frame["historical_target_team_std"] = market_std_median
    frame["historical_market_std_median"] = market_std_median
    frame["historical_market_std_mean"] = market_std_mean
    frame["historical_calibration_start_season"] = int(
        season_audit["season"].min()
    )
    frame["historical_calibration_end_season"] = int(
        season_audit["season"].max()
    )
    frame["historical_calibration_seasons"] = ",".join(
        map(
            str,
            sorted(
                season_audit["season"]
                .dropna()
                .astype(int)
                .unique()
            ),
        )
    )
    frame["median_historical_home_field"] = float(
        season_audit["home_field_points"].median()
    )
    frame["mean_historical_home_field"] = float(
        season_audit["home_field_points"].mean()
    )

    frame["market_dispersion_used_to_rescale_flag"] = 0
    frame["point_mapping_identity_fallback_flag"] = int(
        mapping_audit.loc[0, "identity_fallback_flag"]
    )
    frame["power_rating_cap"] = POWER_RATING_CAP
    frame["cap_applied_count"] = cap_count
    frame["cap_applied_to_team_flag"] = (
        ~np.isclose(
            frame["power_rating_uncapped"],
            frame["power_rating_points"],
            atol=1e-10,
        )
    ).astype(int)

    frame["rating_excludes_home_field_flag"] = 1
    frame["rating_excludes_injuries_flag"] = 1
    frame["rating_excludes_rest_travel_flag"] = 1
    frame["rating_excludes_weather_flag"] = 1

    frame["calibration_method"] = str(
        mapping_audit.loc[0, "calibration_method"]
    )
    frame["calibration_version"] = VERSION
    frame["build_id"] = BUILD_ID
    frame["date_imported"] = (
        dt.datetime.now().isoformat(timespec="seconds")
    )

    output_columns = [
        "season",
        "team",
        "power_rating_rank",
        "power_rating_points",
        "points_above_average",
        "neutral_field_rating",
        "power_rating_uncapped",
        "native_structural_point_index",
        "selected_point_slope",
        "player_name",
        "offense_strength",
        "defense_strength",
        "special_teams_strength",
        "overall_strength",
        "overall_strength_z",
        "point_ready_index",
        "overall_strength_rank",
        "qb_context_adjustment",
        "ol_continuity_strength_adjustment",
        "overall_quality_adjusted_completeness",
        "historical_target_team_std",
        "historical_market_std_median",
        "historical_market_std_mean",
        "historical_calibration_start_season",
        "historical_calibration_end_season",
        "historical_calibration_seasons",
        "median_historical_home_field",
        "mean_historical_home_field",
        "market_dispersion_used_to_rescale_flag",
        "point_mapping_identity_fallback_flag",
        "power_rating_cap",
        "cap_applied_count",
        "cap_applied_to_team_flag",
        "rating_excludes_home_field_flag",
        "rating_excludes_injuries_flag",
        "rating_excludes_rest_travel_flag",
        "rating_excludes_weather_flag",
        "calibration_method",
        "calibration_version",
        "build_id",
        "date_imported",
    ]
    return (
        frame[output_columns]
        .sort_values("power_rating_rank")
        .reset_index(drop=True)
    )


# =============================================================================
# VALIDATION, OUTPUT, AND REPORTING
# =============================================================================

def validate_outputs(
    power: pd.DataFrame,
    season_audit: pd.DataFrame,
    team_audit: pd.DataFrame,
    mapping_audit: pd.DataFrame,
) -> None:
    if len(power) != 32 or power["team"].nunique() != 32:
        raise RuntimeError(
            "Power-rating output does not contain 32 unique teams."
        )
    if power["power_rating_points"].isna().any():
        raise RuntimeError("Power ratings contain missing values.")
    if abs(float(power["power_rating_points"].mean())) > 1e-8:
        raise RuntimeError("Power ratings are not centered at zero.")
    if (
        power["power_rating_points"].abs().max()
        > POWER_RATING_CAP + 1e-8
    ):
        raise RuntimeError(
            "Power ratings exceed the configured cap."
        )
    if season_audit.empty or team_audit.empty:
        raise RuntimeError(
            "Historical market diagnostic outputs are empty."
        )
    if season_audit["season"].nunique() < 5:
        raise RuntimeError(
            "Fewer than five historical seasons were diagnosed."
        )
    if len(mapping_audit) != 1:
        raise RuntimeError(
            "Point-mapping audit must contain exactly one row."
        )
    if int(
        mapping_audit.loc[
            0, "market_dispersion_used_to_rescale_flag"
        ]
    ) != 0:
        raise RuntimeError(
            "Historical market dispersion was incorrectly used to rescale."
        )
    if not (
        power["market_dispersion_used_to_rescale_flag"].eq(0).all()
    ):
        raise RuntimeError(
            "Power output indicates forbidden market-dispersion rescaling."
        )

    selected_slope = float(
        mapping_audit.loc[0, "selected_point_slope"]
    )
    expected_uncapped = (
        power["native_structural_point_index"]
        * selected_slope
    )
    if not np.allclose(
        power["power_rating_uncapped"],
        expected_uncapped,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Uncapped ratings do not reconcile to native index × slope."
        )

    if int(mapping_audit.loc[0, "identity_fallback_flag"]) == 1:
        native_requested = float(
            mapping_audit.loc[
                0, "native_point_slope_requested"
            ]
        )
        if not np.isclose(
            selected_slope,
            native_requested,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Identity fallback did not preserve requested native slope."
            )

    if (
        power["cap_applied_to_team_flag"].sum() == 0
        and not np.allclose(
            power["power_rating_points"],
            power["power_rating_uncapped"],
            atol=1e-9,
        )
    ):
        raise RuntimeError(
            "Uncapped ratings changed despite no cap application."
        )


def save_outputs(
    engine,
    outputs: dict[str, pd.DataFrame],
    output_dir: Path,
    write_csv: bool,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, frame in outputs.items():
        frame.to_sql(
            table_name,
            con=engine,
            if_exists="replace",
            index=False,
        )
        logger.info(
            "[POWER] Saved %s rows to %s",
            f"{len(frame):,}",
            table_name,
        )

        if write_csv:
            csv_path = output_dir / f"{table_name}.csv"
            frame.to_csv(
                csv_path,
                index=False,
                encoding="utf-8-sig",
            )
            logger.info("[POWER] CSV saved: %s", csv_path)


def add_indexes(engine) -> None:
    statements = [
        f"""
        CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_team
        ON {OUTPUT_TABLE} (team)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{SEASON_AUDIT_TABLE}_season
        ON {SEASON_AUDIT_TABLE} (season)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{TEAM_AUDIT_TABLE}_season_team
        ON {TEAM_AUDIT_TABLE} (season, team)
        """,
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(sql.text(statement))


def print_report(
    power: pd.DataFrame,
    season_audit: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    spread_multiplier: float,
    sign_corr: float,
    schedule_source: str,
    logger: logging.Logger,
) -> None:
    row = mapping_audit.iloc[0]

    logger.info("")
    logger.info("=" * 104)
    logger.info("[POWER] 2026 NFL CANONICAL NEUTRAL-FIELD POWER RATINGS")
    logger.info("=" * 104)
    logger.info("[POWER] Build ID: %s", BUILD_ID)
    logger.info("[POWER] Version: %s", VERSION)
    logger.info("[POWER] Historical schedule source: %s", schedule_source)
    logger.info("[POWER] Spread sign multiplier: %.1f", spread_multiplier)
    logger.info(
        "[POWER] Spread/actual-margin correlation: %.4f",
        sign_corr,
    )
    logger.info(
        "[POWER] Point mapping method: %s",
        row["calibration_method"],
    )
    logger.info(
        "[POWER] Selected structural-to-point slope: %.6f",
        float(row["selected_point_slope"]),
    )
    logger.info(
        "[POWER] Identity fallback used: %s",
        "YES" if int(row["identity_fallback_flag"]) == 1 else "NO",
    )
    logger.info(
        "[POWER] Historical market dispersion used to rescale: NO"
    )
    logger.info(
        "[POWER] Historical market rating median std: %.3f",
        float(row["historical_market_std_median"]),
    )
    logger.info(
        "[POWER] 2026 native-index mean/std: %.6f / %.3f",
        power["native_structural_point_index"].mean(),
        power["native_structural_point_index"].std(ddof=1),
    )
    logger.info(
        "[POWER] 2026 rating mean/std: %.6f / %.3f",
        power["power_rating_points"].mean(),
        power["power_rating_points"].std(ddof=1),
    )
    logger.info(
        "[POWER] Cap applications: %s",
        int(power["cap_applied_to_team_flag"].sum()),
    )
    logger.info(
        "[POWER] Historical median HFA: %.3f",
        season_audit["home_field_points"].median(),
    )

    logger.info("")
    logger.info(
        "[POWER] Historical market diagnostics:\n%s",
        season_audit[
            [
                "season",
                "games",
                "teams",
                "home_field_points",
                "team_rating_std",
                "team_rating_min",
                "team_rating_max",
                "market_fit_rmse",
            ]
        ].to_string(index=False),
    )

    display_columns = [
        "power_rating_rank",
        "team",
        "player_name",
        "power_rating_points",
        "offense_strength",
        "defense_strength",
        "overall_strength",
        "overall_quality_adjusted_completeness",
    ]
    logger.info("")
    logger.info(
        "[POWER] 2026 neutral-field power ratings:\n%s",
        power[display_columns].to_string(index=False),
    )
    logger.info("=" * 104)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    db_path = args.db_path.resolve()
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"

    logger, log_path = configure_logging(log_dir)
    started = dt.datetime.now()

    logger.info("[POWER] Building canonical 2026 NFL power ratings")
    logger.info("[POWER] Build ID: %s", BUILD_ID)
    logger.info("[POWER] Version: %s", VERSION)
    logger.info("[POWER] Database: %s", db_path)
    logger.info("[POWER] Calibration seasons: %s", CALIBRATION_SEASONS)

    engine = get_engine(db_path)

    strength = prepare_team_strength(
        read_table(engine, TEAM_STRENGTH_TABLE)
    )

    raw_schedules, schedule_source = load_historical_schedules()
    schedules = standardize_schedules(raw_schedules)
    logger.info(
        "[POWER] Historical schedule rows: %s via %s",
        f"{len(schedules):,}",
        schedule_source,
    )

    (
        season_audit,
        team_audit,
        spread_multiplier,
        sign_corr,
    ) = build_historical_market_diagnostics(
        schedules,
        schedule_source,
    )

    paired_raw = read_optional_table(
        engine,
        args.paired_calibration_table,
    )
    selected_slope, mapping_audit = select_point_slope(
        paired_raw=paired_raw,
        paired_table_name=args.paired_calibration_table,
        native_point_slope=args.native_point_slope,
        market_std_median=float(
            season_audit["team_rating_std"].median()
        ),
        market_std_mean=float(
            season_audit["team_rating_std"].mean()
        ),
    )

    power = build_power_ratings(
        strength=strength,
        selected_slope=selected_slope,
        season_audit=season_audit,
        mapping_audit=mapping_audit,
    )

    validate_outputs(
        power=power,
        season_audit=season_audit,
        team_audit=team_audit,
        mapping_audit=mapping_audit,
    )

    outputs = {
        OUTPUT_TABLE: power,
        SEASON_AUDIT_TABLE: season_audit,
        TEAM_AUDIT_TABLE: team_audit,
        MAPPING_AUDIT_TABLE: mapping_audit,
    }
    save_outputs(
        engine=engine,
        outputs=outputs,
        output_dir=output_dir,
        write_csv=not args.no_csv,
        logger=logger,
    )
    add_indexes(engine)

    print_report(
        power=power,
        season_audit=season_audit,
        mapping_audit=mapping_audit,
        spread_multiplier=spread_multiplier,
        sign_corr=sign_corr,
        schedule_source=schedule_source,
        logger=logger,
    )

    elapsed = (dt.datetime.now() - started).total_seconds()
    logger.info(
        "[POWER] Completed successfully in %.2f seconds",
        elapsed,
    )
    logger.info("[POWER] Log saved: %s", log_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[POWER] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[POWER] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
