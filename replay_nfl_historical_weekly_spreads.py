#!/usr/bin/env python
"""
Replay the rebuilt canonical NFL model week by week for 2020-2025.

This is the decisive historical validation stage for the rewritten model.
It consumes each isolated season database created by Stages 1-5 and reuses the
approved canonical form and spread-prediction scripts without changing their
model formulas.

For target season S and prediction week W:
- the immutable preseason prior is the rebuilt S structural power table;
- form uses only completed S games through W-1;
- the process model is calibrated only on 2018 through S-1;
- the independent spread is frozen before any market field is attached;
- the historical market line is attached afterward;
- grading occurs only after prediction output has been saved;
- Week 1 is excluded because historical QB1 was established from Week 1 usage;
- Week 18 is projected but excluded from default betting summaries.

Primary outputs
---------------
Inside every <database-root>/<season>.sqlite:
- nfl_rebuilt_weekly_replay_predictions
- nfl_rebuilt_weekly_form_history
- nfl_rebuilt_weekly_replay_run_audit
- nfl_rebuilt_weekly_replay_threshold_summary
- nfl_rebuilt_weekly_replay_season_summary

Combined output database:
- <database-root>/nfl_rebuilt_historical_replay.sqlite

The production database and production scripts are never modified.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


BUILD_ID = "NFL_REBUILT_HISTORICAL_WEEKLY_REPLAY_CANONICAL_V4"
VERSION = "v4_1_dynamic_form_process_history_floor"

EXPECTED_FORM_BUILD_ID = "NFL_2026_FORM_RATING_CANONICAL_V3"
EXPECTED_FORM_VERSION = "v3_current_structural_prior_asof_opponent_adjusted_form"
EXPECTED_PREDICT_BUILD_ID = "NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4"
EXPECTED_PREDICT_VERSION = "v4_structural_form_hfa_fail_closed_lineage"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_TARGET_SEASONS = (2023, 2024, 2025)
DEFAULT_FIRST_WEEK = 2
DEFAULT_LAST_WEEK = 17
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
DEFAULT_MAX_DISAGREEMENT = 5.0
DEFAULT_FLAT_STAKE = 500.0
DEFAULT_BANKROLL = 50_000.0
DEFAULT_PRICE = -110

# The production form builder requires at least 800 historical games across
# eight calibration seasons.  Historical replays before 2022 necessarily have
# shorter prior-only windows, so retain a conservative 200-game requirement
# per available calibration season, capped at the production 800-game floor.
# The form builder applies this game count to two team-game rows per game.
FORM_PROCESS_MINIMUM_GAMES_PER_SEASON = 200
FORM_PROCESS_MINIMUM_GAMES_CAP = 800
FORM_PROCESS_MINIMUM_CALIBRATION_SEASONS = 2

CONTEXT_TABLE = "nfl_historical_reconstruction_context"
SCHEDULE_TABLE = "nfl_schedule_target"
POWER_TABLE = "nfl_power_ratings_target"
POWER_CALIBRATION_TABLE = "nfl_power_rating_calibration_season_audit_target"

FORM_TABLE = "nfl_form_ratings_replay_current"
FORM_SNAPSHOT_TABLE = "nfl_preseason_power_snapshot_replay"
FORM_GAME_AUDIT_TABLE = "nfl_form_game_audit_replay_current"
FORM_TEAM_GAME_AUDIT_TABLE = "nfl_form_team_game_audit_replay_current"
FORM_PROCESS_MODEL_TABLE = "nfl_form_process_model_replay"
FORM_FEATURE_CACHE_TABLE = "nfl_form_historical_feature_cache_replay"

PREDICTION_TABLE = "nfl_weekly_spread_predictions_replay_current"
PREDICTION_HISTORY_TABLE = "nfl_weekly_spread_predictions_replay_history"
PREDICTION_RUN_AUDIT_TABLE = "nfl_weekly_spread_run_audit_replay_current"
PREDICTION_RUN_HISTORY_TABLE = "nfl_weekly_spread_run_audit_replay_history"
MANUAL_ADJUSTMENT_TABLE = "nfl_weekly_manual_adjustments_replay"
SCHEDULE_FACTOR_TABLE = "nfl_schedule_factors_replay"

REPLAY_PREDICTIONS_TABLE = "nfl_rebuilt_weekly_replay_predictions"
REPLAY_FORM_HISTORY_TABLE = "nfl_rebuilt_weekly_form_history"
REPLAY_RUN_AUDIT_TABLE = "nfl_rebuilt_weekly_replay_run_audit"
REPLAY_THRESHOLD_TABLE = "nfl_rebuilt_weekly_replay_threshold_summary"
REPLAY_SEASON_TABLE = "nfl_rebuilt_weekly_replay_season_summary"
COMBINED_DB_NAME = "nfl_rebuilt_historical_replay.sqlite"

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}

# Approximate home-stadium coordinates are used only for the predictor's small
# current-trip 2,000-mile flag. Neutral/international games do not receive a
# distance flag because the isolated schedule does not retain exact venue data.
TEAM_COORDINATES = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7868),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3240, -81.6373), "KC": (39.0489, -94.4839),
    "LAC": (33.9535, -118.3392), "LAR": (33.9535, -118.3392),
    "LV": (36.0908, -115.1830), "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575), "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

TEAM_REGION = {
    "ARI": "WEST", "DEN": "WEST", "LAC": "WEST", "LAR": "WEST",
    "LV": "WEST", "SEA": "WEST", "SF": "WEST",
    "CHI": "CENTRAL", "DAL": "CENTRAL", "GB": "CENTRAL",
    "HOU": "CENTRAL", "KC": "CENTRAL", "MIN": "CENTRAL",
    "NO": "CENTRAL", "TEN": "CENTRAL",
}
for _team in TEAM_COORDINATES:
    TEAM_REGION.setdefault(_team, "EAST")


@dataclass(frozen=True)
class RuntimeScript:
    key: str
    source_path: Path
    patched_path: Path
    source_sha256: str


def parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not seasons:
        raise argparse.ArgumentTypeError("At least one target season is required.")
    return seasons


def parse_thresholds(value: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in value.split(",") if x.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--database-root",
        type=Path,
        default=None,
        help="Directory containing isolated <season>.sqlite databases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for replay CSV outputs.",
    )
    parser.add_argument("--target-seasons", type=parse_seasons, default=DEFAULT_TARGET_SEASONS)
    parser.add_argument("--first-week", type=int, default=DEFAULT_FIRST_WEEK)
    parser.add_argument("--last-week", type=int, default=DEFAULT_LAST_WEEK)
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--maximum-market-disagreement", type=float, default=DEFAULT_MAX_DISAGREEMENT)
    parser.add_argument("--flat-stake", type=float, default=DEFAULT_FLAT_STAKE)
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.database_root = (
        args.database_root.resolve()
        if args.database_root is not None
        else (args.project_root / "backtests").resolve()
    )
    args.output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (args.project_root / "outputs").resolve()
    )
    args.python = args.python.resolve() if args.python else Path(sys.executable).resolve()
    if not (2 <= args.first_week <= 18):
        parser.error("--first-week must be between 2 and 18.")
    if not (args.first_week <= args.last_week <= 18):
        parser.error("--last-week must be between first-week and 18.")
    if args.maximum_market_disagreement <= 0:
        parser.error("--maximum-market-disagreement must be positive.")
    return args


def now_string() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def normalize_team(value: Any) -> str:
    text = "" if value is None else str(value).upper().strip()
    return TEAM_ALIASES.get(text, text)


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone() is not None


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
    frame.columns = [str(c).lower().strip() for c in frame.columns]
    return frame


def drop_tables(connection: sqlite3.Connection, names: Iterable[str]) -> None:
    for name in names:
        escaped = name.replace('"', '""')
        connection.execute(f'DROP TABLE IF EXISTS "{escaped}"')
    connection.commit()


def append_schema_evolving(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if not table_exists(connection, table):
        frame.to_sql(table, connection, if_exists="replace", index=False)
        return
    existing = [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()]
    for column in frame.columns:
        if column not in existing:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}"')
            existing.append(column)
    aligned = frame.copy()
    for column in existing:
        if column not in aligned.columns:
            aligned[column] = None
    aligned[existing].to_sql(table, connection, if_exists="append", index=False)


def assignment_nodes(source: str) -> dict[str, ast.Assign | ast.AnnAssign]:
    tree = ast.parse(source)
    nodes: dict[str, ast.Assign | ast.AnnAssign] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
    return nodes


def patch_assignments(source: str, replacements: dict[str, str]) -> str:
    nodes = assignment_nodes(source)
    missing = sorted(set(replacements) - set(nodes))
    if missing:
        raise RuntimeError(f"Patch assignments not found: {missing}")
    lines = source.splitlines()
    edits = []
    for name, expression in replacements.items():
        node = nodes[name]
        edits.append((node.lineno - 1, node.end_lineno or node.lineno, f"{name} = {expression}"))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    return "\n".join(lines) + "\n"


def verify_source(path: Path, build_id: str, version: str, allow_mismatch: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if not allow_mismatch:
        if build_id not in source:
            raise RuntimeError(f"{path.name} missing approved Build ID {build_id}.")
        if version not in source:
            raise RuntimeError(f"{path.name} missing approved version {version}.")
    return source


def form_process_minimum_games(calibration_seasons: list[int]) -> int:
    if len(calibration_seasons) < FORM_PROCESS_MINIMUM_CALIBRATION_SEASONS:
        raise RuntimeError(
            "Historical form replay requires at least two prior calibration "
            f"seasons; found {calibration_seasons}."
        )
    return min(
        FORM_PROCESS_MINIMUM_GAMES_CAP,
        FORM_PROCESS_MINIMUM_GAMES_PER_SEASON * len(calibration_seasons),
    )


def literal_assignment(source: str, name: str) -> Any:
    node = assignment_nodes(source).get(name)
    if node is None:
        raise RuntimeError(f"Patched assignment is missing: {name}")
    try:
        return ast.literal_eval(node.value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Patched assignment is not a literal: {name}") from exc


def validate_form_patch(
    patched: str,
    season: int,
    calibration: list[int],
    minimum_historical_games: int,
) -> None:
    expected = {
        "SEASON": season,
        "CALIBRATION_SEASONS": calibration,
        "MIN_HISTORICAL_GAMES": minimum_historical_games,
        "PRESEASON_POWER_TABLE": POWER_TABLE,
        "OUTPUT_TABLE": FORM_TABLE,
        "PROCESS_MODEL_TABLE": FORM_PROCESS_MODEL_TABLE,
        "HISTORICAL_FEATURE_CACHE_TABLE": FORM_FEATURE_CACHE_TABLE,
    }
    mismatches = {
        name: (literal_assignment(patched, name), value)
        for name, value in expected.items()
        if literal_assignment(patched, name) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Historical form patch contract failed for {season}: {mismatches}"
        )


def patch_form(source_path: Path, target: Path, season: int, allow_mismatch: bool) -> RuntimeScript:
    source = verify_source(source_path, EXPECTED_FORM_BUILD_ID, EXPECTED_FORM_VERSION, allow_mismatch)
    calibration = list(range(2018, season))
    minimum_historical_games = form_process_minimum_games(calibration)
    replacements = {
        "SEASON": repr(season),
        "BUILD_ID": repr(f"NFL_FORM_RATING_{season}_HISTORICAL_REPLAY_V2"),
        "VERSION": repr(
            "v2_prior_only_historical_replay_using_current_structural_canonical_v3"
        ),
        "PROCESS_MODEL_VERSION": repr(
            f"v2_1_2018_{season - 1}_process_to_points_min_{minimum_historical_games}"
        ),
        "PRESEASON_POWER_TABLE": repr(POWER_TABLE),
        "PRESEASON_SNAPSHOT_TABLE": repr(FORM_SNAPSHOT_TABLE),
        "OUTPUT_TABLE": repr(FORM_TABLE),
        "GAME_AUDIT_TABLE": repr(FORM_GAME_AUDIT_TABLE),
        "TEAM_GAME_AUDIT_TABLE": repr(FORM_TEAM_GAME_AUDIT_TABLE),
        "PROCESS_MODEL_TABLE": repr(FORM_PROCESS_MODEL_TABLE),
        "HISTORICAL_FEATURE_CACHE_TABLE": repr(FORM_FEATURE_CACHE_TABLE),
        "CALIBRATION_SEASONS": repr(calibration),
        "MIN_HISTORICAL_GAMES": repr(minimum_historical_games),
    }
    patched = patch_assignments(source, replacements)
    validate_form_patch(
        patched,
        season,
        calibration,
        minimum_historical_games,
    )
    target.write_text(patched, encoding="utf-8")
    return RuntimeScript("form", source_path, target, hashlib.sha256(source_path.read_bytes()).hexdigest())


def patch_predictor(source_path: Path, target: Path, season: int, allow_mismatch: bool) -> RuntimeScript:
    source = verify_source(source_path, EXPECTED_PREDICT_BUILD_ID, EXPECTED_PREDICT_VERSION, allow_mismatch)
    replacements = {
        "SEASON": repr(season),
        "BUILD_ID": repr(f"NFL_WEEKLY_POWER_SPREADS_{season}_HISTORICAL_REPLAY_V2"),
        "VERSION": repr("v3_historical_replay_using_canonical_predictor_v4"),
        "EXPECTED_FORM_BUILD_ID": repr(
            f"NFL_FORM_RATING_{season}_HISTORICAL_REPLAY_V2"
        ),
        "EXPECTED_FORM_VERSION": repr(
            "v2_prior_only_historical_replay_using_current_structural_canonical_v3"
        ),
        "FORM_RATING_TABLE": repr(FORM_TABLE),
        "POWER_RATING_TABLE": repr(POWER_TABLE),
        "POWER_CALIBRATION_TABLE": repr(POWER_CALIBRATION_TABLE),
        "MANUAL_ADJUSTMENT_TABLE": repr(MANUAL_ADJUSTMENT_TABLE),
        "SCHEDULE_FACTOR_TABLE": repr(SCHEDULE_FACTOR_TABLE),
        "OUTPUT_TABLE": repr(PREDICTION_TABLE),
        "OUTPUT_HISTORY_TABLE": repr(PREDICTION_HISTORY_TABLE),
        "RUN_AUDIT_TABLE": repr(PREDICTION_RUN_AUDIT_TABLE),
        "RUN_AUDIT_HISTORY_TABLE": repr(PREDICTION_RUN_HISTORY_TABLE),
        "MODEL_VARIANT": repr("STRUCTURAL_FORM_HFA_HISTORICAL_REPLAY_V2"),
    }
    patched = patch_assignments(source, replacements)
    target.write_text(patched, encoding="utf-8")
    return RuntimeScript("predictor", source_path, target, hashlib.sha256(source_path.read_bytes()).hexdigest())


def compile_script(python_executable: Path, path: Path) -> None:
    # Compile in memory. Writing the normal __pycache__ artifact can exceed the
    # Windows path limit inside timestamped isolated replay directories.
    del python_executable
    try:
        source = path.read_text(encoding="utf-8", errors="strict")
        compile(source, str(path), "exec")
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"Compilation failed for {path}: {exc}") from exc


def prepare_runtime_scripts(args: argparse.Namespace, season: int, runtime_root: Path) -> tuple[RuntimeScript, RuntimeScript]:
    scripts = runtime_root / "patched_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    form = patch_form(
        args.project_root / "build_nfl_2026_form_rating.py",
        scripts / "build_nfl_2026_form_rating.py",
        season,
        args.allow_source_mismatch,
    )
    predictor = patch_predictor(
        args.project_root / "predict_nfl_weekly_power_spreads_2026.py",
        scripts / "predict_nfl_weekly_power_spreads_2026.py",
        season,
        args.allow_source_mismatch,
    )
    compile_script(args.python, form.patched_path)
    compile_script(args.python, predictor.patched_path)
    return form, predictor


def preflight_all(args: argparse.Namespace) -> None:
    root = args.database_root / "runtime" / "replay_preflight"
    if root.exists():
        shutil.rmtree(root)
    count = 0
    for season in args.target_seasons:
        season_root = root / str(season)
        prepare_runtime_scripts(args, season, season_root)
        count += 2
    print(f"[REPLAY] Preflight passed: {count} patched canonical children compiled.")
    shutil.rmtree(root)


def standardize_schedule(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]
    aliases = {
        "season": ["season"], "week": ["week"],
        "game_id": ["game_id", "id"], "game_date": ["game_date", "gameday", "date"],
        "home_team": ["home_team", "home"], "away_team": ["away_team", "away"],
        "home_score": ["home_score", "home_points"], "away_score": ["away_score", "away_points"],
        "neutral_site": ["neutral_site", "neutral"], "spread_line": ["spread_line", "closing_spread", "market_spread"],
        "game_type": ["game_type", "season_type"],
    }
    def find(names: list[str]) -> Optional[str]:
        for name in names:
            if name in out.columns:
                return name
        return None
    result = pd.DataFrame(index=out.index)
    for key, names in aliases.items():
        column = find(names)
        result[key] = out[column] if column else None
    result["season"] = pd.to_numeric(result["season"], errors="coerce").fillna(season).astype(int)
    result = result[result["season"].eq(season)].copy()
    result["week"] = pd.to_numeric(result["week"], errors="coerce").astype("Int64")
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce")
    result["home_team"] = result["home_team"].map(normalize_team)
    result["away_team"] = result["away_team"].map(normalize_team)
    result["home_score"] = pd.to_numeric(result["home_score"], errors="coerce")
    result["away_score"] = pd.to_numeric(result["away_score"], errors="coerce")
    result["spread_line"] = pd.to_numeric(result["spread_line"], errors="coerce")
    result["neutral_site"] = pd.to_numeric(result["neutral_site"], errors="coerce").fillna(0).astype(int)
    result["game_type"] = result["game_type"].fillna("REG").astype(str).str.upper()
    result = result[result["week"].notna() & result["home_team"].notna() & result["away_team"].notna()].copy()
    result["week"] = result["week"].astype(int)
    missing_id = result["game_id"].isna() | result["game_id"].astype(str).isin(["", "None", "nan"])
    result.loc[missing_id, "game_id"] = (
        result.loc[missing_id, "season"].astype(str) + "_" +
        result.loc[missing_id, "week"].astype(str) + "_" +
        result.loc[missing_id, "away_team"] + "_" + result.loc[missing_id, "home_team"]
    )
    result["completed"] = (result["home_score"].notna() & result["away_score"].notna()).astype(int)
    return result.sort_values(["week", "game_date", "game_id"]).drop_duplicates("game_id").reset_index(drop=True)


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.7613 * 2 * math.asin(min(1.0, math.sqrt(value)))


def build_schedule_factors(schedule: pd.DataFrame) -> pd.DataFrame:
    history: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    ordered = schedule.sort_values(["game_date", "week", "game_id"]).copy()
    for _, game in ordered.iterrows():
        home = str(game["home_team"])
        away = str(game["away_team"])
        date = pd.Timestamp(game["game_date"]).normalize() if pd.notna(game["game_date"]) else pd.NaT
        neutral = int(game.get("neutral_site", 0) or 0)
        home_hist = history.get(home, [])
        away_hist = history.get(away, [])
        home_prev = home_hist[-1] if home_hist else None
        away_prev = away_hist[-1] if away_hist else None
        home_rest = (date - home_prev["date"]).days if home_prev and pd.notna(date) else np.nan
        away_rest = (date - away_prev["date"]).days if away_prev and pd.notna(date) else np.nan
        rest_diff = home_rest - away_rest if pd.notna(home_rest) and pd.notna(away_rest) else np.nan
        home_bye = int(pd.notna(home_rest) and home_rest >= 12)
        away_bye = int(pd.notna(away_rest) and away_rest >= 12)
        current_away_flags = [1] + [int(item["is_away"]) for item in reversed(away_hist[-3:])]
        road_last4 = int(sum(current_away_flags))
        consecutive_road = 1
        for item in reversed(away_hist):
            if item["is_away"]:
                consecutive_road += 1
            else:
                break
        back_to_back_away = int(bool(away_prev and away_prev["is_away"]))
        three_of_four = int(len(current_away_flags) >= 4 and road_last4 >= 3)
        travel = np.nan
        travel_2000 = 0
        west_east = 0
        west_central = 0
        if neutral == 0 and away in TEAM_COORDINATES and home in TEAM_COORDINATES:
            travel = haversine_miles(TEAM_COORDINATES[away], TEAM_COORDINATES[home])
            travel_2000 = int(travel >= 2000.0)
            if away_prev and pd.notna(away_rest) and away_rest <= 10:
                previous_region = away_prev["venue_region"]
                current_region = TEAM_REGION.get(home, "EAST")
                west_east = int(previous_region == "WEST" and current_region == "EAST")
                west_central = int(previous_region == "WEST" and current_region == "CENTRAL")
        rows.append({
            "season": int(game["season"]), "week": int(game["week"]), "game_id": str(game["game_id"]),
            "home_team": home, "away_team": away, "travel_miles": travel,
            "home_days_rest": home_rest, "away_days_rest": away_rest, "rest_differential": rest_diff,
            "home_rest_advantage_3plus": int(pd.notna(rest_diff) and rest_diff >= 3),
            "away_rest_advantage_3plus": int(pd.notna(rest_diff) and rest_diff <= -3),
            "travel_dist_2000_plus": travel_2000, "back_to_back_west_east": west_east,
            "back_to_back_west_central": west_central, "back_to_back_away": back_to_back_away,
            "three_of_four_away": three_of_four, "bye_before_away": away_bye,
            "bye_before_home": home_bye, "international_game": 0,
            "away_consecutive_road_games": consecutive_road, "away_road_games_last4": road_last4,
            "location": home, "factor_method": "historical_replay_schedule_derived_v1",
        })
        home_record = {"date": date, "is_away": False, "venue_region": TEAM_REGION.get(home, "EAST")}
        away_record = {"date": date, "is_away": True, "venue_region": TEAM_REGION.get(home, "EAST")}
        history.setdefault(home, []).append(home_record)
        history.setdefault(away, []).append(away_record)
    return pd.DataFrame(rows)


def prepare_prediction_schedule(schedule: pd.DataFrame, path: Path) -> None:
    frame = schedule.copy()
    frame["home_score"] = np.nan
    frame["away_score"] = np.nan
    frame["game_type"] = "REG"
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_child(command: list[str], cwd: Path, label: str) -> None:
    print("\n" + "-" * 112)
    print(f"[REPLAY] Running {label}: {subprocess.list2cmdline(command)}")
    print("-" * 112)
    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = int(process.wait())
    if code != 0:
        raise RuntimeError(f"{label} failed with return code {code}.")


def prediction_date_for_week(schedule: pd.DataFrame, week: int) -> pd.Timestamp:
    dates = schedule.loc[schedule["week"].eq(week), "game_date"].dropna()
    if dates.empty:
        raise RuntimeError(f"Week {week} has no game dates.")
    return pd.Timestamp(dates.min()).normalize() - pd.Timedelta(days=1)


def grade_predictions(predictions: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    actual = schedule.loc[schedule["week"].eq(week), [
        "game_id", "home_score", "away_score", "spread_line", "completed"
    ]].copy()
    actual = actual.rename(columns={"spread_line": "source_market_home_margin"})
    out = predictions.merge(actual, on="game_id", how="left", validate="one_to_one", suffixes=("", "_actual"))
    out["actual_home_margin"] = pd.to_numeric(out["home_score"], errors="coerce") - pd.to_numeric(out["away_score"], errors="coerce")
    market = pd.to_numeric(out["available_market_home_margin"], errors="coerce")
    out["home_cover_margin"] = out["actual_home_margin"] - market
    selected = out["selected_side"].astype(str).str.upper()
    out["selected_cover_margin"] = np.where(selected.eq("HOME"), out["home_cover_margin"], -out["home_cover_margin"])
    out["ats_result"] = np.where(
        out["selected_cover_margin"].isna(), "N",
        np.where(out["selected_cover_margin"].gt(1e-9), "W", np.where(out["selected_cover_margin"].lt(-1e-9), "L", "P")),
    )
    out["ats_profit_units"] = np.where(out["ats_result"].eq("W"), 100.0 / 110.0, np.where(out["ats_result"].eq("L"), -1.0, 0.0))
    out["replay_target_season"] = season
    out["replay_prediction_week"] = week
    out["replay_build_id"] = BUILD_ID
    out["replay_version"] = VERSION
    return out


def wilson_lower(wins: int, losses: int, z: float = 1.959963984540054) -> float:
    n = wins + losses
    if n <= 0:
        return np.nan
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def max_drawdown(values: pd.Series) -> float:
    cumulative = pd.to_numeric(values, errors="coerce").fillna(0.0).cumsum()
    if cumulative.empty:
        return 0.0
    running = cumulative.cummax()
    return float((running - cumulative).max())


def summarize_thresholds(predictions: pd.DataFrame, thresholds: Iterable[float], maximum: float, scope: str) -> pd.DataFrame:
    rows = []
    base = predictions.copy()
    base = base[
        pd.to_numeric(base["available_market_home_margin"], errors="coerce").notna()
        & base["ats_result"].isin(["W", "L", "P"])
    ].copy()
    base["absolute_spread_difference_points"] = pd.to_numeric(base["absolute_spread_difference_points"], errors="coerce")
    for threshold in thresholds:
        sample = base[
            base["absolute_spread_difference_points"].ge(float(threshold))
            & base["absolute_spread_difference_points"].le(float(maximum))
        ].sort_values(["season", "week", "game_date", "game_id"]).copy()
        wins = int(sample["ats_result"].eq("W").sum())
        losses = int(sample["ats_result"].eq("L").sum())
        pushes = int(sample["ats_result"].eq("P").sum())
        bets = len(sample)
        profit = float(sample["ats_profit_units"].sum())
        graded = wins + losses
        predicted_prob = pd.to_numeric(sample["model_selected_cover_probability"], errors="coerce")
        outcome = sample["ats_result"].eq("W").astype(float)
        nonpush = sample["ats_result"].isin(["W", "L"])
        brier = float(np.mean((predicted_prob[nonpush] - outcome[nonpush]) ** 2)) if nonpush.any() else np.nan
        rows.append({
            "scope": scope, "edge_threshold": float(threshold), "maximum_disagreement": float(maximum),
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "ats_win_rate": wins / graded if graded else np.nan,
            "ats_profit_units": profit, "ats_roi": profit / bets if bets else np.nan,
            "wilson_lower_95": wilson_lower(wins, losses),
            "average_selected_probability": float(predicted_prob.mean()) if bets else np.nan,
            "average_probability_edge": float(pd.to_numeric(sample["probability_edge"], errors="coerce").mean()) if bets else np.nan,
            "probability_brier_score": brier,
            "flat_maximum_drawdown_units": max_drawdown(sample["ats_profit_units"]),
            "build_id": BUILD_ID, "version": VERSION, "created_at": now_string(),
        })
    return pd.DataFrame(rows)


def summarize_seasons(predictions: pd.DataFrame, primary_threshold: float, maximum: float) -> pd.DataFrame:
    rows = []
    for season, frame in predictions.groupby("season"):
        model_error = pd.to_numeric(frame["projected_home_margin"], errors="coerce") - pd.to_numeric(frame["actual_home_margin"], errors="coerce")
        sample = frame[
            pd.to_numeric(frame["absolute_spread_difference_points"], errors="coerce").ge(primary_threshold)
            & pd.to_numeric(frame["absolute_spread_difference_points"], errors="coerce").le(maximum)
            & frame["ats_result"].isin(["W", "L", "P"])
        ].copy()
        wins = int(sample["ats_result"].eq("W").sum())
        losses = int(sample["ats_result"].eq("L").sum())
        pushes = int(sample["ats_result"].eq("P").sum())
        bets = len(sample)
        profit = float(sample["ats_profit_units"].sum())
        rows.append({
            "season": int(season), "model_mae": float(model_error.abs().mean()),
            "model_rmse": float(np.sqrt(np.mean(np.square(model_error.dropna())))),
            "bets": bets, "wins": wins, "losses": losses, "pushes": pushes,
            "ats_win_rate": wins / (wins + losses) if wins + losses else np.nan,
            "ats_profit_units": profit, "ats_roi": profit / bets if bets else np.nan,
            "primary_threshold": primary_threshold, "maximum_disagreement": maximum,
            "build_id": BUILD_ID, "version": VERSION, "created_at": now_string(),
        })
    return pd.DataFrame(rows)


def validate_week(predictions: pd.DataFrame, form: pd.DataFrame, week: int) -> None:
    if predictions.empty:
        raise RuntimeError(f"Week {week} prediction output is empty.")
    if predictions["prediction_uses_market_inputs"].ne(0).any():
        raise RuntimeError(f"Week {week} market leakage detected.")
    if predictions["independent_projection_hash"].isna().any():
        raise RuntimeError(f"Week {week} independent hashes are missing.")
    through = pd.to_numeric(form["through_week"], errors="coerce").dropna()
    if not through.empty and int(through.max()) > week - 1:
        raise RuntimeError(f"Week {week} form contains future games through Week {int(through.max())}.")
    rating_week = pd.concat([
        pd.to_numeric(predictions["home_rating_through_week"], errors="coerce"),
        pd.to_numeric(predictions["away_rating_through_week"], errors="coerce"),
    ]).dropna()
    if not rating_week.empty and int(rating_week.max()) > week - 1:
        raise RuntimeError(f"Week {week} predictions used future rating week.")


def build_one_season(args: argparse.Namespace, season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_path = args.database_root / f"{season}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    runtime = args.database_root / "runtime" / f"{season}_weekly_replay"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "outputs").mkdir(exist_ok=True)
    (runtime / "logs").mkdir(exist_ok=True)
    form_script, predictor_script = prepare_runtime_scripts(args, season, runtime)

    with sqlite3.connect(db_path, timeout=60) as connection:
        connection.execute("PRAGMA busy_timeout=60000")
        context = read_table(connection, CONTEXT_TABLE)
        if len(context) != 1 or int(context.iloc[0].get("structural_inputs_ready_flag", 0)) != 1:
            raise RuntimeError(f"{season} preseason structural reconstruction is not ready.")
        schedule = standardize_schedule(read_table(connection, SCHEDULE_TABLE), season)
        if schedule.empty or schedule["home_team"].nunique() < 20:
            raise RuntimeError(f"{season} schedule is invalid.")
        if schedule.loc[schedule["week"].between(args.first_week, args.last_week), "spread_line"].notna().sum() == 0:
            raise RuntimeError(f"{season} has no historical spread lines.")
        schedule.to_sql("nfl_schedule", connection, if_exists="replace", index=False)
        factors = build_schedule_factors(schedule)
        factors.to_sql(SCHEDULE_FACTOR_TABLE, connection, if_exists="replace", index=False)
        drop_tables(connection, [
            FORM_TABLE, FORM_SNAPSHOT_TABLE, FORM_GAME_AUDIT_TABLE, FORM_TEAM_GAME_AUDIT_TABLE,
            FORM_PROCESS_MODEL_TABLE, FORM_FEATURE_CACHE_TABLE, PREDICTION_TABLE,
            PREDICTION_HISTORY_TABLE, PREDICTION_RUN_AUDIT_TABLE, PREDICTION_RUN_HISTORY_TABLE,
            MANUAL_ADJUSTMENT_TABLE, REPLAY_PREDICTIONS_TABLE, REPLAY_FORM_HISTORY_TABLE,
            REPLAY_RUN_AUDIT_TABLE, REPLAY_THRESHOLD_TABLE, REPLAY_SEASON_TABLE,
        ])
        # factors were dropped only if listed; restore after clean-up.
        factors.to_sql(SCHEDULE_FACTOR_TABLE, connection, if_exists="replace", index=False)

    prediction_schedule_path = runtime / f"nfl_prediction_schedule_{season}.csv"
    factor_path = runtime / f"nfl_schedule_factors_{season}.csv"
    prepare_prediction_schedule(schedule, prediction_schedule_path)
    factors.to_csv(factor_path, index=False, encoding="utf-8-sig")

    all_predictions = []
    all_forms = []
    audit_rows = []
    available_weeks = sorted(
        week for week in schedule["week"].unique().tolist()
        if args.first_week <= int(week) <= args.last_week
    )
    for week in available_weeks:
        week = int(week)
        pred_date = prediction_date_for_week(schedule, week)
        started = dt.datetime.now()
        form_command = [
            str(args.python), "-u", str(form_script.patched_path),
            "--project-root", str(runtime), "--db-path", str(db_path),
            "--as-of-date", pred_date.date().isoformat(),
            "--through-week", str(week - 1), "--no-csv",
        ]
        run_child(form_command, runtime, f"{season} Week {week} form through Week {week - 1}")
        predictor_command = [
            str(args.python), "-u", str(predictor_script.patched_path),
            "--project-root", str(runtime), "--db-path", str(db_path),
            "--schedule-path", str(prediction_schedule_path),
            "--schedule-factors-path", str(factor_path),
            "--week", str(week), "--as-of-date", pred_date.date().isoformat(),
            "--market-line-preference", "current",
            "--minimum-spread-difference", str(min(args.thresholds)),
            "--maximum-market-disagreement", str(args.maximum_market_disagreement),
            "--flat-stake", str(args.flat_stake), "--bankroll", str(args.bankroll),
            "--default-spread-price", str(DEFAULT_PRICE), "--no-csv",
        ]
        run_child(predictor_command, runtime, f"{season} Week {week} predictor")
        with sqlite3.connect(db_path, timeout=60) as connection:
            form = read_table(connection, FORM_TABLE)
            predictions = read_table(connection, PREDICTION_TABLE)
            validate_week(predictions, form, week)
            graded = grade_predictions(predictions, schedule, season, week)
            form_snapshot = form.copy()
            form_snapshot["replay_prediction_week"] = week
            form_snapshot["replay_prediction_as_of_date"] = pred_date.date().isoformat()
            form_snapshot["replay_build_id"] = BUILD_ID
            append_schema_evolving(connection, REPLAY_PREDICTIONS_TABLE, graded)
            append_schema_evolving(connection, REPLAY_FORM_HISTORY_TABLE, form_snapshot)
            elapsed = (dt.datetime.now() - started).total_seconds()
            audit = pd.DataFrame([{
                "run_id": str(uuid.uuid4()), "season": season, "week": week,
                "form_through_week": week - 1, "prediction_as_of_date": pred_date.date().isoformat(),
                "games": len(graded), "markets_attached": int(graded["market_attached_after_prediction_freeze"].sum()),
                "market_leakage_rows": int(graded["prediction_uses_market_inputs"].ne(0).sum()),
                "graded_games": int(graded["ats_result"].isin(["W", "L", "P"]).sum()),
                "elapsed_seconds": elapsed, "build_id": BUILD_ID, "version": VERSION,
                "completed_at": now_string(),
            }])
            append_schema_evolving(connection, REPLAY_RUN_AUDIT_TABLE, audit)
            connection.commit()
        all_predictions.append(graded)
        all_forms.append(form_snapshot)
        audit_rows.append(audit)
        print(
            f"[REPLAY] {season} Week {week}: games={len(graded)} | "
            f"markets={int(graded['market_attached_after_prediction_freeze'].sum())} | "
            f"form_through={week - 1} | leakage=0"
        )

    predictions_all = pd.concat(all_predictions, ignore_index=True, sort=False)
    forms_all = pd.concat(all_forms, ignore_index=True, sort=False)
    audits_all = pd.concat(audit_rows, ignore_index=True, sort=False)
    thresholds = summarize_thresholds(predictions_all, args.thresholds, args.maximum_market_disagreement, f"SEASON_{season}")
    seasons = summarize_seasons(predictions_all, 2.5, args.maximum_market_disagreement)
    with sqlite3.connect(db_path) as connection:
        thresholds.to_sql(REPLAY_THRESHOLD_TABLE, connection, if_exists="replace", index=False)
        seasons.to_sql(REPLAY_SEASON_TABLE, connection, if_exists="replace", index=False)
        context_columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{CONTEXT_TABLE}")').fetchall()}
        additions = {
            "weekly_replay_ready_flag": "INTEGER", "weekly_replay_games": "INTEGER",
            "weekly_replay_market_games": "INTEGER", "weekly_replay_build_id": "TEXT",
            "weekly_replay_version": "TEXT", "weekly_replay_completed_at": "TEXT",
        }
        for column, sql_type in additions.items():
            if column not in context_columns:
                connection.execute(f'ALTER TABLE "{CONTEXT_TABLE}" ADD COLUMN "{column}" {sql_type}')
        connection.execute(
            f"UPDATE {CONTEXT_TABLE} SET weekly_replay_ready_flag=1, weekly_replay_games=?, "
            "weekly_replay_market_games=?, weekly_replay_build_id=?, weekly_replay_version=?, "
            "weekly_replay_completed_at=?, structural_inputs_pending=''",
            (len(predictions_all), int(predictions_all["available_market_home_margin"].notna().sum()), BUILD_ID, VERSION, now_string()),
        )
        connection.commit()
    if not args.no_csv:
        output_dir = args.output_root / "historical_weekly_replay" / str(season)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_all.to_csv(output_dir / f"nfl_rebuilt_weekly_replay_predictions_{season}.csv", index=False, encoding="utf-8-sig")
        thresholds.to_csv(output_dir / f"nfl_rebuilt_weekly_replay_threshold_summary_{season}.csv", index=False, encoding="utf-8-sig")
    if not args.keep_runtime:
        shutil.rmtree(runtime)
    return predictions_all, forms_all, audits_all


def save_combined(args: argparse.Namespace, predictions: pd.DataFrame, forms: pd.DataFrame, audits: pd.DataFrame) -> Path:
    db_path = args.database_root / COMBINED_DB_NAME
    if db_path.exists():
        db_path.unlink()
    season_span = f"{min(args.target_seasons)}_{max(args.target_seasons)}"
    overall = summarize_thresholds(
        predictions,
        args.thresholds,
        args.maximum_market_disagreement,
        f"ALL_{season_span}",
    )
    holdout = summarize_thresholds(
        predictions[predictions["season"].isin([2024, 2025])].copy(),
        args.thresholds, args.maximum_market_disagreement, "LOCKED_2024_2025",
    )
    threshold_summary = pd.concat([overall, holdout], ignore_index=True)
    season_summary = summarize_seasons(predictions, 2.5, args.maximum_market_disagreement)
    with sqlite3.connect(db_path) as connection:
        predictions.to_sql(REPLAY_PREDICTIONS_TABLE, connection, if_exists="replace", index=False)
        forms.to_sql(REPLAY_FORM_HISTORY_TABLE, connection, if_exists="replace", index=False)
        audits.to_sql(REPLAY_RUN_AUDIT_TABLE, connection, if_exists="replace", index=False)
        threshold_summary.to_sql(REPLAY_THRESHOLD_TABLE, connection, if_exists="replace", index=False)
        season_summary.to_sql(REPLAY_SEASON_TABLE, connection, if_exists="replace", index=False)
    if not args.no_csv:
        output_dir = args.output_root / "historical_weekly_replay"
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_dir / f"nfl_rebuilt_weekly_replay_predictions_{season_span}.csv", index=False, encoding="utf-8-sig")
        threshold_summary.to_csv(output_dir / f"nfl_rebuilt_weekly_replay_threshold_summary_{season_span}.csv", index=False, encoding="utf-8-sig")
        season_summary.to_csv(output_dir / f"nfl_rebuilt_weekly_replay_season_summary_{season_span}.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 112)
    print("[REPLAY] REBUILT MODEL HISTORICAL REPLAY SUMMARY")
    print("=" * 112)
    print("[REPLAY] New structural model historically replayed: YES")
    print("[REPLAY] Week 1 graded: NO")
    print("[REPLAY] Market used in independent projection: NO")
    print("[REPLAY] Process calibration uses target/future seasons: NO")
    print("\n[REPLAY] Locked 2024-2025 threshold results:")
    print(holdout.to_string(index=False))
    print("\n[REPLAY] Primary-threshold season results:")
    print(season_summary.to_string(index=False))
    print(f"\n[REPLAY] Combined database: {db_path}")
    return db_path


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()
    print("[REPLAY] Starting rebuilt canonical historical weekly replay")
    print(f"[REPLAY] Build ID: {BUILD_ID}")
    print(f"[REPLAY] Version: {VERSION}")
    print(f"[REPLAY] Target seasons: {list(args.target_seasons)}")
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    preflight_all(args)
    if args.preflight_only:
        print("[REPLAY] Preflight-only run complete; databases unchanged.")
        return 0
    prediction_frames = []
    form_frames = []
    audit_frames = []
    for season in args.target_seasons:
        print("\n" + "=" * 112)
        print(f"[REPLAY] Replaying {season} from Week {args.first_week} through Week {args.last_week}")
        print("=" * 112)
        predictions, forms, audits = build_one_season(args, int(season))
        prediction_frames.append(predictions)
        form_frames.append(forms)
        audit_frames.append(audits)
    combined_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    combined_forms = pd.concat(form_frames, ignore_index=True, sort=False)
    combined_audits = pd.concat(audit_frames, ignore_index=True, sort=False)
    save_combined(args, combined_predictions, combined_forms, combined_audits)
    print(f"[REPLAY] Completed successfully in {(dt.datetime.now() - started).total_seconds():.2f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[REPLAY] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[REPLAY] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
