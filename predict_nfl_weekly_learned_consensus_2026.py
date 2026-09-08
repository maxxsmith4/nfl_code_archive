#!/usr/bin/env python
"""Produce the canonical 2026 learned structural-consensus weekly projection.

All three projections are completed before a market is attached:

1. learned eight-unit positive ridge;
2. learned 26-slot positive ridge; and
3. nonlinear unit/form/process/personnel model.

The fair margin is the mean of the unit and nonlinear projections.  The frozen
prospective gate requires at least a two-point edge, no more than 3.5 points of
range across the three projections, and all three models selecting the same
side.  Week 1 is outside the validating backtest scope and is stake-ineligible
by default.  The explicit --allow-week1-stakes override permits otherwise
qualifying Week 1 bets while preserving an unvalidated-scope audit label.

This script replaces the legacy structural/HFA branch.  It deliberately keeps
the established schedule, market, price, history-table, and CSV contracts so
the separate Circa engine is not modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sqlite3
import sys
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

import backtest_nfl_learned_structural_weights as learned
import backtest_nfl_nonlinear_matchup_consensus as nonlinear
import build_backtest_nfl_weekly_matchup_residual as matchup_history
import build_nfl_learned_consensus_2026 as frozen
import predict_nfl_weekly_power_spreads_2026 as legacy


SEASON = 2026
BUILD_ID = "NFL_WEEKLY_LEARNED_CONSENSUS_2026_CANONICAL_V1"
VERSION = "v1_3_readable_execution_csv_scope_fix"
MODEL_VARIANT = "LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS"

EXPECTED_BUNDLE_BUILD_ID = frozen.BUILD_ID
EXPECTED_BUNDLE_VERSION = frozen.VERSION
EXPECTED_FORM_BUILD_ID = legacy.EXPECTED_FORM_BUILD_ID
EXPECTED_FORM_VERSION = legacy.EXPECTED_FORM_VERSION

OUTPUT_TABLE = legacy.OUTPUT_TABLE
OUTPUT_HISTORY_TABLE = legacy.OUTPUT_HISTORY_TABLE
RUN_AUDIT_TABLE = legacy.RUN_AUDIT_TABLE
RUN_AUDIT_HISTORY_TABLE = legacy.RUN_AUDIT_HISTORY_TABLE

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--db-path", "--database", dest="db_path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--source-cache-db", type=Path)
    parser.add_argument("--week", "--prediction-week", dest="week", type=int, required=True)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--schedule-path", type=Path, default=None)
    parser.add_argument("--schedule-factors-path", type=Path, default=None)
    parser.add_argument("--market-path", type=Path, default=None)
    parser.add_argument(
        "--market-line-preference", choices=("current", "opening"), default="current"
    )
    parser.add_argument("--current-pbp-path", type=Path, default=None)
    parser.add_argument("--current-snaps-path", type=Path, default=None)
    parser.add_argument("--bankroll", type=float, default=50_000.0)
    parser.add_argument("--flat-stake", type=float, default=500.0)
    parser.add_argument(
        "--minimum-spread-difference",
        type=float,
        default=frozen.FROZEN_EDGE_THRESHOLD,
    )
    parser.add_argument(
        "--maximum-market-disagreement",
        type=float,
        default=frozen.FROZEN_MAXIMUM_PROJECTION_RANGE,
        help="Compatibility name; this is the maximum three-model projection range.",
    )
    parser.add_argument("--quarter-kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--max-kelly-bet-fraction", type=float, default=0.05)
    parser.add_argument("--default-spread-price", type=int, default=-110)
    parser.add_argument(
        "--allow-week1-stakes",
        action="store_true",
        help=(
            "Permit Week 1 stakes that pass the frozen consensus gate. "
            "Week 1 was not included in the validating backtest, so these "
            "bets are explicitly labeled outside validated scope."
        ),
    )
    parser.add_argument("--include-week18", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.database_root = (args.database_root or args.project_root / "backtests").expanduser().resolve()
    args.db_path = args.db_path.expanduser().resolve()
    args.model_path = (
        args.model_path or args.project_root / "models" / "nfl_learned_consensus_2026.joblib"
    ).expanduser().resolve()
    args.source_cache_db = (
        args.source_cache_db
        or args.database_root / matchup_history.CACHE_DATABASE_NAME
    ).expanduser().resolve()
    for name in (
        "schedule_path",
        "schedule_factors_path",
        "market_path",
        "current_pbp_path",
        "current_snaps_path",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if not 1 <= args.week <= 18:
        parser.error("--week must be between 1 and 18.")
    if not np.isclose(args.minimum_spread_difference, frozen.FROZEN_EDGE_THRESHOLD):
        parser.error(
            f"The canonical gate is frozen at edge >= {frozen.FROZEN_EDGE_THRESHOLD:.1f}."
        )
    if not np.isclose(
        args.maximum_market_disagreement, frozen.FROZEN_MAXIMUM_PROJECTION_RANGE
    ):
        parser.error(
            "The canonical gate is frozen at three-model range <= "
            f"{frozen.FROZEN_MAXIMUM_PROJECTION_RANGE:.1f}."
        )
    if args.bankroll <= 0 or args.flat_stake < 0:
        parser.error("--bankroll must be positive and --flat-stake cannot be negative.")
    if not 0 <= args.quarter_kelly_multiplier <= 1:
        parser.error("--quarter-kelly-multiplier must be between zero and one.")
    if not 0 < args.max_kelly_bet_fraction <= 1:
        parser.error("--max-kelly-bet-fraction must be in (0, 1].")
    parsed = pd.to_datetime(args.as_of_date or dt.date.today(), errors="coerce")
    if pd.isna(parsed):
        parser.error("--as-of-date must be a valid date.")
    args.as_of_date = pd.Timestamp(parsed).normalize().date().isoformat()
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone() is not None


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def one_value(frame: pd.DataFrame, column: str, label: str) -> Any:
    if column not in frame.columns:
        raise RuntimeError(f"{label} is missing {column}.")
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one {column}; found={values}."
        )
    return values[0]


def validate_form_contract(
    connection: sqlite3.Connection,
    week: int,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    """Validate only the live inputs used by the replacement model.

    The legacy predictor also required the form table to reproduce the current
    legacy power-rating hash.  That lineage is obsolete here: unit/slot inputs
    come from the independently frozen learned-model snapshot.  We still carry
    the old identifiers for compatibility, but do not make them a dependency.
    """
    form = read_table(connection, legacy.FORM_RATING_TABLE)
    if len(form) != 32 or form["team"].astype(str).nunique() != 32:
        raise RuntimeError("The live form table must contain 32 unique teams.")
    if str(one_value(form, "build_id", "form table")) != EXPECTED_FORM_BUILD_ID:
        raise RuntimeError("The form table has the wrong build ID.")
    if str(one_value(form, "form_version", "form table")) != EXPECTED_FORM_VERSION:
        raise RuntimeError("The form table has the wrong version.")
    through_week = int(one_value(form, "through_week", "form table"))
    if through_week != week - 1:
        raise RuntimeError(
            f"Form table is through Week {through_week}; Week {week} requires {week - 1}."
        )
    if "no_lookahead_filter_applied_flag" not in form.columns or pd.to_numeric(
        form["no_lookahead_filter_applied_flag"], errors="coerce"
    ).fillna(0).ne(1).any():
        raise RuntimeError("The form table is not marked as point-in-time safe.")
    form_as_of = pd.to_datetime(form["as_of_date"], errors="coerce").dt.normalize()
    if form_as_of.isna().any() or not form_as_of.eq(as_of_date.normalize()).all():
        raise RuntimeError(
            f"Form as-of date must equal {as_of_date.date().isoformat()}."
        )
    games = pd.to_numeric(form["games_played"], errors="coerce")
    if games.isna().any() or games.gt(week - 1).any():
        raise RuntimeError("Form games played exceed the point-in-time week boundary.")
    return {
        "form_build_id": EXPECTED_FORM_BUILD_ID,
        "form_version": EXPECTED_FORM_VERSION,
        "form_as_of_date": as_of_date.date().isoformat(),
        "form_through_week": through_week,
        "form_date_imported": str(one_value(form, "date_imported", "form table")),
        "preseason_snapshot_hash": str(
            one_value(form, "snapshot_hash", "form table")
        ),
        "structural_power_build_id": str(
            one_value(form, "current_structural_build_id", "form table")
        ),
        "structural_power_version": str(
            one_value(form, "current_structural_version", "form table")
        ),
        "structural_power_date_imported": str(
            one_value(form, "current_structural_date_imported", "form table")
        ),
        "structural_power_snapshot_hash": str(
            one_value(form, "current_structural_snapshot_hash", "form table")
        ),
    }


def validate_bundle(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen model bundle is missing: {path}. Run build_nfl_learned_consensus_2026.py first."
        )
    bundle = joblib.load(path)
    expected = {
        "build_id": EXPECTED_BUNDLE_BUILD_ID,
        "version": EXPECTED_BUNDLE_VERSION,
        "season": SEASON,
        "market_features_in_projection": 0,
        "structural_inputs_frozen_during_2026": 1,
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise RuntimeError(
                f"Frozen bundle mismatch for {key}: expected={value!r}, found={bundle.get(key)!r}."
            )
    gate = bundle.get("gate", {})
    gate_expected = {
        "ensemble": frozen.FROZEN_ENSEMBLE,
        "edge_threshold": frozen.FROZEN_EDGE_THRESHOLD,
        "maximum_projection_range": frozen.FROZEN_MAXIMUM_PROJECTION_RANGE,
        "minimum_agreement": frozen.FROZEN_MINIMUM_AGREEMENT,
    }
    for key, value in gate_expected.items():
        actual = gate.get(key)
        if isinstance(value, float):
            valid = actual is not None and np.isclose(float(actual), value)
        else:
            valid = actual == value
        if not valid:
            raise RuntimeError(f"Frozen gate mismatch for {key}.")
    model_hash = frozen.sha256_file(path)
    return bundle, model_hash


def load_historical_cache(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        pbp = pd.read_sql_query("SELECT * FROM pbp_source WHERE season = 2025", connection)
        snaps = pd.read_sql_query("SELECT * FROM snap_source WHERE season = 2025", connection)
        audit = pd.read_sql_query("SELECT * FROM cache_audit", connection)
    finally:
        connection.close()
    if len(audit) != 1:
        raise RuntimeError("The historical matchup source cache has no unique audit row.")
    accepted_cache_builds = {
        "NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V2",
        "NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V3",
    }
    if str(audit.iloc[0].get("build_id", "")) not in accepted_cache_builds:
        raise RuntimeError("The historical matchup source cache has an unrecognized build ID.")
    if len(pbp) < 40_000 or len(snaps) < 20_000:
        raise RuntimeError("The source cache must contain complete 2025 PBP and snap counts.")
    return pbp, snaps


def load_current_raw(
    week: int,
    pbp_path: Path | None,
    snaps_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if week == 1:
        return pd.DataFrame(), pd.DataFrame(), "not_required_week1", "not_required_week1"

    if pbp_path is not None:
        raw_pbp = pd.read_csv(pbp_path, low_memory=False)
        pbp_source = str(pbp_path)
    else:
        raw_pbp, pbp_source = matchup_history.load_package_pbp([SEASON])
    if snaps_path is not None:
        raw_snaps = pd.read_csv(snaps_path, low_memory=False)
        snap_source = str(snaps_path)
    else:
        raw_snaps, snap_source = matchup_history.load_package_snaps([SEASON])

    original_pbp_seasons = matchup_history.PBP_SOURCE_SEASONS
    original_snap_seasons = matchup_history.SNAP_SOURCE_SEASONS
    try:
        matchup_history.PBP_SOURCE_SEASONS = (*original_pbp_seasons, SEASON)
        matchup_history.SNAP_SOURCE_SEASONS = (*original_snap_seasons, SEASON)
        current_pbp = matchup_history.standardize_pbp(raw_pbp)
        current_snaps = matchup_history.standardize_snaps(raw_snaps)
    finally:
        matchup_history.PBP_SOURCE_SEASONS = original_pbp_seasons
        matchup_history.SNAP_SOURCE_SEASONS = original_snap_seasons

    current_pbp = current_pbp[
        current_pbp["season"].eq(SEASON) & current_pbp["week"].lt(week)
    ].copy()
    current_snaps = current_snaps[
        current_snaps["season"].eq(SEASON) & current_snaps["week"].lt(week)
    ].copy()
    if current_pbp.empty:
        raise RuntimeError(
            f"No 2026 PBP through Week {week - 1} is available; refusing a stale Week {week} projection."
        )
    if current_snaps.empty:
        raise RuntimeError(
            f"No 2026 snap counts through Week {week - 1} are available; refusing a stale Week {week} projection."
        )
    return current_pbp, current_snaps, pbp_source, snap_source


def build_process_ratings(
    team_games: pd.DataFrame,
    teams: list[str],
    week: int,
    rating_alpha: float,
) -> pd.DataFrame:
    settings = SimpleNamespace(
        current_decay=matchup_history.DEFAULT_CURRENT_DECAY,
        prior_season_weight=matchup_history.DEFAULT_PRIOR_SEASON_WEIGHT,
        prior_decay=matchup_history.DEFAULT_PRIOR_DECAY,
    )
    history = team_games[
        team_games["season"].eq(SEASON - 1)
        | (team_games["season"].eq(SEASON) & team_games["week"].lt(week))
    ].copy()
    weights = matchup_history.observation_weights(history, SEASON, week, settings)
    current_games = (
        history[history["season"].eq(SEASON)]
        .groupby("offense_team")["game_id"]
        .nunique()
        .to_dict()
    )
    models = {
        metric: matchup_history.fit_metric_model(
            history, column, teams, rating_alpha, weights
        )
        for metric, (column, _) in matchup_history.METRIC_SPECS.items()
    }
    special = matchup_history.weighted_special_teams_ratings(history, teams, weights)
    rows: list[dict[str, Any]] = []
    for team in teams:
        row: dict[str, Any] = {
            "team": team,
            "current_games": int(current_games.get(team, 0)),
            "special_teams_rating": float(special.get(team, 0.0)),
        }
        for metric, (league_mean, offense, defense) in models.items():
            row[f"{metric}_league_mean"] = float(league_mean)
            row[f"{metric}_offense_rating"] = float(offense.get(team, 0.0))
            row[f"{metric}_defense_allow_rating"] = float(defense.get(team, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_personnel(
    snaps: pd.DataFrame,
    team_games: pd.DataFrame,
    teams: list[str],
    week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if week == 1:
        rows = [{"team": team, "last_completed_game_week": np.nan} for team in teams]
        for row in rows:
            for name in (
                "qb_recent_epa",
                "qb_recent_cpoe",
                "qb_changed_from_week_one",
                "qb_changed_last_game",
                "qb_snap_share",
                "offense_continuity_last2",
                "offense_rolling_stability",
                "missing_core_offense_share",
                "defense_continuity_last2",
                "defense_rolling_stability",
                "missing_core_defense_share",
                "ol_continuity_last2",
                "ol_rolling_stability",
                "missing_core_ol_share",
            ):
                row[name] = np.nan
        return pd.DataFrame(rows), pd.DataFrame()

    settings = SimpleNamespace(
        current_decay=matchup_history.DEFAULT_CURRENT_DECAY,
        prior_season_weight=matchup_history.DEFAULT_PRIOR_SEASON_WEIGHT,
        prior_decay=matchup_history.DEFAULT_PRIOR_DECAY,
    )
    snapshots = matchup_history.build_snapshots(snaps)
    crosswalk, crosswalk_audit = matchup_history.build_qb_identity_crosswalk(team_games)
    rows: list[dict[str, Any]] = []
    for team in teams:
        team_snapshots = snapshots.get((SEASON, team), [])
        prior = [row for row in team_snapshots if int(row["week"]) < week]
        if not prior:
            raise RuntimeError(
                f"No point-in-time 2026 snap snapshot exists for {team} before Week {week}."
            )
        current = prior[-1]
        previous = prior[-2] if len(prior) >= 2 else None
        rolling_prior = prior[max(0, len(prior) - 4) : -1]
        week_one = next((row for row in team_snapshots if row["week"] == 1), prior[0])
        identity_key = matchup_history.qb_name_match_key(current["qb_name"])
        pbp_key = crosswalk.get(identity_key, "")
        qb_epa, qb_cpoe, qb_games = matchup_history.weighted_qb_performance(
            team_games, SEASON, week, pbp_key, settings
        )
        if not pbp_key or not np.isfinite(qb_epa):
            raise RuntimeError(
                f"The active QB for {team} could not be linked to prior-only PBP before Week {week}."
            )
        row: dict[str, Any] = {
            "team": team,
            "last_completed_game_week": int(current["week"]),
            "qb_recent_epa": float(qb_epa),
            "qb_recent_cpoe": float(qb_cpoe) if np.isfinite(qb_cpoe) else np.nan,
            "qb_recent_games": int(qb_games),
            "qb_snap_share": float(current["qb_snap_share"]),
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
            current_values = current[label]
            previous_values = previous[label] if previous is not None else {}
            rolling_values = matchup_history.average_share_dictionary(
                [snapshot[label] for snapshot in rolling_prior]
            )
            row[f"{label}_continuity_last2"] = (
                matchup_history.weighted_overlap(current_values, previous_values)
                if previous is not None
                else 1.0
            )
            row[f"{label}_rolling_stability"] = (
                matchup_history.weighted_overlap(current_values, rolling_values)
                if rolling_values
                else 1.0
            )
            row[f"missing_core_{label}_share"] = (
                matchup_history.missing_core_share(rolling_values, current_values)
                if rolling_values
                else 0.0
            )
        rows.append(row)
    personnel = pd.DataFrame(rows)
    if personnel["last_completed_game_week"].ge(week).any():
        raise RuntimeError("Personnel features include the prediction week or a future week.")
    return personnel, crosswalk_audit


def build_live_matchup_features(
    games: pd.DataFrame,
    bundle: dict[str, Any],
    week: int,
    source_cache_db: Path,
    current_pbp_path: Path | None,
    current_snaps_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    prior_pbp, prior_snaps = load_historical_cache(source_cache_db)
    current_pbp, current_snaps, pbp_source, snap_source = load_current_raw(
        week, current_pbp_path, current_snaps_path
    )
    pbp = pd.concat([prior_pbp, current_pbp], ignore_index=True, sort=False)
    snaps = pd.concat([prior_snaps, current_snaps], ignore_index=True, sort=False)
    team_games = matchup_history.build_team_game_features(pbp)
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    all_teams = sorted(
        set(team_games.loc[team_games["season"].eq(SEASON - 1), "offense_team"])
        | set(teams)
    )
    if len(all_teams) != 32:
        raise RuntimeError(f"Expected 32 teams in process history; found {len(all_teams)}.")
    rating_alpha = float(bundle["nonlinear_model"]["parameters"]["rating_alpha"])
    ratings = build_process_ratings(team_games, all_teams, week, rating_alpha)
    personnel, qb_audit = build_personnel(snaps, team_games, teams, week)
    rating_index = ratings.set_index("team")
    personnel_index = personnel.set_index("team")

    rows: list[dict[str, Any]] = []
    for game in games.itertuples(index=False):
        home = str(game.home_team)
        away = str(game.away_team)
        if home not in rating_index.index or away not in rating_index.index:
            raise RuntimeError(f"Missing process ratings for {away} at {home}.")
        home_rating = rating_index.loc[home]
        away_rating = rating_index.loc[away]
        row: dict[str, Any] = {
            "game_id": str(game.game_id),
            "home_team": home,
            "away_team": away,
            "home_current_games": int(home_rating["current_games"]),
            "away_current_games": int(away_rating["current_games"]),
            "home_last_completed_game_week": personnel_index.loc[home, "last_completed_game_week"],
            "away_last_completed_game_week": personnel_index.loc[away, "last_completed_game_week"],
        }
        for metric, (_, higher_is_good) in matchup_history.METRIC_SPECS.items():
            league_mean = float(home_rating[f"{metric}_league_mean"])
            home_expected = (
                league_mean
                + float(home_rating[f"{metric}_offense_rating"])
                + float(away_rating[f"{metric}_defense_allow_rating"])
            )
            away_expected = (
                league_mean
                + float(away_rating[f"{metric}_offense_rating"])
                + float(home_rating[f"{metric}_defense_allow_rating"])
            )
            if higher_is_good:
                row[f"{metric}_advantage"] = home_expected - away_expected
            elif metric == "sack_rate":
                row["sack_advantage"] = away_expected - home_expected
            elif metric == "turnover_rate":
                row["turnover_advantage"] = away_expected - home_expected
        row["special_teams_advantage"] = float(
            home_rating["special_teams_rating"] - away_rating["special_teams_rating"]
        )
        home_personnel = personnel_index.loc[home]
        away_personnel = personnel_index.loc[away]
        row["qb_recent_epa_advantage"] = home_personnel["qb_recent_epa"] - away_personnel["qb_recent_epa"]
        row["qb_recent_cpoe_advantage"] = home_personnel["qb_recent_cpoe"] - away_personnel["qb_recent_cpoe"]
        row["qb_week1_stability_advantage"] = away_personnel["qb_changed_from_week_one"] - home_personnel["qb_changed_from_week_one"]
        row["qb_last_game_stability_advantage"] = away_personnel["qb_changed_last_game"] - home_personnel["qb_changed_last_game"]
        row["qb_snap_share_advantage"] = home_personnel["qb_snap_share"] - away_personnel["qb_snap_share"]
        for label in ("offense", "defense", "ol"):
            row[f"{label}_continuity_advantage"] = home_personnel[f"{label}_continuity_last2"] - away_personnel[f"{label}_continuity_last2"]
            row[f"{label}_stability_advantage"] = home_personnel[f"{label}_rolling_stability"] - away_personnel[f"{label}_rolling_stability"]
            row[f"core_{label}_health_advantage"] = away_personnel[f"missing_core_{label}_share"] - home_personnel[f"missing_core_{label}_share"]
        rows.append(row)
    output = pd.DataFrame(rows)
    if week > 1:
        if output["home_current_games"].gt(week - 1).any() or output["away_current_games"].gt(week - 1).any():
            raise RuntimeError("Process ratings contain current/future-week games.")
        for column in ("home_last_completed_game_week", "away_last_completed_game_week"):
            invalid = output[column].notna() & output[column].ge(week)
            if invalid.any():
                raise RuntimeError(f"Point-in-time failure in {column}.")
    audit = {
        "pbp_source": pbp_source,
        "snap_source": snap_source,
        "prior_pbp_rows": int(len(prior_pbp)),
        "current_pbp_rows": int(len(current_pbp)),
        "prior_snap_rows": int(len(prior_snaps)),
        "current_snap_rows": int(len(current_snaps)),
        "team_game_rows": int(len(team_games)),
        "rating_alpha": rating_alpha,
        "data_through_week": week - 1,
    }
    return output, audit, qb_audit


def prepare_projection_frame(
    games: pd.DataFrame,
    form: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    unit_snapshot = bundle["unit_snapshot"].copy()
    slot_snapshot = bundle["slot_snapshot"].copy()
    unit_snapshot.columns = [str(column).lower() for column in unit_snapshot.columns]
    slot_snapshot.columns = [str(column) for column in slot_snapshot.columns]
    form = form.copy()
    form["team"] = form["team"].astype(str).str.upper().str.strip()
    unit_index = unit_snapshot.set_index("team")
    slot_index = slot_snapshot.set_index("team")
    form_index = form.set_index("team")
    slot_names = tuple(bundle["slot_names"])
    rows: list[dict[str, Any]] = []
    for game in games.to_dict("records"):
        home = str(game["home_team"])
        away = str(game["away_team"])
        for team in (home, away):
            if team not in unit_index.index or team not in slot_index.index or team not in form_index.index:
                raise RuntimeError(f"Missing frozen structural/form input for {team}.")
        row = dict(game)
        for side, team in (("home", home), ("away", away)):
            row[f"{side}_games_played"] = float(form_index.loc[team, "games_played"])
            for unit in learned.UNIT_NAMES:
                row[f"{side}_unit_{unit}"] = float(unit_index.loc[team, unit])
            for slot in slot_names:
                row[f"{side}_slot_{slot}"] = float(slot_index.loc[team, slot])
            for component in learned.FORM_COMPONENTS:
                row[f"{side}_{component}"] = float(form_index.loc[team, component])
        rows.append(row)
    return pd.DataFrame(rows), slot_names


def ensure_feature_contract(actual: pd.DataFrame, expected: Iterable[str], label: str) -> pd.DataFrame:
    expected = list(expected)
    missing = set(expected) - set(actual.columns)
    extra = set(actual.columns) - set(expected)
    if missing or extra:
        raise RuntimeError(
            f"{label} feature contract mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if any(any(token in name.lower() for token in nonlinear.MARKET_TOKENS) for name in expected):
        raise RuntimeError(f"A market-derived feature entered {label}.")
    return actual[expected]


def feature_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    payload = frame[list(columns)].to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_independent_projections(
    connection: sqlite3.Connection,
    games: pd.DataFrame,
    bundle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    form = read_table(connection, legacy.FORM_RATING_TABLE)
    through = pd.to_numeric(form["through_week"], errors="coerce").dropna().unique()
    if len(through) != 1 or int(through[0]) != args.week - 1:
        raise RuntimeError(
            f"Form table must be frozen through Week {args.week - 1}; found={through}."
        )
    structural, slot_names = prepare_projection_frame(games, form, bundle)
    variants = {variant.name: variant for variant in learned.VARIANTS}
    output = structural[["game_id", "home_team", "away_team"]].copy()
    for variant_name, output_column in (
        (frozen.UNIT_VARIANT, "unit_projection"),
        (frozen.SLOT_VARIANT, "slot_projection"),
    ):
        details = bundle["linear_models"][variant_name]
        features = learned.make_features(
            structural,
            variants[variant_name],
            slot_names,
            details["parameters"]["transition_k"],
            details["parameters"]["maximum_current_season_weight"],
        )
        features = ensure_feature_contract(features, details["feature_names"], variant_name)
        output[output_column] = details["model"].predict(features)

    matchup, matchup_audit, qb_audit = build_live_matchup_features(
        games,
        bundle,
        args.week,
        args.source_cache_db,
        args.current_pbp_path,
        args.current_snaps_path,
    )
    nonlinear_frame = structural[["game_id", "home_team", "away_team", "neutral_site"]].copy()
    for unit in learned.UNIT_NAMES:
        nonlinear_frame[f"unit_{unit}"] = (
            structural[f"home_unit_{unit}"] - structural[f"away_unit_{unit}"]
        )
    for component in learned.FORM_COMPONENTS:
        nonlinear_frame[f"form_{component}"] = (
            structural[f"home_{component}"] - structural[f"away_{component}"]
        )
    nonlinear_frame["home_field"] = 1.0 - pd.to_numeric(
        structural["neutral_site"], errors="coerce"
    ).fillna(0).clip(0, 1)
    nonlinear_frame = nonlinear_frame.merge(
        matchup, on=["game_id", "home_team", "away_team"], how="left", validate="one_to_one"
    )
    details = bundle["nonlinear_model"]
    nonlinear_features = ensure_feature_contract(
        nonlinear_frame[list(details["feature_names"])],
        details["feature_names"],
        "NONLINEAR",
    )
    output["nonlinear_projection"] = details["model"].predict(nonlinear_features)
    output["consensus_projection"] = output[
        ["unit_projection", "nonlinear_projection"]
    ].mean(axis=1)
    output["projection_range"] = (
        output[["unit_projection", "slot_projection", "nonlinear_projection"]].max(axis=1)
        - output[["unit_projection", "slot_projection", "nonlinear_projection"]].min(axis=1)
    )
    allowed = [*details["feature_names"]]
    output["nonlinear_feature_hash"] = feature_hash(nonlinear_features, allowed)
    return output, matchup_audit, qb_audit


def projection_hash(row: pd.Series, model_hash: str) -> str:
    payload = {
        "game_id": str(row["game_id"]),
        "model_sha256": model_hash,
        "structural_snapshot_hash": str(row["learned_structural_snapshot_hash"]),
        "unit_projection": round(float(row["unit_projection"]), 12),
        "slot_projection": round(float(row["slot_projection"]), 12),
        "nonlinear_projection": round(float(row["nonlinear_projection"]), 12),
        "consensus_projection": round(float(row["projected_home_margin"]), 12),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def apply_consensus(
    scaffold: pd.DataFrame,
    independent: pd.DataFrame,
    bundle: dict[str, Any],
    model_hash: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    frame = scaffold.merge(
        independent, on=["game_id", "home_team", "away_team"], how="inner", validate="one_to_one"
    )
    if len(frame) != len(scaffold):
        raise RuntimeError("Independent projection rows do not reconcile to the schedule.")
    frame["legacy_structural_form_hfa_projection_audit_only"] = frame["projected_home_margin"]
    frame["legacy_additive_components_used_in_final_projection"] = 0
    frame["projected_home_margin"] = frame["consensus_projection"]
    frame["projected_spread"] = [
        legacy.spread_label(home, away, margin)
        for home, away, margin in zip(
            frame["home_team"], frame["away_team"], frame["projected_home_margin"]
        )
    ]
    frame["build_id"] = BUILD_ID
    frame["version"] = VERSION
    frame["model_variant"] = MODEL_VARIANT
    frame["rating_source"] = "frozen_unit_slot_plus_prior_only_form_process_personnel"
    frame["learned_bundle_build_id"] = bundle["build_id"]
    frame["learned_bundle_version"] = bundle["version"]
    frame["learned_bundle_created_at"] = bundle["created_at"]
    frame["learned_bundle_sha256"] = model_hash
    frame["learned_structural_snapshot_hash"] = bundle["structural_snapshot_hash"]
    frame["deployment_status"] = bundle["deployment_status"]
    frame["ensemble_method"] = frozen.FROZEN_ENSEMBLE
    frame["minimum_model_agreement"] = frozen.FROZEN_MINIMUM_AGREEMENT
    frame["minimum_spread_difference_points"] = frozen.FROZEN_EDGE_THRESHOLD
    frame["maximum_market_disagreement_points"] = frozen.FROZEN_MAXIMUM_PROJECTION_RANGE
    frame["maximum_projection_range_points"] = frozen.FROZEN_MAXIMUM_PROJECTION_RANGE
    frame["model_market_edge_home_points"] = (
        frame["projected_home_margin"] - frame["available_market_home_margin"]
    )
    frame["absolute_spread_difference_points"] = frame[
        "model_market_edge_home_points"
    ].abs()
    for model_name in ("unit", "slot", "nonlinear"):
        frame[f"{model_name}_market_edge_home_points"] = (
            frame[f"{model_name}_projection"] - frame["available_market_home_margin"]
        )

    ensemble_sign = np.sign(frame["model_market_edge_home_points"])
    signs = pd.DataFrame(
        {
            name: np.sign(frame[f"{name}_market_edge_home_points"])
            for name in ("unit", "slot", "nonlinear")
        },
        index=frame.index,
    )
    frame["model_agreement_count"] = signs.eq(ensemble_sign, axis=0).sum(axis=1)
    frame["unanimous_model_side"] = frame["model_agreement_count"].eq(3).astype(int)
    frame["selected_side"] = np.select(
        [ensemble_sign.gt(0), ensemble_sign.lt(0)], ["HOME", "AWAY"], default="NONE"
    )
    frame["selected_team"] = np.select(
        [frame["selected_side"].eq("HOME"), frame["selected_side"].eq("AWAY")],
        [frame["home_team"], frame["away_team"]],
        default="",
    )
    frame["selected_market_line"] = [
        legacy.selected_market_line_label(team, home, away, margin)
        for team, home, away, margin in zip(
            frame["selected_team"],
            frame["home_team"],
            frame["away_team"],
            frame["available_market_home_margin"],
        )
    ]

    probability = bundle["probability_calibration"]
    residual_mean = float(probability["mean"])
    residual_sigma = float(probability["sigma"])
    home_probability: list[float] = []
    away_probability: list[float] = []
    selected_probability: list[float] = []
    selected_prices: list[int] = []
    price_fallbacks: list[int] = []
    break_evens: list[float] = []
    probability_edges: list[float] = []
    full_kelly: list[float] = []
    for row in frame.itertuples(index=False):
        edge = row.model_market_edge_home_points
        if pd.isna(edge):
            home_probability.append(np.nan)
            away_probability.append(np.nan)
            selected_probability.append(np.nan)
            selected_prices.append(int(args.default_spread_price))
            price_fallbacks.append(1)
            break_evens.append(np.nan)
            probability_edges.append(np.nan)
            full_kelly.append(0.0)
            continue
        home_p = float(np.clip(normal_cdf((float(edge) + residual_mean) / residual_sigma), 0.01, 0.99))
        away_p = 1.0 - home_p
        if row.selected_side == "HOME":
            selected_p = home_p
            raw_price = row.market_home_price
        elif row.selected_side == "AWAY":
            selected_p = away_p
            raw_price = row.market_away_price
        else:
            selected_p = np.nan
            raw_price = np.nan
        price, fallback_used = legacy.sanitize_american_price(
            raw_price, fallback=args.default_spread_price
        )
        break_even = legacy.implied_probability_from_american(price)
        probability_edge = selected_p - break_even if pd.notna(selected_p) else np.nan
        net_win = legacy.net_profit_per_unit(price)
        kelly = (
            (net_win * selected_p - (1.0 - selected_p)) / net_win
            if pd.notna(selected_p)
            else 0.0
        )
        home_probability.append(home_p)
        away_probability.append(away_p)
        selected_probability.append(selected_p)
        selected_prices.append(price)
        price_fallbacks.append(fallback_used)
        break_evens.append(break_even)
        probability_edges.append(probability_edge)
        full_kelly.append(float(np.clip(kelly, 0.0, 1.0)))

    frame["probability_method"] = probability["method"]
    frame["probability_source"] = "frozen_consensus_oof_predictions_2022_2025"
    frame["probability_is_model_specific_calibration"] = 1
    frame["historical_residual_rows"] = int(probability["rows"])
    frame["historical_residual_mean"] = residual_mean
    frame["historical_residual_sigma"] = residual_sigma
    frame["home_cover_probability"] = home_probability
    frame["away_cover_probability"] = away_probability
    frame["model_selected_cover_probability"] = selected_probability
    frame["selected_spread_price"] = selected_prices
    frame["price_fallback_used"] = price_fallbacks
    frame["market_break_even_probability"] = break_evens
    frame["probability_edge"] = probability_edges
    frame["full_kelly_fraction"] = full_kelly

    decisions: list[str] = []
    qualifies: list[int] = []
    for row in frame.itertuples(index=False):
        if int(row.completed):
            decision = "COMPLETED_GAME"
        elif not int(row.market_attached_after_prediction_freeze):
            decision = "MARKET_UNAVAILABLE"
        elif int(row.week) == 1 and not args.allow_week1_stakes:
            decision = "WEEK1_OUTSIDE_BACKTEST_SCOPE"
        elif int(row.week) == 18 and not args.include_week18:
            decision = "WEEK18_EXCLUDED"
        elif float(row.absolute_spread_difference_points) < frozen.FROZEN_EDGE_THRESHOLD:
            decision = "NO_BET_BELOW_THRESHOLD"
        elif float(row.projection_range) > frozen.FROZEN_MAXIMUM_PROJECTION_RANGE:
            decision = "NO_BET_MODEL_DISAGREEMENT"
        elif int(row.model_agreement_count) < frozen.FROZEN_MINIMUM_AGREEMENT:
            decision = "NO_BET_SIDE_DISAGREEMENT"
        else:
            decision = (
                "BET_WEEK1_OUTSIDE_VALIDATED_BACKTEST"
                if int(row.week) == 1
                else "BET"
            )
        decisions.append(decision)
        qualifies.append(
            int(decision in {"BET", "BET_WEEK1_OUTSIDE_VALIDATED_BACKTEST"})
        )
    frame["decision"] = decisions
    frame["qualifies_for_stake"] = qualifies
    frame["week1_stake_override_enabled"] = int(args.allow_week1_stakes)
    frame["stake_validation_scope"] = np.where(
        frame["week"].eq(1),
        "OUTSIDE_VALIDATED_BACKTEST_WEEK1",
        "VALIDATED_BACKTEST_WEEKS_2_17",
    )
    frame["recommendation"] = np.where(
        frame["qualifies_for_stake"].eq(1),
        "BET " + frame["selected_market_line"].astype(str),
        frame["decision"],
    )
    frame["starting_bankroll"] = float(args.bankroll)
    frame["flat_stake_setting"] = float(args.flat_stake)
    frame["flat_recommended_stake"] = np.where(
        frame["qualifies_for_stake"].eq(1), float(args.flat_stake), 0.0
    )
    frame["fractional_kelly_multiplier"] = float(args.quarter_kelly_multiplier)
    frame["uncapped_fractional_kelly_fraction"] = (
        frame["full_kelly_fraction"] * args.quarter_kelly_multiplier
    )
    frame["max_kelly_bet_fraction"] = float(args.max_kelly_bet_fraction)
    frame["actual_fractional_kelly_fraction"] = frame[
        "uncapped_fractional_kelly_fraction"
    ].clip(lower=0.0, upper=args.max_kelly_bet_fraction)
    frame["fractional_kelly_stake_raw"] = np.where(
        frame["qualifies_for_stake"].eq(1),
        frame["starting_bankroll"] * frame["actual_fractional_kelly_fraction"],
        0.0,
    )
    frame["fractional_kelly_recommended_stake"] = np.where(
        frame["qualifies_for_stake"].eq(1),
        frame["fractional_kelly_stake_raw"].round(0),
        0.0,
    )
    frame["independent_projection_hash"] = [
        projection_hash(row, model_hash) for _, row in frame.iterrows()
    ]
    frame["prediction_uses_market_inputs"] = 0
    frame["prediction_timestamp"] = now_string()
    frame["date_imported"] = now_string()
    return frame


def validate_output(
    frame: pd.DataFrame,
    games: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    if len(frame) != len(games) or frame["game_id"].nunique() != len(frame):
        raise RuntimeError("Prediction output does not reconcile to scheduled games.")
    if frame["prediction_uses_market_inputs"].ne(0).any():
        raise RuntimeError("A projection is marked as using a market input.")
    if set(frame["build_id"].astype(str)) != {BUILD_ID}:
        raise RuntimeError("Prediction build ID mismatch.")
    if set(frame["model_variant"].astype(str)) != {MODEL_VARIANT}:
        raise RuntimeError("Prediction model variant mismatch.")
    if frame[["unit_projection", "slot_projection", "nonlinear_projection", "projected_home_margin"]].isna().any().any():
        raise RuntimeError("A model projection is missing.")
    if not np.allclose(
        frame["projected_home_margin"],
        frame[["unit_projection", "nonlinear_projection"]].mean(axis=1),
        atol=1e-10,
    ):
        raise RuntimeError("Consensus projection does not reconcile to unit/nonlinear mean.")
    recalculated_range = (
        frame[["unit_projection", "slot_projection", "nonlinear_projection"]].max(axis=1)
        - frame[["unit_projection", "slot_projection", "nonlinear_projection"]].min(axis=1)
    )
    if not np.allclose(frame["projection_range"], recalculated_range, atol=1e-10):
        raise RuntimeError("Projection range does not reconcile.")
    invalid_bet = frame["qualifies_for_stake"].eq(1) & (
        frame["absolute_spread_difference_points"].lt(frozen.FROZEN_EDGE_THRESHOLD)
        | frame["projection_range"].gt(frozen.FROZEN_MAXIMUM_PROJECTION_RANGE)
        | frame["model_agreement_count"].lt(frozen.FROZEN_MINIMUM_AGREEMENT)
        | (frame["week"].eq(1) & (not args.allow_week1_stakes))
    )
    if invalid_bet.any():
        raise RuntimeError("A stake bypassed the frozen consensus gate.")
    week1_bets = frame["qualifies_for_stake"].eq(1) & frame["week"].eq(1)
    if week1_bets.any():
        if not args.allow_week1_stakes:
            raise RuntimeError("A Week 1 stake was created without the explicit override.")
        if frame.loc[week1_bets, "decision"].ne(
            "BET_WEEK1_OUTSIDE_VALIDATED_BACKTEST"
        ).any():
            raise RuntimeError("A Week 1 stake is missing its unvalidated-scope label.")


def print_report(frame: pd.DataFrame, matchup_audit: dict[str, Any]) -> None:
    print("=" * 140)
    print("[LEARNED_PREDICT] 2026 NFL LEARNED STRUCTURAL + NONLINEAR CONSENSUS")
    print("=" * 140)
    print(f"[LEARNED_PREDICT] Build ID: {BUILD_ID}")
    print(f"[LEARNED_PREDICT] Version: {VERSION}")
    print("[LEARNED_PREDICT] Market used in independent projections: NO")
    print(
        "[LEARNED_PREDICT] Gate: edge >= 2.0 | range <= 3.5 | agreement = 3/3"
    )
    print(
        "[LEARNED_PREDICT] Week 1 staking: "
        + (
            "ENABLED - OUTSIDE VALIDATED BACKTEST SCOPE"
            if int(frame["week"].iloc[0]) == 1
            and int(frame["week1_stake_override_enabled"].iloc[0]) == 1
            else "DEFAULT POLICY"
        )
    )
    print(
        f"[LEARNED_PREDICT] Week {int(frame['week'].iloc[0])}: games={len(frame)} "
        f"markets={int(frame['market_attached_after_prediction_freeze'].sum())} "
        f"bets={int(frame['qualifies_for_stake'].sum())}"
    )
    print(
        "[LEARNED_PREDICT] Point-in-time sources: "
        f"PBP={matchup_audit['pbp_source']} | snaps={matchup_audit['snap_source']} | "
        f"through_week={matchup_audit['data_through_week']}"
    )
    display = frame[
        [
            "away_team",
            "home_team",
            "unit_projection",
            "slot_projection",
            "nonlinear_projection",
            "projected_home_margin",
            "projection_range",
            "available_market_home_margin",
            "model_market_edge_home_points",
            "model_agreement_count",
            "selected_team",
            "decision",
        ]
    ].copy()
    print(display.to_string(index=False))
    print("=" * 140)


def build_execution_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the concise, execution-facing weekly CSV.

    The complete prediction and lineage record remains in SQLite.  This view
    intentionally keeps the three component projections and the two consensus
    checks so a no-bet decision can be understood without opening the audit
    tables.
    """
    required = {
        "week",
        "game_date",
        "away_team",
        "home_team",
        "neutral_site",
        "away_power_rating_points",
        "home_power_rating_points",
        "neutral_rating_difference",
        "unit_projection",
        "slot_projection",
        "nonlinear_projection",
        "projected_home_margin",
        "projected_spread",
        "available_market_spread",
        "projection_range",
        "model_agreement_count",
        "absolute_spread_difference_points",
        "selected_team",
        "qualifies_for_stake",
        "decision",
        "selected_market_line",
        "selected_spread_price",
        "fractional_kelly_recommended_stake",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Execution CSV is missing required fields: {missing}")

    parsed_game_date = pd.to_datetime(frame["game_date"], errors="coerce")
    game_date = parsed_game_date.dt.strftime("%Y-%m-%d").fillna("")
    game_time = parsed_game_date.dt.strftime("%I:%M %p").fillna("").str.lstrip("0")
    midnight = parsed_game_date.notna() & parsed_game_date.dt.strftime("%H:%M:%S").eq(
        "00:00:00"
    )
    game_time = game_time.mask(midnight, "")

    matchup = frame["away_team"].astype(str) + " @ " + frame["home_team"].astype(str)
    matchup = matchup + np.where(
        pd.to_numeric(frame["neutral_site"], errors="coerce").fillna(0).eq(1),
        " (Neutral)",
        "",
    )

    sportsbook = pd.Series("", index=frame.index, dtype="object")
    for source_column in (
        "sportsbook",
        "bookmaker_key",
        "available_market_line_source",
        "market_source",
    ):
        if source_column not in frame.columns:
            continue
        candidate = frame[source_column].fillna("").astype(str).str.strip()
        sportsbook = sportsbook.mask(sportsbook.eq("") & candidate.ne(""), candidate)

    decision_labels = {
        "BET": "WAGER",
        "BET_WEEK1_OUTSIDE_VALIDATED_BACKTEST": (
            "WAGER - WEEK 1 OUTSIDE VALIDATED BACKTEST"
        ),
        "COMPLETED_GAME": "NO WAGER - GAME COMPLETED",
        "MARKET_UNAVAILABLE": "NO WAGER - MARKET UNAVAILABLE",
        "WEEK1_OUTSIDE_BACKTEST_SCOPE": "NO WAGER - WEEK 1 OVERRIDE NOT ENABLED",
        "WEEK18_EXCLUDED": "NO WAGER - WEEK 18 EXCLUDED",
        "NO_BET_BELOW_THRESHOLD": "NO WAGER - EDGE BELOW 2.0",
        "NO_BET_MODEL_DISAGREEMENT": (
            "NO WAGER - PROJECTION RANGE ABOVE 3.5"
        ),
        "NO_BET_SIDE_DISAGREEMENT": (
            "NO WAGER - MODELS SELECT DIFFERENT SIDES"
        ),
    }
    readable_decision = (
        frame["decision"].fillna("").astype(str).map(decision_labels)
    )
    readable_decision = readable_decision.fillna(
        frame["decision"].fillna("").astype(str)
    )

    output = pd.DataFrame(
        {
            "week": pd.to_numeric(frame["week"], errors="raise").astype(int),
            "game_date": game_date,
            "game_time": game_time,
            "matchup": matchup,
            "away_team": frame["away_team"].astype(str),
            "home_team": frame["home_team"].astype(str),
            "away_legacy_power_rating": pd.to_numeric(
                frame["away_power_rating_points"], errors="coerce"
            ).round(3),
            "home_legacy_power_rating": pd.to_numeric(
                frame["home_power_rating_points"], errors="coerce"
            ).round(3),
            "legacy_neutral_rating_difference": pd.to_numeric(
                frame["neutral_rating_difference"], errors="coerce"
            ).round(3),
            "unit_model_home_margin": pd.to_numeric(
                frame["unit_projection"], errors="coerce"
            ).round(3),
            "slot_model_home_margin": pd.to_numeric(
                frame["slot_projection"], errors="coerce"
            ).round(3),
            "nonlinear_model_home_margin": pd.to_numeric(
                frame["nonlinear_projection"], errors="coerce"
            ).round(3),
            "final_model_home_margin": pd.to_numeric(
                frame["projected_home_margin"], errors="coerce"
            ).round(3),
            "final_model_spread": frame["projected_spread"].fillna("").astype(str),
            "market_line": frame["available_market_spread"].fillna("").astype(str),
            "projection_range_points": pd.to_numeric(
                frame["projection_range"], errors="coerce"
            ).round(3),
            "model_side_agreement": (
                pd.to_numeric(frame["model_agreement_count"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                + "/3"
            ),
            "absolute_edge_points": pd.to_numeric(
                frame["absolute_spread_difference_points"], errors="coerce"
            ).round(3),
            "selected_team": frame["selected_team"].fillna("").astype(str),
            "wager": np.where(frame["qualifies_for_stake"].eq(1), "YES", "NO"),
            "decision": readable_decision,
            "wager_line": frame["selected_market_line"].fillna("").astype(str),
            "wager_price": pd.to_numeric(
                frame["selected_spread_price"], errors="coerce"
            ).astype("Int64"),
            "recommended_stake": pd.to_numeric(
                frame["fractional_kelly_recommended_stake"], errors="coerce"
            ).fillna(0.0).round(0),
            "sportsbook": sportsbook,
        },
        index=frame.index,
    )
    if len(output) != len(frame):
        raise RuntimeError("Execution CSV row count does not reconcile.")
    return output


def write_execution_csv(
    predictions: pd.DataFrame,
    current_csv: Path | None,
    snapshot_csv: Path | None,
) -> None:
    if current_csv is None and snapshot_csv is None:
        return
    output = build_execution_csv(predictions)
    for path in (current_csv, snapshot_csv):
        if path is not None:
            output.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    args = parse_args()
    bundle, model_hash = validate_bundle(args.model_path)
    connection = legacy.connect_database(args.db_path)
    run_id = str(uuid.uuid4())
    try:
        ratings, rating_source = legacy.load_live_ratings(connection)
        hfa, hfa_source = legacy.load_home_field(connection, None)
        residual_info = legacy.load_probability_noise(connection)
        schedule = legacy.load_primary_schedule(
            connection, args.project_root, args.schedule_path
        )
        factors = legacy.load_schedule_factors(
            connection, args.project_root, args.schedule_factors_path
        )
        schedule = legacy.attach_factors(schedule, factors)
        week = legacy.select_prediction_week(schedule, args.week, args.as_of_date)
        as_of_date = legacy.prediction_date(args.as_of_date)
        rating_lineage = validate_form_contract(connection, week, as_of_date)
        model_games = schedule[schedule["week"].eq(week)].copy()
        if model_games.empty:
            raise RuntimeError(f"No games found for 2026 Week {week}.")
        independent, matchup_audit, qb_audit = generate_independent_projections(
            connection, model_games, bundle, args
        )

        market_schedule = legacy.merge_market_source(schedule, args.market_path)
        market_schedule = legacy.choose_market_line(
            market_schedule, args.market_line_preference
        )
        games = market_schedule[market_schedule["week"].eq(week)].copy()
        manual = legacy.load_manual_adjustments(connection, week)
        legacy_args = SimpleNamespace(
            default_spread_price=args.default_spread_price,
            minimum_spread_difference=frozen.FROZEN_EDGE_THRESHOLD,
            maximum_market_disagreement=frozen.FROZEN_MAXIMUM_PROJECTION_RANGE,
            quarter_kelly_multiplier=args.quarter_kelly_multiplier,
            max_kelly_bet_fraction=args.max_kelly_bet_fraction,
            bankroll=args.bankroll,
            flat_stake=args.flat_stake,
            include_week18=args.include_week18,
        )
        scaffold = legacy.build_predictions(
            games=games,
            ratings=ratings,
            week=week,
            rating_source=rating_source,
            hfa=hfa,
            hfa_source=hfa_source,
            residual_info=residual_info,
            manual=manual,
            args=legacy_args,
            run_id=run_id,
            rating_lineage=rating_lineage,
        )
        predictions = apply_consensus(
            scaffold, independent, bundle, model_hash, args
        )
        validate_output(predictions, games, args)

        completed = dt.datetime.now(dt.timezone.utc)
        sportsbook_counts = (
            predictions["sportsbook"].dropna().astype(str).loc[lambda x: x.str.strip().ne("")].value_counts().to_dict()
        )
        market_ages = pd.to_numeric(predictions["market_line_age_minutes"], errors="coerce").dropna()
        audit = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "season": SEASON,
                    "week": week,
                    "build_id": BUILD_ID,
                    "version": VERSION,
                    "model_variant": MODEL_VARIANT,
                    "status": "SUCCESS",
                    "deployment_status": bundle["deployment_status"],
                    "games": int(len(predictions)),
                    "markets_attached": int(predictions["market_attached_after_prediction_freeze"].sum()),
                    "qualifying_bets": int(predictions["qualifies_for_stake"].sum()),
                    "sportsbooks_json": json.dumps(sportsbook_counts, sort_keys=True),
                    "maximum_market_line_age_minutes": float(market_ages.max()) if not market_ages.empty else None,
                    "circa_lines_attached": int(pd.to_numeric(predictions["circa_line_flag"], errors="coerce").fillna(0).sum()),
                    "rating_source": "frozen_unit_slot_plus_prior_only_form_process_personnel",
                    **rating_lineage,
                    "schedule_source": " | ".join(sorted(predictions["schedule_source"].astype(str).unique())),
                    "factor_source": " | ".join(sorted(predictions["factor_source"].astype(str).unique())),
                    "hfa_points": None,
                    "hfa_source": "embedded_and_learned_in_each_projection",
                    "probability_method": bundle["probability_calibration"]["method"],
                    "probability_source": "frozen_consensus_oof_predictions_2022_2025",
                    "probability_is_model_specific_calibration": 1,
                    "historical_residual_rows": bundle["probability_calibration"]["rows"],
                    "historical_residual_sigma": bundle["probability_calibration"]["sigma"],
                    "prediction_uses_market_inputs": 0,
                    "market_attached_after_prediction_freeze": int(predictions["market_attached_after_prediction_freeze"].sum() > 0),
                    "learned_bundle_build_id": bundle["build_id"],
                    "learned_bundle_version": bundle["version"],
                    "learned_bundle_sha256": model_hash,
                    "learned_structural_snapshot_hash": bundle["structural_snapshot_hash"],
                    "ensemble_method": frozen.FROZEN_ENSEMBLE,
                    "minimum_edge_points": frozen.FROZEN_EDGE_THRESHOLD,
                    "maximum_projection_range_points": frozen.FROZEN_MAXIMUM_PROJECTION_RANGE,
                    "minimum_agreement": frozen.FROZEN_MINIMUM_AGREEMENT,
                    "week1_stake_override_enabled": int(args.allow_week1_stakes),
                    "stake_validation_scope": (
                        "OUTSIDE_VALIDATED_BACKTEST_WEEK1"
                        if week == 1
                        else "VALIDATED_BACKTEST_WEEKS_2_17"
                    ),
                    "data_through_week": matchup_audit["data_through_week"],
                    "pbp_source": matchup_audit["pbp_source"],
                    "snap_source": matchup_audit["snap_source"],
                    "qb_crosswalk_rows": int(len(qb_audit)),
                    "started_at": started.isoformat(timespec="seconds"),
                    "completed_at": completed.isoformat(timespec="seconds"),
                    "elapsed_seconds": (completed - started).total_seconds(),
                    "date_imported": now_string(),
                }
            ]
        )
        current_csv, snapshot_csv = legacy.save_outputs(
            connection, predictions, audit, args.project_root, args.no_csv
        )
        write_execution_csv(predictions, current_csv, snapshot_csv)
        print_report(predictions, matchup_audit)
        print(f"[LEARNED_PREDICT] Saved {len(predictions)} rows to {OUTPUT_TABLE}")
        if current_csv is not None:
            print(f"[LEARNED_PREDICT] CSV: {current_csv}")
        if snapshot_csv is not None:
            print(f"[LEARNED_PREDICT] Snapshot: {snapshot_csv}")
        print(
            f"[LEARNED_PREDICT] Completed in {(dt.datetime.now(dt.timezone.utc) - started).total_seconds():.2f} seconds"
        )
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[LEARNED_PREDICT] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[LEARNED_PREDICT] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
