#!/usr/bin/env python
"""Market-free nonlinear NFL matchup and consensus research backtest.

The nonlinear projection is trained on every reconstructed game, including
games without a valid Circa contest line. Circa is attached only after the
out-of-fold projection is frozen. The wagering-gate audit is separate: it may
use prior out-of-fold ATS results, but never the season it is grading.

This script requires the output of backtest_nfl_learned_structural_weights.py.
It does not modify any source or production database.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import backtest_nfl_learned_structural_weights as learned


BUILD_ID = "NFL_NONLINEAR_MATCHUP_CONSENSUS_BACKTEST_CANONICAL_V1"
VERSION = "v1_0_full_game_market_free_nested_consensus_audit"

MATCHUP_DATABASE = "nfl_weekly_matchup_residual.sqlite"
LEARNED_DATABASE = "nfl_learned_structural_weights_backtest.sqlite"
OUTPUT_DATABASE = "nfl_nonlinear_matchup_consensus_backtest.sqlite"
MATCHUP_TABLE = "nfl_matchup_game_matrix"
LEARNED_PREDICTION_TABLE = "nfl_learned_structural_oof_predictions"

UNIT_VARIANT = "LEARNED_UNIT_STRUCTURAL"
SLOT_VARIANT = "LEARNED_SLOT_STRUCTURAL"

DEFAULT_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
DEFAULT_MAXIMUM_RANGES = (1.5, 2.5, 3.5, 5.0, 7.5, 999.0)
RATING_ALPHA_GRID = (10.0, 25.0, 50.0)
NONLINEAR_L2_GRID = (10.0, 30.0, 100.0)
NONLINEAR_LEAVES_GRID = (7, 15)
TARGET_CAP_GRID = (21.0, 28.0, 35.0)
AGREEMENT_GRID = (2, 3)
ENSEMBLES = ("mean_all", "median_all", "mean_unit_nonlinear")

PROCESS_FEATURES = (
    "pass_epa_advantage",
    "rush_epa_advantage",
    "early_down_epa_advantage",
    "success_rate_advantage",
    "explosive_rate_advantage",
    "sack_advantage",
    "turnover_advantage",
    "cpoe_advantage",
    "special_teams_advantage",
)
PERSONNEL_FEATURES = (
    "qb_recent_epa_advantage",
    "qb_recent_cpoe_advantage",
    "qb_week1_stability_advantage",
    "qb_last_game_stability_advantage",
    "qb_snap_share_advantage",
    "offense_continuity_advantage",
    "offense_stability_advantage",
    "core_offense_health_advantage",
    "defense_continuity_advantage",
    "defense_stability_advantage",
    "core_defense_health_advantage",
    "ol_continuity_advantage",
    "ol_stability_advantage",
    "core_ol_health_advantage",
)
POINT_IN_TIME_AUDIT_COLUMNS = (
    "home_current_games",
    "away_current_games",
    "home_last_completed_game_week",
    "away_last_completed_game_week",
)
MARKET_TOKENS = (
    "market",
    "circa",
    "spread",
    "line",
    "price",
    "odds",
    "cover",
    "residual",
)


@dataclass(frozen=True)
class NonlinearParameters:
    rating_alpha: float
    l2_regularization: float
    max_leaf_nodes: int
    target_margin_cap: float


@dataclass(frozen=True)
class Gate:
    ensemble: str
    edge_threshold: float
    maximum_projection_range: float
    minimum_agreement: int


def parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def parse_float_csv(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one number.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--replay-db", type=Path)
    parser.add_argument("--matchup-db", type=Path)
    parser.add_argument("--learned-weights-db", type=Path)
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--target-seasons", type=parse_int_csv, default=DEFAULT_SEASONS)
    parser.add_argument("--test-start-season", type=int, default=2022)
    parser.add_argument("--gate-test-start-season", type=int, default=2023)
    parser.add_argument("--minimum-week", type=int, default=2)
    parser.add_argument("--maximum-week", type=int, default=17)
    parser.add_argument("--thresholds", type=parse_float_csv, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--no-csv", action="store_true")
    return parser


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.database_root = args.database_root.expanduser().resolve()
    args.replay_db = (
        args.replay_db or args.database_root / learned.REPLAY_DATABASE
    ).expanduser().resolve()
    args.circa_db = (args.database_root / learned.CIRCA_DATABASE).resolve()
    args.matchup_db = (
        args.matchup_db or args.database_root / MATCHUP_DATABASE
    ).expanduser().resolve()
    args.learned_weights_db = (
        args.learned_weights_db or args.database_root / LEARNED_DATABASE
    ).expanduser().resolve()
    args.output_db = (
        args.output_db or args.database_root / OUTPUT_DATABASE
    ).expanduser().resolve()
    source_paths = {
        args.replay_db,
        args.matchup_db,
        args.learned_weights_db,
        *(args.database_root / f"{season}.sqlite" for season in args.target_seasons),
    }
    for source in source_paths:
        if not source.exists():
            raise FileNotFoundError(source)
    if args.output_db in source_paths:
        raise ValueError("The output database must be separate from every source database.")
    if tuple(sorted(args.target_seasons)) != tuple(args.target_seasons):
        raise ValueError("--target-seasons must be unique and in ascending order.")
    if len(set(args.target_seasons)) != len(args.target_seasons):
        raise ValueError("--target-seasons contains duplicates.")
    if args.test_start_season < min(args.target_seasons) + 2:
        raise ValueError("Two earlier seasons are required before the first OOF test season.")
    if args.gate_test_start_season <= args.test_start_season:
        raise ValueError("The gate test must begin after the first base-model OOF season.")
    if args.gate_test_start_season not in args.target_seasons:
        raise ValueError("--gate-test-start-season must be a target season.")
    if args.minimum_week < 2 or args.maximum_week < args.minimum_week:
        raise ValueError("The valid regular-season replay window begins at Week 2.")
    if any(value <= 0 for value in args.thresholds):
        raise ValueError("All edge thresholds must be positive.")
    return args


def read_sql(database: Path, query: str, parameters: Sequence[object] = ()) -> pd.DataFrame:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(query, connection, params=parameters)
    finally:
        connection.close()


def table_columns(database: Path, table: str) -> set[str]:
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            raise RuntimeError(f"Missing required table {table!r} in {database}")
        return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    finally:
        connection.close()


def require_columns(database: Path, table: str, required: Iterable[str]) -> None:
    missing = set(required) - table_columns(database, table)
    if missing:
        raise RuntimeError(f"{database.name}:{table} is missing columns: {sorted(missing)}")


def load_structural_inputs(args: argparse.Namespace) -> pd.DataFrame:
    shared = SimpleNamespace(
        database_root=args.database_root,
        replay_db=args.replay_db,
        circa_db=args.circa_db,
        target_seasons=args.target_seasons,
        minimum_week=args.minimum_week,
        maximum_week=args.maximum_week,
    )
    games = learned.load_games(shared)
    games = learned.attach_form(shared, games)
    games = learned.attach_units(shared, games)
    required = [
        "actual_home_margin",
        *(f"{side}_unit_{unit}" for side in ("home", "away") for unit in learned.UNIT_NAMES),
        *(f"{side}_{component}" for side in ("home", "away") for component in learned.FORM_COMPONENTS),
    ]
    missing = games[required].isna().sum()
    if (missing > 0).any():
        raise RuntimeError(f"Missing structural inputs:\n{missing[missing > 0].to_string()}")
    for unit in learned.UNIT_NAMES:
        games[f"unit_{unit}"] = games[f"home_unit_{unit}"] - games[f"away_unit_{unit}"]
    for component in learned.FORM_COMPONENTS:
        games[f"form_{component}"] = games[f"home_{component}"] - games[f"away_{component}"]
    games["home_field"] = 1.0 - games["neutral_site"].fillna(0).clip(0, 1)
    return games


def nonlinear_feature_names() -> tuple[str, ...]:
    names = (
        *(f"unit_{unit}" for unit in learned.UNIT_NAMES),
        *(f"form_{component}" for component in learned.FORM_COMPONENTS),
        *PROCESS_FEATURES,
        *PERSONNEL_FEATURES,
        "home_field",
    )
    forbidden = [
        name for name in names if any(token in name.lower() for token in MARKET_TOKENS)
    ]
    if forbidden:
        raise RuntimeError(f"Market-derived features entered the projection: {forbidden}")
    return names


def load_matchup_matrix(
    args: argparse.Namespace,
    structural: pd.DataFrame,
    rating_alpha: float,
) -> pd.DataFrame:
    required = (
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "rating_alpha",
        *POINT_IN_TIME_AUDIT_COLUMNS,
        *PROCESS_FEATURES,
        *PERSONNEL_FEATURES,
    )
    require_columns(args.matchup_db, MATCHUP_TABLE, required)
    placeholders = ",".join("?" for _ in args.target_seasons)
    matrix = read_sql(
        args.matchup_db,
        f"SELECT {', '.join(required)} FROM {MATCHUP_TABLE} "
        f"WHERE season IN ({placeholders}) AND week BETWEEN ? AND ? AND rating_alpha = ?",
        (*args.target_seasons, args.minimum_week, args.maximum_week, rating_alpha),
    )
    keys = ["season", "week", "game_id", "home_team", "away_team"]
    if matrix.duplicated(keys).any():
        raise RuntimeError(f"Duplicate matchup rows at rating_alpha={rating_alpha}")
    if matrix["home_current_games"].gt(matrix["week"] - 1).any():
        raise RuntimeError("Home matchup ratings contain current/future-week games.")
    if matrix["away_current_games"].gt(matrix["week"] - 1).any():
        raise RuntimeError("Away matchup ratings contain current/future-week games.")
    for column in ("home_last_completed_game_week", "away_last_completed_game_week"):
        invalid = matrix[column].notna() & matrix[column].ge(matrix["week"])
        if invalid.any():
            raise RuntimeError(f"Point-in-time failure in {column}.")
    structural_columns = [
        *keys,
        "game_date",
        "actual_home_margin",
        *(f"unit_{unit}" for unit in learned.UNIT_NAMES),
        *(f"form_{component}" for component in learned.FORM_COMPONENTS),
        "home_field",
    ]
    output = structural[structural_columns].merge(matrix, on=keys, how="inner", validate="one_to_one")
    if len(output) != len(structural):
        missing = structural[keys].merge(matrix[keys], on=keys, how="left", indicator=True)
        missing = missing[missing["_merge"].eq("left_only")]
        raise RuntimeError(
            f"Matchup matrix is missing {len(missing)} replay games at rating_alpha={rating_alpha}."
        )
    return output.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def build_nonlinear_model(parameters: NonlinearParameters) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=200,
                    max_leaf_nodes=parameters.max_leaf_nodes,
                    min_samples_leaf=30,
                    l2_regularization=parameters.l2_regularization,
                    random_state=20260906,
                ),
            ),
        ]
    )


def parameter_grid() -> Iterable[NonlinearParameters]:
    for rating_alpha, l2, leaves, target_cap in itertools.product(
        RATING_ALPHA_GRID,
        NONLINEAR_L2_GRID,
        NONLINEAR_LEAVES_GRID,
        TARGET_CAP_GRID,
    ):
        yield NonlinearParameters(rating_alpha, l2, leaves, target_cap)


def choose_nonlinear_parameters(
    matrices: dict[float, pd.DataFrame],
    feature_names: Sequence[str],
    test_season: int,
) -> tuple[NonlinearParameters, float]:
    scored: list[tuple[float, NonlinearParameters]] = []
    first_season = min(next(iter(matrices.values()))["season"])
    for parameters in parameter_grid():
        frame = matrices[parameters.rating_alpha]
        errors: list[float] = []
        for validation_season in range(first_season + 1, test_season):
            train = frame["season"] < validation_season
            validate = frame["season"] == validation_season
            model = build_nonlinear_model(parameters)
            model.fit(
                frame.loc[train, feature_names],
                frame.loc[train, "actual_home_margin"].clip(
                    -parameters.target_margin_cap, parameters.target_margin_cap
                ),
            )
            prediction = model.predict(frame.loc[validate, feature_names])
            errors.extend(
                np.abs(prediction - frame.loc[validate, "actual_home_margin"]).tolist()
            )
        if not errors:
            raise RuntimeError(f"No inner-validation rows were available before {test_season}.")
        scored.append((float(np.mean(errors)), parameters))
    return min(
        scored,
        key=lambda item: (
            item[0],
            -item[1].l2_regularization,
            item[1].max_leaf_nodes,
            item[1].target_margin_cap,
            -item[1].rating_alpha,
        ),
    )[1], min(item[0] for item in scored)


def fit_nonlinear_oof(
    matrices: dict[float, pd.DataFrame],
    feature_names: Sequence[str],
    test_seasons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for test_season in test_seasons:
        parameters, inner_mae = choose_nonlinear_parameters(
            matrices, feature_names, test_season
        )
        frame = matrices[parameters.rating_alpha]
        train = frame["season"] < test_season
        test = frame["season"] == test_season
        model = build_nonlinear_model(parameters)
        model.fit(
            frame.loc[train, feature_names],
            frame.loc[train, "actual_home_margin"].clip(
                -parameters.target_margin_cap, parameters.target_margin_cap
            ),
        )
        projected = model.predict(frame.loc[test, feature_names])
        result = frame.loc[
            test,
            [
                "season",
                "week",
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "actual_home_margin",
            ],
        ].copy()
        result["nonlinear_projection"] = projected
        predictions.append(result)
        test_mae = float(
            np.mean(np.abs(projected - result["actual_home_margin"]))
        )
        folds.append(
            {
                "test_season": test_season,
                "training_seasons": ",".join(
                    str(season) for season in sorted(frame.loc[train, "season"].unique())
                ),
                "training_games": int(train.sum()),
                "test_games": int(test.sum()),
                "inner_validation_mae": inner_mae,
                "rating_alpha": parameters.rating_alpha,
                "l2_regularization": parameters.l2_regularization,
                "max_leaf_nodes": parameters.max_leaf_nodes,
                "target_margin_cap": parameters.target_margin_cap,
                "test_margin_mae": test_mae,
                "market_features_in_projection": 0,
                "future_seasons_in_training": 0,
            }
        )
        print(
            f"[CONSENSUS] Nonlinear {test_season}: train={int(train.sum())} "
            f"test={int(test.sum())} rating_alpha={parameters.rating_alpha:.0f} "
            f"l2={parameters.l2_regularization:.0f} leaves={parameters.max_leaf_nodes} "
            f"target_cap={parameters.target_margin_cap:.0f} test_MAE={test_mae:.3f}",
            flush=True,
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def load_learned_oof(
    args: argparse.Namespace, test_seasons: Sequence[int]
) -> pd.DataFrame:
    required = (
        "season",
        "week",
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "actual_home_margin",
        "model_variant",
        "projected_home_margin",
        "circa_home_margin",
        "strict_circa_line_valid",
        "line_source",
        "build_id",
        "version",
    )
    require_columns(args.learned_weights_db, LEARNED_PREDICTION_TABLE, required)
    placeholders = ",".join("?" for _ in test_seasons)
    frame = read_sql(
        args.learned_weights_db,
        f"SELECT {', '.join(required)} FROM {LEARNED_PREDICTION_TABLE} "
        f"WHERE season IN ({placeholders}) AND model_variant IN (?, ?)",
        (*test_seasons, UNIT_VARIANT, SLOT_VARIANT),
    )
    if set(frame["build_id"]) != {learned.BUILD_ID}:
        raise RuntimeError("Learned-weight database build ID does not match the required script.")
    if set(frame["version"]) != {learned.VERSION}:
        raise RuntimeError("Learned-weight database version does not match the required script.")
    keys = ["season", "week", "game_id"]
    if frame.duplicated([*keys, "model_variant"]).any():
        raise RuntimeError("Duplicate learned-weight OOF predictions.")
    counts = frame.groupby(keys)["model_variant"].nunique()
    if not counts.eq(2).all():
        raise RuntimeError("Every game must have both unit and slot OOF projections.")
    identity = frame[
        [
            *keys,
            "game_date",
            "home_team",
            "away_team",
            "actual_home_margin",
            "circa_home_margin",
            "strict_circa_line_valid",
            "line_source",
        ]
    ].drop_duplicates(keys)
    pivot = frame.pivot(
        index=keys, columns="model_variant", values="projected_home_margin"
    ).reset_index()
    pivot = pivot.rename(
        columns={UNIT_VARIANT: "unit_projection", SLOT_VARIANT: "slot_projection"}
    )
    return identity.merge(pivot, on=keys, how="inner", validate="one_to_one")


def build_consensus_matrix(
    nonlinear: pd.DataFrame, learned_oof: pd.DataFrame
) -> pd.DataFrame:
    keys = ["season", "week", "game_id"]
    nonlinear_keep = nonlinear[
        [*keys, "actual_home_margin", "nonlinear_projection"]
    ].rename(columns={"actual_home_margin": "nonlinear_actual_home_margin"})
    frame = learned_oof.merge(nonlinear_keep, on=keys, how="inner", validate="one_to_one")
    if len(frame) != len(learned_oof) or len(frame) != len(nonlinear):
        raise RuntimeError("Nonlinear and learned-weight OOF game universes do not match.")
    if not np.allclose(
        frame["actual_home_margin"], frame["nonlinear_actual_home_margin"], equal_nan=True
    ):
        raise RuntimeError("Actual margins disagree between source databases.")
    frame = frame.drop(columns="nonlinear_actual_home_margin")
    projection_columns = (
        "unit_projection",
        "slot_projection",
        "nonlinear_projection",
    )
    for column in projection_columns:
        frame[column.replace("projection", "edge")] = (
            frame[column] - frame["circa_home_margin"]
        )
    edge_columns = tuple(
        column.replace("projection", "edge") for column in projection_columns
    )
    frame["mean_all_projection"] = frame[list(projection_columns)].mean(axis=1)
    frame["median_all_projection"] = frame[list(projection_columns)].median(axis=1)
    frame["mean_unit_nonlinear_projection"] = frame[
        ["unit_projection", "nonlinear_projection"]
    ].mean(axis=1)
    frame["projection_range"] = (
        frame[list(projection_columns)].max(axis=1)
        - frame[list(projection_columns)].min(axis=1)
    )
    frame["projection_standard_deviation"] = frame[list(projection_columns)].std(
        axis=1, ddof=0
    )
    for ensemble in ENSEMBLES:
        edge = f"{ensemble}_edge"
        frame[edge] = frame[f"{ensemble}_projection"] - frame["circa_home_margin"]
        selected_sign = np.sign(frame[edge]).replace(0, 1)
        frame[f"{ensemble}_agreement"] = np.sign(frame[list(edge_columns)]).eq(
            selected_sign, axis=0
        ).sum(axis=1)
    return frame.sort_values(["game_date", "season", "week", "game_id"]).reset_index(drop=True)


def wilson_interval(wins: int, losses: int, z: float = 1.959963984540054) -> tuple[float, float]:
    decisions = wins + losses
    if decisions == 0:
        return math.nan, math.nan
    probability = wins / decisions
    denominator = 1.0 + z * z / decisions
    center = (probability + z * z / (2.0 * decisions)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / decisions
            + z * z / (4.0 * decisions * decisions)
        )
        / denominator
    )
    return center - half, center + half


def gate_mask(frame: pd.DataFrame, gate: Gate) -> pd.Series:
    edge = frame[f"{gate.ensemble}_edge"]
    agreement = frame[f"{gate.ensemble}_agreement"]
    return (
        frame["strict_circa_line_valid"].eq(1)
        & frame["circa_home_margin"].notna()
        & frame["actual_home_margin"].notna()
        & edge.abs().ge(gate.edge_threshold)
        & frame["projection_range"].le(gate.maximum_projection_range)
        & agreement.ge(gate.minimum_agreement)
    )


def grade_gate(frame: pd.DataFrame, gate: Gate) -> dict[str, object]:
    bets = frame[gate_mask(frame, gate)].copy()
    edge = bets[f"{gate.ensemble}_edge"]
    bets["selected_cover_margin"] = np.sign(edge) * (
        bets["actual_home_margin"] - bets["circa_home_margin"]
    )
    bets["profit"] = np.select(
        [bets["selected_cover_margin"] > 0, bets["selected_cover_margin"] < 0],
        [1.0 / 1.1, -1.0],
        default=0.0,
    )
    wins = int((bets["selected_cover_margin"] > 0).sum())
    losses = int((bets["selected_cover_margin"] < 0).sum())
    pushes = int((bets["selected_cover_margin"] == 0).sum())
    decisions = wins + losses
    lower, upper = wilson_interval(wins, losses)
    equity = bets.sort_values(["game_date", "season", "week", "game_id"])[
        "profit"
    ].cumsum()
    drawdown = equity.cummax().sub(equity)
    profit = float(bets["profit"].sum())
    return {
        "ensemble": gate.ensemble,
        "edge_threshold": gate.edge_threshold,
        "maximum_projection_range": gate.maximum_projection_range,
        "minimum_agreement": gate.minimum_agreement,
        "bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_win_rate": wins / decisions if decisions else math.nan,
        "ats_profit_units": profit,
        "ats_roi": profit / len(bets) if len(bets) else math.nan,
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "maximum_drawdown_units": float(drawdown.max()) if len(drawdown) else 0.0,
    }


def enumerate_gates(thresholds: Sequence[float]) -> Iterable[Gate]:
    for ensemble, threshold, maximum_range, agreement in itertools.product(
        ENSEMBLES,
        thresholds,
        DEFAULT_MAXIMUM_RANGES,
        AGREEMENT_GRID,
    ):
        yield Gate(ensemble, threshold, maximum_range, agreement)


def summarize_posthoc_grid(
    frame: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("OOF_ALL", frame)]
    scopes.extend(
        (f"SEASON_{int(season)}", season_frame)
        for season, season_frame in frame.groupby("season", sort=True)
    )
    for scope, scope_frame in scopes:
        for gate in enumerate_gates(thresholds):
            rows.append(
                {
                    "scope": scope,
                    "research_status": "POSTHOC_GRID_NOT_DEPLOYABLE",
                    **grade_gate(scope_frame, gate),
                }
            )
    return pd.DataFrame(rows)


def bayesian_gate_score(summary: dict[str, object]) -> float:
    wins = int(summary["wins"])
    losses = int(summary["losses"])
    alpha = 25.0 + wins
    beta = 25.0 + losses
    mean = alpha / (alpha + beta)
    standard_deviation = math.sqrt(
        alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    )
    return mean - 0.5 * standard_deviation


def select_walk_forward_gates(
    frame: pd.DataFrame,
    thresholds: Sequence[float],
    base_test_start: int,
    gate_test_seasons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    bet_frames: list[pd.DataFrame] = []
    for test_season in gate_test_seasons:
        training = frame[
            frame["season"].between(base_test_start, test_season - 1)
        ]
        testing = frame[frame["season"].eq(test_season)]
        prior_oof_seasons = test_season - base_test_start
        minimum_training_bets = max(40, 20 * prior_oof_seasons)
        candidates: list[
            tuple[float, int, float, float, int, str, Gate, dict[str, object]]
        ] = []
        for gate in enumerate_gates(thresholds):
            summary = grade_gate(training, gate)
            if int(summary["bets"]) < minimum_training_bets:
                continue
            score = bayesian_gate_score(summary)
            candidates.append(
                (
                    score,
                    int(summary["bets"]),
                    -gate.edge_threshold,
                    -gate.maximum_projection_range,
                    gate.minimum_agreement,
                    gate.ensemble,
                    gate,
                    summary,
                )
            )
        if not candidates:
            raise RuntimeError(f"No gate met the training-volume floor for {test_season}.")
        selected = max(candidates, key=lambda item: item[:6])
        score, _, _, _, _, _, gate, training_summary = selected
        test_summary = grade_gate(testing, gate)
        fold_rows.append(
            {
                "test_season": test_season,
                "prior_oof_seasons": ",".join(
                    str(season) for season in range(base_test_start, test_season)
                ),
                "minimum_training_bets": minimum_training_bets,
                "selection_score": score,
                **{f"training_{key}": value for key, value in training_summary.items()},
                **{f"test_{key}": value for key, value in test_summary.items()},
            }
        )
        selected_bets = testing[gate_mask(testing, gate)].copy()
        selected_edge = selected_bets[f"{gate.ensemble}_edge"]
        selected_bets["selected_ensemble"] = gate.ensemble
        selected_bets["selected_edge_threshold"] = gate.edge_threshold
        selected_bets["selected_maximum_projection_range"] = (
            gate.maximum_projection_range
        )
        selected_bets["selected_minimum_agreement"] = gate.minimum_agreement
        selected_bets["selected_side"] = np.where(selected_edge > 0, "HOME", "AWAY")
        selected_bets["selected_team"] = np.where(
            selected_edge > 0, selected_bets["home_team"], selected_bets["away_team"]
        )
        selected_bets["selected_edge"] = selected_edge.abs()
        selected_bets["selected_market_line"] = np.where(
            selected_edge > 0,
            -selected_bets["circa_home_margin"],
            selected_bets["circa_home_margin"],
        )
        selected_bets["selected_cover_margin"] = np.sign(selected_edge) * (
            selected_bets["actual_home_margin"]
            - selected_bets["circa_home_margin"]
        )
        selected_bets["ats_result"] = np.select(
            [
                selected_bets["selected_cover_margin"] > 0,
                selected_bets["selected_cover_margin"] < 0,
            ],
            ["WIN", "LOSS"],
            default="PUSH",
        )
        selected_bets["ats_profit_units"] = np.select(
            [
                selected_bets["selected_cover_margin"] > 0,
                selected_bets["selected_cover_margin"] < 0,
            ],
            [1.0 / 1.1, -1.0],
            default=0.0,
        )
        bet_frames.append(selected_bets)
        print(
            f"[CONSENSUS] Gate {test_season}: {gate.ensemble} "
            f"edge>={gate.edge_threshold:.1f} range<={gate.maximum_projection_range:.1f} "
            f"agreement>={gate.minimum_agreement} | "
            f"test={test_summary['wins']}-{test_summary['losses']}-{test_summary['pushes']}",
            flush=True,
        )
    bets = pd.concat(bet_frames, ignore_index=True) if bet_frames else pd.DataFrame()
    if bets.empty:
        aggregate = pd.DataFrame(
            [{"scope": "WALK_FORWARD_ALL", "bets": 0, "wins": 0, "losses": 0, "pushes": 0}]
        )
    else:
        wins = int(bets["ats_result"].eq("WIN").sum())
        losses = int(bets["ats_result"].eq("LOSS").sum())
        pushes = int(bets["ats_result"].eq("PUSH").sum())
        lower, upper = wilson_interval(wins, losses)
        profit = float(bets["ats_profit_units"].sum())
        ordered = bets.sort_values(["game_date", "season", "week", "game_id"])
        equity = ordered["ats_profit_units"].cumsum()
        drawdown = equity.cummax().sub(equity)
        aggregate = pd.DataFrame(
            [
                {
                    "scope": "WALK_FORWARD_ALL",
                    "bets": len(bets),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "ats_win_rate": wins / (wins + losses) if wins + losses else math.nan,
                    "ats_profit_units": profit,
                    "ats_roi": profit / len(bets),
                    "wilson_lower_95": lower,
                    "wilson_upper_95": upper,
                    "maximum_drawdown_units": float(drawdown.max()),
                }
            ]
        )
    return pd.DataFrame(fold_rows), bets, aggregate


def nonlinear_threshold_summary(
    frame: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    temporary = frame.copy()
    temporary["nonlinear_edge"] = (
        temporary["nonlinear_projection"] - temporary["circa_home_margin"]
    )
    temporary["nonlinear_agreement"] = 3
    temporary["projection_range"] = 0.0
    rows: list[dict[str, object]] = []
    for scope, group in [
        ("OOF_ALL", temporary),
        *(
            (f"SEASON_{int(season)}", season_frame)
            for season, season_frame in temporary.groupby("season", sort=True)
        ),
    ]:
        for threshold in thresholds:
            summary = grade_gate(
                group, Gate("nonlinear", threshold, 999.0, 1)
            )
            rows.append({"scope": scope, **summary})
    return pd.DataFrame(rows)


def add_metadata(frames: Sequence[pd.DataFrame], run_id: str, created_at: str) -> None:
    for frame in frames:
        frame["run_id"] = run_id
        frame["build_id"] = BUILD_ID
        frame["version"] = VERSION
        frame["created_at"] = created_at


def write_results(
    args: argparse.Namespace,
    tables: dict[str, pd.DataFrame],
) -> None:
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.output_db) as connection:
        for table, frame in tables.items():
            frame.to_sql(table, connection, if_exists="replace", index=False)
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if check != "ok":
        raise RuntimeError(f"Output database integrity check failed: {check}")
    if not args.no_csv:
        csv_root = args.output_db.with_suffix("")
        csv_root.mkdir(parents=True, exist_ok=True)
        for table, frame in tables.items():
            frame.to_csv(csv_root / f"{table}.csv", index=False)


def main() -> int:
    started = datetime.now(timezone.utc)
    args = resolve_paths(build_parser().parse_args())
    print("=" * 112)
    print("[CONSENSUS] NFL NONLINEAR MATCHUP + MULTI-MODEL CONSENSUS BACKTEST")
    print("=" * 112)
    print(f"[CONSENSUS] Build ID: {BUILD_ID}")
    print(f"[CONSENSUS] Version: {VERSION}")
    print(f"[CONSENSUS] Seasons: {list(args.target_seasons)}")
    print("[CONSENSUS] Market inputs in independent projections: NO")
    print("[CONSENSUS] Market used in prior-only wagering gate: YES")
    print("[CONSENSUS] Source databases modified: NO")

    structural = load_structural_inputs(args)
    feature_names = nonlinear_feature_names()
    matrices = {
        rating_alpha: load_matchup_matrix(args, structural, rating_alpha)
        for rating_alpha in RATING_ALPHA_GRID
    }
    test_seasons = tuple(
        season for season in args.target_seasons if season >= args.test_start_season
    )
    nonlinear, nonlinear_folds = fit_nonlinear_oof(
        matrices, feature_names, test_seasons
    )
    learned_oof = load_learned_oof(args, test_seasons)
    consensus = build_consensus_matrix(nonlinear, learned_oof)
    nonlinear_summary = nonlinear_threshold_summary(consensus, args.thresholds)
    posthoc_grid = summarize_posthoc_grid(consensus, args.thresholds)
    gate_test_seasons = tuple(
        season for season in test_seasons if season >= args.gate_test_start_season
    )
    gate_folds, gate_bets, gate_summary = select_walk_forward_gates(
        consensus,
        args.thresholds,
        args.test_start_season,
        gate_test_seasons,
    )

    frozen_research_gate = Gate("mean_unit_nonlinear", 2.0, 3.5, 3)
    frozen_rows = [
        {
            "scope": "OOF_ALL",
            "research_status": "POSTHOC_CANDIDATE_FREEZE_FOR_PROSPECTIVE_TEST_ONLY",
            **grade_gate(consensus, frozen_research_gate),
        }
    ]
    frozen_rows.extend(
        {
            "scope": f"SEASON_{int(season)}",
            "research_status": "POSTHOC_CANDIDATE_FREEZE_FOR_PROSPECTIVE_TEST_ONLY",
            **grade_gate(group, frozen_research_gate),
        }
        for season, group in consensus.groupby("season", sort=True)
    )
    frozen_summary = pd.DataFrame(frozen_rows)

    run_id = started.strftime("nfl_nonlinear_consensus_%Y%m%dT%H%M%SZ")
    created_at = started.isoformat()
    audit = pd.DataFrame(
        [
            {"audit_item": "target_seasons", "audit_value": ",".join(map(str, args.target_seasons))},
            {"audit_item": "base_oof_test_seasons", "audit_value": ",".join(map(str, test_seasons))},
            {"audit_item": "gate_oof_test_seasons", "audit_value": ",".join(map(str, gate_test_seasons))},
            {"audit_item": "full_replay_training_rows", "audit_value": str(len(structural))},
            {
                "audit_item": "strict_circa_grading_rows",
                "audit_value": str(int(consensus["strict_circa_line_valid"].eq(1).sum())),
            },
            {"audit_item": "market_used_in_projection", "audit_value": "NO"},
            {"audit_item": "future_seasons_used_in_projection_training", "audit_value": "NO"},
            {"audit_item": "market_used_in_wagering_gate", "audit_value": "PRIOR_OOF_ONLY"},
            {"audit_item": "future_seasons_used_in_gate_selection", "audit_value": "NO"},
            {"audit_item": "pooled_grid_deployable", "audit_value": "NO"},
            {"audit_item": "source_databases_modified", "audit_value": "NO"},
        ]
    )
    tables = {
        "nfl_nonlinear_matchup_oof_predictions": nonlinear,
        "nfl_nonlinear_matchup_fold_audit": nonlinear_folds,
        "nfl_nonlinear_matchup_threshold_summary": nonlinear_summary,
        "nfl_consensus_oof_predictions": consensus,
        "nfl_consensus_posthoc_gate_grid": posthoc_grid,
        "nfl_consensus_frozen_research_candidate": frozen_summary,
        "nfl_consensus_walk_forward_gate_audit": gate_folds,
        "nfl_consensus_walk_forward_bets": gate_bets,
        "nfl_consensus_walk_forward_summary": gate_summary,
        "nfl_consensus_run_audit": audit,
    }
    add_metadata(list(tables.values()), run_id, created_at)
    write_results(args, tables)

    nonlinear_primary = nonlinear_summary[
        nonlinear_summary["scope"].eq("OOF_ALL")
        & nonlinear_summary["edge_threshold"].isin((1.0, 2.0, 4.0))
    ][
        [
            "edge_threshold",
            "bets",
            "wins",
            "losses",
            "pushes",
            "ats_win_rate",
            "ats_roi",
            "wilson_lower_95",
            "wilson_upper_95",
        ]
    ]
    print("\n[CONSENSUS] Nonlinear projection — OOF 2022-2025:")
    print(nonlinear_primary.to_string(index=False))
    print("\n[CONSENSUS] Fixed research candidate — not deployable from this backtest:")
    print(
        frozen_summary[
            [
                "scope",
                "bets",
                "wins",
                "losses",
                "pushes",
                "ats_win_rate",
                "ats_roi",
                "wilson_lower_95",
                "wilson_upper_95",
            ]
        ].to_string(index=False)
    )
    print("\n[CONSENSUS] Strict walk-forward learned gate:")
    print(
        gate_summary[
            [
                "scope",
                "bets",
                "wins",
                "losses",
                "pushes",
                "ats_win_rate",
                "ats_roi",
                "wilson_lower_95",
                "wilson_upper_95",
            ]
        ].to_string(index=False)
    )
    print(f"\n[CONSENSUS] Output database: {args.output_db}")
    print("[CONSENSUS] Integrity check: ok")
    print(
        f"[CONSENSUS] Completed in "
        f"{(datetime.now(timezone.utc) - started).total_seconds():.2f} seconds"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CONSENSUS] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CONSENSUS] FAILED: {exc}", file=sys.stderr)
        raise
