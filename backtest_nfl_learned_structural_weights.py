#!/usr/bin/env python
"""Leakage-controlled NFL structural-weight research backtest.

This script learns, rather than hard-codes:

* position-unit or projected-starter-slot weights;
* prior-season versus current-season contribution weights;
* season/recent and process/result form weights;
* home-field advantage;
* the current-season transition curve g / (g + k), subject to a learned cap;
* ridge shrinkage and the training-margin cap.

The independent projection is trained only on actual scoring margin. Circa is
attached after each frozen out-of-fold projection and is used only for ATS
grading. The first honest test season defaults to 2022, leaving 2020 for the
first training window and 2021 for the first inner validation window.

Source databases are read-only. Results are written to a separate SQLite file.
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
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUILD_ID = "NFL_LEARNED_STRUCTURAL_WEIGHTS_BACKTEST_CANONICAL_V1"
VERSION = "v1_0_market_free_nested_season_forward"

REPLAY_DATABASE = "nfl_rebuilt_historical_replay.sqlite"
CIRCA_DATABASE = "nfl_circa_contest_lines.sqlite"
REPLAY_PREDICTION_TABLE = "nfl_rebuilt_weekly_replay_predictions"
REPLAY_FORM_TABLE = "nfl_rebuilt_weekly_form_history"
UNIT_TABLE = "nfl_position_group_ratings_target"
DEPTH_TABLE = "nfl_projected_depth_chart_target"
CIRCA_TABLE = "nfl_circa_game_lines"

UNIT_NAMES = ("qb", "rb", "wr_te", "ol", "dl_edge", "lb", "db", "st")
FORM_COMPONENTS = (
    "season_process_rating",
    "season_result_rating",
    "recent_process_rating",
    "recent_result_rating",
)
SCHEDULE_FEATURES = (
    "home_field",
    "rest_advantage",
    "away_travel_1000",
    "away_extreme_travel",
    "away_direction_burden",
    "away_schedule_burden",
)
MARKET_TOKENS = ("market", "circa", "spread", "line", "price", "odds", "cover")

DEFAULT_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5)
K_GRID = (3.0, 5.0, 8.0, 12.0)
CURRENT_CAP_GRID = (0.65, 0.75, 0.85)
ALPHA_GRID = (30.0, 100.0, 300.0, 1000.0)
TARGET_CAP_GRID = (21.0, 28.0, 35.0)


@dataclass(frozen=True)
class Variant:
    name: str
    structural_source: str
    include_form: bool


VARIANTS = (
    Variant("LEARNED_UNIT_STRUCTURAL", "unit", False),
    Variant("LEARNED_UNIT_STRUCTURAL_FORM", "unit", True),
    Variant("LEARNED_SLOT_STRUCTURAL", "slot", False),
    Variant("LEARNED_SLOT_STRUCTURAL_FORM", "slot", True),
)


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
    parser.add_argument("--circa-db", type=Path)
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--target-seasons", type=parse_int_csv, default=DEFAULT_SEASONS)
    parser.add_argument("--test-start-season", type=int, default=2022)
    parser.add_argument("--minimum-week", type=int, default=2)
    parser.add_argument("--maximum-week", type=int, default=17)
    parser.add_argument("--thresholds", type=parse_float_csv, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--no-csv", action="store_true")
    return parser


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.database_root = args.database_root.expanduser().resolve()
    args.replay_db = (args.replay_db or args.database_root / REPLAY_DATABASE).expanduser().resolve()
    args.circa_db = (args.circa_db or args.database_root / CIRCA_DATABASE).expanduser().resolve()
    args.output_db = (
        args.output_db or args.database_root / "nfl_learned_structural_weights_backtest.sqlite"
    ).expanduser().resolve()
    sources = {args.replay_db, args.circa_db, *(args.database_root / f"{season}.sqlite" for season in args.target_seasons)}
    if args.output_db in sources:
        raise ValueError("The output database must be separate from every source database.")
    if tuple(sorted(args.target_seasons)) != tuple(args.target_seasons):
        raise ValueError("--target-seasons must be in ascending order.")
    if len(set(args.target_seasons)) != len(args.target_seasons):
        raise ValueError("--target-seasons contains duplicates.")
    if args.test_start_season not in args.target_seasons:
        raise ValueError("--test-start-season must be included in --target-seasons.")
    first = min(args.target_seasons)
    if args.test_start_season < first + 2:
        raise ValueError(
            "Nested season-forward validation needs at least two seasons before the first test season."
        )
    if args.minimum_week < 2:
        raise ValueError("Week 1 is intentionally excluded; --minimum-week must be at least 2.")
    if args.maximum_week < args.minimum_week:
        raise ValueError("--maximum-week must not precede --minimum-week.")
    if any(threshold <= 0 for threshold in args.thresholds):
        raise ValueError("Every threshold must be positive.")
    for source in sources:
        if source == args.output_db:
            continue
        if not source.exists():
            raise FileNotFoundError(source)
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


def load_games(args: argparse.Namespace) -> pd.DataFrame:
    prediction_columns = (
        "season, week, game_id, game_date, home_team, away_team, actual_home_margin, "
        "projected_home_margin AS baseline_projection, neutral_site, home_days_rest, "
        "away_days_rest, travel_miles, travel_dist_2000_plus, back_to_back_west_east, "
        "back_to_back_west_central, back_to_back_away, three_of_four_away, "
        "form_through_week, home_rating_through_week, away_rating_through_week, "
        "prediction_uses_market_inputs"
    )
    require_columns(
        args.replay_db,
        REPLAY_PREDICTION_TABLE,
        [item.split(" AS ")[0].strip() for item in prediction_columns.split(",")],
    )
    placeholders = ",".join("?" for _ in args.target_seasons)
    games = read_sql(
        args.replay_db,
        f"SELECT {prediction_columns} FROM {REPLAY_PREDICTION_TABLE} "
        f"WHERE season IN ({placeholders}) AND week BETWEEN ? AND ?",
        (*args.target_seasons, args.minimum_week, args.maximum_week),
    )
    if games.empty:
        raise RuntimeError("The replay query returned no games.")
    if games.duplicated(["season", "week", "game_id"]).any():
        raise RuntimeError("Replay predictions contain duplicate season/week/game_id rows.")
    if games["prediction_uses_market_inputs"].fillna(0).astype(int).ne(0).any():
        raise RuntimeError("A replay projection is marked as using market inputs.")
    for column in ("form_through_week", "home_rating_through_week", "away_rating_through_week"):
        if games[column].astype(int).ne(games["week"].astype(int) - 1).any():
            raise RuntimeError(f"Point-in-time failure: {column} is not exactly week - 1.")
    return games


def attach_form(args: argparse.Namespace, games: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "through_week",
        "team",
        "games_played",
        "no_lookahead_filter_applied_flag",
        *FORM_COMPONENTS,
    ]
    require_columns(args.replay_db, REPLAY_FORM_TABLE, required)
    placeholders = ",".join("?" for _ in args.target_seasons)
    form = read_sql(
        args.replay_db,
        f"SELECT {', '.join(required)} FROM {REPLAY_FORM_TABLE} "
        f"WHERE season IN ({placeholders})",
        args.target_seasons,
    )
    if form.duplicated(["season", "through_week", "team"]).any():
        raise RuntimeError("Form history contains duplicate season/through_week/team rows.")
    if form["no_lookahead_filter_applied_flag"].fillna(0).astype(int).ne(1).any():
        raise RuntimeError("A form-history row is not marked leakage-safe.")
    output = games
    for side in ("home", "away"):
        renamed = form.drop(columns="no_lookahead_filter_applied_flag").rename(
            columns={
                "through_week": "form_through_week",
                "team": f"{side}_team",
                **{
                    column: f"{side}_{column}"
                    for column in ("games_played", *FORM_COMPONENTS)
                },
            }
        )
        output = output.merge(
            renamed,
            on=["season", "form_through_week", f"{side}_team"],
            how="left",
            validate="many_to_one",
        )
    return output


def attach_units(args: argparse.Namespace, games: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    required = ("season", "team", "unit_name", "unit_rating")
    for season in args.target_seasons:
        database = args.database_root / f"{season}.sqlite"
        require_columns(database, UNIT_TABLE, required)
        unit = read_sql(database, f"SELECT {', '.join(required)} FROM {UNIT_TABLE}")
        unit["unit_name"] = unit["unit_name"].str.lower()
        unexpected = set(unit["unit_name"].dropna()) - set(UNIT_NAMES)
        if unexpected:
            raise RuntimeError(f"Unexpected units in {database.name}: {sorted(unexpected)}")
        if unit.duplicated(["season", "team", "unit_name"]).any():
            raise RuntimeError(f"Duplicate team-unit rows in {database.name}")
        wide = unit.pivot(index=["season", "team"], columns="unit_name", values="unit_rating").reset_index()
        if set(UNIT_NAMES) - set(wide.columns):
            raise RuntimeError(f"Incomplete unit matrix in {database.name}")
        for name in UNIT_NAMES:
            wide[name] -= wide.groupby("season")[name].transform("mean")
        parts.append(wide)
    units = pd.concat(parts, ignore_index=True)
    output = games
    for side in ("home", "away"):
        renamed = units.rename(
            columns={"team": f"{side}_team", **{name: f"{side}_unit_{name}" for name in UNIT_NAMES}}
        )
        output = output.merge(
            renamed,
            on=["season", f"{side}_team"],
            how="left",
            validate="many_to_one",
        )
    return output


def attach_slots(args: argparse.Namespace, games: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    parts: list[pd.DataFrame] = []
    required = (
        "season",
        "team",
        "starter_slot",
        "is_projected_starter",
        "performance_grade",
        "replacement_baseline",
    )
    for season in args.target_seasons:
        database = args.database_root / f"{season}.sqlite"
        require_columns(database, DEPTH_TABLE, required)
        depth = read_sql(
            database,
            f"SELECT {', '.join(required)} FROM {DEPTH_TABLE} "
            "WHERE is_projected_starter = 1 AND starter_slot IS NOT NULL AND starter_slot <> ''",
        )
        if depth.duplicated(["season", "team", "starter_slot"]).any():
            raise RuntimeError(f"Duplicate projected starter slots in {database.name}")
        depth["slot_value"] = depth["performance_grade"] - depth["replacement_baseline"]
        wide = depth.pivot(
            index=["season", "team"], columns="starter_slot", values="slot_value"
        ).reset_index()
        parts.append(wide)
    slots = pd.concat(parts, ignore_index=True, sort=False)
    slot_names = tuple(sorted(set(slots.columns) - {"season", "team"}))
    if not slot_names:
        raise RuntimeError("No projected starter slots were loaded.")
    for slot in slot_names:
        slots[slot] = slots[slot].fillna(slots.groupby("season")[slot].transform("median"))
        slots[slot] = slots[slot].fillna(0.0)
        slots[slot] -= slots.groupby("season")[slot].transform("mean")
    output = games
    for side in ("home", "away"):
        renamed = slots.rename(
            columns={
                "team": f"{side}_team",
                **{slot: f"{side}_slot_{slot}" for slot in slot_names},
            }
        )
        output = output.merge(
            renamed,
            on=["season", f"{side}_team"],
            how="left",
            validate="many_to_one",
        )
    return output, slot_names


def attach_circa(args: argparse.Namespace, games: pd.DataFrame) -> pd.DataFrame:
    required = (
        "season",
        "week",
        "game_id",
        "circa_home_margin",
        "strict_circa_line_valid",
        "line_source",
    )
    require_columns(args.circa_db, CIRCA_TABLE, required)
    lines = read_sql(args.circa_db, f"SELECT {', '.join(required)} FROM {CIRCA_TABLE}")
    if lines.duplicated(["season", "week", "game_id"]).any():
        raise RuntimeError("Circa table contains duplicate season/week/game_id rows.")
    return games.merge(
        lines,
        on=["season", "week", "game_id"],
        how="left",
        validate="one_to_one",
    )


def validate_modeling_inputs(frame: pd.DataFrame, slot_names: Sequence[str]) -> None:
    required = [
        "actual_home_margin",
        "baseline_projection",
        "home_games_played",
        "away_games_played",
        *(f"{side}_unit_{name}" for side in ("home", "away") for name in UNIT_NAMES),
        *(f"{side}_slot_{slot}" for side in ("home", "away") for slot in slot_names),
        *(f"{side}_{name}" for side in ("home", "away") for name in FORM_COMPONENTS),
    ]
    missing = frame[required].isna().sum()
    if (missing > 0).any():
        raise RuntimeError(f"Missing modeling inputs:\n{missing[missing > 0].to_string()}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise RuntimeError("A modeling input contains a non-finite value.")


def season_weights(frame: pd.DataFrame, k: float, cap: float) -> tuple[pd.Series, pd.Series]:
    home_games = frame["home_games_played"].astype(float)
    away_games = frame["away_games_played"].astype(float)
    home = (home_games / (home_games + k)).clip(upper=cap)
    away = (away_games / (away_games + k)).clip(upper=cap)
    return home, away


def make_features(
    frame: pd.DataFrame,
    variant: Variant,
    slot_names: Sequence[str],
    k: float,
    cap: float,
) -> pd.DataFrame:
    home_weight, away_weight = season_weights(frame, k, cap)
    output = pd.DataFrame(index=frame.index)
    names = UNIT_NAMES if variant.structural_source == "unit" else slot_names
    prefix = "unit" if variant.structural_source == "unit" else "slot"
    for name in names:
        home = frame[f"home_{prefix}_{name}"]
        away = frame[f"away_{prefix}_{name}"]
        output[f"prior_{name}"] = (1.0 - home_weight) * home - (1.0 - away_weight) * away
        output[f"current_{name}"] = home_weight * home - away_weight * away
    if variant.include_form:
        for component in FORM_COMPONENTS:
            output[f"form_{component}"] = (
                home_weight * frame[f"home_{component}"]
                - away_weight * frame[f"away_{component}"]
            )
    output["home_field"] = 1.0 - frame["neutral_site"].fillna(0).clip(0, 1)
    output["rest_advantage"] = (
        frame["home_days_rest"].fillna(7) - frame["away_days_rest"].fillna(7)
    ).clip(-7, 7)
    output["away_travel_1000"] = frame["travel_miles"].fillna(0).clip(0, 5000) / 1000.0
    output["away_extreme_travel"] = frame["travel_dist_2000_plus"].fillna(0).clip(0, 1)
    output["away_direction_burden"] = (
        frame["back_to_back_west_east"].fillna(0)
        + frame["back_to_back_west_central"].fillna(0)
    ).clip(0, 1)
    output["away_schedule_burden"] = (
        frame["back_to_back_away"].fillna(0) + frame["three_of_four_away"].fillna(0)
    ).clip(0, 2)
    forbidden = [name for name in output if any(token in name.lower() for token in MARKET_TOKENS)]
    if forbidden:
        raise RuntimeError(f"Market-derived features entered the model: {forbidden}")
    return output


def fit_model(x: pd.DataFrame, y: pd.Series, alpha: float, target_cap: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "ridge",
                Ridge(alpha=alpha, fit_intercept=False, positive=True, solver="lbfgs"),
            ),
        ]
    )
    model.fit(x, y.clip(-target_cap, target_cap))
    return model


def choose_parameters(
    frame: pd.DataFrame,
    variant: Variant,
    slot_names: Sequence[str],
    test_season: int,
) -> dict[str, float]:
    feature_cache: dict[tuple[float, float], pd.DataFrame] = {}
    scored: list[tuple[float, float, float, float, float]] = []
    for k, cap, alpha, target_cap in itertools.product(
        K_GRID, CURRENT_CAP_GRID, ALPHA_GRID, TARGET_CAP_GRID
    ):
        key = (k, cap)
        if key not in feature_cache:
            feature_cache[key] = make_features(frame, variant, slot_names, k, cap)
        x = feature_cache[key]
        errors: list[float] = []
        for validation_season in range(min(frame["season"]) + 1, test_season):
            train = frame["season"] < validation_season
            validate = frame["season"] == validation_season
            if not train.any() or not validate.any():
                continue
            model = fit_model(x.loc[train], frame.loc[train, "actual_home_margin"], alpha, target_cap)
            predicted = model.predict(x.loc[validate])
            errors.extend(np.abs(predicted - frame.loc[validate, "actual_home_margin"]).tolist())
        if not errors:
            raise RuntimeError(f"No inner-validation rows were available before {test_season}.")
        scored.append((float(np.mean(errors)), k, cap, alpha, target_cap))
    best = min(scored, key=lambda row: (row[0], row[3], row[1], row[2], row[4]))
    return {
        "inner_validation_mae": best[0],
        "transition_k": best[1],
        "maximum_current_season_weight": best[2],
        "ridge_alpha": best[3],
        "target_margin_cap": best[4],
    }


def feature_group(name: str) -> str:
    if name.startswith("prior_"):
        return "PRIOR_STRUCTURAL"
    if name.startswith("current_"):
        return "CURRENT_STRUCTURAL"
    if name.startswith("form_"):
        return "CURRENT_FORM"
    return "HFA_SCHEDULE"


def fit_out_of_fold(
    frame: pd.DataFrame,
    variants: Sequence[Variant],
    slot_names: Sequence[str],
    test_seasons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    for variant in variants:
        print(f"[LEARN] Variant: {variant.name}", flush=True)
        for test_season in test_seasons:
            parameters = choose_parameters(frame, variant, slot_names, test_season)
            x = make_features(
                frame,
                variant,
                slot_names,
                parameters["transition_k"],
                parameters["maximum_current_season_weight"],
            )
            train = frame["season"] < test_season
            test = frame["season"] == test_season
            model = fit_model(
                x.loc[train],
                frame.loc[train, "actual_home_margin"],
                parameters["ridge_alpha"],
                parameters["target_margin_cap"],
            )
            projected = model.predict(x.loc[test])
            result = frame.loc[test].copy()
            result["model_variant"] = variant.name
            result["projected_home_margin"] = projected
            result["model_market_edge_home_points"] = projected - result["circa_home_margin"]
            predictions.append(result)

            test_mae = float(np.mean(np.abs(projected - result["actual_home_margin"])))
            folds.append(
                {
                    "model_variant": variant.name,
                    "test_season": test_season,
                    "training_seasons": ",".join(
                        str(season) for season in sorted(frame.loc[train, "season"].unique())
                    ),
                    "training_games": int(train.sum()),
                    "test_games": int(test.sum()),
                    **parameters,
                    "test_margin_mae": test_mae,
                    "market_features_in_projection": 0,
                    "future_seasons_in_training": 0,
                }
            )

            scale = model.named_steps["scale"].scale_
            standardized = model.named_steps["ridge"].coef_
            raw = standardized / scale
            coefficient_frame = pd.DataFrame(
                {
                    "feature": x.columns,
                    "standardized_coefficient": standardized,
                    "raw_coefficient": raw,
                }
            )
            coefficient_frame["feature_group"] = coefficient_frame["feature"].map(feature_group)
            group_sum = coefficient_frame.groupby("feature_group")["raw_coefficient"].transform("sum")
            coefficient_frame["within_group_weight"] = np.where(
                group_sum > 0, coefficient_frame["raw_coefficient"] / group_sum, 0.0
            )
            for row in coefficient_frame.to_dict("records"):
                coefficients.append(
                    {
                        "model_variant": variant.name,
                        "test_season": test_season,
                        **row,
                    }
                )
            print(
                f"[LEARN] {test_season}: train={int(train.sum())} test={int(test.sum())} "
                f"k={parameters['transition_k']:.1f} "
                f"cap={parameters['maximum_current_season_weight']:.2f} "
                f"alpha={parameters['ridge_alpha']:.0f} "
                f"test_MAE={test_mae:.3f}",
                flush=True,
            )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(folds),
        pd.DataFrame(coefficients),
    )


def add_baseline_predictions(frame: pd.DataFrame, test_seasons: Sequence[int]) -> pd.DataFrame:
    baseline = frame[frame["season"].isin(test_seasons)].copy()
    baseline["model_variant"] = "EXISTING_STRUCTURAL_FORM_BASELINE"
    baseline["projected_home_margin"] = baseline["baseline_projection"]
    baseline["model_market_edge_home_points"] = (
        baseline["projected_home_margin"] - baseline["circa_home_margin"]
    )
    return baseline


def wilson_interval(wins: int, losses: int, z: float = 1.959963984540054) -> tuple[float, float]:
    n = wins + losses
    if n == 0:
        return math.nan, math.nan
    p = wins / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - half, center + half


def grade_threshold(group: pd.DataFrame, threshold: float) -> dict[str, object]:
    eligible = group[
        group["strict_circa_line_valid"].eq(1)
        & group["circa_home_margin"].notna()
        & group["actual_home_margin"].notna()
    ].copy()
    eligible["edge"] = eligible["model_market_edge_home_points"]
    bets = eligible[eligible["edge"].abs().ge(threshold)].copy()
    bets["selected_cover_margin"] = np.sign(bets["edge"]) * (
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
    profit = float(bets["profit"].sum())
    ordered = bets.sort_values(["game_date", "season", "week", "game_id"])
    equity = ordered["profit"].cumsum()
    drawdown = equity.cummax().sub(equity)
    lower, upper = wilson_interval(wins, losses)
    return {
        "edge_threshold": threshold,
        "eligible_games": len(eligible),
        "bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_win_rate": wins / decisions if decisions else math.nan,
        "ats_profit_units": profit,
        "ats_roi": profit / len(bets) if len(bets) else math.nan,
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "average_absolute_edge": float(bets["edge"].abs().mean()) if len(bets) else math.nan,
        "maximum_drawdown_units": float(drawdown.max()) if len(drawdown) else 0.0,
    }


def summarize_thresholds(predictions: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, variant_frame in predictions.groupby("model_variant", sort=True):
        scopes = [("OOF_ALL", variant_frame)]
        scopes.extend(
            (f"SEASON_{int(season)}", season_frame)
            for season, season_frame in variant_frame.groupby("season", sort=True)
        )
        for scope, scope_frame in scopes:
            for threshold in thresholds:
                rows.append(
                    {
                        "model_variant": variant,
                        "scope": scope,
                        **grade_threshold(scope_frame, threshold),
                    }
                )
    return pd.DataFrame(rows)


def summarize_seasons(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, season), group in predictions.groupby(["model_variant", "season"], sort=True):
        rows.append(
            {
                "model_variant": variant,
                "season": int(season),
                "games": len(group),
                "model_mae": float(
                    np.mean(np.abs(group["projected_home_margin"] - group["actual_home_margin"]))
                ),
                "market_mae": float(
                    np.mean(
                        np.abs(
                            group.loc[group["strict_circa_line_valid"].eq(1), "circa_home_margin"]
                            - group.loc[group["strict_circa_line_valid"].eq(1), "actual_home_margin"]
                        )
                    )
                ),
                "strict_circa_games": int(group["strict_circa_line_valid"].eq(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_metadata(frames: Sequence[pd.DataFrame], run_id: str, created_at: str) -> None:
    for frame in frames:
        frame["run_id"] = run_id
        frame["build_id"] = BUILD_ID
        frame["version"] = VERSION
        frame["created_at"] = created_at


def write_results(
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    coefficients: pd.DataFrame,
    thresholds: pd.DataFrame,
    seasons: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    tables = {
        "nfl_learned_structural_oof_predictions": predictions,
        "nfl_learned_structural_fold_audit": folds,
        "nfl_learned_structural_coefficients": coefficients,
        "nfl_learned_structural_threshold_summary": thresholds,
        "nfl_learned_structural_season_summary": seasons,
        "nfl_learned_structural_run_audit": audit,
    }
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
    print("=" * 108)
    print("[LEARN] NFL LEARNED STRUCTURAL WEIGHTS — NESTED SEASON-FORWARD BACKTEST")
    print("=" * 108)
    print(f"[LEARN] Build ID: {BUILD_ID}")
    print(f"[LEARN] Version: {VERSION}")
    print(f"[LEARN] Seasons: {list(args.target_seasons)}")
    print(f"[LEARN] First out-of-fold test season: {args.test_start_season}")
    print("[LEARN] Market inputs in independent projection: NO")
    print("[LEARN] Source databases modified: NO")

    frame = load_games(args)
    frame = attach_form(args, frame)
    frame = attach_units(args, frame)
    frame, slot_names = attach_slots(args, frame)
    frame = attach_circa(args, frame)
    validate_modeling_inputs(frame, slot_names)
    test_seasons = tuple(
        season for season in args.target_seasons if season >= args.test_start_season
    )
    if not test_seasons:
        raise RuntimeError("No out-of-fold test seasons were selected.")

    learned, folds, coefficients = fit_out_of_fold(frame, VARIANTS, slot_names, test_seasons)
    baseline = add_baseline_predictions(frame, test_seasons)
    predictions = pd.concat([learned, baseline], ignore_index=True, sort=False)
    prediction_columns = [
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
        "model_market_edge_home_points",
    ]
    predictions = predictions[prediction_columns]
    threshold_summary = summarize_thresholds(predictions, args.thresholds)
    season_summary = summarize_seasons(predictions)

    run_id = started.strftime("nfl_learned_weights_%Y%m%dT%H%M%SZ")
    created_at = started.isoformat()
    audit = pd.DataFrame(
        [
            {"audit_item": "target_seasons", "audit_value": ",".join(map(str, args.target_seasons))},
            {"audit_item": "test_seasons", "audit_value": ",".join(map(str, test_seasons))},
            {"audit_item": "week_range", "audit_value": f"{args.minimum_week}-{args.maximum_week}"},
            {"audit_item": "source_game_rows", "audit_value": str(len(frame))},
            {
                "audit_item": "strict_circa_rows",
                "audit_value": str(int(frame["strict_circa_line_valid"].eq(1).sum())),
            },
            {"audit_item": "projected_starter_slots", "audit_value": ",".join(slot_names)},
            {"audit_item": "week_1_graded", "audit_value": "NO"},
            {"audit_item": "market_used_in_projection", "audit_value": "NO"},
            {"audit_item": "future_seasons_used_in_fold_training", "audit_value": "NO"},
            {"audit_item": "hyperparameters_selected_on_ats", "audit_value": "NO"},
            {"audit_item": "source_databases_modified", "audit_value": "NO"},
        ]
    )
    add_metadata(
        [predictions, folds, coefficients, threshold_summary, season_summary, audit],
        run_id,
        created_at,
    )
    write_results(
        args,
        predictions,
        folds,
        coefficients,
        threshold_summary,
        season_summary,
        audit,
    )

    primary = threshold_summary[
        threshold_summary["scope"].eq("OOF_ALL")
        & threshold_summary["edge_threshold"].isin((1.0, 2.5, 5.0))
    ][
        [
            "model_variant",
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
    print("\n[LEARN] Out-of-fold threshold summary:")
    print(primary.to_string(index=False))
    print(f"\n[LEARN] Output database: {args.output_db}")
    print(f"[LEARN] Integrity check: ok")
    print(f"[LEARN] Completed in {(datetime.now(timezone.utc) - started).total_seconds():.2f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[LEARN] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[LEARN] FAILED: {exc}", file=sys.stderr)
        raise
