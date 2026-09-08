#!/usr/bin/env python
"""
Final read-only audit of the rebuilt 2022-2025 NFL weekly spread replay.

This script independently recomputes the primary backtest statistics directly
from nfl_rebuilt_weekly_replay_predictions, compares the locked 2024-2025
holdout with the legacy benchmark, checks leakage/duplication contracts, reviews
probability calibration, and simulates the recorded flat and fractional-Kelly
stakes.

It does not modify any SQLite database.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BUILD_ID = "NFL_REBUILT_HISTORICAL_REPLAY_FINAL_AUDIT_V1"
VERSION = "v1_independent_recalculation_legacy_comparison_calibration_staking"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_LEGACY_DB = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")
DEFAULT_REBUILT_DB_NAME = "nfl_rebuilt_historical_replay.sqlite"

PREDICTIONS_TABLE = "nfl_rebuilt_weekly_replay_predictions"
SAVED_THRESHOLD_TABLE = "nfl_rebuilt_weekly_replay_threshold_summary"
SAVED_SEASON_TABLE = "nfl_rebuilt_weekly_replay_season_summary"
LEGACY_THRESHOLD_TABLE = "nfl_weekly_power_spread_backtest_threshold_summary"
LEGACY_SEASON_TABLE = "nfl_weekly_power_spread_backtest_season_summary"

DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
DEFAULT_PRIMARY_THRESHOLD = 2.5
DEFAULT_MAXIMUM_DISAGREEMENT = 5.0
DEFAULT_PRICE = -110.0
DEFAULT_STARTING_BANKROLL = 50_000.0
DEFAULT_FLAT_STAKE = 500.0

# Locked legacy benchmark previously reproduced from the old STRUCTURAL_FORM_HFA
# historical warehouse. This is included even if the legacy summary table is
# unavailable so the comparison cannot silently disappear.
LOCKED_LEGACY = {
    "bets": 166,
    "wins": 90,
    "losses": 75,
    "pushes": 1,
    "ats_profit_units": 6.818181818181818,
    "ats_roi": 0.04107338444687842,
    "ats_win_rate": 0.5454545454545454,
    "model_mae_2024": 10.674897,
    "model_mae_2025": 10.794050,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--rebuilt-db", type=Path, default=None)
    parser.add_argument("--legacy-db", type=Path, default=DEFAULT_LEGACY_DB)
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--maximum-disagreement", type=float, default=DEFAULT_MAXIMUM_DISAGREEMENT)
    parser.add_argument("--starting-bankroll", type=float, default=DEFAULT_STARTING_BANKROLL)
    parser.add_argument("--flat-stake", type=float, default=DEFAULT_FLAT_STAKE)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.rebuilt_db = (
        args.rebuilt_db.resolve()
        if args.rebuilt_db is not None
        else (args.project_root / "backtests" / DEFAULT_REBUILT_DB_NAME).resolve()
    )
    args.legacy_db = args.legacy_db.resolve()
    if args.primary_threshold < 0:
        parser.error("--primary-threshold must be nonnegative.")
    if args.maximum_disagreement < args.primary_threshold:
        parser.error("--maximum-disagreement must be >= --primary-threshold.")
    if args.starting_bankroll <= 0:
        parser.error("--starting-bankroll must be positive.")
    if args.flat_stake < 0:
        parser.error("--flat-stake must be nonnegative.")
    return args


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
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def to_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def wilson_lower(wins: int, losses: int, z: float = 1.959963984540054) -> float:
    n = wins + losses
    if n <= 0:
        return np.nan
    p = wins / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (center - margin) / denominator


def max_drawdown(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).fillna(0.0)
    cumulative = series.cumsum()
    if cumulative.empty:
        return 0.0
    return float((cumulative.cummax() - cumulative).max())


def american_profit(stake: float, price: float) -> float:
    if not np.isfinite(stake) or stake <= 0:
        return 0.0
    if not np.isfinite(price) or price == 0:
        price = DEFAULT_PRICE
    if price < 0:
        return stake * 100.0 / abs(price)
    return stake * price / 100.0


def eligible_sample(
    predictions: pd.DataFrame,
    threshold: float,
    maximum: float,
    seasons: Iterable[int] | None = None,
) -> pd.DataFrame:
    sample = predictions.copy()
    if seasons is not None:
        sample = sample[sample["season"].isin(list(seasons))].copy()
    edge = to_numeric(sample, "absolute_spread_difference_points")
    market = to_numeric(sample, "available_market_home_margin")
    sample = sample[
        to_numeric(sample, "week").lt(18)
        & edge.ge(float(threshold))
        & edge.le(float(maximum))
        & market.notna()
        & sample["ats_result"].astype(str).isin(["W", "L", "P"])
    ].copy()
    sort_columns = [column for column in ["season", "week", "game_date", "game_id"] if column in sample.columns]
    return sample.sort_values(sort_columns).reset_index(drop=True)


def summarize_sample(sample: pd.DataFrame) -> dict[str, float | int]:
    wins = int(sample["ats_result"].eq("W").sum())
    losses = int(sample["ats_result"].eq("L").sum())
    pushes = int(sample["ats_result"].eq("P").sum())
    bets = int(len(sample))
    graded = wins + losses
    unit_profit = float(to_numeric(sample, "ats_profit_units", 0.0).fillna(0.0).sum())
    probability = to_numeric(sample, "model_selected_cover_probability")
    outcome = sample["ats_result"].eq("W").astype(float)
    nonpush = sample["ats_result"].isin(["W", "L"])
    probability_nonpush = probability[nonpush]
    outcome_nonpush = outcome[nonpush]
    valid_probability = probability_nonpush.notna()
    if valid_probability.any():
        p = probability_nonpush[valid_probability].clip(1e-9, 1.0 - 1e-9)
        y = outcome_nonpush[valid_probability]
        brier = float(np.mean(np.square(p - y)))
        log_loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    else:
        brier = np.nan
        log_loss = np.nan
    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_win_rate": wins / graded if graded else np.nan,
        "ats_profit_units": unit_profit,
        "ats_roi": unit_profit / bets if bets else np.nan,
        "wilson_lower_95": wilson_lower(wins, losses),
        "average_selected_probability": float(probability.mean()) if bets else np.nan,
        "average_probability_edge": float(to_numeric(sample, "probability_edge").mean()) if bets else np.nan,
        "probability_brier_score": brier,
        "probability_log_loss": log_loss,
        "flat_maximum_drawdown_units": max_drawdown(to_numeric(sample, "ats_profit_units", 0.0)),
    }


def build_threshold_summary(predictions: pd.DataFrame, maximum: float) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for scope, seasons in [("ALL_2022_2025", None), ("LOCKED_2024_2025", [2024, 2025])]:
        for threshold in DEFAULT_THRESHOLDS:
            sample = eligible_sample(predictions, threshold, maximum, seasons)
            rows.append({
                "scope": scope,
                "edge_threshold": threshold,
                "maximum_disagreement": maximum,
                **summarize_sample(sample),
            })
    return pd.DataFrame(rows)


def build_season_summary(predictions: pd.DataFrame, threshold: float, maximum: float) -> pd.DataFrame:
    rows = []
    for season, frame in predictions.groupby("season", sort=True):
        projected = to_numeric(frame, "projected_home_margin")
        actual = to_numeric(frame, "actual_home_margin")
        error = projected - actual
        sample = eligible_sample(frame, threshold, maximum)
        metrics = summarize_sample(sample)
        rows.append({
            "season": int(season),
            "model_mae": float(error.abs().mean()),
            "model_rmse": float(np.sqrt(np.mean(np.square(error.dropna())))),
            **{key: metrics[key] for key in [
                "bets", "wins", "losses", "pushes", "ats_win_rate",
                "ats_profit_units", "ats_roi", "flat_maximum_drawdown_units",
            ]},
        })
    return pd.DataFrame(rows)


def build_edge_bucket_summary(predictions: pd.DataFrame, maximum: float) -> pd.DataFrame:
    holdout = predictions[predictions["season"].isin([2024, 2025])].copy()
    edge = to_numeric(holdout, "absolute_spread_difference_points")
    buckets = [
        ("2.50-2.99", 2.5, 3.0, False),
        ("3.00-3.99", 3.0, 4.0, False),
        ("4.00-5.00", 4.0, maximum, True),
    ]
    rows = []
    for label, lower, upper, inclusive_upper in buckets:
        mask = edge.ge(lower) & (edge.le(upper) if inclusive_upper else edge.lt(upper))
        frame = holdout[mask].copy()
        frame = frame[
            to_numeric(frame, "week").lt(18)
            & to_numeric(frame, "available_market_home_margin").notna()
            & frame["ats_result"].isin(["W", "L", "P"])
        ].copy()
        rows.append({"edge_bucket": label, **summarize_sample(frame)})
    return pd.DataFrame(rows)


def build_probability_bins(predictions: pd.DataFrame, threshold: float, maximum: float) -> pd.DataFrame:
    sample = eligible_sample(predictions, threshold, maximum, [2024, 2025])
    probability = to_numeric(sample, "model_selected_cover_probability")
    sample = sample[probability.notna()].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["probability_bin"] = pd.cut(
        to_numeric(sample, "model_selected_cover_probability"),
        bins=[0.0, 0.55, 0.575, 0.60, 0.625, 0.65, 1.0],
        labels=["<=55%", "55-57.5%", "57.5-60%", "60-62.5%", "62.5-65%", ">65%"],
        include_lowest=True,
        right=True,
    )
    rows = []
    for label, frame in sample.groupby("probability_bin", observed=False):
        if frame.empty:
            continue
        metrics = summarize_sample(frame)
        rows.append({
            "probability_bin": str(label),
            "average_predicted_probability": float(to_numeric(frame, "model_selected_cover_probability").mean()),
            **metrics,
            "calibration_gap": metrics["ats_win_rate"] - float(to_numeric(frame, "model_selected_cover_probability").mean()),
        })
    return pd.DataFrame(rows)


def simulate_staking(
    sample: pd.DataFrame,
    starting_bankroll: float,
    flat_stake_setting: float,
) -> pd.DataFrame:
    sample = sample.copy().reset_index(drop=True)
    price = to_numeric(sample, "selected_price", DEFAULT_PRICE).fillna(DEFAULT_PRICE)
    results = sample["ats_result"].astype(str)

    strategies: list[tuple[str, pd.Series]] = []
    if "flat_recommended_stake" in sample.columns:
        flat_stakes = to_numeric(sample, "flat_recommended_stake", 0.0).fillna(0.0)
        if flat_stakes.sum() <= 0:
            flat_stakes = pd.Series(flat_stake_setting, index=sample.index, dtype=float)
    else:
        flat_stakes = pd.Series(flat_stake_setting, index=sample.index, dtype=float)
    strategies.append(("FLAT_RECORDED_OR_SETTING", flat_stakes))

    if "fractional_kelly_recommended_stake" in sample.columns:
        kelly_stakes = to_numeric(sample, "fractional_kelly_recommended_stake", 0.0).fillna(0.0)
        strategies.append(("RECORDED_FRACTIONAL_KELLY", kelly_stakes))

    rows = []
    for name, stakes in strategies:
        profits = []
        risked = 0.0
        for result, stake, wager_price in zip(results, stakes, price):
            stake = max(float(stake), 0.0)
            risked += stake
            if result == "W":
                profits.append(american_profit(stake, float(wager_price)))
            elif result == "L":
                profits.append(-stake)
            else:
                profits.append(0.0)
        profit_series = pd.Series(profits, dtype=float)
        cumulative = profit_series.cumsum()
        drawdown = float((cumulative.cummax() - cumulative).max()) if not cumulative.empty else 0.0
        net = float(profit_series.sum())
        rows.append({
            "strategy": name,
            "bets": int((stakes > 0).sum()),
            "total_risked": risked,
            "net_profit": net,
            "ending_bankroll": starting_bankroll + net,
            "bankroll_return": net / starting_bankroll,
            "staking_roi": net / risked if risked else np.nan,
            "maximum_drawdown": drawdown,
            "maximum_drawdown_pct_start": drawdown / starting_bankroll,
            "average_stake": float(stakes[stakes > 0].mean()) if (stakes > 0).any() else 0.0,
            "maximum_stake": float(stakes.max()) if len(stakes) else 0.0,
        })
    return pd.DataFrame(rows)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season", "week", "game_id", "projected_home_margin",
        "available_market_home_margin", "absolute_spread_difference_points",
        "selected_side", "ats_result", "ats_profit_units",
        "prediction_uses_market_inputs", "independent_projection_hash",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Prediction table is missing columns: {missing}")

    duplicate_mask = predictions.duplicated(["season", "week", "game_id"], keep=False)
    week_one_rows = int(to_numeric(predictions, "week").eq(1).sum())
    leakage_rows = int(to_numeric(predictions, "prediction_uses_market_inputs", 0.0).fillna(0.0).ne(0).sum())
    missing_hash_rows = int(predictions["independent_projection_hash"].isna().sum())

    form_violation = pd.Series(False, index=predictions.index)
    if "home_rating_through_week" in predictions.columns:
        form_violation |= to_numeric(predictions, "home_rating_through_week").gt(to_numeric(predictions, "week") - 1)
    if "away_rating_through_week" in predictions.columns:
        form_violation |= to_numeric(predictions, "away_rating_through_week").gt(to_numeric(predictions, "week") - 1)

    market_attach_mismatch = pd.Series(False, index=predictions.index)
    if "market_attached_after_prediction_freeze" in predictions.columns:
        market_present = to_numeric(predictions, "available_market_home_margin").notna()
        market_attach_mismatch = market_present & to_numeric(
            predictions, "market_attached_after_prediction_freeze", 0.0
        ).fillna(0.0).ne(1)

    audit = pd.DataFrame([
        {"check": "prediction_rows_positive", "passed": int(len(predictions) > 0), "violations": int(len(predictions) == 0)},
        {"check": "unique_season_week_game", "passed": int(not duplicate_mask.any()), "violations": int(duplicate_mask.sum())},
        {"check": "week1_excluded", "passed": int(week_one_rows == 0), "violations": week_one_rows},
        {"check": "market_not_used_in_prediction", "passed": int(leakage_rows == 0), "violations": leakage_rows},
        {"check": "independent_projection_hash_present", "passed": int(missing_hash_rows == 0), "violations": missing_hash_rows},
        {"check": "rating_through_week_prior_only", "passed": int(not form_violation.any()), "violations": int(form_violation.sum())},
        {"check": "market_attached_after_freeze", "passed": int(not market_attach_mismatch.any()), "violations": int(market_attach_mismatch.sum())},
        {"check": "four_target_seasons_present", "passed": int(set(to_numeric(predictions, "season").dropna().astype(int)) == {2022, 2023, 2024, 2025}), "violations": int(set(to_numeric(predictions, "season").dropna().astype(int)) != {2022, 2023, 2024, 2025})},
    ])
    return audit


def load_legacy_summary(legacy_db: Path, primary_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not legacy_db.exists():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True) as connection:
        thresholds = read_table(connection, LEGACY_THRESHOLD_TABLE) if table_exists(connection, LEGACY_THRESHOLD_TABLE) else pd.DataFrame()
        seasons = read_table(connection, LEGACY_SEASON_TABLE) if table_exists(connection, LEGACY_SEASON_TABLE) else pd.DataFrame()
    if not thresholds.empty:
        scope_column = "sample_scope" if "sample_scope" in thresholds.columns else "scope"
        thresholds = thresholds[
            thresholds[scope_column].astype(str).str.upper().str.contains("HOLDOUT")
            & np.isclose(to_numeric(thresholds, "edge_threshold"), primary_threshold)
        ].copy()
    return thresholds, seasons


def compare_primary(
    new_metrics: dict[str, float | int],
    legacy_threshold: pd.DataFrame,
) -> pd.DataFrame:
    if not legacy_threshold.empty:
        row = legacy_threshold.iloc[0]
        legacy = {
            "bets": int(row.get("bets", LOCKED_LEGACY["bets"])),
            "wins": int(row.get("wins", LOCKED_LEGACY["wins"])),
            "losses": int(row.get("losses", LOCKED_LEGACY["losses"])),
            "pushes": int(row.get("pushes", LOCKED_LEGACY["pushes"])),
            "ats_win_rate": float(row.get("ats_win_rate", LOCKED_LEGACY["ats_win_rate"])),
            "ats_profit_units": float(row.get("ats_profit_units", LOCKED_LEGACY["ats_profit_units"])),
            "ats_roi": float(row.get("ats_roi", LOCKED_LEGACY["ats_roi"])),
        }
    else:
        legacy = {key: LOCKED_LEGACY[key] for key in [
            "bets", "wins", "losses", "pushes", "ats_win_rate", "ats_profit_units", "ats_roi"
        ]}

    rows = []
    for metric in ["bets", "wins", "losses", "pushes", "ats_win_rate", "ats_profit_units", "ats_roi"]:
        old = legacy[metric]
        new = new_metrics[metric]
        rows.append({
            "metric": metric,
            "legacy": old,
            "rebuilt": new,
            "difference": float(new) - float(old),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()

    print("[FINAL_AUDIT] Auditing rebuilt historical weekly replay")
    print(f"[FINAL_AUDIT] Build ID: {BUILD_ID}")
    print(f"[FINAL_AUDIT] Version: {VERSION}")
    print(f"[FINAL_AUDIT] Rebuilt DB: {args.rebuilt_db}")

    if not args.rebuilt_db.exists():
        raise FileNotFoundError(args.rebuilt_db)

    with sqlite3.connect(f"file:{args.rebuilt_db}?mode=ro", uri=True) as connection:
        predictions = read_table(connection, PREDICTIONS_TABLE)
        saved_thresholds = read_table(connection, SAVED_THRESHOLD_TABLE) if table_exists(connection, SAVED_THRESHOLD_TABLE) else pd.DataFrame()
        saved_seasons = read_table(connection, SAVED_SEASON_TABLE) if table_exists(connection, SAVED_SEASON_TABLE) else pd.DataFrame()

    predictions["season"] = to_numeric(predictions, "season").astype("Int64")
    predictions["week"] = to_numeric(predictions, "week").astype("Int64")

    validation = validate_predictions(predictions)
    if validation["passed"].ne(1).any():
        print("\n[FINAL_AUDIT] VALIDATION FAILURES")
        print(validation.to_string(index=False))
        raise RuntimeError("Rebuilt replay validation failed.")

    threshold_summary = build_threshold_summary(predictions, args.maximum_disagreement)
    season_summary = build_season_summary(predictions, args.primary_threshold, args.maximum_disagreement)
    edge_buckets = build_edge_bucket_summary(predictions, args.maximum_disagreement)
    probability_bins = build_probability_bins(predictions, args.primary_threshold, args.maximum_disagreement)

    primary_sample = eligible_sample(
        predictions,
        args.primary_threshold,
        args.maximum_disagreement,
        [2024, 2025],
    )
    primary_metrics = summarize_sample(primary_sample)
    staking = simulate_staking(
        primary_sample,
        args.starting_bankroll,
        args.flat_stake,
    )

    legacy_threshold, legacy_seasons = load_legacy_summary(args.legacy_db, args.primary_threshold)
    comparison = compare_primary(primary_metrics, legacy_threshold)

    # Cross-check independently recomputed summaries against the replay's saved tables.
    saved_reconciliation_rows = []
    if not saved_thresholds.empty:
        saved_scope = saved_thresholds[
            saved_thresholds["scope"].astype(str).eq("LOCKED_2024_2025")
            & np.isclose(to_numeric(saved_thresholds, "edge_threshold"), args.primary_threshold)
        ]
        if len(saved_scope) == 1:
            saved_row = saved_scope.iloc[0]
            for metric in ["bets", "wins", "losses", "pushes", "ats_profit_units", "ats_roi"]:
                recalculated = float(primary_metrics[metric])
                saved_value = float(saved_row[metric])
                saved_reconciliation_rows.append({
                    "metric": metric,
                    "saved_value": saved_value,
                    "recalculated_value": recalculated,
                    "difference": recalculated - saved_value,
                    "passed": int(math.isclose(recalculated, saved_value, rel_tol=1e-9, abs_tol=1e-9)),
                })
    saved_reconciliation = pd.DataFrame(saved_reconciliation_rows)
    if not saved_reconciliation.empty and saved_reconciliation["passed"].ne(1).any():
        raise RuntimeError("Saved replay summary does not reconcile to raw predictions.")

    print("\n" + "=" * 126)
    print("[FINAL_AUDIT] REBUILT MODEL — LOCKED 2024-2025 HOLDOUT")
    print("=" * 126)
    holdout_thresholds = threshold_summary[threshold_summary["scope"].eq("LOCKED_2024_2025")]
    print(holdout_thresholds.to_string(index=False))

    print("\n[FINAL_AUDIT] Primary 2.5-to-5.0 point comparison versus legacy:")
    print(comparison.to_string(index=False))

    print("\n[FINAL_AUDIT] Rebuilt primary-threshold season results:")
    print(season_summary.to_string(index=False))

    print("\n[FINAL_AUDIT] Rebuilt holdout edge buckets:")
    print(edge_buckets.to_string(index=False))

    print("\n[FINAL_AUDIT] Probability calibration bins:")
    print(probability_bins.to_string(index=False) if not probability_bins.empty else "No probability rows available.")

    print("\n[FINAL_AUDIT] Recorded staking simulation:")
    print(staking.to_string(index=False))

    print("\n[FINAL_AUDIT] Leakage and integrity checks:")
    print(validation.to_string(index=False))

    output_dir = args.project_root / "outputs" / "historical_weekly_replay" / "final_audit"
    if not args.no_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        threshold_summary.to_csv(output_dir / "rebuilt_threshold_summary.csv", index=False, encoding="utf-8-sig")
        season_summary.to_csv(output_dir / "rebuilt_season_summary.csv", index=False, encoding="utf-8-sig")
        edge_buckets.to_csv(output_dir / "rebuilt_edge_bucket_summary.csv", index=False, encoding="utf-8-sig")
        probability_bins.to_csv(output_dir / "rebuilt_probability_calibration_bins.csv", index=False, encoding="utf-8-sig")
        staking.to_csv(output_dir / "rebuilt_staking_summary.csv", index=False, encoding="utf-8-sig")
        comparison.to_csv(output_dir / "rebuilt_vs_legacy_primary_comparison.csv", index=False, encoding="utf-8-sig")
        validation.to_csv(output_dir / "rebuilt_integrity_audit.csv", index=False, encoding="utf-8-sig")
        if not saved_reconciliation.empty:
            saved_reconciliation.to_csv(output_dir / "saved_summary_reconciliation.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 126)
    print("[FINAL_AUDIT] Audit completed successfully")
    print(f"[FINAL_AUDIT] Raw replay rows: {len(predictions):,}")
    print(f"[FINAL_AUDIT] Locked primary bets: {int(primary_metrics['bets']):,}")
    print(f"[FINAL_AUDIT] Locked primary ROI: {float(primary_metrics['ats_roi']):.4%}")
    print(f"[FINAL_AUDIT] New structural model historically replayed: YES")
    print(f"[FINAL_AUDIT] Market used in independent projection: NO")
    print(f"[FINAL_AUDIT] Completed in {(dt.datetime.now() - started).total_seconds():.2f} seconds")
    if not args.no_csv:
        print(f"[FINAL_AUDIT] CSV output: {output_dir}")
    print("=" * 126)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[FINAL_AUDIT] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[FINAL_AUDIT] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
