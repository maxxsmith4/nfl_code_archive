"""Build 2022-2025 defensive-front metrics and a 2026 current-roster view.

This script consumes the validated canonical advanced-history table. It does
not reload play-by-play and it does not perform name-based identity matching.
All joins use canonical GSIS player_id values.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sql

SEASON = 2026
HIST_SEASONS = [2022, 2023, 2024, 2025]
BUILD_ID = "NFL_DEFENSIVE_FRONT_2026_CANONICAL_V3"
METRICS_VERSION = "v3_canonical_2022_2025_snap_denominator_shrunk"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

MASTER_TABLE = "nfl_player_master_2026"
ADVANCED_HISTORY_TABLE = "nfl_player_advanced_stats_2022_2025"
OUTPUT_TABLE = "nfl_defensive_front_metrics_2022_2025"
CURRENT_OUTPUT_TABLE = "nfl_defensive_front_metrics_current_roster_2026"

DEF_FRONT_GROUPS = ["EDGE", "DL", "LB", "LB_EDGE"]
SEASON_WEIGHTS = {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40}
PASS_RUSH_SHARE = {"EDGE": 0.58, "DL": 0.48, "LB_EDGE": 0.42, "LB": 0.20}
RUN_DEFENSE_SHARE = {"EDGE": 0.36, "DL": 0.46, "LB_EDGE": 0.42, "LB": 0.52}

FINAL_COLUMNS = [
    "season", "player_id", "player_name", "team", "position", "position_group",
    "defense_event_games", "defense_snaps", "pass_rush_opportunities_proxy",
    "run_defense_opportunities_proxy", "sacks", "half_sacks", "qb_hits", "tfl",
    "tackles", "forced_fumbles", "fumble_recoveries", "pressure_events_proxy",
    "pressure_rate_proxy", "sack_rate_proxy", "qb_hit_rate_proxy", "tfl_rate_proxy",
    "run_stop_proxy", "run_stop_rate_proxy", "front_playmaking_score_raw",
    "pressure_score_0_100", "sack_score_0_100", "run_defense_score_0_100",
    "tackling_score_0_100", "front_composite_score_raw", "front_composite_score",
    "seasons_observed", "available_season_weight", "history_available",
    "current_team", "current_position", "current_position_group", "master_matched",
    "front_metrics_version", "date_imported",
]

NUMERIC_METRIC_COLUMNS = [
    "defense_event_games", "defense_snaps", "pass_rush_opportunities_proxy",
    "run_defense_opportunities_proxy", "sacks", "half_sacks", "qb_hits", "tfl",
    "tackles", "forced_fumbles", "fumble_recoveries", "pressure_events_proxy",
    "pressure_rate_proxy", "sack_rate_proxy", "qb_hit_rate_proxy", "tfl_rate_proxy",
    "run_stop_proxy", "run_stop_rate_proxy", "front_playmaking_score_raw",
    "pressure_score_0_100", "sack_score_0_100", "run_defense_score_0_100",
    "tackling_score_0_100", "front_composite_score_raw", "front_composite_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


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
    frame.columns = [str(c).lower().strip() for c in frame.columns]
    return frame


def clean_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("\u200b", "").replace("\ufeff", "")
    return None if text.lower() in {"", "nan", "none", "null", "na"} else text


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return pd.Series(np.where(den > 0, num / den, 0.0), index=num.index, dtype=float)


def percentile(values: pd.Series, eligible: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(50.0, index=values.index, dtype=float)
    valid = eligible & values.notna()
    if valid.sum() >= 2:
        out.loc[valid] = values.loc[valid].rank(method="average", pct=True) * 100.0
    return out.clip(0.0, 100.0)


def empirical_bayes_rate(events: pd.Series, opportunities: pd.Series, prior_opportunities: float) -> pd.Series:
    ev = pd.to_numeric(events, errors="coerce").fillna(0.0)
    opp = pd.to_numeric(opportunities, errors="coerce").fillna(0.0)
    total_opp = float(opp.sum())
    prior_rate = float(ev.sum() / total_opp) if total_opp > 0 else 0.0
    return (ev + prior_rate * prior_opportunities) / (opp + prior_opportunities)


def load_master(engine: sql.Engine) -> pd.DataFrame:
    master = read_table(engine, MASTER_TABLE)
    for col in ["player_id", "player_name", "team", "position", "position_group"]:
        if col not in master.columns:
            master[col] = None
    master["player_id"] = master["player_id"].map(clean_id)
    master["position_group"] = master["position_group"].astype(str).str.upper().str.strip()
    master = master[master["player_id"].notna()].copy()
    master = master.drop_duplicates("player_id", keep="first")
    return master[["player_id", "player_name", "team", "position", "position_group"]]


def load_history(engine: sql.Engine) -> pd.DataFrame:
    history = read_table(engine, ADVANCED_HISTORY_TABLE)
    for col in ["season", "player_id", "player_name", "team", "position", "position_group"]:
        if col not in history.columns:
            history[col] = None
    history["season"] = pd.to_numeric(history["season"], errors="coerce")
    history["player_id"] = history["player_id"].map(clean_id)
    history["position_group"] = history["position_group"].astype(str).str.upper().str.strip()
    history = history[
        history["season"].isin(HIST_SEASONS)
        & history["player_id"].notna()
        & history["position_group"].isin(DEF_FRONT_GROUPS)
    ].copy()
    history["season"] = history["season"].astype(int)
    history = history.sort_values(["season", "player_id"]).drop_duplicates(["season", "player_id"], keep="first")
    return history


def score_historical(history: pd.DataFrame) -> pd.DataFrame:
    out = history[["season", "player_id", "player_name", "team", "position", "position_group"]].copy()
    out["defense_event_games"] = numeric(history, "games")
    defense_snaps = numeric(history, "defense_snaps")
    total_snaps = numeric(history, "total_snaps")
    out["defense_snaps"] = np.where(defense_snaps > 0, defense_snaps, total_snaps)

    out["sacks"] = numeric(history, "def_sacks")
    out["half_sacks"] = numeric(history, "def_half_sacks")
    out["qb_hits"] = numeric(history, "def_qb_hits")
    out["tfl"] = numeric(history, "def_tfl")
    out["tackles"] = numeric(history, "def_tackles")
    out["forced_fumbles"] = numeric(history, "def_forced_fumbles")
    out["fumble_recoveries"] = numeric(history, "def_fumble_recoveries")

    out["pass_rush_opportunities_proxy"] = np.maximum(
        1.0,
        out["defense_snaps"] * out["position_group"].map(PASS_RUSH_SHARE).fillna(0.25),
    )
    out["run_defense_opportunities_proxy"] = np.maximum(
        1.0,
        out["defense_snaps"] * out["position_group"].map(RUN_DEFENSE_SHARE).fillna(0.40),
    )

    sack_equivalents = out["sacks"] + 0.5 * out["half_sacks"]
    non_sack_hits = np.maximum(0.0, out["qb_hits"] - sack_equivalents)
    out["pressure_events_proxy"] = sack_equivalents + 0.35 * non_sack_hits
    out["run_stop_proxy"] = out["tfl"] + 0.12 * out["tackles"] + 0.50 * out["forced_fumbles"]

    scored_parts: list[pd.DataFrame] = []
    for (_, _), group in out.groupby(["season", "position_group"], sort=True):
        group = group.copy()
        eligible = group["defense_snaps"] >= 75
        group["pressure_rate_proxy"] = empirical_bayes_rate(
            group["pressure_events_proxy"], group["pass_rush_opportunities_proxy"], 150.0
        )
        group["sack_rate_proxy"] = empirical_bayes_rate(
            sack_equivalents.loc[group.index], group["pass_rush_opportunities_proxy"], 175.0
        )
        group["qb_hit_rate_proxy"] = empirical_bayes_rate(
            group["qb_hits"], group["pass_rush_opportunities_proxy"], 150.0
        )
        group["tfl_rate_proxy"] = empirical_bayes_rate(
            group["tfl"], group["run_defense_opportunities_proxy"], 175.0
        )
        group["run_stop_rate_proxy"] = empirical_bayes_rate(
            group["run_stop_proxy"], group["run_defense_opportunities_proxy"], 175.0
        )
        tackle_rate = empirical_bayes_rate(group["tackles"], group["defense_snaps"].clip(lower=1.0), 200.0)

        group["pressure_score_0_100"] = percentile(group["pressure_rate_proxy"], eligible)
        group["sack_score_0_100"] = percentile(group["sack_rate_proxy"], eligible)
        group["run_defense_score_0_100"] = percentile(group["run_stop_rate_proxy"], eligible)
        group["tackling_score_0_100"] = percentile(tackle_rate, eligible)
        group["front_playmaking_score_raw"] = (
            100.0 * group["pressure_rate_proxy"]
            + 75.0 * group["sack_rate_proxy"]
            + 50.0 * group["run_stop_rate_proxy"]
        )

        pg = str(group["position_group"].iloc[0])
        weights = {
            "EDGE": (0.45, 0.25, 0.20, 0.10),
            "DL": (0.35, 0.15, 0.40, 0.10),
            "LB_EDGE": (0.30, 0.20, 0.30, 0.20),
            "LB": (0.15, 0.10, 0.45, 0.30),
        }.get(pg, (0.25, 0.15, 0.35, 0.25))
        group["front_composite_score_raw"] = (
            weights[0] * group["pressure_score_0_100"]
            + weights[1] * group["sack_score_0_100"]
            + weights[2] * group["run_defense_score_0_100"]
            + weights[3] * group["tackling_score_0_100"]
        )
        confidence = group["defense_snaps"] / (group["defense_snaps"] + 225.0)
        group["front_composite_score"] = np.where(
            group["defense_snaps"] > 0,
            50.0 + confidence * (group["front_composite_score_raw"] - 50.0),
            0.0,
        ).clip(0.0, 100.0)
        scored_parts.append(group)

    scored = pd.concat(scored_parts, ignore_index=True, sort=False) if scored_parts else pd.DataFrame(columns=FINAL_COLUMNS)
    scored["seasons_observed"] = 1
    scored["available_season_weight"] = scored["season"].map(SEASON_WEIGHTS).fillna(0.0)
    # A handful of snaps is not a model-usable historical sample.
    scored["history_available"] = (scored["defense_snaps"] >= 75).astype(int)
    scored["current_team"] = None
    scored["current_position"] = None
    scored["current_position_group"] = None
    scored["master_matched"] = False
    scored["front_metrics_version"] = METRICS_VERSION
    scored["date_imported"] = pd.Timestamp(dt.date.today())
    for col in FINAL_COLUMNS:
        if col not in scored.columns:
            scored[col] = None
    return scored[FINAL_COLUMNS]


def weighted_rollup(historical: pd.DataFrame) -> pd.DataFrame:
    if historical.empty:
        return pd.DataFrame(columns=["player_id"])
    hist = historical.copy()
    hist["base_weight"] = hist["season"].map(SEASON_WEIGHTS).fillna(0.0)
    hist["observed_weight"] = np.where(hist["history_available"] > 0, hist["base_weight"], 0.0)
    weighted_columns = [col for col in NUMERIC_METRIC_COLUMNS if col in hist.columns]
    hist["observed_season"] = (hist["observed_weight"] > 0).astype(int)
    weighted = hist[["player_id", "season", "observed_weight", "observed_season", *weighted_columns]].copy()
    for col in weighted_columns:
        weighted[f"weighted__{col}"] = pd.to_numeric(weighted[col], errors="coerce").fillna(0.0) * weighted["observed_weight"]

    agg_spec: dict[str, tuple[str, str]] = {
        "seasons_observed": ("observed_season", "sum"),
        "available_season_weight": ("observed_weight", "sum"),
    }
    for col in weighted_columns:
        agg_spec[f"sum__{col}"] = (f"weighted__{col}", "sum")
    rolled = weighted.groupby("player_id", dropna=False).agg(**agg_spec).reset_index()
    for col in weighted_columns:
        rolled[col] = np.where(
            rolled["available_season_weight"] > 0,
            rolled[f"sum__{col}"] / rolled["available_season_weight"],
            0.0,
        )
        rolled = rolled.drop(columns=[f"sum__{col}"])
    rolled["history_available"] = (rolled["available_season_weight"] > 0).astype(int)
    return rolled


def build_current(master: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    current = master[master["position_group"].isin(DEF_FRONT_GROUPS)].copy()
    current = current.rename(columns={
        "player_name": "current_player_name", "team": "current_team",
        "position": "current_position", "position_group": "current_position_group",
    })
    rolled = weighted_rollup(historical)
    out = current.merge(rolled, on="player_id", how="left", validate="one_to_one")
    out["season"] = SEASON
    out["player_name"] = out["current_player_name"]
    out["team"] = out["current_team"]
    out["position"] = out["current_position"]
    out["position_group"] = out["current_position_group"]
    out["master_matched"] = True

    for col in [*NUMERIC_METRIC_COLUMNS, "seasons_observed", "available_season_weight", "history_available"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["history_available"] = (out["history_available"] > 0).astype(int)
    out["front_metrics_version"] = METRICS_VERSION
    out["date_imported"] = pd.Timestamp(dt.date.today())
    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    final = out[FINAL_COLUMNS].copy()
    if final["player_id"].duplicated().any():
        raise RuntimeError("Current defensive-front table contains duplicate player_id values.")
    return final


def save_frame(frame: pd.DataFrame, engine: sql.Engine, table_name: str, csv_path: Path | None) -> None:
    frame.to_sql(table_name, con=engine, if_exists="replace", index=False)
    if csv_path is not None:
        frame.to_csv(csv_path, index=False)


def main() -> int:
    args = parse_args()
    output_dir = args.project_root / "outputs"
    log_dir = args.project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[FRONT] Building canonical defensive-front metrics")
    print(f"[FRONT] Build ID: {BUILD_ID}")
    print(f"[FRONT] Historical seasons: {HIST_SEASONS}")
    print(f"[FRONT] Database: {args.db_path}")

    engine = get_engine(args.db_path)
    master = load_master(engine)
    history = load_history(engine)
    historical = score_historical(history)
    current = build_current(master, historical)

    seasons = sorted(set(pd.to_numeric(historical["season"], errors="coerce").dropna().astype(int)))
    if seasons != HIST_SEASONS:
        raise RuntimeError(f"Defensive-front history seasons mismatch. Expected {HIST_SEASONS}, found {seasons}")

    history_csv = None if args.no_csv else output_dir / "nfl_defensive_front_metrics_2022_2025.csv"
    current_csv = None if args.no_csv else output_dir / "nfl_defensive_front_metrics_current_roster_2026.csv"
    summary_csv = None if args.no_csv else output_dir / "nfl_defensive_front_metrics_summary_2026.csv"
    save_frame(historical, engine, OUTPUT_TABLE, history_csv)
    save_frame(current, engine, CURRENT_OUTPUT_TABLE, current_csv)

    summary = current.groupby("position_group", dropna=False).agg(
        players=("player_id", "count"),
        with_history=("history_available", "sum"),
        avg_defense_snaps=("defense_snaps", "mean"),
        avg_front_score=("front_composite_score", "mean"),
        max_front_score=("front_composite_score", "max"),
    ).reset_index()
    if summary_csv is not None:
        summary.to_csv(summary_csv, index=False)

    with_history = int(current["history_available"].sum())
    log_path = log_dir / "load_nfl_defensive_front_metrics.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Run timestamp: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Build ID: {BUILD_ID}\n")
        handle.write(f"Historical rows: {len(historical)}\n")
        handle.write(f"Current rows: {len(current)}\n")
        handle.write(f"Current with history: {with_history}\n")

    print(f"[FRONT] Historical rows: {len(historical):,}")
    print(f"[FRONT] Current front players: {len(current):,}")
    print(f"[FRONT] Current players with history: {with_history:,}/{len(current):,}")
    print(f"[FRONT] Saved table: {CURRENT_OUTPUT_TABLE}")
    print(f"[FRONT] Log saved: {log_path}")
    print("\n[FRONT] Position summary:")
    print(summary.to_string(index=False))
    print("\n[FRONT] Top 25 current front ratings:")
    cols = ["player_name", "team", "position", "position_group", "defense_snaps", "pressure_rate_proxy", "run_stop_rate_proxy", "front_composite_score"]
    print(current.sort_values("front_composite_score", ascending=False).head(25)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
