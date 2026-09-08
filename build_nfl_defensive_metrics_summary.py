"""Combine canonical front and coverage metrics for current 2026 defenders."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sql

SEASON = 2026
BUILD_ID = "NFL_DEFENSIVE_SUMMARY_2026_CANONICAL_V3"
METRICS_VERSION = "v3_available_component_front_coverage_blend"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

MASTER_TABLE = "nfl_player_master_2026"
FRONT_TABLE = "nfl_defensive_front_metrics_current_roster_2026"
COVERAGE_TABLE = "nfl_coverage_metrics_current_roster_2026"
OUTPUT_TABLE = "nfl_defensive_player_metrics_current_roster_2026"

DEFENSIVE_GROUPS = ["EDGE", "DL", "LB", "LB_EDGE", "DB"]
FRONT_BASE_WEIGHT = {"EDGE": 0.90, "DL": 0.85, "LB_EDGE": 0.60, "LB": 0.50, "DB": 0.10}
COVERAGE_BASE_WEIGHT = {"EDGE": 0.10, "DL": 0.15, "LB_EDGE": 0.40, "LB": 0.50, "DB": 0.90}

FINAL_COLUMNS = [
    "season", "player_id", "player_name", "team", "position", "position_group",
    "front_event_games", "coverage_event_games", "defensive_event_games",
    "front_history_available", "coverage_history_available", "defensive_history_available",
    "front_component_weight", "coverage_component_weight", "defense_snaps",
    "front_composite_score", "front_composite_score_raw", "pressure_score_0_100",
    "sack_score_0_100", "run_defense_score_0_100", "pressure_rate_proxy",
    "sack_rate_proxy", "run_stop_rate_proxy", "pressure_events_proxy", "sacks",
    "qb_hits", "tfl", "coverage_composite_score", "coverage_composite_score_raw",
    "coverage_score_0_100", "ball_hawk_score_0_100", "tackling_score_0_100",
    "coverage_playmaking_rate", "interceptions", "pass_defended",
    "defensive_advanced_score_raw", "defensive_advanced_score",
    "defensive_metrics_version", "date_imported",
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


def load_master(engine: sql.Engine) -> pd.DataFrame:
    master = read_table(engine, MASTER_TABLE)
    for col in ["player_id", "player_name", "team", "position", "position_group"]:
        if col not in master.columns:
            master[col] = None
    master["player_id"] = master["player_id"].map(clean_id)
    master["position_group"] = master["position_group"].astype(str).str.upper().str.strip()
    master = master[
        master["player_id"].notna() & master["position_group"].isin(DEFENSIVE_GROUPS)
    ].copy()
    master = master.drop_duplicates("player_id", keep="first")
    if master["player_id"].duplicated().any():
        raise RuntimeError("Duplicate defensive player_id values in master.")
    return master[["player_id", "player_name", "team", "position", "position_group"]]


def prepare_front(front: pd.DataFrame) -> pd.DataFrame:
    frame = front.copy()
    if "player_id" not in frame.columns:
        frame["player_id"] = None
    frame["player_id"] = frame["player_id"].map(clean_id)
    mapping = {
        "defense_event_games": "front_event_games",
        "history_available": "front_history_available",
        "defense_snaps": "front_defense_snaps",
    }
    frame = frame.rename(columns=mapping)
    keep = [
        "player_id", "front_event_games", "front_history_available", "front_defense_snaps",
        "front_composite_score", "front_composite_score_raw", "pressure_score_0_100",
        "sack_score_0_100", "run_defense_score_0_100", "pressure_rate_proxy",
        "sack_rate_proxy", "run_stop_rate_proxy", "pressure_events_proxy", "sacks",
        "qb_hits", "tfl",
    ]
    for col in keep:
        if col not in frame.columns:
            frame[col] = 0.0 if col != "player_id" else None
    frame = frame[keep].drop_duplicates("player_id", keep="first")
    return frame


def prepare_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    frame = coverage.copy()
    if "player_id" not in frame.columns:
        frame["player_id"] = None
    frame["player_id"] = frame["player_id"].map(clean_id)
    mapping = {
        "coverage_event_games": "coverage_event_games",
        "history_available": "coverage_history_available",
        "defense_snaps": "coverage_defense_snaps",
    }
    frame = frame.rename(columns=mapping)
    keep = [
        "player_id", "coverage_event_games", "coverage_history_available", "coverage_defense_snaps",
        "coverage_composite_score", "coverage_composite_score_raw", "coverage_score_0_100",
        "ball_hawk_score_0_100", "tackling_score_0_100", "coverage_playmaking_rate",
        "interceptions", "pass_defended",
    ]
    for col in keep:
        if col not in frame.columns:
            frame[col] = 0.0 if col != "player_id" else None
    frame = frame[keep].drop_duplicates("player_id", keep="first")
    return frame


def build_summary(master: pd.DataFrame, front: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    out = master.merge(prepare_front(front), on="player_id", how="left", validate="one_to_one")
    out = out.merge(prepare_coverage(coverage), on="player_id", how="left", validate="one_to_one")

    numeric_cols = [
        "front_event_games", "coverage_event_games", "front_history_available",
        "coverage_history_available", "front_defense_snaps", "coverage_defense_snaps",
        "front_composite_score", "front_composite_score_raw", "pressure_score_0_100",
        "sack_score_0_100", "run_defense_score_0_100", "pressure_rate_proxy",
        "sack_rate_proxy", "run_stop_rate_proxy", "pressure_events_proxy", "sacks",
        "qb_hits", "tfl", "coverage_composite_score", "coverage_composite_score_raw",
        "coverage_score_0_100", "ball_hawk_score_0_100", "tackling_score_0_100",
        "coverage_playmaking_rate", "interceptions", "pass_defended",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["front_history_available"] = (out["front_history_available"] > 0).astype(int)
    out["coverage_history_available"] = (out["coverage_history_available"] > 0).astype(int)
    out["defensive_history_available"] = (
        (out["front_history_available"] > 0) | (out["coverage_history_available"] > 0)
    ).astype(int)
    out["defensive_event_games"] = np.maximum(out["front_event_games"], out["coverage_event_games"])
    out["defense_snaps"] = np.maximum(out["front_defense_snaps"], out["coverage_defense_snaps"])

    front_base = out["position_group"].map(FRONT_BASE_WEIGHT).fillna(0.0)
    coverage_base = out["position_group"].map(COVERAGE_BASE_WEIGHT).fillna(0.0)
    front_effective = front_base * out["front_history_available"]
    coverage_effective = coverage_base * out["coverage_history_available"]
    total_effective = front_effective + coverage_effective
    out["front_component_weight"] = np.where(total_effective > 0, front_effective / total_effective, 0.0)
    out["coverage_component_weight"] = np.where(total_effective > 0, coverage_effective / total_effective, 0.0)

    out["defensive_advanced_score_raw"] = (
        out["front_component_weight"] * out["front_composite_score"]
        + out["coverage_component_weight"] * out["coverage_composite_score"]
    )
    out["defensive_advanced_score"] = np.where(
        out["defensive_history_available"] > 0,
        out["defensive_advanced_score_raw"].clip(0.0, 100.0),
        0.0,
    )

    out["season"] = SEASON
    out["defensive_metrics_version"] = METRICS_VERSION
    out["date_imported"] = pd.Timestamp(dt.date.today())
    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    final = out[FINAL_COLUMNS].copy()

    if final["player_id"].duplicated().any():
        raise RuntimeError("Defensive summary contains duplicate player_id values.")
    if len(final) != len(master):
        raise RuntimeError(f"Defensive summary row mismatch: expected {len(master)}, found {len(final)}")
    scores = pd.to_numeric(final["defensive_advanced_score"], errors="coerce")
    if ((scores < 0) | (scores > 100)).any():
        raise RuntimeError("Defensive score outside 0-100 range.")
    history_without_score = (final["defensive_history_available"] > 0) & (scores <= 0)
    if history_without_score.any():
        examples = final.loc[history_without_score, ["player_id", "player_name", "position_group"]].head(10)
        raise RuntimeError(f"Defenders with history received non-positive scores:\n{examples.to_string(index=False)}")
    return final


def main() -> int:
    args = parse_args()
    output_dir = args.project_root / "outputs"
    log_dir = args.project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[DEF_SUMMARY] Building canonical defensive summary")
    print(f"[DEF_SUMMARY] Build ID: {BUILD_ID}")
    print(f"[DEF_SUMMARY] Database: {args.db_path}")

    engine = get_engine(args.db_path)
    master = load_master(engine)
    front = read_table(engine, FRONT_TABLE)
    coverage = read_table(engine, COVERAGE_TABLE)
    summary = build_summary(master, front, coverage)

    summary.to_sql(OUTPUT_TABLE, con=engine, if_exists="replace", index=False)
    output_csv = output_dir / "nfl_defensive_player_metrics_current_roster_2026.csv"
    position_csv = output_dir / "nfl_defensive_player_metrics_position_summary_2026.csv"
    top_csv = output_dir / "nfl_defensive_player_metrics_top_players_2026.csv"
    if not args.no_csv:
        summary.to_csv(output_csv, index=False)

    position_summary = summary.groupby("position_group", dropna=False).agg(
        players=("player_id", "count"),
        with_history=("defensive_history_available", "sum"),
        front_available=("front_history_available", "sum"),
        coverage_available=("coverage_history_available", "sum"),
        avg_defensive_score=("defensive_advanced_score", "mean"),
        max_defensive_score=("defensive_advanced_score", "max"),
    ).reset_index()
    top_players = summary[summary["defensive_history_available"] > 0].sort_values(
        "defensive_advanced_score", ascending=False
    ).head(150)
    if not args.no_csv:
        position_summary.to_csv(position_csv, index=False)
        top_players.to_csv(top_csv, index=False)

    with_history = int(summary["defensive_history_available"].sum())
    log_path = log_dir / "build_nfl_defensive_metrics_summary.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Run timestamp: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Build ID: {BUILD_ID}\n")
        handle.write(f"Master defensive rows: {len(master)}\n")
        handle.write(f"Front rows: {len(front)}\n")
        handle.write(f"Coverage rows: {len(coverage)}\n")
        handle.write(f"Output rows: {len(summary)}\n")
        handle.write(f"With defensive history: {with_history}\n")

    print(f"[DEF_SUMMARY] Output rows: {len(summary):,}")
    print(f"[DEF_SUMMARY] Players with defensive history: {with_history:,}/{len(summary):,}")
    print(f"[DEF_SUMMARY] Saved table: {OUTPUT_TABLE}")
    print(f"[DEF_SUMMARY] Log saved: {log_path}")
    print("\n[DEF_SUMMARY] Position summary:")
    print(position_summary.to_string(index=False))
    print("\n[DEF_SUMMARY] Top 30 defensive players:")
    cols = ["player_name", "team", "position", "position_group", "defense_snaps", "front_composite_score", "coverage_composite_score", "defensive_advanced_score"]
    print(top_players.head(30)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
