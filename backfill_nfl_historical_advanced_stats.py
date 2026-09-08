#!/usr/bin/env python
"""Backfill rolling historical advanced-player inputs for the canonical NFL model.

This Stage 2 historical-reconstruction script imports the approved production
`nfl_player_advanced_stats.py`, builds the shared 2018-2024 raw player-season
history once, and writes only the legal four-year window to each isolated
season database:

    2022 <- 2018-2021
    2023 <- 2019-2022
    2024 <- 2020-2023
    2025 <- 2021-2024

The production database is never written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import sqlite3
import sys
import traceback
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

BUILD_ID = "NFL_HISTORICAL_ADVANCED_STATS_CANONICAL_V1"
VERSION = "v1_reuse_v3_2_final_rolling_four_year_windows"
EXPECTED_BUILD_ID = "NFL_ADVANCED_STATS_2026_V3_2_FINAL"
EXPECTED_VERSION = "v3_2_final_snap_participation_identity"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_TARGET_SEASONS = (2022, 2023, 2024, 2025)
HISTORY_YEARS = 4

CONTEXT_TABLE = "nfl_historical_reconstruction_context"
MASTER_TABLE = "nfl_player_master_target"
CROSSWALK_TABLE = "nfl_player_crosswalk"
CACHE_DB_NAME = "nfl_historical_advanced_cache.sqlite"
CACHE_RAW_TABLE = "nfl_player_advanced_stats_raw_union"
CACHE_SNAP_TABLE = "nfl_player_advanced_stats_snap_id_audit_union"
CACHE_AUDIT_TABLE = "nfl_historical_advanced_cache_audit"
HISTORY_TABLE = "nfl_player_advanced_stats_history"
CURRENT_TABLE = "nfl_player_advanced_stats_current_roster_target"
SNAP_TABLE = "nfl_player_advanced_stats_snap_id_audit"
READINESS_TABLE = "nfl_historical_advanced_stats_readiness_audit"

POSITION_PROFILES = {
    "QB": (0.10, 0.20, 0.30, 0.40),
    "RB": (0.02, 0.08, 0.25, 0.65),
    "WR_TE": (0.05, 0.15, 0.30, 0.50),
    "OL": (0.15, 0.20, 0.30, 0.35),
    "EDGE": (0.10, 0.15, 0.30, 0.45),
    "DL": (0.10, 0.15, 0.30, 0.45),
    "LB_EDGE": (0.10, 0.15, 0.30, 0.45),
    "LB": (0.10, 0.15, 0.30, 0.45),
    "DB": (0.05, 0.15, 0.30, 0.50),
    "ST": (0.10, 0.20, 0.30, 0.40),
    "OTHER": (0.10, 0.20, 0.30, 0.40),
}
DEFAULT_PROFILE = (0.10, 0.20, 0.30, 0.40)


def parse_seasons(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not result:
        raise argparse.ArgumentTypeError("At least one target season is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--canonical-script", type=Path, default=None)
    parser.add_argument("--target-seasons", type=parse_seasons, default=DEFAULT_TARGET_SEASONS)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--allow-build-id-mismatch", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.canonical_script = (
        args.canonical_script.resolve()
        if args.canonical_script is not None
        else args.project_root / "nfl_player_advanced_stats.py"
    )
    return args


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone() is not None


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not table_exists(conn, table):
        raise RuntimeError(f"Missing required table: {table}")
    frame = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    frame.columns = [str(c).lower().strip() for c in frame.columns]
    return frame


def add_column(conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
    columns = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
    if column not in columns:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {sql_type}')


def frame_hash(frame: pd.DataFrame) -> str:
    columns = [c for c in ["season", "player_id", "team", "position", "total_snaps"] if c in frame.columns]
    work = frame[columns].copy()
    for column in columns:
        work[column] = work[column].astype(str)
    return hashlib.sha256(work.sort_values(columns).to_csv(index=False).encode()).hexdigest()


def import_canonical(path: Path, allow_mismatch: bool) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("canonical_advanced", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_id = getattr(module, "ADVANCED_BUILD_ID", getattr(module, "BUILD_ID", None))
    version = getattr(module, "ADVANCED_VERSION", getattr(module, "VERSION", None))
    if not allow_mismatch and (build_id != EXPECTED_BUILD_ID or version != EXPECTED_VERSION):
        raise RuntimeError(
            f"Canonical advanced file mismatch: build={build_id!r}, version={version!r}; "
            f"expected {EXPECTED_BUILD_ID} / {EXPECTED_VERSION}"
        )
    required = [
        "get_engine", "load_master", "load_weekly_data", "load_snap_counts",
        "canonicalize_snap_ids", "load_ngs_data", "load_pbp_data",
        "build_advanced_stats", "apply_master_identity", "build_current_roster_view",
    ]
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"Canonical file missing functions: {missing}")
    return module


def configure(module: ModuleType, target: int, history: list[int], db: Path, root: Path) -> None:
    module.SEASON = target
    module.HIST_SEASONS = list(history)
    if hasattr(module, "HISTORICAL_SEASONS"):
        module.HISTORICAL_SEASONS = tuple(history)
    if hasattr(module, "RECENT_SEASON"):
        module.RECENT_SEASON = history[-1]
    module.PROJECT_ROOT = root
    module.DB_PATH = str(db)
    module.MASTER_TABLE = MASTER_TABLE
    if hasattr(module, "CROSSWALK_TABLE"):
        module.CROSSWALK_TABLE = CROSSWALK_TABLE
    module.OUTPUT_DIR = root / "outputs" / "historical_advanced" / str(target)
    module.LOG_DIR = root / "logs"
    module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    module.LOG_DIR.mkdir(parents=True, exist_ok=True)
    module.POSITION_SEASON_WEIGHTS = {
        group: {season: profile[i] for i, season in enumerate(history)}
        for group, profile in POSITION_PROFILES.items()
    }
    module.DEFAULT_SEASON_WEIGHTS = {
        season: DEFAULT_PROFILE[i] for i, season in enumerate(history)
    }


def load_mapping(module: ModuleType, engine: Any) -> pd.DataFrame:
    if callable(getattr(module, "load_crosswalk_pfr_map", None)):
        return module.load_crosswalk_pfr_map(engine)
    if callable(getattr(module, "load_nflverse_player_ids", None)) and callable(
        getattr(module, "build_pfr_to_gsis_map", None)
    ):
        return module.build_pfr_to_gsis_map(module.load_nflverse_player_ids())
    raise RuntimeError("Canonical file exposes no supported PFR-to-GSIS mapping loader")


def copy_crosswalk_to_cache(root: Path, cache_db: Path) -> None:
    candidates = sorted((root / "backtests").glob("20*.sqlite"))
    if not candidates:
        raise RuntimeError("No isolated season DBs found. Run prepare_nfl_historical_reconstruction.py first")
    with sqlite3.connect(candidates[-1]) as source, sqlite3.connect(cache_db) as target:
        if not table_exists(source, CROSSWALK_TABLE):
            raise RuntimeError(f"{candidates[-1]} is missing {CROSSWALK_TABLE}")
        read_table(source, CROSSWALK_TABLE).to_sql(CROSSWALK_TABLE, target, if_exists="replace", index=False)


def build_cache(module: ModuleType, root: Path, canonical_script: Path, cache_db: Path, union: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_db.exists():
        cache_db.unlink()
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    copy_crosswalk_to_cache(root, cache_db)
    configure(module, union[-1] + 1, union[-4:], cache_db, root)
    module.HIST_SEASONS = union
    if hasattr(module, "HISTORICAL_SEASONS"):
        module.HISTORICAL_SEASONS = tuple(union)
    engine = module.get_engine()

    print(f"[HIST_ADV] Loading shared canonical source window: {union}")
    weekly = module.load_weekly_data(union)
    raw_snaps = module.load_snap_counts(union)
    mapping = load_mapping(module, engine)
    snaps, snap_audit = module.canonicalize_snap_ids(raw_snaps, mapping)
    ngs = module.load_ngs_data(union)
    pbp = module.load_pbp_data(union)
    raw = pd.DataFrame(module.build_advanced_stats(weekly, snaps, ngs, pbp)).copy()
    raw.columns = [str(c).lower().strip() for c in raw.columns]
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    raw = raw[raw["season"].isin(union)].copy()
    raw["season"] = raw["season"].astype(int)
    found = sorted(raw["season"].unique())
    missing = sorted(set(union) - set(found))
    if missing:
        raise RuntimeError(f"Canonical advanced cache missing seasons {missing}; found {found}")
    raw["historical_cache_build_id"] = BUILD_ID
    raw["historical_cache_version"] = VERSION
    snap_audit = pd.DataFrame(snap_audit).copy()
    snap_audit.columns = [str(c).lower().strip() for c in snap_audit.columns]
    snap_audit["historical_cache_build_id"] = BUILD_ID
    snap_audit["historical_cache_version"] = VERSION
    audit = pd.DataFrame([{
        "union_seasons": ",".join(map(str, union)),
        "advanced_rows": len(raw),
        "advanced_players": raw["player_id"].nunique() if "player_id" in raw.columns else None,
        "canonical_script": str(canonical_script),
        "canonical_script_sha256": hashlib.sha256(canonical_script.read_bytes()).hexdigest(),
        "canonical_build_id": getattr(module, "ADVANCED_BUILD_ID", None),
        "canonical_version": getattr(module, "ADVANCED_VERSION", None),
        "raw_hash": frame_hash(raw),
        "build_id": BUILD_ID,
        "version": VERSION,
        "created_at": now(),
    }])
    with sqlite3.connect(cache_db) as conn:
        raw.to_sql(CACHE_RAW_TABLE, conn, if_exists="replace", index=False)
        snap_audit.to_sql(CACHE_SNAP_TABLE, conn, if_exists="replace", index=False)
        audit.to_sql(CACHE_AUDIT_TABLE, conn, if_exists="replace", index=False)
    return raw, snap_audit


def load_cache(cache_db: Path, union: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(cache_db) as conn:
        raw = read_table(conn, CACHE_RAW_TABLE)
        snap = read_table(conn, CACHE_SNAP_TABLE)
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")
    missing = sorted(set(union) - set(raw["season"].dropna().astype(int).unique()))
    if missing:
        raise RuntimeError(f"Existing cache missing {missing}; rerun with --rebuild-cache")
    return raw, snap


def validate_context(db: Path, target: int) -> tuple[pd.DataFrame, int]:
    if not db.exists():
        raise FileNotFoundError(db)
    with sqlite3.connect(db) as conn:
        context = read_table(conn, CONTEXT_TABLE)
        master = read_table(conn, MASTER_TABLE)
        if not table_exists(conn, CROSSWALK_TABLE):
            raise RuntimeError(f"{db} missing {CROSSWALK_TABLE}")
    if len(context) != 1 or int(context.iloc[0]["target_season"]) != target:
        raise RuntimeError(f"Invalid historical context in {db}")
    if int(context.iloc[0]["personnel_context_strict_flag"]) != 1:
        raise RuntimeError(f"{target} personnel context is not strict")
    return context, len(master)


def update_context(conn: sqlite3.Connection, rows: int, current_rows: int, history_hash: str) -> None:
    additions = {
        "advanced_history_ready_flag": "INTEGER",
        "advanced_history_rows": "INTEGER",
        "advanced_current_roster_rows": "INTEGER",
        "advanced_history_hash": "TEXT",
        "advanced_history_build_id": "TEXT",
        "advanced_history_version": "TEXT",
        "advanced_history_completed_at": "TEXT",
        "structural_inputs_pending": "TEXT",
    }
    for column, sql_type in additions.items():
        add_column(conn, CONTEXT_TABLE, column, sql_type)
    conn.execute(
        f"""UPDATE {CONTEXT_TABLE}
        SET missing_required_history_seasons='',
            advanced_history_ready_flag=1,
            advanced_history_rows=?,
            advanced_current_roster_rows=?,
            advanced_history_hash=?,
            advanced_history_build_id=?,
            advanced_history_version=?,
            advanced_history_completed_at=?,
            structural_inputs_ready_flag=0,
            structural_inputs_pending=?""",
        (
            rows, current_rows, history_hash, BUILD_ID, VERSION, now(),
            "historical_qb_ratings,historical_defensive_metrics,historical_depth_chart,"
            "historical_ol_continuity,historical_team_units,historical_team_strength",
        ),
    )


def build_target(module: ModuleType, root: Path, target: int, raw_union: pd.DataFrame, snap_union: pd.DataFrame, no_csv: bool) -> dict[str, Any]:
    history_seasons = list(range(target - HISTORY_YEARS, target))
    db = root / "backtests" / f"{target}.sqlite"
    context, master_rows = validate_context(db, target)
    configure(module, target, history_seasons, db, root)
    master = module.load_master(module.get_engine())
    window = raw_union[pd.to_numeric(raw_union["season"], errors="coerce").isin(history_seasons)].copy()
    found = sorted(pd.to_numeric(window["season"], errors="coerce").dropna().astype(int).unique())
    missing = sorted(set(history_seasons) - set(found))
    if missing:
        raise RuntimeError(f"{target} rolling window missing seasons {missing}")
    canonical_input = window.drop(
        columns=["historical_cache_build_id", "historical_cache_version"],
        errors="ignore",
    )
    history = pd.DataFrame(module.apply_master_identity(canonical_input, master)).copy()
    current = pd.DataFrame(module.build_current_roster_view(history, master)).copy()
    history.columns = [str(c).lower().strip() for c in history.columns]
    current.columns = [str(c).lower().strip() for c in current.columns]
    if len(current) != master_rows:
        raise RuntimeError(f"{target} current advanced/master mismatch: {len(current)} vs {master_rows}")
    if "player_id" not in current.columns or current["player_id"].nunique() != len(current):
        raise RuntimeError(f"{target} current advanced view has invalid player IDs")
    history["historical_target_season"] = target
    history["historical_history_start_season"] = history_seasons[0]
    history["historical_history_end_season"] = history_seasons[-1]
    history["historical_wrapper_build_id"] = BUILD_ID
    history["historical_wrapper_version"] = VERSION
    current["historical_target_season"] = target
    current["historical_recent_season"] = history_seasons[-1]
    current["historical_history_start_season"] = history_seasons[0]
    current["historical_history_end_season"] = history_seasons[-1]
    current["historical_wrapper_build_id"] = BUILD_ID
    current["historical_wrapper_version"] = VERSION
    snap = snap_union.copy()
    snap["historical_target_season"] = target
    history_hash = frame_hash(history)
    history_mirror = f"nfl_player_advanced_stats_{history_seasons[0]}_{history_seasons[-1]}"
    current_mirror = f"nfl_player_advanced_stats_current_roster_{target}"
    readiness = pd.DataFrame([{
        "target_season": target,
        "history_start_season": history_seasons[0],
        "history_end_season": history_seasons[-1],
        "history_seasons": ",".join(map(str, history_seasons)),
        "history_rows": len(history),
        "history_players": history["player_id"].nunique() if "player_id" in history.columns else None,
        "current_roster_rows": len(current),
        "master_rows_expected": master_rows,
        "all_required_history_seasons_present": 1,
        "current_roster_reconciles_to_master": 1,
        "history_hash": history_hash,
        "advanced_history_ready_flag": 1,
        "overall_structural_ready_flag": 0,
        "remaining_structural_inputs": "QB,defense,depth,OL,units,strength",
        "build_id": BUILD_ID,
        "version": VERSION,
        "created_at": now(),
    }])
    with sqlite3.connect(db) as conn:
        history.to_sql(HISTORY_TABLE, conn, if_exists="replace", index=False)
        current.to_sql(CURRENT_TABLE, conn, if_exists="replace", index=False)
        snap.to_sql(SNAP_TABLE, conn, if_exists="replace", index=False)
        readiness.to_sql(READINESS_TABLE, conn, if_exists="replace", index=False)
        history.to_sql(history_mirror, conn, if_exists="replace", index=False)
        current.to_sql(current_mirror, conn, if_exists="replace", index=False)
        update_context(conn, len(history), len(current), history_hash)
        conn.commit()
    if not no_csv:
        out = root / "outputs" / "historical_advanced" / str(target)
        out.mkdir(parents=True, exist_ok=True)
        history.to_csv(out / f"{history_mirror}.csv", index=False, encoding="utf-8-sig")
        current.to_csv(out / f"{current_mirror}.csv", index=False, encoding="utf-8-sig")
        readiness.to_csv(out / "nfl_historical_advanced_stats_readiness_audit.csv", index=False, encoding="utf-8-sig")
    return readiness.iloc[0].to_dict()


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()
    run_id = str(uuid.uuid4())
    print("[HIST_ADV] Building rolling historical advanced stats")
    print(f"[HIST_ADV] Build ID: {BUILD_ID}")
    print(f"[HIST_ADV] Version: {VERSION}")
    print(f"[HIST_ADV] Target seasons: {list(args.target_seasons)}")
    module = import_canonical(args.canonical_script, args.allow_build_id_mismatch)
    union = list(range(min(args.target_seasons) - HISTORY_YEARS, max(args.target_seasons)))
    cache_db = args.project_root / "backtests" / CACHE_DB_NAME
    if args.rebuild_cache or not cache_db.exists():
        raw_union, snap_union = build_cache(module, args.project_root, args.canonical_script, cache_db, union)
    else:
        print(f"[HIST_ADV] Reusing cache: {cache_db}")
        raw_union, snap_union = load_cache(cache_db, union)
    rows = []
    for target in args.target_seasons:
        print("\n" + "=" * 108)
        print(f"[HIST_ADV] Building {target}")
        print("=" * 108)
        row = build_target(module, args.project_root, target, raw_union, snap_union, args.no_csv)
        row["run_id"] = run_id
        rows.append(row)
        print(
            f"[HIST_ADV] {target}: history={row['history_start_season']}-{row['history_end_season']} | "
            f"rows={row['history_rows']:,} | current={row['current_roster_rows']:,} | advanced_ready=1"
        )
    manifest = pd.DataFrame(rows)
    manifest_path = args.project_root / "outputs" / "nfl_historical_advanced_stats_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 108)
    print("[HIST_ADV] SUMMARY")
    print("=" * 108)
    print(manifest[[
        "target_season", "history_start_season", "history_end_season", "history_rows",
        "history_players", "current_roster_rows", "advanced_history_ready_flag",
        "overall_structural_ready_flag",
    ]].to_string(index=False))
    print("[HIST_ADV] Production database modified: NO")
    print("[HIST_ADV] Next: historical QB and defensive metric reconstruction")
    print(f"[HIST_ADV] Completed in {(dt.datetime.now() - started).total_seconds():.2f} seconds")
    if not args.no_csv:
        print(f"[HIST_ADV] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[HIST_ADV] Cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[HIST_ADV] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
