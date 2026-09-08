#!/usr/bin/env python
"""
Build historical QB and defensive metrics for isolated NFL reconstruction DBs.

Stage 3 of the canonical historical reconstruction.

The wrapper does not copy or simplify production formulas. For each target
season it verifies the approved canonical source Build IDs, creates temporary
season-patched copies, executes those copies against backtests/<season>.sqlite,
and validates the resulting QB, defensive-front, coverage, and defensive-summary
tables. Production scripts and the production database are never modified.

Required previous stages
------------------------
1. prepare_nfl_historical_reconstruction.py
2. backfill_nfl_historical_advanced_stats.py

Canonical source files
----------------------
- load_rbsdm_qb_ratings.py
- load_nfl_defensive_front_metrics.py
- load_nfl_coverage_metrics.py
- build_nfl_defensive_metrics_summary.py

Primary outputs in each isolated DB
-----------------------------------
- nfl_qb_rbsdm_ratings_raw_history
- nfl_qb_rbsdm_ratings_target
- nfl_qb_rbsdm_local_unmatched_audit_history
- nfl_defensive_front_metrics_history
- nfl_defensive_front_metrics_current_roster_target
- nfl_coverage_metrics_history
- nfl_coverage_metrics_current_roster_target
- nfl_defensive_player_metrics_current_roster_target
- nfl_projected_qb_starter_ratings_target
- nfl_historical_qb_defense_readiness_audit
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BUILD_ID = "NFL_HISTORICAL_QB_DEFENSE_CANONICAL_V2"
VERSION = "v2_qb_v3_isolated_database_root_prior_only"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_TARGET_SEASONS = (2023, 2024, 2025)
HISTORY_WINDOW_YEARS = 4
ROLLING_WEIGHTS = (0.10, 0.20, 0.30, 0.40)

CONTEXT_TABLE = "nfl_historical_reconstruction_context"
MASTER_TABLE = "nfl_player_master_target"
STARTER_QB_TABLE = "nfl_projected_qb_starters_target"
ADVANCED_HISTORY_TABLE = "nfl_player_advanced_stats_history"

QB_RAW_TABLE = "nfl_qb_rbsdm_ratings_raw_history"
QB_TABLE = "nfl_qb_rbsdm_ratings_target"
QB_UNMATCHED_TABLE = "nfl_qb_rbsdm_local_unmatched_audit_history"
FRONT_HISTORY_TABLE = "nfl_defensive_front_metrics_history"
FRONT_CURRENT_TABLE = "nfl_defensive_front_metrics_current_roster_target"
COVERAGE_HISTORY_TABLE = "nfl_coverage_metrics_history"
COVERAGE_CURRENT_TABLE = "nfl_coverage_metrics_current_roster_target"
DEFENSE_SUMMARY_TABLE = "nfl_defensive_player_metrics_current_roster_target"
STARTER_RATING_TABLE = "nfl_projected_qb_starter_ratings_target"
READINESS_TABLE = "nfl_historical_qb_defense_readiness_audit"
STEP_AUDIT_TABLE = "nfl_historical_qb_defense_step_audit"

FRONT_GROUPS = {"EDGE", "DL", "LB", "LB_EDGE"}
COVERAGE_GROUPS = {"DB", "LB", "LB_EDGE"}
DEFENSIVE_GROUPS = FRONT_GROUPS | COVERAGE_GROUPS

EXPECTED_SOURCES = {
    "qb": {
        "filename": "load_rbsdm_qb_ratings.py",
        "build_id": "NFL_RBSDM_QB_2026_CANONICAL_V3",
        "version": "v3_single_metric_stabilization_unshrunk_composite",
    },
    "front": {
        "filename": "load_nfl_defensive_front_metrics.py",
        "build_id": "NFL_DEFENSIVE_FRONT_2026_CANONICAL_V3",
        "version": "v3_canonical_2022_2025_snap_denominator_shrunk",
    },
    "coverage": {
        "filename": "load_nfl_coverage_metrics.py",
        "build_id": "NFL_COVERAGE_2026_CANONICAL_V3",
        "version": "v3_canonical_2022_2025_snap_denominator_shrunk",
    },
    "summary": {
        "filename": "build_nfl_defensive_metrics_summary.py",
        "build_id": "NFL_DEFENSIVE_SUMMARY_2026_CANONICAL_V3",
        "version": "v3_available_component_front_coverage_blend",
    },
}


@dataclass(frozen=True)
class PatchedScript:
    key: str
    source_path: Path
    patched_path: Path
    source_sha256: str


def parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not seasons:
        raise argparse.ArgumentTypeError("At least one target season is required.")
    return seasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--database-root",
        type=Path,
        default=None,
        help="Directory containing the isolated <season>.sqlite databases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for historical manifests and optional CSV exports.",
    )
    parser.add_argument("--target-seasons", type=parse_seasons, default=DEFAULT_TARGET_SEASONS)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--keep-patched-scripts", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
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
    return args


def now_string() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


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


def add_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    sql_type: str,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }
    if column_name not in columns:
        connection.execute(
            f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {sql_type}'
        )


def stable_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    usable = [c for c in columns if c in frame.columns]
    if frame.empty or not usable:
        return hashlib.sha256(b"").hexdigest()
    work = frame[usable].copy()
    for column in usable:
        work[column] = work[column].astype(str)
    return hashlib.sha256(
        work.sort_values(usable).to_csv(index=False).encode("utf-8")
    ).hexdigest()


def replace_assignment(text: str, variable: str, expression: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(variable)}\s*=\s*.*$")
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level assignment for {variable}; found {len(matches)}."
        )
    return pattern.sub(f"{variable} = {expression}", text, count=1)


def verified_source(path: Path, expected_build: str, expected_version: str, allow: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not allow:
        if expected_build not in text:
            raise RuntimeError(
                f"{path.name} is not the approved source. Missing Build ID {expected_build}."
            )
        if expected_version not in text:
            raise RuntimeError(
                f"{path.name} is not the approved source. Missing version {expected_version}."
            )
    return text


def patch_source(
    *,
    key: str,
    source_path: Path,
    patched_path: Path,
    target_season: int,
    history_seasons: list[int],
    allow_mismatch: bool,
) -> PatchedScript:
    spec = EXPECTED_SOURCES[key]
    text = verified_source(
        source_path,
        spec["build_id"],
        spec["version"],
        allow_mismatch,
    )
    text = replace_assignment(text, "SEASON", repr(target_season))
    text = replace_assignment(text, "MASTER_TABLE", repr(MASTER_TABLE))

    if re.search(r"(?m)^HIST_SEASONS\s*=", text):
        text = replace_assignment(text, "HIST_SEASONS", repr(history_seasons))
    if re.search(r"(?m)^RECENT_SEASON\s*=", text):
        text = replace_assignment(text, "RECENT_SEASON", repr(history_seasons[-1]))
    if re.search(r"(?m)^SEASON_WEIGHTS\s*=", text):
        weights = {
            season: float(ROLLING_WEIGHTS[index])
            for index, season in enumerate(history_seasons)
        }
        text = replace_assignment(text, "SEASON_WEIGHTS", repr(weights))

    if key == "qb":
        text = text.replace(
            'hist[hist["season"].isin([2023, 2024, 2025])]',
            'hist[hist["season"].isin(HIST_SEASONS[-3:])]',
        )
        if 'hist[hist["season"].isin(HIST_SEASONS[-3:])]' not in text:
            raise RuntimeError(
                "QB V3 source does not contain the approved dynamic three-year patch point."
            )
        replacements = {
            "ADVANCED_HISTORY_TABLE": ADVANCED_HISTORY_TABLE,
            "RAW_OUTPUT_TABLE": QB_RAW_TABLE,
            "OUTPUT_TABLE": QB_TABLE,
            "UNMATCHED_TABLE": QB_UNMATCHED_TABLE,
        }
    elif key == "front":
        replacements = {
            "ADVANCED_HISTORY_TABLE": ADVANCED_HISTORY_TABLE,
            "OUTPUT_TABLE": FRONT_HISTORY_TABLE,
            "CURRENT_OUTPUT_TABLE": FRONT_CURRENT_TABLE,
        }
    elif key == "coverage":
        replacements = {
            "ADVANCED_HISTORY_TABLE": ADVANCED_HISTORY_TABLE,
            "OUTPUT_TABLE": COVERAGE_HISTORY_TABLE,
            "CURRENT_OUTPUT_TABLE": COVERAGE_CURRENT_TABLE,
        }
    else:
        replacements = {
            "FRONT_TABLE": FRONT_CURRENT_TABLE,
            "COVERAGE_TABLE": COVERAGE_CURRENT_TABLE,
            "OUTPUT_TABLE": DEFENSE_SUMMARY_TABLE,
        }

    for variable, table_name in replacements.items():
        text = replace_assignment(text, variable, repr(table_name))

    patched_path.parent.mkdir(parents=True, exist_ok=True)
    patched_path.write_text(text, encoding="utf-8")
    return PatchedScript(
        key=key,
        source_path=source_path,
        patched_path=patched_path,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )


def prepare_runtime(
    project_root: Path,
    database_root: Path,
    target_season: int,
    history_seasons: list[int],
    allow_mismatch: bool,
) -> tuple[Path, list[PatchedScript], list[str]]:
    runtime_root = database_root / "runtime" / f"{target_season}_qb_defense"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    scripts_dir = runtime_root / "patched_scripts"
    inputs_dir = runtime_root / "inputs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs: list[str] = []
    source_inputs = project_root / "inputs"
    for season in history_seasons:
        source = source_inputs / f"rbsdm_qb_{season}.csv"
        if source.exists():
            shutil.copy2(source, inputs_dir / source.name)
            copied_inputs.append(str(source))

    scripts: list[PatchedScript] = []
    for key, spec in EXPECTED_SOURCES.items():
        scripts.append(
            patch_source(
                key=key,
                source_path=project_root / spec["filename"],
                patched_path=scripts_dir / spec["filename"],
                target_season=target_season,
                history_seasons=history_seasons,
                allow_mismatch=allow_mismatch,
            )
        )
    return runtime_root, scripts, copied_inputs


def run_child(
    *,
    script: PatchedScript,
    python_executable: Path,
    runtime_root: Path,
    db_path: Path,
    run_id: str,
    connection: sqlite3.Connection,
) -> None:
    command = [
        str(python_executable),
        "-u",
        str(script.patched_path),
        "--project-root",
        str(runtime_root),
        "--db-path",
        str(db_path),
        "--no-csv",
    ]
    print("\n" + "-" * 110)
    print(f"[HIST_QB_DEF] Running {script.key}: {subprocess.list2cmdline(command)}")
    print("-" * 110)

    started = dt.datetime.now()
    process = subprocess.Popen(
        command,
        cwd=str(runtime_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "NFL_HISTORICAL_QB_DEFENSE_RUN_ID": run_id,
        },
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = int(process.wait())
    completed = dt.datetime.now()

    pd.DataFrame(
        [
            {
                "run_id": run_id,
                "step_key": script.key,
                "source_script": str(script.source_path),
                "source_sha256": script.source_sha256,
                "patched_script": str(script.patched_path),
                "return_code": return_code,
                "status": "SUCCESS" if return_code == 0 else "FAILED",
                "started_at": started.isoformat(timespec="seconds"),
                "completed_at": completed.isoformat(timespec="seconds"),
                "elapsed_seconds": (completed - started).total_seconds(),
                "wrapper_build_id": BUILD_ID,
                "wrapper_version": VERSION,
            }
        ]
    ).to_sql(STEP_AUDIT_TABLE, connection, if_exists="append", index=False)
    connection.commit()

    if return_code != 0:
        raise RuntimeError(
            f"Historical {script.key} step failed with return code {return_code}."
        )


def group_count(master: pd.DataFrame, groups: set[str]) -> int:
    values = master["position_group"].fillna("").astype(str).str.upper().str.strip()
    return int(values.isin(groups).sum())


def validate_range(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    for column in columns:
        if column not in frame.columns:
            raise RuntimeError(f"{label} is missing {column}.")
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.notna() & (values.lt(0) | values.gt(100))
        if bad.any():
            raise RuntimeError(f"{label}.{column} contains a score outside 0-100.")


def historical_seasons(frame: pd.DataFrame) -> list[int]:
    if "season" not in frame.columns:
        return []
    return sorted(
        pd.to_numeric(frame["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


def validate_outputs(
    connection: sqlite3.Connection,
    target_season: int,
    expected_history: list[int],
) -> dict[str, Any]:
    master = read_table(connection, MASTER_TABLE)
    starters = read_table(connection, STARTER_QB_TABLE)
    qb_raw = read_table(connection, QB_RAW_TABLE)
    qbs = read_table(connection, QB_TABLE)
    front_history = read_table(connection, FRONT_HISTORY_TABLE)
    front_current = read_table(connection, FRONT_CURRENT_TABLE)
    coverage_history = read_table(connection, COVERAGE_HISTORY_TABLE)
    coverage_current = read_table(connection, COVERAGE_CURRENT_TABLE)
    defense = read_table(connection, DEFENSE_SUMMARY_TABLE)

    for label, frame in {
        "QB ratings": qbs,
        "front current": front_current,
        "coverage current": coverage_current,
        "defensive summary": defense,
    }.items():
        if "player_id" not in frame.columns or frame["player_id"].duplicated().any():
            raise RuntimeError(f"{label} lacks unique canonical player_id values.")

    for label, frame in {
        "QB": qb_raw,
        "front": front_history,
        "coverage": coverage_history,
    }.items():
        found = historical_seasons(frame)
        if found != expected_history:
            raise RuntimeError(
                f"{target_season} {label} seasons mismatch. "
                f"Expected {expected_history}; found {found}."
            )

    expected_counts = {
        "qb": group_count(master, {"QB"}),
        "front": group_count(master, FRONT_GROUPS),
        "coverage": group_count(master, COVERAGE_GROUPS),
        "defense": group_count(master, DEFENSIVE_GROUPS),
    }
    actual_counts = {
        "qb": len(qbs),
        "front": len(front_current),
        "coverage": len(coverage_current),
        "defense": len(defense),
    }
    if expected_counts != actual_counts:
        raise RuntimeError(
            f"{target_season} current-roster count mismatch. "
            f"Expected {expected_counts}; found {actual_counts}."
        )

    starter_ids = set(starters["player_id"].dropna().astype(str))
    rating_ids = set(qbs["player_id"].dropna().astype(str))
    if len(starter_ids) != 32:
        raise RuntimeError(f"{target_season} has {len(starter_ids)} starting QB IDs; expected 32.")
    missing_starters = sorted(starter_ids - rating_ids)
    if missing_starters:
        raise RuntimeError(
            f"{target_season} starting QBs missing rating rows: {missing_starters}"
        )

    validate_range(
        qbs,
        ["qb_multi_year_score", "qb_recent_score", "qb_rating_override"],
        "QB ratings",
    )
    validate_range(front_current, ["front_composite_score"], "front current")
    validate_range(
        coverage_current,
        ["coverage_composite_score"],
        "coverage current",
    )
    validate_range(
        defense,
        ["defensive_advanced_score"],
        "defensive summary",
    )

    starter_ratings = starters[["team", "player_id", "player_name"]].merge(
        qbs[
            [
                "player_id",
                "qb_multi_year_score",
                "qb_recent_score",
                "qb_rating_override",
                "qb_multi_year_available",
                "qb_recent_available",
            ]
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    starter_ratings["target_season"] = target_season
    starter_ratings.to_sql(
        STARTER_RATING_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )

    defenders_with_history = int(
        pd.to_numeric(
            defense.get("defensive_history_available", pd.Series(0, index=defense.index)),
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .sum()
    )

    return {
        "target_season": target_season,
        "history_start_season": expected_history[0],
        "history_end_season": expected_history[-1],
        "history_seasons": ",".join(map(str, expected_history)),
        "master_rows": len(master),
        "qb_rating_rows": len(qbs),
        "qb_raw_rows": len(qb_raw),
        "starting_qbs_with_ratings": 32,
        "front_history_rows": len(front_history),
        "front_current_rows": len(front_current),
        "coverage_history_rows": len(coverage_history),
        "coverage_current_rows": len(coverage_current),
        "defensive_summary_rows": len(defense),
        "defenders_with_history": defenders_with_history,
        "qb_hash": stable_hash(
            qbs,
            ["player_id", "qb_multi_year_score", "qb_recent_score", "qb_rating_override"],
        ),
        "defense_hash": stable_hash(
            defense,
            [
                "player_id",
                "front_composite_score",
                "coverage_composite_score",
                "defensive_advanced_score",
            ],
        ),
    }


def write_mirrors(
    connection: sqlite3.Connection,
    target_season: int,
    history: list[int],
) -> None:
    mirrors = {
        QB_TABLE: f"nfl_qb_rbsdm_ratings_{target_season}",
        QB_RAW_TABLE: f"nfl_qb_rbsdm_ratings_raw_{history[0]}_{history[-1]}",
        FRONT_HISTORY_TABLE: f"nfl_defensive_front_metrics_{history[0]}_{history[-1]}",
        FRONT_CURRENT_TABLE: f"nfl_defensive_front_metrics_current_roster_{target_season}",
        COVERAGE_HISTORY_TABLE: f"nfl_coverage_metrics_{history[0]}_{history[-1]}",
        COVERAGE_CURRENT_TABLE: f"nfl_coverage_metrics_current_roster_{target_season}",
        DEFENSE_SUMMARY_TABLE: f"nfl_defensive_player_metrics_current_roster_{target_season}",
    }
    for source, target in mirrors.items():
        read_table(connection, source).to_sql(
            target,
            connection,
            if_exists="replace",
            index=False,
        )


def update_context(connection: sqlite3.Connection, metrics: dict[str, Any]) -> None:
    additions = {
        "qb_defense_ready_flag": "INTEGER",
        "qb_rating_rows": "INTEGER",
        "starting_qbs_with_ratings": "INTEGER",
        "defensive_summary_rows": "INTEGER",
        "defenders_with_history": "INTEGER",
        "qb_rating_hash": "TEXT",
        "defensive_rating_hash": "TEXT",
        "qb_defense_build_id": "TEXT",
        "qb_defense_version": "TEXT",
        "qb_defense_completed_at": "TEXT",
        "structural_inputs_pending": "TEXT",
    }
    for column, sql_type in additions.items():
        add_column(connection, CONTEXT_TABLE, column, sql_type)

    connection.execute(
        f"""
        UPDATE {CONTEXT_TABLE}
        SET qb_defense_ready_flag = 1,
            qb_rating_rows = ?,
            starting_qbs_with_ratings = ?,
            defensive_summary_rows = ?,
            defenders_with_history = ?,
            qb_rating_hash = ?,
            defensive_rating_hash = ?,
            qb_defense_build_id = ?,
            qb_defense_version = ?,
            qb_defense_completed_at = ?,
            structural_inputs_ready_flag = 0,
            structural_inputs_pending = ?
        """,
        (
            metrics["qb_rating_rows"],
            metrics["starting_qbs_with_ratings"],
            metrics["defensive_summary_rows"],
            metrics["defenders_with_history"],
            metrics["qb_hash"],
            metrics["defense_hash"],
            BUILD_ID,
            VERSION,
            now_string(),
            (
                "historical_player_performance,historical_depth_chart,"
                "historical_ol_continuity,historical_team_units,"
                "historical_team_strength,historical_power_ratings"
            ),
        ),
    )


def export_csvs(
    connection: sqlite3.Connection,
    output_root: Path,
    target_season: int,
    no_csv: bool,
) -> None:
    if no_csv:
        return
    output_dir = (
        output_root / "historical_qb_defense" / str(target_season)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name in (
        QB_TABLE,
        STARTER_RATING_TABLE,
        FRONT_CURRENT_TABLE,
        COVERAGE_CURRENT_TABLE,
        DEFENSE_SUMMARY_TABLE,
        READINESS_TABLE,
    ):
        if table_exists(connection, table_name):
            read_table(connection, table_name).to_csv(
                output_dir / f"{table_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()
    run_id = str(uuid.uuid4())

    print("[HIST_QB_DEF] Building historical QB and defense")
    print(f"[HIST_QB_DEF] Build ID: {BUILD_ID}")
    print(f"[HIST_QB_DEF] Version: {VERSION}")
    print(f"[HIST_QB_DEF] Target seasons: {list(args.target_seasons)}")

    if not args.python.exists():
        raise FileNotFoundError(args.python)

    manifest_rows: list[dict[str, Any]] = []

    for target_season in args.target_seasons:
        history = list(range(target_season - HISTORY_WINDOW_YEARS, target_season))
        db_path = args.database_root / f"{target_season}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(db_path)

        print("\n" + "=" * 110)
        print(f"[HIST_QB_DEF] Building {target_season} from {history}")
        print("=" * 110)

        runtime_root, scripts, copied_inputs = prepare_runtime(
            args.project_root,
            args.database_root,
            target_season,
            history,
            args.allow_source_mismatch,
        )

        with sqlite3.connect(db_path, timeout=60) as connection:
            connection.execute("PRAGMA busy_timeout = 60000")
            context = read_table(connection, CONTEXT_TABLE)
            if int(context.iloc[0].get("advanced_history_ready_flag", 0)) != 1:
                raise RuntimeError(
                    f"{target_season} advanced history is not ready."
                )

            for script in scripts:
                run_child(
                    script=script,
                    python_executable=args.python,
                    runtime_root=runtime_root,
                    db_path=db_path,
                    run_id=run_id,
                    connection=connection,
                )

            metrics = validate_outputs(connection, target_season, history)
            metrics.update(
                {
                    "run_id": run_id,
                    "copied_local_rbsdm_inputs": " | ".join(copied_inputs),
                    "qb_defense_ready_flag": 1,
                    "overall_structural_ready_flag": 0,
                    "build_id": BUILD_ID,
                    "version": VERSION,
                    "created_at": now_string(),
                }
            )
            pd.DataFrame([metrics]).to_sql(
                READINESS_TABLE,
                connection,
                if_exists="replace",
                index=False,
            )
            write_mirrors(connection, target_season, history)
            update_context(connection, metrics)
            connection.commit()
            export_csvs(
                connection,
                args.output_root,
                target_season,
                args.no_csv,
            )

        manifest_rows.append(metrics)
        print(
            f"[HIST_QB_DEF] {target_season}: "
            f"QB={metrics['qb_rating_rows']:,} | "
            f"starters={metrics['starting_qbs_with_ratings']}/32 | "
            f"front={metrics['front_current_rows']:,} | "
            f"coverage={metrics['coverage_current_rows']:,} | "
            f"defense={metrics['defensive_summary_rows']:,} | "
            f"with_history={metrics['defenders_with_history']:,}"
        )

        if not args.keep_patched_scripts:
            shutil.rmtree(runtime_root, ignore_errors=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = (
        args.output_root / "nfl_historical_qb_defense_manifest.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_csv:
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("[HIST_QB_DEF] SUMMARY")
    print("=" * 110)
    print(
        manifest[
            [
                "target_season",
                "history_start_season",
                "history_end_season",
                "qb_rating_rows",
                "starting_qbs_with_ratings",
                "front_current_rows",
                "coverage_current_rows",
                "defensive_summary_rows",
                "defenders_with_history",
                "qb_defense_ready_flag",
                "overall_structural_ready_flag",
            ]
        ].to_string(index=False)
    )
    print("[HIST_QB_DEF] Production scripts modified: NO")
    print("[HIST_QB_DEF] Production database modified: NO")
    print(
        "[HIST_QB_DEF] Next stage: historical player-performance and "
        "depth-chart reconstruction."
    )
    print(
        f"[HIST_QB_DEF] Completed in "
        f"{(dt.datetime.now() - started).total_seconds():.2f} seconds"
    )
    if not args.no_csv:
        print(f"[HIST_QB_DEF] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[HIST_QB_DEF] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[HIST_QB_DEF] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
