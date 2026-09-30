#!/usr/bin/env python
"""
Run the complete canonical 2026 NFL structural-rating refresh.

CADENCE
-------
Run this script infrequently, not as the normal weekly job.

Appropriate triggers:
- material roster transactions;
- authoritative starting-quarterback changes;
- meaningful projected depth-chart changes;
- an intentional refresh of historical player inputs;
- a methodology/code change to any structural model stage;
- a full preseason rebuild.

Do NOT run merely because another NFL week finished. The weekly form runner
handles completed-game information separately and preserves the immutable
preseason snapshot.

CANONICAL SEQUENCE
------------------
1. load_nfl_rosters.py
2. build_nfl_player_master.py
3. build_nfl_player_crosswalk.py
4. nfl_player_advanced_stats.py
5. load_rbsdm_qb_ratings.py
6. load_nfl_defensive_front_metrics.py
7. load_nfl_coverage_metrics.py
8. build_nfl_defensive_metrics_summary.py
9. build_nfl_player_performance.py
10. build_nfl_projected_depth_chart.py
11. build_nfl_ol_continuity.py
12. build_nfl_team_unit_ratings.py
13. build_nfl_team_strength.py
14. build_nfl_power_ratings.py

The live player-performance experiment is intentionally excluded.

After this runner succeeds, run:
    run_nfl_weekly_form.py

That updates the live form table and records any drift between the immutable
preseason prior and the newly rebuilt current structural rating.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEASON = 2026
RUNNER_VERSION = "canonical_structural_refresh_v1_1_source_outage_partial"
SOURCE_UNAVAILABLE_EXIT_CODE = 20
SOURCE_STATUS_TABLE = "nfl_live_roster_refresh_status_2026"
DEPTH_DEPENDENT_STEPS = {
    "ol_continuity", "team_units", "team_strength", "power_ratings",
}

PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

LOCK_NAME = ".run_nfl_structural_refresh.lock"
RUN_AUDIT_TABLE = "nfl_structural_refresh_runner_audit_2026"
STEP_AUDIT_TABLE = "nfl_structural_refresh_step_audit_2026"


@dataclass(frozen=True)
class ExpectedTable:
    table_name: str
    minimum_rows: int = 1
    unique_column: str | None = None
    expected_unique: int | None = None


@dataclass(frozen=True)
class PipelineStep:
    key: str
    label: str
    filename: str
    expected_tables: tuple[ExpectedTable, ...]


PIPELINE_STEPS = (
    PipelineStep(
        "rosters",
        "Refresh 2026 rosters",
        "load_nfl_rosters.py",
        (
            ExpectedTable(
                "nfl_rosters_2026_raw",
                minimum_rows=2500,
            ),
        ),
    ),
    PipelineStep(
        "player_master",
        "Build canonical player master",
        "build_nfl_player_master.py",
        (
            ExpectedTable(
                "nfl_player_master_2026",
                minimum_rows=2500,
            ),
        ),
    ),
    PipelineStep(
        "crosswalk",
        "Build permanent player crosswalk",
        "build_nfl_player_crosswalk.py",
        (
            ExpectedTable(
                "nfl_player_crosswalk",
                minimum_rows=5000,
            ),
        ),
    ),
    PipelineStep(
        "advanced_stats",
        "Build canonical 2022-2025 advanced history",
        "nfl_player_advanced_stats.py",
        (
            ExpectedTable(
                "nfl_player_advanced_stats_2022_2025",
                minimum_rows=7000,
            ),
            ExpectedTable(
                "nfl_player_advanced_stats_current_roster_2026",
                minimum_rows=2500,
            ),
        ),
    ),
    PipelineStep(
        "qb_ratings",
        "Build canonical quarterback ratings",
        "load_rbsdm_qb_ratings.py",
        (
            ExpectedTable(
                "nfl_qb_rbsdm_ratings_2026",
                minimum_rows=80,
            ),
        ),
    ),
    PipelineStep(
        "defensive_front",
        "Build defensive-front metrics",
        "load_nfl_defensive_front_metrics.py",
        (
            ExpectedTable(
                "nfl_defensive_front_metrics_current_roster_2026",
                minimum_rows=600,
            ),
        ),
    ),
    PipelineStep(
        "coverage",
        "Build coverage metrics",
        "load_nfl_coverage_metrics.py",
        (
            ExpectedTable(
                "nfl_coverage_metrics_current_roster_2026",
                minimum_rows=700,
            ),
        ),
    ),
    PipelineStep(
        "defensive_summary",
        "Build combined defensive metrics",
        "build_nfl_defensive_metrics_summary.py",
        (
            ExpectedTable(
                "nfl_defensive_player_metrics_current_roster_2026",
                minimum_rows=1100,
            ),
        ),
    ),
    PipelineStep(
        "player_performance",
        "Build canonical player-performance inputs",
        "build_nfl_player_performance.py",
        (
            ExpectedTable(
                "nfl_player_performance_inputs_2026",
                minimum_rows=2500,
            ),
            ExpectedTable(
                "nfl_player_performance_seasonal_2022_2025",
                minimum_rows=7000,
            ),
        ),
    ),
    PipelineStep(
        "depth_chart",
        "Build authoritative projected depth chart",
        "build_nfl_projected_depth_chart.py",
        (
            ExpectedTable(
                "nfl_projected_depth_chart_2026",
                minimum_rows=2500,
            ),
        ),
    ),
    PipelineStep(
        "ol_continuity",
        "Build offensive-line continuity",
        "build_nfl_ol_continuity.py",
        (
            ExpectedTable(
                "nfl_ol_continuity_2026",
                minimum_rows=32,
                unique_column="team",
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        "team_units",
        "Build replacement-anchored team unit ratings",
        "build_nfl_team_unit_ratings.py",
        (
            ExpectedTable(
                "nfl_team_unit_ratings_2026",
                minimum_rows=32,
                unique_column="team",
                expected_unique=32,
            ),
            ExpectedTable(
                "nfl_position_group_ratings_2026",
                minimum_rows=256,
            ),
        ),
    ),
    PipelineStep(
        "team_strength",
        "Build team strength",
        "build_nfl_team_strength.py",
        (
            ExpectedTable(
                "nfl_team_strength_2026",
                minimum_rows=32,
                unique_column="team",
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        "power_ratings",
        "Build neutral-field power ratings",
        "build_nfl_power_ratings.py",
        (
            ExpectedTable(
                "nfl_power_ratings_2026",
                minimum_rows=32,
                unique_column="team",
                expected_unique=32,
            ),
        ),
    ),
)


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )

    def write(self, message: str = "") -> None:
        text = str(message)
        print(text, flush=True)
        self.handle.write(text + "\n")

    def raw(self, message: str) -> None:
        print(message, end="", flush=True)
        self.handle.write(message)

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 104)
        self.write(title)
        self.write("=" * 104)

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    keys = [step.key for step in PIPELINE_STEPS]

    parser.add_argument(
        "--start-at",
        choices=keys,
        default=None,
        help="Start at a named step after an earlier failure.",
    )
    parser.add_argument(
        "--stop-after",
        choices=keys,
        default=None,
        help="Stop after a named step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and print commands without changing data.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python interpreter override. Defaults to the current interpreter.",
    )
    return parser.parse_args()


def selected_steps(args: argparse.Namespace) -> list[PipelineStep]:
    steps = list(PIPELINE_STEPS)

    if args.start_at is not None:
        index = next(
            i for i, step in enumerate(steps)
            if step.key == args.start_at
        )
        steps = steps[index:]

    if args.stop_after is not None:
        original = list(PIPELINE_STEPS)
        stop_index = next(
            i for i, step in enumerate(original)
            if step.key == args.stop_after
        )
        allowed = {step.key for step in original[: stop_index + 1]}
        steps = [step for step in steps if step.key in allowed]

    if not steps:
        raise RuntimeError("No structural-refresh steps were selected.")
    return steps


def acquire_lock(path: Path, run_id: str) -> None:
    payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        descriptor = os.open(
            str(path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        existing = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        raise RuntimeError(
            "Another structural refresh may be active.\n"
            f"Lock: {path}\n"
            f"Contents: {existing}"
        ) from exc

    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def release_lock(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=60,
    )
    connection.execute("PRAGMA busy_timeout = 60000")
    return connection


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def validate_expected_table(
    connection: sqlite3.Connection,
    expected: ExpectedTable,
) -> dict[str, int | str]:
    if not table_exists(connection, expected.table_name):
        raise RuntimeError(
            f"Expected output table was not created: "
            f"{expected.table_name}"
        )

    escaped = expected.table_name.replace('"', '""')
    row_count = int(
        connection.execute(
            f'SELECT COUNT(*) FROM "{escaped}"'
        ).fetchone()[0]
    )

    if row_count < expected.minimum_rows:
        raise RuntimeError(
            f"{expected.table_name} has only {row_count:,} rows; "
            f"minimum expected is {expected.minimum_rows:,}."
        )

    unique_count = None
    if expected.unique_column is not None:
        escaped_column = expected.unique_column.replace('"', '""')
        unique_count = int(
            connection.execute(
                f'SELECT COUNT(DISTINCT "{escaped_column}") '
                f'FROM "{escaped}"'
            ).fetchone()[0]
        )
        if (
            expected.expected_unique is not None
            and unique_count != expected.expected_unique
        ):
            raise RuntimeError(
                f"{expected.table_name}.{expected.unique_column} "
                f"has {unique_count} unique values; expected "
                f"{expected.expected_unique}."
            )

    return {
        "table_name": expected.table_name,
        "rows": row_count,
        "unique_count": (
            unique_count if unique_count is not None else -1
        ),
    }


def validate_source_outage(
    connection: sqlite3.Connection,
    step_started_utc: dt.datetime,
) -> str:
    """Accept exit 20 only for a recorded outage from this depth attempt."""
    if not table_exists(connection, SOURCE_STATUS_TABLE):
        raise RuntimeError("Depth exit 20 has no availability refresh marker.")
    rows = connection.execute(
        f'SELECT season, status, attempted_at_utc, error FROM "{SOURCE_STATUS_TABLE}"'
    ).fetchall()
    if len(rows) != 1 or int(rows[0][0]) != SEASON or rows[0][1] != "SOURCE_UNAVAILABLE":
        raise RuntimeError("Depth exit 20 lacks a valid SOURCE_UNAVAILABLE marker.")
    try:
        attempted = dt.datetime.fromisoformat(str(rows[0][2]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("Depth outage marker has an invalid timestamp.") from exc
    if attempted.tzinfo is None or not (
        step_started_utc - dt.timedelta(seconds=5)
        <= attempted.astimezone(dt.timezone.utc)
        <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)
    ):
        raise RuntimeError("Depth outage marker is not from this refresh attempt.")
    reason = str(rows[0][3] or "").strip()
    if not reason:
        raise RuntimeError("Depth outage marker has no source failure reason.")
    return reason


def validate_existing_structure(
    connection: sqlite3.Connection,
    logger: TeeLogger,
) -> None:
    """Require a complete prior 32-team snapshot before partial completion."""
    for step in PIPELINE_STEPS:
        if step.key != "depth_chart" and step.key not in DEPTH_DEPENDENT_STEPS:
            continue
        for expected in step.expected_tables:
            metrics = validate_expected_table(connection, expected)
            logger.write(
                f"[STRUCTURAL] Retained {metrics['table_name']}: "
                f"rows={metrics['rows']:,}; not refreshed."
            )

    depth = connection.execute(
        """SELECT COUNT(*), COUNT(DISTINCT player_id), COUNT(DISTINCT team),
                  SUM(CASE WHEN qb_authoritative_starter=1 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN ol_authoritative_starter=1 THEN 1 ELSE 0 END),
                  MAX(date_imported)
           FROM nfl_projected_depth_chart_2026 WHERE season=?""",
        (SEASON,),
    ).fetchone()
    if (int(depth[0]) != int(depth[1]) or int(depth[2]) != 32
            or int(depth[3] or 0) != 32 or int(depth[4] or 0) != 160):
        raise RuntimeError(
            "Prior depth snapshot is incomplete or has duplicate players; "
            "cannot complete using frozen structural inputs."
        )
    logger.write(
        "[STRUCTURAL] Retained depth snapshot: 32 teams, 32 QB starters, "
        f"160 OL starters; last build={depth[5]}."
    )


def insert_audit(
    connection: sqlite3.Connection,
    table_name: str,
    record: dict,
) -> None:
    columns = list(record)
    placeholders = ", ".join("?" for _ in columns)
    quoted_columns = ", ".join(
        f'"{column.replace(chr(34), chr(34) * 2)}"'
        for column in columns
    )

    definitions = []
    for column, value in record.items():
        if isinstance(value, int):
            sql_type = "INTEGER"
        elif isinstance(value, float):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        definitions.append(
            f'"{column.replace(chr(34), chr(34) * 2)}" {sql_type}'
        )

    connection.execute(
        f'CREATE TABLE IF NOT EXISTS "{table_name}" '
        f'({", ".join(definitions)})'
    )
    connection.execute(
        f'INSERT INTO "{table_name}" '
        f'({quoted_columns}) VALUES ({placeholders})',
        [record[column] for column in columns],
    )
    connection.commit()


def run_step(
    step: PipelineStep,
    python_executable: Path,
    logger: TeeLogger,
    run_id: str,
    connection: sqlite3.Connection,
) -> str:
    script_path = PROJECT_ROOT / step.filename
    command = [
        str(python_executable),
        "-u",
        str(script_path),
    ]

    logger.section(
        f"[STRUCTURAL] {step.key}: {step.label}"
    )
    logger.write(
        "[STRUCTURAL] Command: "
        + subprocess.list2cmdline(command)
    )

    started = dt.datetime.now()
    started_utc = dt.datetime.now(dt.timezone.utc)
    started_perf = time.perf_counter()

    attempts = 2 if step.key == "depth_chart" else 1
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            logger.write("[STRUCTURAL] Retrying depth source once after exit 20.")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "NFL_STRUCTURAL_REFRESH_RUN_ID": run_id,
            },
        )

        assert process.stdout is not None
        for line in process.stdout:
            logger.raw(line)

        return_code = int(process.wait())
        if return_code != SOURCE_UNAVAILABLE_EXIT_CODE:
            break

    elapsed = time.perf_counter() - started_perf
    completed = dt.datetime.now()

    source_outage = step.key == "depth_chart" and return_code == SOURCE_UNAVAILABLE_EXIT_CODE
    status = "SOURCE_UNAVAILABLE" if source_outage else "SUCCESS" if return_code == 0 else "FAILED"
    if source_outage:
        try:
            reason = validate_source_outage(connection, started_utc)
            validate_existing_structure(connection, logger)
        except Exception:
            status = "FAILED"
            raise
        finally:
            insert_audit(
                connection, STEP_AUDIT_TABLE,
                {
                    "run_id": run_id, "runner_version": RUNNER_VERSION,
                    "step_key": step.key, "script_filename": step.filename,
                    "status": status, "return_code": return_code,
                    "started_at": started.isoformat(timespec="seconds"),
                    "completed_at": completed.isoformat(timespec="seconds"),
                    "elapsed_seconds": float(elapsed),
                },
            )
        logger.write(f"[STRUCTURAL] Source unavailable: {reason}")
        logger.write(
            "[STRUCTURAL] Current depth was not rebuilt. Subsequent depth-dependent "
            "stages will retain the validated prior snapshot."
        )
        return status

    insert_audit(
        connection,
        STEP_AUDIT_TABLE,
        {
            "run_id": run_id,
            "runner_version": RUNNER_VERSION,
            "step_key": step.key,
            "script_filename": step.filename,
            "status": status,
            "return_code": return_code,
            "started_at": started.isoformat(timespec="seconds"),
            "completed_at": completed.isoformat(timespec="seconds"),
            "elapsed_seconds": float(elapsed),
        },
    )

    if return_code != 0:
        raise RuntimeError(
            f"Structural refresh failed at {step.key} "
            f"with return code {return_code}."
        )

    for expected in step.expected_tables:
        metrics = validate_expected_table(
            connection,
            expected,
        )
        logger.write(
            f"[STRUCTURAL] Validated {metrics['table_name']}: "
            f"rows={metrics['rows']:,}"
        )
    return status


def validate_environment(
    steps: Iterable[PipelineStep],
    python_executable: Path,
) -> None:
    if not PROJECT_ROOT.exists():
        raise RuntimeError(
            f"Project root does not exist: {PROJECT_ROOT}"
        )
    if not DB_PATH.exists():
        raise RuntimeError(
            f"SQLite database does not exist: {DB_PATH}"
        )
    if not python_executable.exists():
        raise RuntimeError(
            f"Python executable does not exist: {python_executable}"
        )

    missing = [
        step.filename
        for step in steps
        if not (PROJECT_ROOT / step.filename).exists()
    ]
    if missing:
        raise RuntimeError(
            "Missing canonical scripts:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )


def validate_final_power(
    connection: sqlite3.Connection,
) -> None:
    table = "nfl_power_ratings_2026"
    if not table_exists(connection, table):
        return

    rows = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT team) AS teams,
            AVG(power_rating_points) AS rating_mean,
            MIN(power_rating_points) AS rating_min,
            MAX(power_rating_points) AS rating_max
        FROM {table}
        """
    ).fetchone()

    if int(rows[0]) != 32 or int(rows[1]) != 32:
        raise RuntimeError(
            "Final power table does not contain 32 unique teams."
        )
    if abs(float(rows[2])) > 1e-6:
        raise RuntimeError(
            f"Final power ratings are not zero-centered: "
            f"mean={float(rows[2]):.8f}"
        )


def main() -> int:
    args = parse_args()
    steps = selected_steps(args)
    python_executable = (
        args.python.resolve()
        if args.python is not None
        else Path(sys.executable).resolve()
    )

    validate_environment(steps, python_executable)

    if args.dry_run:
        print("[STRUCTURAL] DRY RUN")
        for step in steps:
            command = [
                str(python_executable),
                "-u",
                str(PROJECT_ROOT / step.filename),
            ]
            print(subprocess.list2cmdline(command))
        return 0

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    log_path = (
        log_dir
        / f"run_nfl_structural_refresh_{timestamp}_{run_id[:8]}.log"
    )
    lock_path = PROJECT_ROOT / LOCK_NAME

    logger = TeeLogger(log_path)
    connection = connect_database()
    started = dt.datetime.now()
    started_perf = time.perf_counter()
    status = "FAILED"
    failed_step = None
    partial_source_outage = False

    try:
        acquire_lock(lock_path, run_id)

        logger.section(
            "[STRUCTURAL] CANONICAL 2026 NFL STRUCTURAL REFRESH"
        )
        logger.write(f"[STRUCTURAL] Runner: {RUNNER_VERSION}")
        logger.write(f"[STRUCTURAL] Run ID: {run_id}")
        logger.write(f"[STRUCTURAL] Python: {python_executable}")
        logger.write(f"[STRUCTURAL] Project: {PROJECT_ROOT}")
        logger.write(f"[STRUCTURAL] Database: {DB_PATH}")
        logger.write(
            "[STRUCTURAL] Steps: "
            + ", ".join(step.key for step in steps)
        )

        for step in steps:
            failed_step = step.key
            if partial_source_outage and step.key in DEPTH_DEPENDENT_STEPS:
                skipped_at = dt.datetime.now().isoformat(timespec="seconds")
                insert_audit(
                    connection, STEP_AUDIT_TABLE,
                    {
                        "run_id": run_id, "runner_version": RUNNER_VERSION,
                        "step_key": step.key, "script_filename": step.filename,
                        "status": "SKIPPED_SOURCE_UNAVAILABLE", "return_code": None,
                        "started_at": skipped_at, "completed_at": skipped_at,
                        "elapsed_seconds": 0.0,
                    },
                )
                logger.write(
                    f"[STRUCTURAL] Skipped {step.key}: prior validated output "
                    "retained after availability source outage."
                )
                failed_step = None
                continue
            step_status = run_step(
                step,
                python_executable,
                logger,
                run_id,
                connection,
            )
            if step_status == "SOURCE_UNAVAILABLE":
                partial_source_outage = True
            failed_step = None

        validate_final_power(connection)
        status = "PARTIAL_SOURCE_UNAVAILABLE" if partial_source_outage else "SUCCESS"

        if partial_source_outage:
            logger.section("[STRUCTURAL] PARTIAL COMPLETION — SOURCE UNAVAILABLE")
            logger.write(
                "[STRUCTURAL] Upstream player inputs completed; the prior 32-team "
                "depth, OL, unit, strength, and power tables were validated and retained."
            )
            logger.write(
                "[STRUCTURAL] No current depth or roster-adjusted lines are certified. "
                "Run the normal weekly pipeline for its baseline-only fallback. "
                "Rerun --start-at depth_chart after the source recovers."
            )
        else:
            logger.section("[STRUCTURAL] REFRESH COMPLETE")
            logger.write(
                "[STRUCTURAL] The immutable preseason form snapshot "
                "was not replaced."
            )
            logger.write(
                "[STRUCTURAL] Next command: run_nfl_weekly_form.py"
            )
        return 0

    except Exception as exc:
        logger.section("[STRUCTURAL] REFRESH FAILED")
        logger.write(f"[STRUCTURAL] Failed step: {failed_step}")
        logger.write(f"[STRUCTURAL] Error: {exc}")
        logger.write(traceback.format_exc())
        return 1

    finally:
        completed = dt.datetime.now()
        elapsed = time.perf_counter() - started_perf
        try:
            insert_audit(
                connection,
                RUN_AUDIT_TABLE,
                {
                    "run_id": run_id,
                    "runner_version": RUNNER_VERSION,
                    "status": status,
                    "failed_step": failed_step,
                    "started_at": started.isoformat(timespec="seconds"),
                    "completed_at": completed.isoformat(timespec="seconds"),
                    "elapsed_seconds": float(elapsed),
                    "selected_steps": ",".join(
                        step.key for step in steps
                    ),
                    "log_path": str(log_path),
                },
            )
        except Exception:
            pass

        connection.close()
        release_lock(lock_path)
        logger.write(
            f"[STRUCTURAL] Log saved: {log_path}"
        )
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
