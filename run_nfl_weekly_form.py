#!/usr/bin/env python3
"""Build and verify the canonical learned-consensus weekly projection.

This is the orchestration entry point between the point-in-time NFL inputs and
the Circa top-five workflow. It rebuilds form, verifies the frozen 2026 learned
model, generates market-independent fair lines, and checks exact row lineage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SEASON = 2026
BUILD_ID = "NFL_WEEKLY_FORM_RUNNER_CANONICAL_V2"
VERSION = "v2_form_v3_frozen_learned_consensus_fail_closed_postflight"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

FORM_SCRIPT = "build_nfl_2026_form_rating.py"
MODEL_SCRIPT = "build_nfl_learned_consensus_2026.py"
SPREAD_SCRIPT = "predict_nfl_weekly_learned_consensus_2026.py"
CIRCA_SCRIPT = "predict_nfl_circa_top5_2026.py"

FORM_TABLE = "nfl_2026_form_ratings"
PREDICTION_TABLE = "nfl_weekly_power_spread_predictions_2026"
RUN_AUDIT_TABLE = "nfl_weekly_power_spread_run_audit_2026"

EXPECTED_FORM_BUILD_ID = "NFL_2026_FORM_RATING_CANONICAL_V3"
EXPECTED_FORM_VERSION = (
    "v3_current_structural_prior_asof_opponent_adjusted_form"
)
EXPECTED_SPREAD_BUILD_ID = "NFL_WEEKLY_LEARNED_CONSENSUS_2026_CANONICAL_V1"
EXPECTED_SPREAD_VERSION = "v1_0_frozen_learned_weights_point_in_time_consensus"
EXPECTED_MODEL_VARIANT = "LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS"
EXPECTED_BUNDLE_BUILD_ID = "NFL_LEARNED_CONSENSUS_2026_CANONICAL_V1"
EXPECTED_BUNDLE_VERSION = "v1_0_2020_2025_frozen_market_free_consensus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
    parser.add_argument(
        "--db-path",
        "--database",
        dest="db_path",
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Required YYYY-MM-DD lineage date; defaults to today.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for child scripts.",
    )
    parser.add_argument(
        "--verification-only",
        action="store_true",
        help="Do not rebuild; verify the current SQLite outputs only.",
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.week <= 18:
        parser.error("--week must be between 1 and 18.")
    parsed = pd.to_datetime(
        args.as_of_date if args.as_of_date else dt.date.today(),
        errors="coerce",
    )
    if pd.isna(parsed):
        parser.error("--as-of-date must be YYYY-MM-DD.")
    args.as_of_date = pd.Timestamp(parsed).normalize().date().isoformat()
    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    args.python = args.python.resolve()
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_command(
    command: list[str],
    label: str,
    project_root: Path,
) -> None:
    print("=" * 112)
    print(f"[WEEKLY_FORM] {label}")
    print(subprocess.list2cmdline(command))
    print("=" * 112)
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with return code {return_code}."
        )


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    if exists is None:
        raise RuntimeError(f"Required table is missing: {table_name}")
    return pd.read_sql_query(
        f'SELECT * FROM "{table_name}"', connection
    )


def unique_text(frame: pd.DataFrame, column: str, label: str) -> str:
    if column not in frame.columns:
        raise RuntimeError(f"{label} is missing column {column}.")
    values = {
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    }
    if len(values) != 1:
        raise RuntimeError(
            f"{label} must contain one {column}; found={sorted(values)}."
        )
    return next(iter(values))


def unique_integer(frame: pd.DataFrame, column: str, label: str) -> int:
    if column not in frame.columns:
        raise RuntimeError(f"{label} is missing column {column}.")
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise RuntimeError(
            f"{label} must contain one {column}; found={values}."
        )
    return int(values[0])


def exact_date(
    frame: pd.DataFrame,
    column: str,
    expected: str,
    label: str,
) -> None:
    if column not in frame.columns:
        raise RuntimeError(f"{label} is missing column {column}.")
    dates = pd.to_datetime(frame[column], errors="coerce").dt.date
    expected_date = pd.Timestamp(expected).date()
    if dates.isna().any() or set(dates) != {expected_date}:
        raise RuntimeError(
            f"{label} {column} is stale: expected={expected}, "
            f"found={sorted(str(value) for value in dates.dropna().unique())}."
        )


def postflight(args: argparse.Namespace) -> dict[str, Any]:
    if not args.db_path.exists():
        raise FileNotFoundError(args.db_path)
    with sqlite3.connect(args.db_path) as connection:
        form = read_table(connection, FORM_TABLE)
        predictions = read_table(connection, PREDICTION_TABLE)
        audits = read_table(connection, RUN_AUDIT_TABLE)

    if len(form) != 32 or form["team"].nunique() != 32:
        raise RuntimeError("Form table must contain 32 unique teams.")
    if unique_text(form, "build_id", FORM_TABLE) != EXPECTED_FORM_BUILD_ID:
        raise RuntimeError("Form table has the wrong canonical build ID.")
    if unique_text(form, "form_version", FORM_TABLE) != EXPECTED_FORM_VERSION:
        raise RuntimeError("Form table has the wrong canonical version.")
    expected_through = max(0, int(args.week) - 1)
    if unique_integer(form, "through_week", FORM_TABLE) != expected_through:
        raise RuntimeError(
            "Form table has the wrong through-week for this prediction."
        )
    exact_date(form, "as_of_date", args.as_of_date, FORM_TABLE)

    week_values = pd.to_numeric(
        predictions.get("week"), errors="coerce"
    )
    predictions = predictions[week_values.eq(int(args.week))].copy()
    if predictions.empty:
        raise RuntimeError(
            f"No current learned-consensus predictions exist for Week {args.week}."
        )
    if unique_text(
        predictions, "build_id", PREDICTION_TABLE
    ) != EXPECTED_SPREAD_BUILD_ID:
        raise RuntimeError("Weekly predictions have the wrong build ID.")
    if unique_text(
        predictions, "version", PREDICTION_TABLE
    ) != EXPECTED_SPREAD_VERSION:
        raise RuntimeError("Weekly predictions have the wrong version.")
    if unique_text(
        predictions, "model_variant", PREDICTION_TABLE
    ) != EXPECTED_MODEL_VARIANT:
        raise RuntimeError("Weekly predictions are not the learned consensus model.")
    if unique_text(
        predictions, "learned_bundle_build_id", PREDICTION_TABLE
    ) != EXPECTED_BUNDLE_BUILD_ID:
        raise RuntimeError("Weekly predictions have the wrong learned-bundle build ID.")
    if unique_text(
        predictions, "learned_bundle_version", PREDICTION_TABLE
    ) != EXPECTED_BUNDLE_VERSION:
        raise RuntimeError("Weekly predictions have the wrong learned-bundle version.")
    if unique_integer(
        predictions, "prediction_uses_market_inputs", PREDICTION_TABLE
    ) != 0:
        raise RuntimeError("Learned consensus prediction used market inputs.")
    exact_date(
        predictions,
        "prediction_as_of_date",
        args.as_of_date,
        PREDICTION_TABLE,
    )
    exact_date(
        predictions,
        "prediction_timestamp",
        args.as_of_date,
        PREDICTION_TABLE,
    )

    run_id = unique_text(predictions, "run_id", PREDICTION_TABLE)
    audit = audits[audits["run_id"].astype(str).eq(run_id)].copy()
    if len(audit) != 1:
        raise RuntimeError(
            "Current predictions do not match exactly one run-audit row."
        )
    if unique_text(audit, "status", RUN_AUDIT_TABLE) != "SUCCESS":
        raise RuntimeError("Structural run audit is not successful.")
    if unique_integer(audit, "games", RUN_AUDIT_TABLE) != len(predictions):
        raise RuntimeError("Structural run-audit game count does not match.")
    for column in (
        "build_id",
        "version",
        "model_variant",
        "form_build_id",
        "form_version",
        "form_as_of_date",
        "form_through_week",
        "form_date_imported",
        "preseason_snapshot_hash",
        "structural_power_build_id",
        "structural_power_version",
        "structural_power_date_imported",
        "structural_power_snapshot_hash",
        "learned_bundle_build_id",
        "learned_bundle_version",
        "learned_bundle_sha256",
        "learned_structural_snapshot_hash",
    ):
        if unique_text(predictions, column, PREDICTION_TABLE) != unique_text(
            audit, column, RUN_AUDIT_TABLE
        ):
            raise RuntimeError(
                f"Prediction/run-audit lineage mismatch for {column}."
            )
    exact_date(audit, "completed_at", args.as_of_date, RUN_AUDIT_TABLE)

    return {
        "runner_build_id": BUILD_ID,
        "runner_version": VERSION,
        "season": SEASON,
        "week": int(args.week),
        "as_of_date": args.as_of_date,
        "through_week": expected_through,
        "form_rows": len(form),
        "prediction_rows": len(predictions),
        "structural_run_id": run_id,
        "form_build_id": EXPECTED_FORM_BUILD_ID,
        "spread_build_id": EXPECTED_SPREAD_BUILD_ID,
        "structural_power_snapshot_hash": unique_text(
            predictions,
            "structural_power_snapshot_hash",
            PREDICTION_TABLE,
        ),
        "learned_bundle_build_id": unique_text(
            predictions, "learned_bundle_build_id", PREDICTION_TABLE
        ),
        "learned_bundle_sha256": unique_text(
            predictions, "learned_bundle_sha256", PREDICTION_TABLE
        ),
        "verified_at": now_string(),
    }


def main() -> int:
    args = parse_args()
    print(f"[WEEKLY_FORM] Build ID: {BUILD_ID}")
    print(f"[WEEKLY_FORM] Version: {VERSION}")
    print(f"[WEEKLY_FORM] Project root: {args.project_root}")
    print(f"[WEEKLY_FORM] Database: {args.db_path}")
    print(
        f"[WEEKLY_FORM] Week={args.week} | as_of_date={args.as_of_date}"
    )

    scripts = [
        args.project_root / FORM_SCRIPT,
        args.project_root / MODEL_SCRIPT,
        args.project_root / SPREAD_SCRIPT,
        args.project_root / CIRCA_SCRIPT,
    ]
    missing = [str(path) for path in scripts if not path.exists()]
    if missing:
        raise RuntimeError("Missing canonical scripts: " + "; ".join(missing))

    run_command(
        [
            str(args.python),
            "-m",
            "py_compile",
            *(str(path) for path in scripts),
        ],
        "Compile form/learned-consensus/Circa canonical scripts",
        args.project_root,
    )

    if not args.verification_only:
        form_command = [
            str(args.python),
            "-u",
            str(args.project_root / FORM_SCRIPT),
            "--project-root",
            str(args.project_root),
            "--db-path",
            str(args.db_path),
            "--as-of-date",
            args.as_of_date,
        ]
        if int(args.week) > 1:
            form_command.extend(
                ["--through-week", str(int(args.week) - 1)]
            )
        if args.no_csv:
            form_command.append("--no-csv")
        run_command(
            form_command,
            "Rebuild current point-in-time form ratings",
            args.project_root,
        )

        model_command = [
            str(args.python),
            "-u",
            str(args.project_root / MODEL_SCRIPT),
            "--project-root",
            str(args.project_root),
            "--database-root",
            str(args.project_root / "backtests"),
            "--db-path",
            str(args.db_path),
        ]
        if args.no_csv:
            model_command.append("--no-csv")
        run_command(
            model_command,
            "Freeze or verify the 2026 learned consensus model",
            args.project_root,
        )

        spread_command = [
            str(args.python),
            "-u",
            str(args.project_root / SPREAD_SCRIPT),
            "--project-root",
            str(args.project_root),
            "--db-path",
            str(args.db_path),
            "--week",
            str(args.week),
            "--as-of-date",
            args.as_of_date,
        ]
        if args.no_csv:
            spread_command.append("--no-csv")
        run_command(
            spread_command,
            "Generate fresh market-independent learned-consensus lines",
            args.project_root,
        )
    else:
        print(
            "[WEEKLY_FORM] Verification-only mode; no tables were rebuilt."
        )

    summary = postflight(args)
    print("=" * 112)
    print("[WEEKLY_FORM] FINAL VERIFICATION PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[WEEKLY_FORM] FAILED: {exc}", file=sys.stderr)
        raise
