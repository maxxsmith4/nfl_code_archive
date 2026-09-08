#!/usr/bin/env python
"""Run the canonical QB repair, one Circa backtest, and model rebuild chain."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd


BUILD_ID = "NFL_CIRCA_QB_REPAIR_CANONICAL_RUNNER_V1"
VERSION = "v1_2_verification_only_windows_self_test_close"
DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)

RESIDUAL_SCRIPT = "build_backtest_nfl_weekly_matchup_residual.py"
CIRCA_SCRIPT = "build_backtest_nfl_circa_contest_lines.py"
PHASE_SCRIPT = "build_backtest_nfl_circa_season_phase_v3_2.py"
PREDICTOR_SCRIPT = "predict_nfl_circa_top5_2026.py"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rebuild-source-cache", action="store_true")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-from-circa",
        action="store_true",
        help=(
            "Skip the already completed matchup-residual rebuild and resume "
            "with the corrected Circa backtest."
        ),
    )
    resume_group.add_argument(
        "--verification-only",
        action="store_true",
        help=(
            "Skip completed rebuilds and run only the live self-test plus "
            "final database/model lineage verification."
        ),
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    return args


class TeeLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, value: str) -> None:
        print(value, flush=True)
        self.handle.write(value + "\n")

    def close(self) -> None:
        self.handle.close()


def run_command(label: str, command: list[str], log: TeeLog) -> None:
    log.write("=" * 112)
    log.write(f"[QB_REPAIR] {label}")
    log.write(subprocess.list2cmdline(command))
    log.write("=" * 112)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line.rstrip("\r\n"))
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{label} failed with return code {return_code}.")


def read_one_row(database: Path, table: str) -> dict[str, Any]:
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            raise RuntimeError(f"Missing audit table {table} in {database}.")
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected one row in {table}; found {len(frame)}."
        )
    return frame.iloc[0].to_dict()


def assert_positive_qb_metadata(bundle: dict[str, Any], label: str) -> None:
    required = {
        "qb_feature_integrity_passed",
        "qb_side_epa_coverage",
        "qb_side_cpoe_coverage",
        "qb_epa_advantage_std",
        "qb_cpoe_advantage_std",
        "qb_epa_advantage_nonzero_rows",
        "qb_cpoe_advantage_nonzero_rows",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"{label} is missing QB audit fields: {missing}")
    if not (
        int(bundle["qb_feature_integrity_passed"]) == 1
        and float(bundle["qb_side_epa_coverage"]) >= 0.99
        and float(bundle["qb_side_cpoe_coverage"]) >= 0.99
        and float(bundle["qb_epa_advantage_std"]) > 1e-6
        and float(bundle["qb_cpoe_advantage_std"]) > 1e-6
        and int(bundle["qb_epa_advantage_nonzero_rows"]) > 0
        and int(bundle["qb_cpoe_advantage_nonzero_rows"]) > 0
    ):
        raise RuntimeError(f"{label} has invalid QB audit metadata.")


def final_verification(project_root: Path) -> dict[str, Any]:
    residual_database = project_root / "backtests/nfl_weekly_matchup_residual.sqlite"
    circa_database = project_root / "backtests/nfl_circa_contest_lines.sqlite"
    v1_path = project_root / "models/nfl_circa_contest_model_v1.joblib"
    v32_path = project_root / "models/nfl_circa_contest_season_phase_v3_2.joblib"

    residual = read_one_row(residual_database, "nfl_matchup_run_audit")
    circa = read_one_row(circa_database, "nfl_circa_run_audit")
    if int(residual.get("qb_feature_integrity_passed", 0)) != 1:
        raise RuntimeError("Residual database failed the final QB audit.")
    if int(circa.get("circa_qb_feature_integrity_passed", 0)) != 1:
        raise RuntimeError("Circa database failed the final QB audit.")

    v1 = joblib.load(v1_path)
    v32 = joblib.load(v32_path)
    assert_positive_qb_metadata(v1, "Circa V1 bundle")
    assert_positive_qb_metadata(v32, "Circa V3.2 bundle")
    v1_qb_coefficient = float(v1.get("qb_epa_standardized_coefficient", 0.0))
    if abs(v1_qb_coefficient) <= 1e-12:
        raise RuntimeError("Circa V1 QB EPA coefficient is zero.")
    for key in ("source_v1_build_id", "source_v1_version", "source_v1_created_at"):
        expected_key = key.removeprefix("source_v1_")
        if str(v32.get(key, "")) != str(v1.get(expected_key, "")):
            raise RuntimeError(f"V3.2/V1 lineage mismatch for {key}.")
    if abs(float(v32["selected_late_qb_epa_standardized_coefficient"])) <= 1e-12:
        raise RuntimeError("V3.2 selected late QB EPA coefficient is zero.")

    return {
        "residual_build_id": residual.get("build_id"),
        "circa_build_id": circa.get("build_id"),
        "qb_identity_match_rate": residual.get("qb_identity_match_rate"),
        "qb_recent_epa_coverage": residual.get("qb_recent_epa_coverage"),
        "qb_recent_cpoe_coverage": residual.get("qb_recent_cpoe_coverage"),
        "qb_epa_advantage_std": residual.get("qb_epa_advantage_std"),
        "qb_cpoe_advantage_std": residual.get("qb_cpoe_advantage_std"),
        "v1_qb_epa_coefficient": v1_qb_coefficient,
        "v32_qb_epa_coefficient": v32[
            "selected_late_qb_epa_standardized_coefficient"
        ],
        "v32_v1_lineage_matched": True,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    scripts = [
        args.project_root / RESIDUAL_SCRIPT,
        args.project_root / CIRCA_SCRIPT,
        args.project_root / PHASE_SCRIPT,
        args.project_root / PREDICTOR_SCRIPT,
    ]
    missing = [str(path) for path in scripts if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical scripts: " + ", ".join(missing))

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.project_root / "logs" / f"run_nfl_circa_qb_repair_{stamp}.log"
    log = TeeLog(log_path)
    try:
        log.write(f"[QB_REPAIR] Build ID: {BUILD_ID}")
        log.write(f"[QB_REPAIR] Version: {VERSION}")
        log.write(f"[QB_REPAIR] Python: {args.python}")
        log.write(f"[QB_REPAIR] Project root: {args.project_root}")

        compile_command = [str(args.python), "-m", "py_compile", *map(str, scripts)]
        run_command("Compile all canonical scripts", compile_command, log)

        residual_command = [
            str(args.python), "-u", str(scripts[0]),
            "--project-root", str(args.project_root),
        ]
        if args.rebuild_source_cache:
            residual_command.append("--rebuild-cache")
        if args.no_csv:
            residual_command.append("--no-csv")
        if args.verification_only:
            log.write(
                "[QB_REPAIR] Verification-only mode; completed residual, "
                "Circa V1, and V3.2 rebuilds will not be repeated."
            )
        elif args.resume_from_circa:
            log.write(
                "[QB_REPAIR] Resuming from Circa; preserved repaired matchup "
                "residual database will be revalidated downstream."
            )
        else:
            run_command(
                "Rebuild matchup residual with repaired QB identity",
                residual_command,
                log,
            )

        if not args.verification_only:
            circa_command = [
                str(args.python), "-u", str(scripts[1]),
                "--mode", "backtest", "--project-root", str(args.project_root),
                "--allow-incomplete",
                "--backfill-missing-2025-with-reference",
            ]
            if args.no_csv:
                circa_command.append("--no-csv")
            run_command(
                "Run the one corrected Circa V1 backtest",
                circa_command,
                log,
            )

            run_command(
                "Rebuild the V3.2 season-phase ceiling",
                [
                    str(args.python), "-u", str(scripts[2]),
                    "--project-root", str(args.project_root),
                ],
                log,
            )
        run_command(
            "Run live predictor self-test",
            [str(args.python), "-u", str(scripts[3]), "--self-test"],
            log,
        )

        verification = final_verification(args.project_root)
        log.write("=" * 112)
        log.write("[QB_REPAIR] FINAL VERIFICATION PASSED")
        log.write(json.dumps(verification, indent=2, default=str))
        log.write(f"[QB_REPAIR] Log: {log_path}")
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[QB_REPAIR] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[QB_REPAIR] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
