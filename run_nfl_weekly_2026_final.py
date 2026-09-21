#!/usr/bin/env python
"""
Run the canonical 2026 NFL weekly production pipeline.

Official weekly model
---------------------
LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS is the independent weekly spread model.

Canonical sequence
------------------
0. load_nfl_rosters.py, then build_nfl_player_master.py
1. build_nfl_projected_depth_chart.py (automatic depth and injury refresh)
2. build_nfl_ol_continuity.py
3. build_nfl_team_unit_ratings.py
4. build_nfl_team_strength.py
5. build_nfl_power_ratings.py
6. build_nfl_2026_form_rating.py
7. build_nfl_learned_consensus_2026.py
8. load_nfl_live_market_odds.py
9. predict_nfl_weekly_learned_consensus_2026.py

The market loader runs only after the independent structural/form inputs have
been built. Its output is attached by the predictor after the fair spread is
frozen. Supply --market-path to use an existing market file, or
--skip-market-load for an intentional fair-spread-only run.

The Circa contest residual model is not run here. After Circa posts its Thursday
contest board, run predict_nfl_circa_top5_2026.py separately.

The runner:
- uses the same Python interpreter that launched it;
- resolves canonical filenames, including the automatic depth-chart builder;
- streams every child process to the console and a timestamped master log;
- stops after model/data failures; a recorded availability-source outage permits
  a baseline-only run with adjusted projections withheld;
- prevents overlapping runs with a lock file;
- passes only command-line options actually supported by each child script;
- validates expected SQLite outputs after each successful step;
- records run-level and step-level audits in the production SQLite database.

Typical commands
----------------
Full refresh and projection (the week is required):
    python run_nfl_weekly_2026_final.py --week 1

Refresh availability, current units, form and projection:
    python run_nfl_weekly_2026_final.py --form-and-predict-only --week 2

Refresh availability and current units, then predict with existing form:
    python run_nfl_weekly_2026_final.py --predict-only --week 2
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
from typing import Any, Iterable, Optional


SEASON = 2026
BUILD_ID = "NFL_WEEKLY_2026_FINAL_RUNNER_CANONICAL_V5"
VERSION = "v6_5_roster_dns_recovery_and_local_source_inputs"
SOURCE_UNAVAILABLE_EXIT_CODE = 20
REFRESH_STATUS_TABLE = "nfl_live_roster_refresh_status_2026"
LIVE_STRUCTURE_STEPS = {"ol_continuity", "team_units", "team_strength", "power_ratings"}

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

RUN_AUDIT_TABLE = "nfl_weekly_final_runner_audit_2026"
RUN_AUDIT_HISTORY_TABLE = "nfl_weekly_final_runner_audit_history_2026"
STEP_AUDIT_TABLE = "nfl_weekly_final_runner_step_audit_2026"
STEP_AUDIT_HISTORY_TABLE = "nfl_weekly_final_runner_step_audit_history_2026"

LOCK_FILENAME = ".run_nfl_weekly_2026_final.lock"


@dataclass(frozen=True)
class TableExpectation:
    table_name: str
    minimum_rows: int = 1
    unique_candidates: tuple[str, ...] = ()
    expected_unique: Optional[int] = None
    week_candidates: tuple[str, ...] = ()
    require_prediction_week: bool = False


@dataclass(frozen=True)
class PipelineStep:
    key: str
    description: str
    candidates: tuple[str, ...]
    expectations: tuple[TableExpectation, ...]


PIPELINE_STEPS = (
    PipelineStep(
        key="rosters",
        description="Refresh current NFL rosters",
        candidates=("load_nfl_rosters.py",),
        expectations=(TableExpectation("nfl_rosters_2026_raw", minimum_rows=500),),
    ),
    PipelineStep(
        key="player_master",
        description="Refresh canonical player identities",
        candidates=("build_nfl_player_master.py",),
        expectations=(TableExpectation("nfl_player_master_2026", minimum_rows=500),),
    ),
    PipelineStep(
        key="depth_chart",
        description="Build authoritative projected depth chart",
        candidates=(
            "build_nfl_projected_depth_chart.py",
        ),
        expectations=(
            TableExpectation(
                "nfl_projected_depth_chart_2026",
                minimum_rows=500,
            ),
            TableExpectation("nfl_live_roster_source_audit_2026", minimum_rows=32,
                             unique_candidates=("team",), expected_unique=32),
        ),
    ),
    PipelineStep(
        key="ol_continuity",
        description="Build offensive-line continuity",
        candidates=("build_nfl_ol_continuity.py",),
        expectations=(
            TableExpectation(
                "nfl_ol_continuity_2026",
                minimum_rows=32,
                unique_candidates=("team", "team_abbr", "team_code"),
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        key="team_units",
        description="Build team and position-group unit ratings",
        candidates=("build_nfl_team_unit_ratings.py",),
        expectations=(
            TableExpectation(
                "nfl_team_unit_ratings_2026",
                minimum_rows=32,
                unique_candidates=("team", "team_abbr", "team_code"),
                expected_unique=32,
            ),
            TableExpectation(
                "nfl_position_group_ratings_2026",
                minimum_rows=128,
            ),
        ),
    ),
    PipelineStep(
        key="team_strength",
        description="Build structural team strength",
        candidates=("build_nfl_team_strength.py",),
        expectations=(
            TableExpectation(
                "nfl_team_strength_2026",
                minimum_rows=32,
                unique_candidates=("team", "team_abbr", "team_code"),
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        key="power_ratings",
        description="Build neutral-field structural power ratings",
        candidates=("build_nfl_power_ratings.py",),
        expectations=(
            TableExpectation(
                "nfl_power_ratings_2026",
                minimum_rows=32,
                unique_candidates=("team", "team_abbr", "team_code"),
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        key="form_ratings",
        description="Build current-season live form ratings",
        candidates=("build_nfl_2026_form_rating.py",),
        expectations=(
            TableExpectation(
                "nfl_2026_form_ratings",
                minimum_rows=32,
                unique_candidates=("team", "team_abbr", "team_code"),
                expected_unique=32,
            ),
        ),
    ),
    PipelineStep(
        key="learned_model",
        description="Freeze or verify the 2026 learned consensus model",
        candidates=("build_nfl_learned_consensus_2026.py",),
        expectations=(
            TableExpectation(
                "nfl_learned_consensus_model_registry_2026",
                minimum_rows=1,
            ),
        ),
    ),
    PipelineStep(
        key="live_market",
        description="Load audited sharp reference spreads",
        candidates=("load_nfl_live_market_odds.py",),
        expectations=(
            TableExpectation(
                "nfl_live_market_odds_2026",
                minimum_rows=1,
                week_candidates=("week", "game_week", "week_number"),
                require_prediction_week=True,
            ),
        ),
    ),
    PipelineStep(
        key="weekly_predictions",
        description="Produce weekly learned structural-consensus projections",
        candidates=("predict_nfl_weekly_learned_consensus_2026.py",),
        expectations=(
            TableExpectation(
                "nfl_weekly_power_spread_predictions_2026",
                minimum_rows=1,
                week_candidates=("week", "prediction_week", "game_week"),
                require_prediction_week=True,
            ),
        ),
    ),
)


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, message: str = "") -> None:
        text = str(message)
        print(text, flush=True)
        self.handle.write(text + "\n")

    def raw(self, message: str) -> None:
        print(message, end="", flush=True)
        self.handle.write(message)

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 112)
        self.write(title)
        self.write("=" * 112)

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--db-path",
        "--database",
        dest="db_path",
        type=Path,
        default=DEFAULT_DB_PATH,
    )
    parser.add_argument("--week", "--prediction-week", dest="week", type=int)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument(
        "--through-week", type=int, default=None,
        help="Form cutoff; defaults to prediction week minus one (Week 1 uses 0).",
    )
    parser.add_argument("--schedule-path", type=Path, default=None)
    parser.add_argument("--schedule-factors-path", type=Path, default=None)
    parser.add_argument("--market-path", type=Path, default=None)
    parser.add_argument("--current-pbp-path", type=Path, default=None)
    parser.add_argument("--current-snaps-path", type=Path, default=None)
    parser.add_argument("--roster-input-csv", type=Path, default=None)
    parser.add_argument("--diagnose-sources", action="store_true",
                        help="Check network and local source inventory without running or changing model tables.")
    parser.add_argument(
        "--skip-market-load",
        action="store_true",
        help="Do not call the live odds loader; permits fair-only output.",
    )
    parser.add_argument(
        "--odds-api-key-env",
        type=str,
        default="ODDS_API_KEY",
    )
    parser.add_argument(
        "--market-bookmaker-priority",
        type=str,
        default="pinnacle,lowvig,betonlineag",
    )
    parser.add_argument(
        "--market-max-age-minutes",
        type=float,
        default=1440.0,
    )
    parser.add_argument(
        "--market-require-primary-bookmaker",
        action="store_true",
    )
    parser.add_argument(
        "--market-allow-partial-week",
        action="store_true",
    )
    parser.add_argument(
        "--market-line-preference",
        choices=("current", "opening"),
        default="current",
    )
    parser.add_argument("--bankroll", type=float, default=50_000.0)
    parser.add_argument("--flat-stake", type=float, default=500.0)
    parser.add_argument("--minimum-spread-difference", type=float, default=2.0)
    parser.add_argument("--maximum-market-disagreement", type=float, default=3.5)
    parser.add_argument("--quarter-kelly-multiplier", type=float, default=0.25)
    parser.add_argument("--max-kelly-bet-fraction", type=float, default=0.05)
    parser.add_argument("--default-spread-price", type=int, default=-110)
    parser.add_argument("--home-field-points", type=float, default=None)
    parser.add_argument(
        "--allow-week1-stakes",
        action="store_true",
        help=(
            "Permit Week 1 learned-consensus stakes that pass the frozen gate. "
            "They remain explicitly labeled outside validated backtest scope."
        ),
    )
    parser.add_argument("--include-week18", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--prepare-process-model", action="store_true")
    parser.add_argument("--rebuild-process-cache", action="store_true")
    parser.add_argument(
        "--form-and-predict-only",
        action="store_true",
        help="Refresh rosters, master, live depth and units, then run form and predictions.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Refresh rosters, master, live depth and units, then predict using existing form.",
    )
    parser.add_argument(
        "--start-at",
        choices=tuple(step.key for step in PIPELINE_STEPS),
        default=None,
    )
    parser.add_argument(
        "--stop-after",
        choices=tuple(step.key for step in PIPELINE_STEPS),
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-unlock", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    for name in (
        "schedule_path",
        "schedule_factors_path",
        "market_path",
        "current_pbp_path",
        "current_snaps_path",
        "roster_input_csv",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if args.week is not None and not 1 <= args.week <= 18:
        parser.error("--week must be between 1 and 18.")
    if args.through_week is not None and not 0 <= args.through_week <= 18:
        parser.error("--through-week must be between 0 and 18.")
    if args.market_max_age_minutes < 0:
        parser.error("--market-max-age-minutes cannot be negative.")
    if args.skip_market_load and args.market_path is not None:
        parser.error(
            "--skip-market-load cannot be combined with --market-path."
        )
    if args.bankroll <= 0:
        parser.error("--bankroll must be positive.")
    if args.flat_stake < 0:
        parser.error("--flat-stake cannot be negative.")
    if args.maximum_market_disagreement < args.minimum_spread_difference:
        parser.error(
            "--maximum-market-disagreement must be at least "
            "--minimum-spread-difference."
        )
    if args.predict_only and args.form_and_predict_only:
        parser.error(
            "--predict-only and --form-and-predict-only are mutually exclusive."
        )
    return args


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone() is not None


def table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    escaped = table_name.replace('"', '""')
    return [
        str(row[1]).lower().strip()
        for row in connection.execute(f'PRAGMA table_info("{escaped}")')
    ]


def first_existing(
    values: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {str(value).lower().strip(): str(value) for value in values}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def resolve_script(project_root: Path, step: PipelineStep) -> Path:
    for candidate in step.candidates:
        path = project_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing step {step.key!r}. Checked: {list(step.candidates)}"
    )


def supported_help_options(
    python_path: Path,
    script_path: Path,
) -> set[str]:
    result = subprocess.run(
        [str(python_path), str(script_path), "--help"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    options = set()
    for token in help_text.replace(",", " ").split():
        if token.startswith("--"):
            options.add(token.rstrip("]})>,"))
    return options


_NO_OPTION_VALUE = object()


def add_if_supported(
    command: list[str],
    supported: set[str],
    option: str,
    value: Any = _NO_OPTION_VALUE,
    enabled: bool = True,
) -> None:
    """Append a supported CLI option without emitting empty value options.

    Boolean flags call this function without ``value``. Value-taking options
    pass their value explicitly. When an optional value is ``None``, the entire
    option is omitted instead of producing an invalid bare argument such as
    ``--as-of-date``.
    """
    if not enabled or option not in supported:
        return
    if value is None:
        return

    command.append(option)
    if value is not _NO_OPTION_VALUE:
        command.append(str(value))


def default_market_path(
    project_root: Path,
    week: Optional[int],
) -> Optional[Path]:
    """Return the deterministic live-market CSV used by the predictor."""
    if week is None:
        return None
    return (
        project_root
        / "outputs"
        / "nfl_market"
        / f"nfl_weekly_market_2026_week_{int(week):02d}.csv"
    ).resolve()


def effective_market_path(args: argparse.Namespace) -> Optional[Path]:
    """Resolve explicit, automatic, or intentionally absent market input."""
    if args.market_path is not None:
        return args.market_path
    if args.skip_market_load:
        return None
    return default_market_path(args.project_root, args.week)


def build_child_command(
    args: argparse.Namespace,
    step: PipelineStep,
    script_path: Path,
) -> list[str]:
    python_path = Path(sys.executable)
    supported = supported_help_options(python_path, script_path)
    command = [str(python_path), str(script_path)]

    add_if_supported(command, supported, "--project-root", args.project_root)
    if "--db-path" in supported:
        command.extend(["--db-path", str(args.db_path)])
    elif "--database" in supported:
        command.extend(["--database", str(args.db_path)])

    if step.key == "rosters":
        value = getattr(args, "roster_input_csv", None)
        if value is not None and "--input-csv" not in supported:
            raise RuntimeError("The roster loader does not support --input-csv.")
        add_if_supported(command, supported, "--input-csv", value)

    if step.key == "form_ratings":
        if args.through_week is not None and "--through-week" not in supported:
            raise RuntimeError(
                "The form builder does not support the required --through-week cutoff. "
                "Install the updated build_nfl_2026_form_rating.py."
            )
        add_if_supported(command, supported, "--as-of-date", args.as_of_date)
        add_if_supported(command, supported, "--through-week", args.through_week)
        for option, value in (("--current-pbp-path", args.current_pbp_path),
                              ("--schedule-path", args.schedule_path)):
            if value is not None and option not in supported:
                raise RuntimeError(f"Install the corrected form builder to use {option}.")
            add_if_supported(command, supported, option, value)
        add_if_supported(
            command,
            supported,
            "--prepare-process-model",
            enabled=args.prepare_process_model,
        )
        add_if_supported(
            command,
            supported,
            "--rebuild-process-cache",
            enabled=args.rebuild_process_cache,
        )
        add_if_supported(command, supported, "--no-csv", enabled=args.no_csv)

    if step.key == "learned_model":
        add_if_supported(
            command,
            supported,
            "--database-root",
            args.project_root / "backtests",
        )
        add_if_supported(command, supported, "--no-csv", enabled=args.no_csv)

    if step.key == "live_market":
        add_if_supported(command, supported, "--week", args.week)
        add_if_supported(
            command,
            supported,
            "--api-key-env",
            args.odds_api_key_env,
        )
        add_if_supported(
            command,
            supported,
            "--bookmaker-priority",
            args.market_bookmaker_priority,
        )
        add_if_supported(
            command,
            supported,
            "--maximum-line-age-minutes",
            args.market_max_age_minutes,
        )
        add_if_supported(
            command,
            supported,
            "--require-primary-bookmaker",
            enabled=args.market_require_primary_bookmaker,
        )
        add_if_supported(
            command,
            supported,
            "--allow-partial-week",
            enabled=args.market_allow_partial_week,
        )
        add_if_supported(
            command,
            supported,
            "--output-path",
            effective_market_path(args),
        )

    if step.key == "weekly_predictions":
        add_if_supported(
            command,
            supported,
            "--database-root",
            args.project_root / "backtests",
        )
        add_if_supported(
            command,
            supported,
            "--source-cache-db",
            args.project_root
            / "backtests"
            / "nfl_weekly_matchup_source_cache_v2.sqlite",
        )
        if "--week" in supported:
            add_if_supported(command, supported, "--week", args.week)
        else:
            add_if_supported(
                command,
                supported,
                "--prediction-week",
                args.week,
            )
        add_if_supported(command, supported, "--as-of-date", args.as_of_date)
        add_if_supported(command, supported, "--schedule-path", args.schedule_path)
        add_if_supported(
            command,
            supported,
            "--schedule-factors-path",
            args.schedule_factors_path,
        )
        add_if_supported(
            command,
            supported,
            "--market-path",
            effective_market_path(args),
        )
        add_if_supported(
            command,
            supported,
            "--current-pbp-path",
            args.current_pbp_path,
        )
        add_if_supported(
            command,
            supported,
            "--current-snaps-path",
            args.current_snaps_path,
        )
        add_if_supported(
            command,
            supported,
            "--market-line-preference",
            args.market_line_preference,
        )
        add_if_supported(command, supported, "--bankroll", args.bankroll)
        add_if_supported(command, supported, "--flat-stake", args.flat_stake)
        add_if_supported(
            command,
            supported,
            "--minimum-spread-difference",
            args.minimum_spread_difference,
        )
        add_if_supported(
            command,
            supported,
            "--maximum-market-disagreement",
            args.maximum_market_disagreement,
        )
        add_if_supported(
            command,
            supported,
            "--quarter-kelly-multiplier",
            args.quarter_kelly_multiplier,
        )
        add_if_supported(
            command,
            supported,
            "--max-kelly-bet-fraction",
            args.max_kelly_bet_fraction,
        )
        add_if_supported(
            command,
            supported,
            "--default-spread-price",
            args.default_spread_price,
        )
        add_if_supported(
            command,
            supported,
            "--home-field-points",
            args.home_field_points,
        )
        add_if_supported(
            command,
            supported,
            "--include-week18",
            enabled=args.include_week18,
        )
        add_if_supported(
            command,
            supported,
            "--allow-week1-stakes",
            enabled=args.allow_week1_stakes,
        )
        add_if_supported(command, supported, "--no-csv", enabled=args.no_csv)

    return command


def selected_steps(args: argparse.Namespace) -> list[PipelineStep]:
    steps = list(PIPELINE_STEPS)
    if args.predict_only:
        steps = [
            step
            for step in steps
            if step.key in {"rosters", "player_master", "depth_chart", "team_units",
                            "learned_model", "live_market", "weekly_predictions"}
        ]
    elif args.form_and_predict_only:
        steps = [
            step
            for step in steps
            if step.key
            in {"rosters", "player_master", "depth_chart", "team_units", "form_ratings",
                "learned_model", "live_market", "weekly_predictions"}
        ]

    if args.market_path is not None or args.skip_market_load:
        steps = [step for step in steps if step.key != "live_market"]

    selected_keys = [step.key for step in steps]

    if args.start_at is not None:
        if args.start_at not in selected_keys:
            raise RuntimeError(
                f"--start-at {args.start_at!r} is incompatible with the "
                "selected run mode."
            )
        index = selected_keys.index(args.start_at)
        steps = steps[index:]
        selected_keys = [step.key for step in steps]

    if args.stop_after is not None:
        if args.stop_after not in selected_keys:
            raise RuntimeError(
                f"--stop-after {args.stop_after!r} is incompatible with the "
                "selected run mode or occurs before --start-at."
            )
        index = selected_keys.index(args.stop_after)
        steps = steps[: index + 1]

    return steps


def configure_form_cutoff(
    args: argparse.Namespace, steps: list[PipelineStep]
) -> None:
    """Use the frozen weekly boundary even after early games have finished."""
    if args.week is None:
        return
    expected = int(args.week) - 1
    if args.through_week is None:
        args.through_week = expected
    elif any(step.key == "weekly_predictions" for step in steps):
        if int(args.through_week) != expected:
            raise RuntimeError(
                f"Week {args.week} learned predictions require form through Week "
                f"{expected}; received --through-week {args.through_week}."
            )


def validate_form_cutoff(db_path: Path, expected: int) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            'SELECT DISTINCT through_week FROM "nfl_2026_form_ratings"'
        ).fetchall()
    if len(rows) != 1 or rows[0][0] is None:
        raise RuntimeError(f"Form output has missing or inconsistent cutoffs: {rows}")
    try:
        valid = float(rows[0][0]) == expected
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError(
            f"Form output is through Week {rows[0][0]}; expected Week {expected}."
        )
    return {"form_through_week": expected}


def completed_market_week_audit(
    db_path: Path, week: Optional[int], step_started: dt.datetime
) -> Optional[dict[str, Any]]:
    """Allow zero current lines only when this pull audited the whole week inactive."""
    if week is None:
        return None
    table = "nfl_live_market_odds_run_audit_2026"
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        if not table_exists(connection, table):
            return None
        required = {
            "season", "week", "run_id", "schedule_games", "remaining_games",
            "excluded_games", "selected_lines", "missing_games", "retrieved_at_utc",
        }
        if not required.issubset(table_columns(connection, table)):
            return None
        row = connection.execute(
            f'SELECT * FROM "{table}" WHERE season=? AND week=? '
            'ORDER BY retrieved_at_utc DESC LIMIT 1', (SEASON, week)
        ).fetchone()
        if row is None:
            return None
        try:
            retrieved = dt.datetime.fromisoformat(str(row["retrieved_at_utc"]).replace("Z", "+00:00"))
            recent = (
                retrieved.tzinfo is not None
                and retrieved >= step_started.astimezone(dt.timezone.utc) - dt.timedelta(seconds=1)
            )
            complete = (
                int(row["schedule_games"]) > 0
                and int(row["excluded_games"]) == int(row["schedule_games"])
                and all(int(row[key]) == 0 for key in ("remaining_games", "selected_lines", "missing_games"))
            )
        except (ValueError, TypeError, OverflowError):
            return None
        if not recent or not complete:
            return None
        count = connection.execute(
            'SELECT COUNT(*) FROM "nfl_live_market_odds_2026" WHERE season=? AND week=?',
            (SEASON, week),
        ).fetchone()[0]
        if count != 0:
            raise RuntimeError("Inactive-week audit conflicts with remaining market rows.")
    return {"remaining_pregame_games": 0, "selected_lines": 0, "market_run_id": row["run_id"]}


def stream_process(
    command: list[str],
    cwd: Path,
    logger: TeeLogger,
) -> int:
    logger.write(subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.raw(line)
    return int(process.wait())


def validate_expectation(
    db_path: Path,
    expectation: TableExpectation,
    prediction_week: Optional[int],
) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    with sqlite3.connect(db_path) as connection:
        if not table_exists(connection, expectation.table_name):
            raise RuntimeError(
                f"Expected table was not created: {expectation.table_name}"
            )
        escaped = expectation.table_name.replace('"', '""')
        columns = table_columns(connection, expectation.table_name)
        row_count = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{escaped}"'
            ).fetchone()[0]
        )
        if row_count < expectation.minimum_rows:
            raise RuntimeError(
                f"{expectation.table_name} has {row_count:,} rows; "
                f"minimum is {expectation.minimum_rows:,}."
            )

        unique_count = None
        if expectation.expected_unique is not None:
            unique_column = first_existing(
                columns,
                expectation.unique_candidates,
            )
            if unique_column is None:
                raise RuntimeError(
                    f"{expectation.table_name} lacks a recognized unique-team "
                    f"column: {expectation.unique_candidates}"
                )
            unique_count = int(
                connection.execute(
                    f'SELECT COUNT(DISTINCT "{unique_column}") '
                    f'FROM "{escaped}"'
                ).fetchone()[0]
            )
            if unique_count != expectation.expected_unique:
                raise RuntimeError(
                    f"{expectation.table_name}.{unique_column} has "
                    f"{unique_count} unique values; expected "
                    f"{expectation.expected_unique}."
                )

        week_rows = None
        if expectation.require_prediction_week and prediction_week is not None:
            week_column = first_existing(
                columns,
                expectation.week_candidates,
            )
            if week_column is None:
                raise RuntimeError(
                    f"{expectation.table_name} lacks a recognized week column."
                )
            week_rows = int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{escaped}" '
                    f'WHERE CAST("{week_column}" AS INTEGER)=?',
                    (int(prediction_week),),
                ).fetchone()[0]
            )
            if week_rows < 1:
                raise RuntimeError(
                    f"{expectation.table_name} has no rows for Week "
                    f"{prediction_week}."
                )

        return {
            "table_name": expectation.table_name,
            "row_count": row_count,
            "unique_count": unique_count,
            "prediction_week_rows": week_rows,
        }


def validate_source_outage(db_path: Path, step_started: dt.datetime) -> dict[str, Any]:
    """Exit 20 is recoverable only with this attempt's explicit outage marker."""
    with sqlite3.connect(db_path) as connection:
        records = connection.execute(
            f'SELECT status, attempted_at_utc, error FROM "{REFRESH_STATUS_TABLE}"'
        ).fetchall()
    if len(records) != 1 or records[0][0] != "SOURCE_UNAVAILABLE":
        raise RuntimeError("Depth exit 20 lacks a valid SOURCE_UNAVAILABLE refresh marker")
    marker_status, attempted, reason = records[0]
    stamp = dt.datetime.fromisoformat(str(attempted).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise RuntimeError("Source-outage marker must include a UTC offset")
    if stamp.astimezone(dt.timezone.utc) < step_started.astimezone(dt.timezone.utc):
        raise RuntimeError("Source-outage marker predates the current depth attempt")
    return {"table_name": REFRESH_STATUS_TABLE, "status": marker_status,
            "attempted_at_utc": attempted, "error": reason,
            "adjusted_projections": "WITHHELD"}


def require_existing_frozen_model(project_root: Path) -> None:
    """Recovery cannot create a first structural freeze from unrefreshed inputs."""
    required = [project_root / "models" / filename for filename in
                ("nfl_learned_consensus_2026.joblib", "nfl_learned_consensus_2026_metadata.json")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("Baseline-only recovery requires the existing frozen model and metadata: " + ", ".join(missing))


def write_audits(
    db_path: Path,
    run_row: dict[str, Any],
    step_rows: list[dict[str, Any]],
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        import pandas as pd

        run_frame = pd.DataFrame([run_row])
        step_frame = pd.DataFrame(step_rows)

        run_frame.to_sql(
            RUN_AUDIT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        run_frame.to_sql(
            RUN_AUDIT_HISTORY_TABLE,
            connection,
            if_exists="append",
            index=False,
        )
        if not step_frame.empty:
            step_frame.to_sql(
                STEP_AUDIT_TABLE,
                connection,
                if_exists="replace",
                index=False,
            )
            step_frame.to_sql(
                STEP_AUDIT_HISTORY_TABLE,
                connection,
                if_exists="append",
                index=False,
            )


def acquire_lock(lock_path: Path, run_id: str, force_unlock: bool) -> None:
    if lock_path.exists() and force_unlock:
        lock_path.unlink(missing_ok=True)
    if lock_path.exists():
        existing = lock_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            "Another weekly run may already be active. "
            f"Lock: {lock_path} | contents={existing}"
        )
    lock_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_self_test() -> int:
    for week in (1, 2, 18):
        settings = argparse.Namespace(week=week, through_week=None)
        configure_form_cutoff(settings, list(PIPELINE_STEPS))
        if settings.through_week != week - 1:
            raise AssertionError("Prediction-week cutoff resolution failed.")
    try:
        configure_form_cutoff(
            argparse.Namespace(week=1, through_week=1), list(PIPELINE_STEPS)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Current-week form cutoff was not rejected.")
    zero_command: list[str] = []
    add_if_supported(zero_command, {"--through-week"}, "--through-week", 0)
    if zero_command != ["--through-week", "0"]:
        raise AssertionError("Week-zero cutoff was incorrectly omitted.")
    optional_command: list[str] = []
    add_if_supported(
        optional_command,
        {"--as-of-date", "--through-week", "--no-csv"},
        "--as-of-date",
        None,
    )
    add_if_supported(
        optional_command,
        {"--as-of-date", "--through-week", "--no-csv"},
        "--through-week",
        None,
    )
    add_if_supported(
        optional_command,
        {"--as-of-date", "--through-week", "--no-csv"},
        "--no-csv",
        enabled=True,
    )
    if optional_command != ["--no-csv"]:
        raise AssertionError(
            "Optional-argument omission or Boolean-flag handling failed: "
            f"{optional_command}"
        )

    value_command: list[str] = []
    add_if_supported(
        value_command,
        {"--week", "--as-of-date"},
        "--week",
        3,
    )
    add_if_supported(
        value_command,
        {"--week", "--as-of-date"},
        "--as-of-date",
        "2026-09-10",
    )
    if value_command != [
        "--week",
        "3",
        "--as-of-date",
        "2026-09-10",
    ]:
        raise AssertionError(
            f"Value-taking option handling failed: {value_command}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:  # type: ignore[name-defined]
        root = Path(temp_dir)
        db_path = root / "test.sqlite"
        script = root / "dummy.py"
        script.write_text(
            """
import argparse, sqlite3
p=argparse.ArgumentParser()
p.add_argument("--db-path")
p.add_argument("--week", type=int)
a=p.parse_args()
with sqlite3.connect(a.db_path) as c:
    c.execute("CREATE TABLE IF NOT EXISTS nfl_weekly_power_spread_predictions_2026 (week INTEGER)")
    c.execute("DELETE FROM nfl_weekly_power_spread_predictions_2026")
    c.execute("INSERT INTO nfl_weekly_power_spread_predictions_2026 VALUES (?)",(a.week,))
""".strip(),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(script), "--db-path", str(db_path), "--week", "3"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        outcome = validate_expectation(
            db_path,
            TableExpectation(
                "nfl_weekly_power_spread_predictions_2026",
                week_candidates=("week",),
                require_prediction_week=True,
            ),
            3,
        )
        if outcome["prediction_week_rows"] != 1:
            raise AssertionError("Week validation failed.")
    print("[NFL_WEEKLY_FINAL] Self-test passed.")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.diagnose_sources:
        diagnostic = args.project_root / "diagnose_nfl_sources_2026.py"
        if not diagnostic.is_file():
            raise RuntimeError(f"Missing diagnostic script: {diagnostic}")
        return subprocess.call([sys.executable, str(diagnostic), "--project-root", str(args.project_root),
                                "--db-path", str(args.db_path), "--week", str(args.week or 2)])

    project_root = args.project_root
    project_root.mkdir(parents=True, exist_ok=True)
    log_directory = project_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    started_at = dt.datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    log_path = (
        log_directory
        / f"run_nfl_weekly_2026_final_{timestamp}_{run_id[:8]}.log"
    )
    logger = TeeLogger(log_path)
    lock_path = project_root / LOCK_FILENAME

    step_rows: list[dict[str, Any]] = []
    status = "FAILED"
    error_text: Optional[str] = None
    baseline_only = False

    try:
        steps = selected_steps(args)
        if not steps:
            raise RuntimeError("No pipeline steps were selected.")

        prediction_selected = any(
            step.key == "weekly_predictions"
            for step in steps
        )
        if prediction_selected and args.week is None:
            raise RuntimeError(
                "--week is required whenever weekly_predictions is selected. "
                "Example: run_nfl_weekly_2026_final.py --week 1"
            )

        configure_form_cutoff(args, steps)

        acquire_lock(lock_path, run_id, args.force_unlock)

        logger.section("[NFL_WEEKLY_FINAL] RUN CONFIGURATION")
        logger.write(f"Build ID:       {BUILD_ID}")
        logger.write(f"Version:        {VERSION}")
        logger.write(f"Run ID:         {run_id}")
        logger.write(f"Project root:   {project_root}")
        logger.write(f"Database:       {args.db_path}")
        logger.write(f"Prediction week:{args.week}")
        logger.write(f"Form cutoff:    {args.through_week}")
        logger.write(f"Steps:          {[step.key for step in steps]}")
        logger.write(
            "Market input:   "
            + (
                "DISABLED (fair-only)"
                if effective_market_path(args) is None
                else str(effective_market_path(args))
            )
        )
        if any(step.key == "live_market" for step in steps):
            logger.write(
                "Market books:   " + args.market_bookmaker_priority
            )
            logger.write(
                "Market max age: "
                f"{args.market_max_age_minutes:g} minutes"
            )
        logger.write(
            "Official weekly model: LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS / "
            "frozen 2020-2025 models / point-in-time form, process, personnel"
        )
        logger.write("Circa contest model run here: NO")

        resolved: list[tuple[PipelineStep, Path]] = []
        for step in steps:
            resolved.append((step, resolve_script(project_root, step)))

        if args.dry_run:
            logger.section("[NFL_WEEKLY_FINAL] DRY RUN")
            for step, script_path in resolved:
                command = build_child_command(args, step, script_path)
                logger.write(
                    f"{step.key}: {subprocess.list2cmdline(command)}"
                )
            logger.write("Dry run complete; no child scripts executed.")
            status = "DRY_RUN"
            return 0

        for sequence, (step, script_path) in enumerate(resolved, start=1):
            step_started = dt.datetime.now()
            logger.section(
                f"[NFL_WEEKLY_FINAL] STEP {sequence}/{len(resolved)} "
                f"— {step.description}"
            )
            if baseline_only and step.key in LIVE_STRUCTURE_STEPS:
                validations = [
                    {**validate_expectation(args.db_path, expectation, args.week),
                     "validation_scope": "EXISTING_TABLE_NOT_REFRESHED"}
                    for expectation in step.expectations
                ]
                step_finished = dt.datetime.now()
                step_rows.append({
                    "run_id": run_id, "sequence": sequence, "step_key": step.key,
                    "description": step.description, "script_path": str(script_path),
                    "command": "", "exit_code": None, "status": "SKIPPED_SOURCE_UNAVAILABLE",
                    "validation_json": json.dumps(validations, default=str),
                    "started_at": step_started.isoformat(timespec="seconds"),
                    "finished_at": step_finished.isoformat(timespec="seconds"),
                    "elapsed_seconds": (step_finished - step_started).total_seconds(),
                })
                logger.write(f"[NFL_WEEKLY_FINAL] Skipped {step.key}: source unavailable; existing outputs validated, not refreshed.")
                continue
            command = build_child_command(args, step, script_path)
            exit_code = stream_process(command, project_root, logger)
            validations: list[dict[str, Any]] = []
            source_outage = step.key == "depth_chart" and exit_code == SOURCE_UNAVAILABLE_EXIT_CODE
            if source_outage:
                outage = validate_source_outage(args.db_path, step_started)
                validations.append(outage)
                validations.append({
                    **validate_expectation(args.db_path, TableExpectation("nfl_projected_depth_chart_2026", minimum_rows=500), args.week),
                    "validation_scope": "EXISTING_TABLE_NOT_REFRESHED",
                })
                if prediction_selected or any(item.key == "learned_model" for item in steps):
                    require_existing_frozen_model(project_root)
                baseline_only = True
                error_text = f"Availability source unavailable; baseline only: {outage['error']}"
                logger.write("[NFL_WEEKLY_FINAL][WARN] BASELINE ONLY: current availability is unavailable. Preserving existing structural tables, continuing the frozen baseline, and withholding adjusted lines.")
            if exit_code == 0:
                finished_week = (
                    completed_market_week_audit(args.db_path, args.week, step_started)
                    if step.key == "live_market" else None
                )
                if finished_week is not None:
                    validations.append(finished_week)
                for expectation in (() if finished_week is not None else step.expectations):
                    validations.append(
                        validate_expectation(
                            args.db_path,
                            expectation,
                            args.week,
                        )
                    )
                if step.key == "form_ratings" and args.through_week is not None:
                    validations.append(
                        validate_form_cutoff(args.db_path, int(args.through_week))
                    )

            step_finished = dt.datetime.now()
            step_row = {
                "run_id": run_id,
                "sequence": sequence,
                "step_key": step.key,
                "description": step.description,
                "script_path": str(script_path),
                "command": subprocess.list2cmdline(command),
                "exit_code": exit_code,
                "status": "SOURCE_UNAVAILABLE" if source_outage else "SUCCESS" if exit_code == 0 else "FAILED",
                "validation_json": json.dumps(validations, default=str),
                "started_at": step_started.isoformat(timespec="seconds"),
                "finished_at": step_finished.isoformat(timespec="seconds"),
                "elapsed_seconds": (
                    step_finished - step_started
                ).total_seconds(),
            }
            step_rows.append(step_row)

            if exit_code != 0 and not source_outage:
                raise RuntimeError(
                    f"Pipeline stopped after {step.key}; exit code={exit_code}."
                )

            logger.write(
                f"[NFL_WEEKLY_FINAL] Step validated: "
                f"{json.dumps(validations, default=str)}"
            )

        status = "BASELINE_ONLY_SOURCE_UNAVAILABLE" if baseline_only else "SUCCESS"
        logger.section("[NFL_WEEKLY_FINAL] COMPLETED")
        logger.write("Completed in BASELINE ONLY mode; availability was not refreshed and adjusted lines were withheld."
                     if baseline_only else "All selected steps completed and validated.")
        logger.write(f"Master log: {log_path}")
        return 0

    except Exception as exc:
        error_text = str(exc)
        logger.section("[NFL_WEEKLY_FINAL] FAILED")
        logger.write(error_text)
        logger.write(traceback.format_exc())
        raise
    finally:
        finished_at = dt.datetime.now()
        run_row = {
            "run_id": run_id,
            "build_id": BUILD_ID,
            "version": VERSION,
            "status": status,
            "error": error_text,
            "prediction_week": args.week,
            "project_root": str(args.project_root),
            "db_path": str(args.db_path),
            "selected_steps": "|".join(row["step_key"] for row in step_rows),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "log_path": str(log_path),
        }
        try:
            write_audits(args.db_path, run_row, step_rows)
        except Exception as audit_exc:
            logger.write(f"[NFL_WEEKLY_FINAL][WARN] Audit write failed: {audit_exc}")
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.close()


if __name__ == "__main__":
    # tempfile is imported only for the isolated self-test path.
    import tempfile

    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[NFL_WEEKLY_FINAL] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[NFL_WEEKLY_FINAL] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
