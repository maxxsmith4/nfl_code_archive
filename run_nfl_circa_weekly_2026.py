#!/usr/bin/env python
"""Pull the official 2026 Circa board, run picks, and certify data integrity.

The runner reuses the audited official-PDF discovery and OCR implementation in
``build_backtest_nfl_circa_contest_lines.py``.  It never substitutes a
sportsbook line, never accepts a non-Circa host, and does not launch the
production predictor unless every scheduled game for the requested week has a
valid reconciled contest spread.

After prediction, a fail-closed weekly certificate independently verifies the
2026 source-game capture, prior-week cutoff, form, learned-consensus projection,
authoritative QB/OL inputs, frozen model lineage, and both five-pick entries.
The certificate is saved as JSON and appended to two audit tables.  ``--audit-only``
rechecks an already-generated week without fetching a PDF or rerunning models.

Normal use requires no week number: the current regular-season week is derived
from ``identifier.sqlite::nfl_schedule_2026``.  ``--week`` remains available
for an explicit replay or preflight.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd


BUILD_ID = "NFL_CIRCA_WEEKLY_2026_CANONICAL_V1"
VERSION = "v1_2_learned_consensus_confidence_certificate"
SEASON = 2026
SEASON_ROMAN = "VIII"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DATABASE = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)
SCHEDULE_TABLE = "nfl_schedule_2026"
HISTORICAL_BUILDER_FILENAME = "build_backtest_nfl_circa_contest_lines.py"
PREDICTOR_FILENAME = "predict_nfl_circa_top5_2026.py"
EXPECTED_HISTORICAL_BUILD = "NFL_CIRCA_CONTEST_LINES_CANONICAL_V1"
EXPECTED_HISTORICAL_VERSION = "v1_5_qb_identity_integrity_guard"
EXPECTED_PREDICTOR_BUILD = "NFL_CIRCA_TOP5_2026_PRODUCTION_CONFIDENCE_BOARD"
EXPECTED_PREDICTOR_VERSION = (
    "v8_4_learned_consensus_lineage_live_feature_freshness"
)

FORM_TABLE = "nfl_2026_form_ratings"
POWER_TABLE = "nfl_power_ratings_2026"
DEPTH_TABLE = "nfl_projected_depth_chart_2026"
STRUCTURAL_TABLE = "nfl_weekly_power_spread_predictions_2026"
PREDICTOR_AUDIT_TABLE = "nfl_circa_top5_run_audit_2026"
PREDICTOR_AUDIT_HISTORY_TABLE = "nfl_circa_top5_run_audit_history_2026"
PORTFOLIO_TABLE = "nfl_circa_final_portfolio_predictions_2026"
PORTFOLIO_HISTORY_TABLE = "nfl_circa_final_portfolio_prediction_history_2026"
TEAM_GAME_TABLE = "nfl_matchup_team_game_features"
PERSONNEL_TABLE = "nfl_matchup_weekly_personnel_features"
FEATURE_TABLE_CANDIDATES = (
    "nfl_circa_live_features_2026",
    "nfl_matchup_game_matrix_live_2026",
    "nfl_matchup_game_matrix_2026",
    "nfl_matchup_game_matrix",
)
INTEGRITY_RUN_TABLE = "nfl_circa_weekly_integrity_runs_2026"
INTEGRITY_DETAIL_TABLE = "nfl_circa_weekly_integrity_details_2026"
ENTRY_1_POLICY = "TB_V1_P37_D050"
ENTRY_2_POLICY = "CEILING_LATE_HOME_FAVORITE_7P5_VETO"
EXPECTED_FORM_BUILD = "NFL_2026_FORM_RATING_CANONICAL_V3"
EXPECTED_POWER_BUILD = "NFL_POWER_RATINGS_2026_CANONICAL_V2"
EXPECTED_DEPTH_BUILD = "NFL_PROJECTED_DEPTH_CHART_2026_CANONICAL_V5"
EXPECTED_STRUCTURAL_BUILD = "NFL_WEEKLY_LEARNED_CONSENSUS_2026_CANONICAL_V1"
EXPECTED_STRUCTURAL_VERSION = "v1_3_readable_execution_csv_scope_fix"
EXPECTED_STRUCTURAL_MODEL_VARIANT = (
    "LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS"
)
EXPECTED_LEARNED_BUNDLE_BUILD = "NFL_LEARNED_CONSENSUS_2026_CANONICAL_V1"
EXPECTED_LEARNED_BUNDLE_VERSION = (
    "v1_0_2020_2025_frozen_market_free_consensus"
)
EXPECTED_LEARNED_BUNDLE_SHA256 = (
    "5c298a2ef0555612525df2fcc0341d4bbb0c19be5dd6226089f2511f9fcbe524"
)
EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH = (
    "8ece49f9674a90e368743feac3eb15f0b88b23e648fc878ecbee0d0cd508b49f"
)
EXPECTED_V1_MODEL_BUILD = "NFL_CIRCA_CONTEST_LINES_CANONICAL_V1"
EXPECTED_V1_MODEL_VERSION = "v1_5_qb_identity_integrity_guard"
EXPECTED_CEILING_MODEL_BUILD = "NFL_CIRCA_CONTEST_SEASON_PHASE_V3_2_QB_REBUILT"
EXPECTED_CEILING_MODEL_VERSION = "v3_2_1_fixed_architecture_qb_integrity_lineage"
QB_COVERAGE_MINIMUM = 0.99
OL_STARTER_SLOTS = ("LT", "LG", "C", "RG", "RT")

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}


class BoardNotPublished(RuntimeError):
    """The official target-week Circa PDF is not available yet."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--db-path",
        "--database",
        dest="db_path",
        type=Path,
        default=DEFAULT_DATABASE,
    )
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--historical-builder-path", type=Path, default=None)
    parser.add_argument("--predictor-path", type=Path, default=None)
    parser.add_argument("--tesseract-path", type=Path, default=None)
    parser.add_argument(
        "--wait-minutes",
        type=float,
        default=0.0,
        help=(
            "If the board is not posted yet, poll for up to this many minutes. "
            "Use 180 for a scheduled Thursday run."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Pull and validate the board without launching the predictor.",
    )
    parser.add_argument("--rebuild-ocr", action="store_true")
    parser.add_argument(
        "--rebuild-live-features",
        action="store_true",
        help="Forward this request to the production predictor.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Recheck the saved official board, latest exact-week predictor "
            "run, 2026 inputs, and final cards without fetching or predicting."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    for name in (
        "historical_builder_path",
        "predictor_path",
        "tesseract_path",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.week is not None and not 1 <= args.week <= 18:
        parser.error("--week must be between 1 and 18.")
    if args.wait_minutes < 0:
        parser.error("--wait-minutes cannot be negative.")
    if not 30 <= args.poll_seconds <= 3600:
        parser.error("--poll-seconds must be between 30 and 3600.")
    if args.audit_only and (
        args.fetch_only
        or args.rebuild_ocr
        or args.rebuild_live_features
        or args.wait_minutes > 0
    ):
        parser.error(
            "--audit-only cannot be combined with fetching or rebuild options."
        )
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def normalize_team(value: Any) -> str:
    text = "" if value is None else str(value).upper().strip()
    return TEAM_ALIASES.get(text, text)


def first_existing(columns: Any, candidates: tuple[str, ...]) -> Optional[str]:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    """Convert pandas/numpy/path values into stable JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in columns:
        selected[column] = selected[column].map(
            lambda value: "" if pd.isna(value) else str(value).strip()
        )
    selected = selected.sort_values(columns).reset_index(drop=True)
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_sqlite_table(
    database: Path,
    table_name: str,
) -> pd.DataFrame:
    if not database.exists():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database)
    try:
        if not table_exists(connection, table_name):
            raise RuntimeError(f"Missing required table: {database}::{table_name}")
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
    finally:
        connection.close()


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone() is not None


def resolve_script(
    explicit: Optional[Path],
    project_root: Path,
    filename: str,
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    candidates = (
        project_root / filename,
        project_root / "backtests" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find {filename}; checked: "
        + "; ".join(str(path) for path in candidates)
    )


def source_marker(path: Path, name: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']",
        flags=re.MULTILINE,
    )
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"{path} has no literal {name} marker.")
    return match.group(1)


def load_historical_builder(path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        f"circa_historical_{uuid.uuid4().hex}",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import historical Circa builder: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if str(getattr(module, "BUILD_ID", "")) != EXPECTED_HISTORICAL_BUILD:
        raise RuntimeError(
            "Historical Circa builder has the wrong BUILD_ID: "
            f"{getattr(module, 'BUILD_ID', None)!r}"
        )
    if str(getattr(module, "VERSION", "")) != EXPECTED_HISTORICAL_VERSION:
        raise RuntimeError(
            "Historical Circa builder has the wrong VERSION: "
            f"{getattr(module, 'VERSION', None)!r}"
        )

    # Extend the proven archive parser to the live 2026 contest. Caches are
    # isolated from the historical training database and its PDF collection.
    module.SEASON_ROMAN[SEASON] = SEASON_ROMAN
    module.PDF_DIRECTORY_NAME = "nfl_circa_live_2026_pdfs"
    module.OCR_DIRECTORY_NAME = "nfl_circa_live_2026_ocr"
    return module


def verify_predictor(path: Path) -> tuple[str, str]:
    build_id = source_marker(path, "BUILD_ID")
    version = source_marker(path, "VERSION")
    if build_id != EXPECTED_PREDICTOR_BUILD:
        raise RuntimeError(
            f"Production predictor BUILD_ID is {build_id!r}; expected "
            f"{EXPECTED_PREDICTOR_BUILD!r}."
        )
    if version != EXPECTED_PREDICTOR_VERSION:
        raise RuntimeError(
            f"Production predictor VERSION is {version!r}; expected "
            f"{EXPECTED_PREDICTOR_VERSION!r}."
        )
    return build_id, version


def standardize_database_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    aliases = {
        "season": ("season", "season_year", "year"),
        "week": ("week", "game_week", "week_number"),
        "game_id": ("game_id", "id", "gsis_id"),
        "game_type": ("game_type", "season_type"),
        "home_team": ("home_team", "home", "home_team_abbr"),
        "away_team": ("away_team", "away", "away_team_abbr"),
        "gameday": ("gameday", "game_date", "date"),
        "home_score": ("home_score", "score_home", "home_points"),
        "away_score": ("away_score", "score_away", "away_points"),
        "spread_line": (
            "spread_line",
            "closing_spread",
            "market_home_margin",
        ),
    }
    output = pd.DataFrame(index=frame.index)
    for target, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        if source is not None:
            output[target] = frame[source]
        elif target == "season":
            output[target] = SEASON
        elif target == "game_type":
            output[target] = "REG"
        else:
            output[target] = np.nan

    required = ("week", "home_team", "away_team")
    missing = [name for name in required if output[name].isna().all()]
    if missing:
        raise RuntimeError(
            f"{SCHEDULE_TABLE} is missing usable columns: {missing}"
        )

    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output["week"] = pd.to_numeric(output["week"], errors="coerce")
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output["spread_line"] = pd.to_numeric(output["spread_line"], errors="coerce")
    output["home_team"] = output["home_team"].map(normalize_team)
    output["away_team"] = output["away_team"].map(normalize_team)
    output["gameday"] = pd.to_datetime(output["gameday"], errors="coerce")
    regular = output["game_type"].fillna("REG").astype(str).str.upper().isin(
        {"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"}
    )
    output = output[
        regular
        & output["season"].eq(SEASON)
        & output["week"].between(1, 18)
        & output["home_team"].ne("")
        & output["away_team"].ne("")
    ].copy()
    if output.empty:
        raise RuntimeError(f"{SCHEDULE_TABLE} has no 2026 regular-season games.")
    output[["season", "week"]] = output[["season", "week"]].astype(int)

    output["game_id"] = output["game_id"].astype("object")
    generated = (
        output["season"].astype(str)
        + "_W"
        + output["week"].astype(str).str.zfill(2)
        + "_"
        + output["away_team"]
        + "_"
        + output["home_team"]
    )
    missing_game_id = output["game_id"].isna() | output["game_id"].astype(
        str
    ).str.strip().isin({"", "nan", "None"})
    output.loc[missing_game_id, "game_id"] = generated.loc[missing_game_id]
    output["game_id"] = output["game_id"].astype(str).str.strip()
    output["actual_home_margin"] = output["home_score"] - output["away_score"]
    duplicate = output.duplicated(["week", "home_team", "away_team"], keep=False)
    if duplicate.any():
        raise RuntimeError(
            "The 2026 schedule contains duplicate matchups:\n"
            + output.loc[
                duplicate,
                ["week", "away_team", "home_team", "game_id"],
            ].to_string(index=False)
        )
    return output.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)


def load_schedule(
    db_path: Path,
    historical: ModuleType,
) -> tuple[pd.DataFrame, str]:
    if db_path.exists():
        with sqlite3.connect(db_path) as connection:
            if table_exists(connection, SCHEDULE_TABLE):
                raw = pd.read_sql_query(
                    f'SELECT * FROM "{SCHEDULE_TABLE}"',
                    connection,
                )
                return (
                    standardize_database_schedule(raw),
                    f"{db_path}::{SCHEDULE_TABLE}",
                )

    raw, source = historical.load_schedule_from_package([SEASON])
    standardized = historical.standardize_schedule(raw, (SEASON,))
    if standardized.empty:
        raise RuntimeError("The fallback NFL schedule returned no 2026 games.")
    return standardized, source


def detect_current_week(
    schedule: pd.DataFrame,
    explicit_week: Optional[int],
    today: Optional[dt.date] = None,
) -> int:
    if explicit_week is not None:
        return int(explicit_week)
    today = today or dt.datetime.now().astimezone().date()
    candidates: list[tuple[int, int]] = []
    for week, group in schedule.groupby("week", sort=True):
        dates = pd.to_datetime(group["gameday"], errors="coerce").dropna()
        if dates.empty:
            continue
        first = dates.min().date()
        last = dates.max().date()
        window_start = first - dt.timedelta(days=3)
        window_end = last + dt.timedelta(days=1)
        if window_start <= today <= window_end:
            distance = min(abs((today - first).days), abs((today - last).days))
            candidates.append((distance, int(week)))
    if not candidates:
        raise RuntimeError(
            f"Could not infer an NFL week for {today.isoformat()}. "
            "Pass --week explicitly."
        )
    return min(candidates)[1]


def historical_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mode="all",
        project_root=args.project_root,
        schedule_path=None,
        manifest_path=None,
        manual_lines_path=None,
        seasons=(SEASON,),
        tesseract_path=args.tesseract_path,
        ocr_dpi=300,
        ocr_psms=(3, 6, 11, 12),
        thresholds=(0.5,),
        fallback_ats_price=-110.0,
        request_timeout=30.0,
        request_pause=0.08,
        probe_missing=True,
        rebuild_downloads=True,
        rebuild_ocr=bool(args.rebuild_ocr),
        allow_incomplete=False,
        backfill_missing_2025_with_reference=False,
        no_csv=True,
    )


def revision_number(url: Any) -> int:
    match = re.search(r"-(\d+)\.pdf(?:$|[?#])", str(url), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def select_latest_manifest(
    manifest: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    if manifest.empty:
        raise BoardNotPublished(
            f"Official Circa Million VIII Week {week} PDF was not found."
        )
    frame = manifest.copy()
    season = pd.to_numeric(frame.get("season"), errors="coerce")
    weeks = pd.to_numeric(frame.get("week"), errors="coerce")
    status = frame.get("download_status", pd.Series(index=frame.index, dtype=str))
    frame = frame[
        season.eq(SEASON)
        & weeks.eq(week)
        & status.isin({"DOWNLOADED", "CACHED"})
    ].copy()
    if frame.empty:
        raise BoardNotPublished(
            f"Official Circa Million VIII Week {week} PDF is not published yet."
        )
    frame["_wordpress_date"] = pd.to_datetime(
        frame.get("wordpress_date"), errors="coerce", utc=True
    )
    frame["_date_present"] = frame["_wordpress_date"].notna().astype(int)
    frame["_revision"] = frame["url"].map(revision_number)
    frame = frame.sort_values(
        ["_date_present", "_wordpress_date", "_revision", "url"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return frame.head(1).drop(
        columns=["_wordpress_date", "_date_present", "_revision"],
        errors="ignore",
    )


def validate_complete_board(
    game_lines: pd.DataFrame,
    week_audit: pd.DataFrame,
    schedule_week: pd.DataFrame,
    historical: ModuleType,
) -> None:
    if len(week_audit) != 1:
        raise RuntimeError(
            f"Expected one Week {int(schedule_week.week.iloc[0])} audit row; "
            f"received {len(week_audit)}."
        )
    audit = week_audit.iloc[0]
    expected_games = int(schedule_week["game_id"].nunique())
    if (
        int(audit["week_complete"]) != 1
        or int(audit["scheduled_games"]) != expected_games
        or int(audit["valid_games"]) != expected_games
        or int(audit["missing_games"]) != 0
    ):
        unresolved = game_lines[
            pd.to_numeric(game_lines.get("line_valid"), errors="coerce")
            .fillna(0)
            .ne(1)
        ][["away_team", "home_team"]]
        raise RuntimeError(
            "Official Circa board OCR is incomplete; production prediction is "
            "blocked. Unresolved games:\n"
            + unresolved.to_string(index=False)
        )
    if len(game_lines) != expected_games:
        raise RuntimeError(
            f"Parsed board has {len(game_lines)} rows; expected {expected_games}."
        )
    if game_lines.duplicated(["away_team", "home_team"]).any():
        raise RuntimeError("Parsed board contains duplicate matchups.")
    margins = pd.to_numeric(game_lines["circa_home_margin"], errors="coerce")
    if margins.isna().any() or not np.isfinite(margins).all():
        raise RuntimeError("Parsed board contains missing or non-finite spreads.")
    if not np.allclose(margins * 2.0, np.round(margins * 2.0), atol=1e-9):
        raise RuntimeError("Parsed board contains a spread outside half-point increments.")
    manual = pd.to_numeric(
        game_lines.get("manual_override", 0), errors="coerce"
    ).fillna(0)
    if manual.ne(0).any():
        raise RuntimeError("Live automation cannot use manual line overrides.")
    urls = game_lines.get("url", pd.Series(index=game_lines.index, dtype=str))
    if urls.isna().any() or not urls.map(historical.official_circa_pdf_url).all():
        raise RuntimeError("At least one parsed line lacks an official Circa PDF URL.")


def fetch_board_once(
    args: argparse.Namespace,
    historical: ModuleType,
    schedule_week: pd.DataFrame,
    week: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hargs = historical_args(args)
    manifest = historical.discover_and_download(hargs, schedule_week)
    selected_manifest = select_latest_manifest(manifest, week)
    attempts, team_lines, raw_game_lines = historical.ocr_pdf_versions(
        hargs,
        selected_manifest,
        schedule_week,
    )
    if attempts.empty or raw_game_lines.empty:
        raise RuntimeError("The official PDF downloaded but produced no OCR lines.")
    game_lines, week_audit = historical.finalize_lines(
        schedule_week,
        attempts,
        raw_game_lines,
        pd.DataFrame(),
    )
    validate_complete_board(game_lines, week_audit, schedule_week, historical)
    return selected_manifest, attempts, team_lines, game_lines


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_live_outputs(
    args: argparse.Namespace,
    week: int,
    schedule_source: str,
    selected_manifest: pd.DataFrame,
    attempts: pd.DataFrame,
    game_lines: pd.DataFrame,
) -> tuple[Path, Path]:
    inputs = args.project_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    lines_path = inputs / f"nfl_circa_lines_2026_week_{week}.csv"
    audit_path = inputs / f"nfl_circa_lines_2026_week_{week}_audit.json"

    output = game_lines[
        ["week", "away_team", "home_team", "circa_home_margin", "game_id"]
    ].copy()
    output["home_spread"] = -pd.to_numeric(
        output["circa_home_margin"], errors="raise"
    )
    output = output[
        ["week", "away_team", "home_team", "home_spread", "game_id"]
    ].sort_values(["week", "game_id"])
    temporary = lines_path.with_name(f"{lines_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        output.to_csv(temporary, index=False)
        os.replace(temporary, lines_path)
    finally:
        temporary.unlink(missing_ok=True)

    source = selected_manifest.iloc[0]
    source_pdf = Path(str(source["local_path"]))
    audit = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "season": SEASON,
        "week": week,
        "schedule_source": schedule_source,
        "scheduled_games": len(output),
        "valid_games": len(output),
        "complete_board": True,
        "manual_overrides": 0,
        "official_pdf_url": str(source["url"]),
        "official_pdf_path": str(source_pdf),
        "official_pdf_sha256": sha256_file(source_pdf),
        "official_pdf_discovery_method": str(source["discovery_method"]),
        "ocr_attempts": int(attempts["attempt_id"].nunique()),
        "line_csv_path": str(lines_path),
        "line_csv_sha256": sha256_file(lines_path),
        "created_at": now_string(),
    }
    write_json_atomic(audit_path, audit)
    return lines_path, audit_path


def run_predictor(
    args: argparse.Namespace,
    predictor_path: Path,
    lines_path: Path,
    week: int,
) -> str:
    started_at = now_string()
    command = [
        sys.executable,
        "-u",
        str(predictor_path),
        "--project-root",
        str(args.project_root),
        "--db-path",
        str(args.db_path),
        "--week",
        str(week),
        "--circa-lines-path",
        str(lines_path),
        "--approve-contest-forward-test",
    ]
    if args.rebuild_live_features:
        command.append("--rebuild-live-features")
    print("=" * 120)
    print("[CIRCA_WEEKLY] Launching frozen production Top 5 predictor")
    print(subprocess.list2cmdline(command))
    print("=" * 120)
    process = subprocess.Popen(
        command,
        cwd=args.project_root,
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
            f"Production Top 5 predictor failed with return code {return_code}."
        )
    return started_at


def add_integrity_check(
    details: list[dict[str, Any]],
    category: str,
    check_name: str,
    passed: bool,
    expected: Any,
    observed: Any,
    message: str,
    required: bool = True,
) -> None:
    details.append(
        {
            "category": category,
            "check_name": check_name,
            "status": "PASS" if passed else ("FAIL" if required else "INFO"),
            "required": int(required),
            "expected_json": json.dumps(
                json_value(expected), sort_keys=True, separators=(",", ":")
            ),
            "observed_json": json.dumps(
                json_value(observed), sort_keys=True, separators=(",", ":")
            ),
            "message": message,
        }
    )


def frame_for_season_week(
    frame: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    season = pd.to_numeric(frame.get("season"), errors="coerce")
    week_column = first_existing(frame.columns, ("week", "prediction_week"))
    if week_column is None:
        return pd.DataFrame(columns=frame.columns)
    weeks = pd.to_numeric(frame[week_column], errors="coerce")
    return frame[season.eq(SEASON) & weeks.eq(int(week))].copy()


def latest_predictor_run(
    db_path: Path,
    week: int,
    minimum_created_at: Optional[str],
) -> pd.Series:
    candidates: list[pd.DataFrame] = []
    connection = sqlite3.connect(db_path)
    try:
        for table_name in (
            PREDICTOR_AUDIT_HISTORY_TABLE,
            PREDICTOR_AUDIT_TABLE,
        ):
            if not table_exists(connection, table_name):
                continue
            frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
            subset = frame_for_season_week(frame, week)
            if not subset.empty:
                subset["_audit_table"] = table_name
                candidates.append(subset)
    finally:
        connection.close()
    if not candidates:
        raise RuntimeError(
            f"No exact Week {week} production predictor audit exists in {db_path}."
        )
    combined = pd.concat(candidates, ignore_index=True)
    combined["_created"] = pd.to_datetime(
        combined.get("created_at"), errors="coerce", utc=True
    )
    if minimum_created_at is not None:
        minimum = pd.Timestamp(minimum_created_at)
        if minimum.tzinfo is None:
            minimum = minimum.tz_localize("UTC")
        eligible = combined[combined["_created"].ge(minimum)].copy()
        if eligible.empty:
            newest = combined["_created"].max()
            raise RuntimeError(
                "The predictor returned successfully, but no new exact-week "
                f"audit was written after {minimum_created_at}; newest={newest}."
            )
        combined = eligible
    combined = combined.sort_values(["_created", "_audit_table"])
    return combined.iloc[-1]


def portfolio_for_run(
    db_path: Path,
    week: int,
    predictor_run_id: str,
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    connection = sqlite3.connect(db_path)
    try:
        for table_name in (PORTFOLIO_HISTORY_TABLE, PORTFOLIO_TABLE):
            if not table_exists(connection, table_name):
                continue
            frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
            subset = frame_for_season_week(frame, week)
            if "run_id" in subset.columns:
                subset = subset[
                    subset["run_id"].astype(str).eq(predictor_run_id)
                ].copy()
            if not subset.empty:
                candidates.append(subset)
    finally:
        connection.close()
    if not candidates:
        raise RuntimeError(
            f"No final portfolio rows exist for predictor run {predictor_run_id}."
        )
    combined = pd.concat(candidates, ignore_index=True)
    keys = [
        column
        for column in ("run_id", "entry_number", "contest_rank", "game_id")
        if column in combined.columns
    ]
    return combined.drop_duplicates(keys, keep="last").reset_index(drop=True)


def parse_feature_source(source: str) -> tuple[Path, str]:
    if "::" not in source:
        raise RuntimeError(
            "Predictor feature_source does not identify an exact SQLite table: "
            f"{source!r}"
        )
    database_text, table_name = source.rsplit("::", 1)
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
        raise RuntimeError(f"Unsafe feature table name in source: {source!r}")
    return Path(database_text).resolve(), table_name


def file_hash_if_present(value: Any) -> tuple[str, str]:
    path = Path(str(value))
    if not path.exists() or not path.is_file():
        return str(path), ""
    return str(path.resolve()), sha256_file(path)


def load_saved_board(
    args: argparse.Namespace,
    week: int,
) -> tuple[Path, Path, pd.DataFrame, dict[str, Any]]:
    lines_path = (
        args.project_root / "inputs" / f"nfl_circa_lines_2026_week_{week}.csv"
    )
    audit_path = lines_path.with_name(
        f"nfl_circa_lines_2026_week_{week}_audit.json"
    )
    if not lines_path.exists():
        raise FileNotFoundError(
            f"Saved official Week {week} Circa line file is missing: {lines_path}"
        )
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Saved official Week {week} Circa line audit is missing: {audit_path}"
        )
    board = pd.read_csv(lines_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return lines_path, audit_path, board, audit


def expected_team_games(schedule: pd.DataFrame, week: int) -> dict[str, int]:
    prior = schedule[schedule["week"].lt(week)]
    teams = sorted(
        set(schedule["home_team"].astype(str))
        | set(schedule["away_team"].astype(str))
    )
    counts = {team: 0 for team in teams}
    for row in prior.itertuples(index=False):
        counts[str(row.home_team)] += 1
        counts[str(row.away_team)] += 1
    return counts


def table_or_empty(
    database: Path,
    table_name: str,
    category: str,
    details: list[dict[str, Any]],
    allow_empty: bool = False,
) -> pd.DataFrame:
    try:
        frame = read_sqlite_table(database, table_name)
    except Exception as exc:
        add_integrity_check(
            details,
            category,
            f"table_{table_name}",
            False,
            f"nonempty {database}::{table_name}",
            str(exc),
            "Required weekly lineage table is unavailable.",
        )
        return pd.DataFrame()
    add_integrity_check(
        details,
        category,
        f"table_{table_name}",
        allow_empty or not frame.empty,
        "present" if allow_empty else "nonempty",
        {"rows": len(frame), "columns": len(frame.columns)},
        "Required weekly lineage table is present.",
    )
    return frame


def audit_model_files(
    predictor_audit: pd.Series,
    previous_payload: Optional[dict[str, Any]],
    details: list[dict[str, Any]],
) -> dict[str, str]:
    model_path, model_hash = file_hash_if_present(predictor_audit.get("model_path", ""))
    ceiling_path, ceiling_hash = file_hash_if_present(
        predictor_audit.get("ceiling_model_path", "")
    )
    add_integrity_check(
        details,
        "MODEL",
        "frozen_model_files_exist",
        bool(model_hash and ceiling_hash),
        "both V1 and ceiling joblib files exist",
        {
            "v1_path": model_path,
            "v1_sha256": model_hash,
            "ceiling_path": ceiling_path,
            "ceiling_sha256": ceiling_hash,
        },
        "The exact model artifacts used by the predictor are hashable.",
    )
    previous_metrics = (previous_payload or {}).get("metrics", {})
    previous_v1_hash = str(previous_metrics.get("v1_model_sha256", ""))
    previous_ceiling_hash = str(previous_metrics.get("ceiling_model_sha256", ""))
    if previous_v1_hash and previous_ceiling_hash:
        unchanged = (
            model_hash == previous_v1_hash
            and ceiling_hash == previous_ceiling_hash
        )
        add_integrity_check(
            details,
            "MODEL",
            "frozen_model_hashes_unchanged",
            unchanged,
            {
                "v1_sha256": previous_v1_hash,
                "ceiling_sha256": previous_ceiling_hash,
            },
            {"v1_sha256": model_hash, "ceiling_sha256": ceiling_hash},
            "Production model binaries must not change between certified weeks.",
        )
    else:
        add_integrity_check(
            details,
            "MODEL",
            "frozen_model_hashes_unchanged",
            True,
            "prior certificate when available",
            "FIRST_CERTIFICATE",
            "No prior certified model hashes are available for comparison.",
            required=False,
        )
    return {
        "v1_model_path": model_path,
        "v1_model_sha256": model_hash,
        "ceiling_model_path": ceiling_path,
        "ceiling_model_sha256": ceiling_hash,
    }


def build_integrity_certificate(
    args: argparse.Namespace,
    schedule: pd.DataFrame,
    schedule_source: str,
    week: int,
    lines_path: Path,
    lines_audit_path: Path,
    predictor_started_at: Optional[str],
    audit_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    audit_run_id = uuid.uuid4().hex
    created_at = now_string()
    cutoff_week = week - 1
    details: list[dict[str, Any]] = []
    target_schedule = schedule[schedule["week"].eq(week)].copy()
    prior_schedule = schedule[schedule["week"].lt(week)].copy()
    prior_ids = set(prior_schedule["game_id"].astype(str))
    target_ids = set(target_schedule["game_id"].astype(str))
    week_added_ids = set(
        schedule[schedule["week"].eq(cutoff_week)]["game_id"].astype(str)
    ) if cutoff_week > 0 else set()
    team_game_expectation = expected_team_games(schedule, week)

    previous_path = (
        args.project_root
        / "outputs"
        / "audits"
        / f"nfl_circa_weekly_integrity_2026_week_{week - 1}.json"
    )
    previous_payload: Optional[dict[str, Any]] = None
    if week > 1 and previous_path.exists():
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
        if str(previous_payload.get("status", "")) != "PASS":
            previous_payload = None

    board = pd.read_csv(lines_path)
    line_audit = json.loads(lines_audit_path.read_text(encoding="utf-8"))
    board.columns = [str(column).lower().strip() for column in board.columns]
    for column in ("home_team", "away_team"):
        if column in board.columns:
            board[column] = board[column].map(normalize_team)
    board_ids = set(board.get("game_id", pd.Series(dtype=str)).astype(str))
    board_sha = sha256_file(lines_path)
    board_valid = (
        len(board) == len(target_schedule)
        and board_ids == target_ids
        and not board.duplicated(["away_team", "home_team"]).any()
        and int(line_audit.get("season", 0)) == SEASON
        and int(line_audit.get("week", 0)) == week
        and bool(line_audit.get("complete_board", False))
        and int(line_audit.get("manual_overrides", -1)) == 0
        and str(line_audit.get("line_csv_sha256", "")) == board_sha
    )
    add_integrity_check(
        details,
        "CIRCA_BOARD",
        "exact_official_board",
        board_valid,
        {
            "games": len(target_schedule),
            "game_ids": sorted(target_ids),
            "complete_board": True,
            "manual_overrides": 0,
            "line_csv_sha256": board_sha,
        },
        {
            "games": len(board),
            "game_ids": sorted(board_ids),
            "complete_board": line_audit.get("complete_board"),
            "manual_overrides": line_audit.get("manual_overrides"),
            "audit_line_csv_sha256": line_audit.get("line_csv_sha256"),
        },
        "Saved lines must be the complete, unchanged official target-week board.",
    )
    pdf_path = Path(str(line_audit.get("official_pdf_path", "")))
    pdf_url = str(line_audit.get("official_pdf_url", ""))
    pdf_expected_hash = str(line_audit.get("official_pdf_sha256", ""))
    pdf_actual_hash = (
        sha256_file(pdf_path) if pdf_path.exists() and pdf_path.is_file() else ""
    )
    pdf_source_ok = (
        bool(re.fullmatch(r"[0-9a-f]{64}", pdf_expected_hash))
        and pdf_actual_hash == pdf_expected_hash
        and "circasports.com/" in pdf_url.lower()
        and pdf_url.lower().split("?", 1)[0].endswith(".pdf")
    )
    add_integrity_check(
        details,
        "CIRCA_BOARD",
        "official_pdf_provenance",
        pdf_source_ok,
        "existing hash-matched PDF from circasports.com",
        {
            "url": pdf_url,
            "path": str(pdf_path),
            "expected_sha256": pdf_expected_hash,
            "actual_sha256": pdf_actual_hash,
        },
        "The saved line board must remain traceable to the exact official PDF bytes.",
    )
    spread = pd.to_numeric(board.get("home_spread"), errors="coerce")
    spread_valid = (
        len(spread) == len(board)
        and spread.notna().all()
        and np.isfinite(spread).all()
        and np.allclose(spread * 2.0, np.round(spread * 2.0), atol=1e-9)
    )
    add_integrity_check(
        details,
        "CIRCA_BOARD",
        "spread_values",
        bool(spread_valid),
        "finite half-point increments",
        spread.tolist(),
        "Every Circa spread must be finite and use a valid half point.",
    )

    predictor_audit = latest_predictor_run(
        args.db_path, week, predictor_started_at
    )
    predictor_run_id = str(predictor_audit.get("run_id", ""))
    portfolio = portfolio_for_run(args.db_path, week, predictor_run_id)
    add_integrity_check(
        details,
        "MODEL",
        "predictor_build_and_version",
        (
            str(predictor_audit.get("build_id", ""))
            == EXPECTED_PREDICTOR_BUILD
            and str(predictor_audit.get("version", ""))
            == EXPECTED_PREDICTOR_VERSION
        ),
        {
            "build_id": EXPECTED_PREDICTOR_BUILD,
            "version": EXPECTED_PREDICTOR_VERSION,
        },
        {
            "build_id": predictor_audit.get("build_id"),
            "version": predictor_audit.get("version"),
        },
        "The exact frozen production predictor must have generated the cards.",
    )
    model_lineage = {
        "model_build_id": predictor_audit.get("model_build_id"),
        "model_version": predictor_audit.get("model_version"),
        "ceiling_model_build_id": predictor_audit.get("ceiling_model_build_id"),
        "ceiling_model_version": predictor_audit.get("ceiling_model_version"),
        "rating_alpha": predictor_audit.get("rating_alpha"),
    }
    expected_model_lineage = {
        "model_build_id": EXPECTED_V1_MODEL_BUILD,
        "model_version": EXPECTED_V1_MODEL_VERSION,
        "ceiling_model_build_id": EXPECTED_CEILING_MODEL_BUILD,
        "ceiling_model_version": EXPECTED_CEILING_MODEL_VERSION,
        "rating_alpha": 10.0,
    }
    lineage_ok = (
        str(model_lineage["model_build_id"]) == EXPECTED_V1_MODEL_BUILD
        and str(model_lineage["model_version"]) == EXPECTED_V1_MODEL_VERSION
        and str(model_lineage["ceiling_model_build_id"]) == EXPECTED_CEILING_MODEL_BUILD
        and str(model_lineage["ceiling_model_version"]) == EXPECTED_CEILING_MODEL_VERSION
        and math.isclose(float(model_lineage["rating_alpha"]), 10.0, abs_tol=1e-9)
    )
    add_integrity_check(
        details,
        "MODEL",
        "frozen_bundle_lineage",
        lineage_ok,
        expected_model_lineage,
        model_lineage,
        "V1, ceiling, and frozen rating alpha must match the approved production lineage.",
    )
    frozen_flags = {
        "contest_forward_test_override": int(
            predictor_audit.get("contest_forward_test_override", 0)
        ),
        "production_promoted": int(predictor_audit.get("production_promoted", 0)),
        "immutable_model_card_preserved": int(
            predictor_audit.get("immutable_model_card_preserved", 0)
        ),
        "structural_used_to_change_model_card": int(
            predictor_audit.get("structural_used_to_change_model_card", -1)
        ),
        "sportsbook_staking_enabled": int(
            predictor_audit.get("sportsbook_staking_enabled", -1)
        ),
    }
    add_integrity_check(
        details,
        "MODEL",
        "frozen_production_flags",
        frozen_flags == {
            "contest_forward_test_override": 1,
            "production_promoted": 1,
            "immutable_model_card_preserved": 1,
            "structural_used_to_change_model_card": 0,
            "sportsbook_staking_enabled": 0,
        },
        {
            "contest_forward_test_override": 1,
            "production_promoted": 1,
            "immutable_model_card_preserved": 1,
            "structural_used_to_change_model_card": 0,
            "sportsbook_staking_enabled": 0,
        },
        frozen_flags,
        "Learned-consensus review cannot alter either frozen model card and staking remains disabled.",
    )
    model_file_metrics = audit_model_files(
        predictor_audit, previous_payload, details
    )

    feature_source = str(predictor_audit.get("feature_source", ""))
    feature_db, feature_table = parse_feature_source(feature_source)
    feature_matrix_all = table_or_empty(
        feature_db, feature_table, "FEATURE_MATRIX", details
    )
    feature_matrix = frame_for_season_week(feature_matrix_all, week)
    rating_alpha = pd.to_numeric(
        feature_matrix.get("rating_alpha"), errors="coerce"
    )
    if not feature_matrix.empty and rating_alpha.notna().any():
        feature_matrix = feature_matrix[
            rating_alpha.eq(float(predictor_audit.get("rating_alpha", 10.0)))
        ].copy()
    feature_ids = set(feature_matrix.get("game_id", pd.Series(dtype=str)).astype(str))
    add_integrity_check(
        details,
        "FEATURE_MATRIX",
        "exact_target_week_games",
        len(feature_matrix) == len(target_schedule) and feature_ids == target_ids,
        {"rows": len(target_schedule), "game_ids": sorted(target_ids)},
        {"rows": len(feature_matrix), "game_ids": sorted(feature_ids)},
        "The model matrix must contain exactly one row per scheduled target-week game.",
    )
    personnel_complete = pd.to_numeric(
        feature_matrix.get("personnel_complete"), errors="coerce"
    )
    add_integrity_check(
        details,
        "FEATURE_MATRIX",
        "personnel_complete",
        len(personnel_complete) == len(feature_matrix)
        and not feature_matrix.empty
        and personnel_complete.eq(1).all(),
        1.0,
        float(personnel_complete.eq(1).mean()) if len(personnel_complete) else 0.0,
        "Every target game must have complete personnel features.",
    )

    team_games = table_or_empty(
        feature_db,
        TEAM_GAME_TABLE,
        "CAPTURE",
        details,
        allow_empty=(week == 1),
    )
    if not team_games.empty:
        season_values = pd.to_numeric(team_games.get("season"), errors="coerce")
        week_values = pd.to_numeric(team_games.get("week"), errors="coerce")
        current_source = team_games[season_values.eq(SEASON)].copy()
        current_weeks = pd.to_numeric(current_source.get("week"), errors="coerce")
        captured_prior = current_source[current_weeks.lt(week)].copy()
        captured_ids = set(captured_prior["game_id"].astype(str))
        leaking = current_source[current_weeks.ge(week)]
    else:
        captured_prior = pd.DataFrame()
        captured_ids = set()
        leaking = pd.DataFrame()
    source_fingerprint = hashlib.sha256(
        "\n".join(sorted(captured_ids)).encode("utf-8")
    ).hexdigest()
    add_integrity_check(
        details,
        "CAPTURE",
        "all_prior_games_captured",
        captured_ids == prior_ids,
        {"games": len(prior_ids), "game_ids": sorted(prior_ids)},
        {"games": len(captured_ids), "game_ids": sorted(captured_ids)},
        "The 2026 play-by-play-derived source must contain every prior scheduled game.",
    )
    expected_team_game_rows = 2 * len(prior_ids)
    add_integrity_check(
        details,
        "CAPTURE",
        "two_team_rows_per_prior_game",
        len(captured_prior) == expected_team_game_rows,
        expected_team_game_rows,
        len(captured_prior),
        "Each completed game must contribute one offensive row for each team.",
    )
    add_integrity_check(
        details,
        "CAPTURE",
        "no_current_or_future_game_leakage",
        leaking.empty,
        0,
        len(leaking),
        "No Week W or future 2026 result may enter a Week W prediction.",
    )
    prior_scores_complete = (
        prior_schedule[["home_score", "away_score"]].notna().all(axis=None)
        if not prior_schedule.empty
        else True
    )
    add_integrity_check(
        details,
        "CAPTURE",
        "prior_schedule_results_complete",
        bool(prior_scores_complete),
        len(prior_schedule),
        int(
            prior_schedule[["home_score", "away_score"]]
            .notna()
            .all(axis=1)
            .sum()
        ) if not prior_schedule.empty else 0,
        "All schedule games before the prediction week must have final scores.",
    )
    if week > 1 and previous_payload is not None:
        previous_metrics = previous_payload.get("metrics", {})
        previous_ids = set(previous_metrics.get("captured_game_ids", []))
        previous_fingerprint = str(
            previous_metrics.get("source_game_fingerprint", "")
        )
        capture_advanced = (
            previous_ids.issubset(captured_ids)
            and week_added_ids.issubset(captured_ids)
            and source_fingerprint != previous_fingerprint
        )
        add_integrity_check(
            details,
            "CAPTURE",
            "weekly_capture_advanced",
            capture_advanced,
            {
                "retains_previous_games": True,
                "adds_week": cutoff_week,
                "new_game_ids": sorted(week_added_ids),
                "fingerprint_changes": True,
            },
            {
                "retained_previous": previous_ids.issubset(captured_ids),
                "new_games_present": week_added_ids.issubset(captured_ids),
                "previous_fingerprint": previous_fingerprint,
                "current_fingerprint": source_fingerprint,
            },
            "The captured 2026 source must advance from the prior certified week.",
        )
    else:
        add_integrity_check(
            details,
            "CAPTURE",
            "weekly_capture_advanced",
            True,
            "prior certificate when available",
            "WEEK1_OR_FIRST_CERTIFICATE",
            "No earlier certificate is available; exact schedule coverage remains required.",
            required=False,
        )

    personnel = table_or_empty(feature_db, PERSONNEL_TABLE, "PERSONNEL", details)
    personnel_week = frame_for_season_week(personnel, week)
    if not personnel_week.empty and "team" in personnel_week.columns:
        personnel_week["team"] = personnel_week["team"].map(normalize_team)
        actual_history = dict(
            zip(
                personnel_week["team"],
                pd.to_numeric(
                    personnel_week.get("history_games"), errors="coerce"
                ).fillna(-1).astype(int),
            )
        )
    else:
        actual_history = {}
    add_integrity_check(
        details,
        "PERSONNEL",
        "bye_aware_team_history_counts",
        len(personnel_week) == 32 and actual_history == team_game_expectation,
        team_game_expectation,
        actual_history,
        "Every team history count must equal its scheduled completed games, including byes.",
    )

    identity_fields = ("home_qb_identity_matched", "away_qb_identity_matched")
    identity_values = pd.concat(
        [
            pd.to_numeric(feature_matrix.get(column), errors="coerce")
            for column in identity_fields
        ],
        ignore_index=True,
    ) if not feature_matrix.empty else pd.Series(dtype=float)
    identity_rate = float(identity_values.eq(1).mean()) if len(identity_values) else 0.0
    add_integrity_check(
        details,
        "QB",
        "qb_identity_match_rate",
        identity_rate == 1.0,
        1.0,
        identity_rate,
        "All 64 team-side QB identities on a full week must resolve exactly.",
    )
    qb_epa = pd.concat(
        [
            pd.to_numeric(feature_matrix.get("home_qb_recent_epa"), errors="coerce"),
            pd.to_numeric(feature_matrix.get("away_qb_recent_epa"), errors="coerce"),
        ],
        ignore_index=True,
    ) if not feature_matrix.empty else pd.Series(dtype=float)
    qb_cpoe = pd.concat(
        [
            pd.to_numeric(feature_matrix.get("home_qb_recent_cpoe"), errors="coerce"),
            pd.to_numeric(feature_matrix.get("away_qb_recent_cpoe"), errors="coerce"),
        ],
        ignore_index=True,
    ) if not feature_matrix.empty else pd.Series(dtype=float)
    epa_adv = pd.to_numeric(
        feature_matrix.get("qb_recent_epa_advantage"), errors="coerce"
    )
    cpoe_adv = pd.to_numeric(
        feature_matrix.get("qb_recent_cpoe_advantage"), errors="coerce"
    )
    qb_metrics = {
        "epa_coverage": float(qb_epa.notna().mean()) if len(qb_epa) else 0.0,
        "cpoe_coverage": float(qb_cpoe.notna().mean()) if len(qb_cpoe) else 0.0,
        "epa_nonzero_rows": int(epa_adv.fillna(0).abs().gt(1e-12).sum()),
        "cpoe_nonzero_rows": int(cpoe_adv.fillna(0).abs().gt(1e-12).sum()),
        "epa_std": float(epa_adv.std(ddof=0)) if len(epa_adv) else 0.0,
        "cpoe_std": float(cpoe_adv.std(ddof=0)) if len(cpoe_adv) else 0.0,
    }
    if week == 1:
        qb_passed = True
        qb_expected: Any = "Week 1 intentionally uses prior-only power/depth QB context"
    else:
        qb_passed = (
            qb_metrics["epa_coverage"] >= QB_COVERAGE_MINIMUM
            and qb_metrics["cpoe_coverage"] >= QB_COVERAGE_MINIMUM
            and qb_metrics["epa_nonzero_rows"] > 0
            and qb_metrics["cpoe_nonzero_rows"] > 0
            and qb_metrics["epa_std"] > 1e-6
            and qb_metrics["cpoe_std"] > 1e-6
        )
        qb_expected = {
            "coverage_minimum": QB_COVERAGE_MINIMUM,
            "nonzero_rows": ">0",
            "std": ">1e-6",
        }
    add_integrity_check(
        details,
        "QB",
        "qb_epa_cpoe_live_signal",
        qb_passed,
        qb_expected,
        qb_metrics,
        (
            "Week 1 same-season QB EPA/CPOE is intentionally unavailable; "
            "Weeks 2-18 must have covered, non-degenerate live signals."
        ),
    )

    ol_columns = (
        "ol_continuity_advantage",
        "ol_stability_advantage",
        "core_ol_health_advantage",
    )
    ol_series = {
        column: pd.to_numeric(feature_matrix.get(column), errors="coerce")
        for column in ol_columns
    }
    ol_metrics = {
        f"{column}_coverage": float(series.notna().mean()) if len(series) else 0.0
        for column, series in ol_series.items()
    }
    ol_metrics.update(
        {
            f"{column}_std": float(series.std(ddof=0)) if len(series) else 0.0
            for column, series in ol_series.items()
        }
    )
    ol_nonzero = sum(
        int(series.fillna(0).abs().gt(1e-12).sum())
        for series in ol_series.values()
    )
    ol_metrics["combined_nonzero_rows"] = ol_nonzero
    ol_live_passed = (
        all(value == 1.0 for key, value in ol_metrics.items() if key.endswith("_coverage"))
        and (week == 1 or ol_nonzero > 0)
        and (week == 1 or any(
            value > 1e-6 for key, value in ol_metrics.items() if key.endswith("_std")
        ))
    )
    add_integrity_check(
        details,
        "OL",
        "ol_live_signal",
        ol_live_passed,
        "100% coverage and non-degenerate after Week 1",
        ol_metrics,
        "Live OL continuity, stability, and health fields cannot be missing or dead.",
    )

    form = table_or_empty(args.db_path, FORM_TABLE, "FORM", details)
    form_2026 = form[pd.to_numeric(form.get("season"), errors="coerce").eq(SEASON)].copy() if not form.empty else form
    if not form_2026.empty and "team" in form_2026.columns:
        form_2026["team"] = form_2026["team"].map(normalize_team)
    form_through = pd.to_numeric(form_2026.get("through_week"), errors="coerce")
    form_games = dict(
        zip(
            form_2026.get("team", pd.Series(dtype=str)),
            pd.to_numeric(form_2026.get("games_played"), errors="coerce")
            .fillna(-1).astype(int),
        )
    ) if not form_2026.empty else {}
    form_ok = (
        len(form_2026) == 32
        and form_2026.get("team", pd.Series(dtype=str)).nunique() == 32
        and form_through.eq(cutoff_week).all()
        and form_games == team_game_expectation
        and form_2026.get("build_id", pd.Series(dtype=str)).astype(str).eq(EXPECTED_FORM_BUILD).all()
        and pd.to_numeric(
            form_2026.get("no_lookahead_filter_applied_flag"), errors="coerce"
        ).eq(1).all()
    )
    add_integrity_check(
        details,
        "FORM",
        "prior_week_form_cutoff",
        form_ok,
        {
            "teams": 32,
            "through_week": cutoff_week,
            "games_played": team_game_expectation,
            "build_id": EXPECTED_FORM_BUILD,
            "no_lookahead": 1,
        },
        {
            "rows": len(form_2026),
            "teams": form_2026.get("team", pd.Series(dtype=str)).nunique(),
            "through_weeks": sorted(form_through.dropna().unique().tolist()),
            "games_played": form_games,
            "build_ids": sorted(form_2026.get("build_id", pd.Series(dtype=str)).astype(str).unique().tolist()),
        },
        "Form must be a 32-team snapshot calculated only through Week W-1.",
    )

    power = table_or_empty(args.db_path, POWER_TABLE, "POWER", details)
    power_2026 = power[pd.to_numeric(power.get("season"), errors="coerce").eq(SEASON)].copy() if not power.empty else power
    power_qb = pd.to_numeric(power_2026.get("qb_context_adjustment"), errors="coerce")
    power_ol = pd.to_numeric(
        power_2026.get("ol_continuity_strength_adjustment"), errors="coerce"
    )
    exclude_columns = (
        "rating_excludes_home_field_flag",
        "rating_excludes_injuries_flag",
        "rating_excludes_rest_travel_flag",
        "rating_excludes_weather_flag",
    )
    power_ok = (
        len(power_2026) == 32
        and power_2026.get("team", pd.Series(dtype=str)).nunique() == 32
        and power_2026.get("build_id", pd.Series(dtype=str)).astype(str).eq(EXPECTED_POWER_BUILD).all()
        and power_qb.notna().all()
        and power_ol.notna().all()
        and float(power_ol.std(ddof=0)) > 1e-6
        and all(
            pd.to_numeric(power_2026.get(column), errors="coerce").eq(1).all()
            for column in exclude_columns
        )
    )
    add_integrity_check(
        details,
        "POWER",
        "clean_qb_ol_power_inputs",
        power_ok,
        {
            "teams": 32,
            "build_id": EXPECTED_POWER_BUILD,
            "qb_and_ol_coverage": 1.0,
            "ol_std": ">1e-6",
            "qb_context_adjustment": "may be zero when QB quality is embedded in offense strength",
            "structural_exclusion_flags": 1,
        },
        {
            "rows": len(power_2026),
            "teams": power_2026.get("team", pd.Series(dtype=str)).nunique(),
            "qb_coverage": float(power_qb.notna().mean()) if len(power_qb) else 0.0,
            "qb_std": float(power_qb.std(ddof=0)) if len(power_qb) else 0.0,
            "ol_coverage": float(power_ol.notna().mean()) if len(power_ol) else 0.0,
            "ol_std": float(power_ol.std(ddof=0)) if len(power_ol) else 0.0,
        },
        "Preseason power must retain complete cleaned QB/OL lineage and non-degenerate OL adjustments.",
    )

    depth = table_or_empty(args.db_path, DEPTH_TABLE, "DEPTH", details)
    depth_2026 = depth[pd.to_numeric(depth.get("season"), errors="coerce").eq(SEASON)].copy() if not depth.empty else depth
    slots = depth_2026.get("starter_slot", pd.Series(dtype=str)).astype(str)
    starters = depth_2026[
        pd.to_numeric(depth_2026.get("is_projected_starter"), errors="coerce").eq(1)
        & slots.isin(OL_STARTER_SLOTS)
    ].copy() if not depth_2026.empty else depth_2026
    slot_pairs = set(zip(starters.get("team", []), starters.get("starter_slot", [])))
    expected_slot_pairs = {
        (team, slot) for team in team_game_expectation for slot in OL_STARTER_SLOTS
    }
    unit_grade = pd.to_numeric(starters.get("unit_quality_grade"), errors="coerce")
    performance_grade = pd.to_numeric(starters.get("performance_grade"), errors="coerce")
    depth_ok = (
        len(starters) == 160
        and slot_pairs == expected_slot_pairs
        and pd.to_numeric(starters.get("ol_authoritative_starter"), errors="coerce").eq(1).all()
        and pd.to_numeric(starters.get("likely_unavailable"), errors="coerce").eq(0).all()
        and pd.to_numeric(starters.get("ol_quality_discount_applied"), errors="coerce").eq(0).all()
        and np.allclose(unit_grade, performance_grade, equal_nan=False, atol=1e-9)
        and starters.get("build_id", pd.Series(dtype=str)).astype(str).eq(EXPECTED_DEPTH_BUILD).all()
    )
    add_integrity_check(
        details,
        "DEPTH",
        "authoritative_clean_ol_starters",
        depth_ok,
        {
            "starters": 160,
            "slots_per_team": list(OL_STARTER_SLOTS),
            "authoritative": 1,
            "unavailable": 0,
            "second_shrink": 0,
            "build_id": EXPECTED_DEPTH_BUILD,
        },
        {
            "starters": len(starters),
            "slot_pairs": len(slot_pairs),
            "teams": starters.get("team", pd.Series(dtype=str)).nunique(),
        },
        "The OL layer must contain exactly one clean authoritative starter per slot and team.",
    )
    depth_imported = pd.to_datetime(
        depth_2026.get("date_imported"), errors="coerce", utc=True
    )
    latest_depth_imported = (
        depth_imported.max() if len(depth_imported) else pd.NaT
    )
    certificate_time = pd.Timestamp(created_at)
    depth_age_days = (
        float((certificate_time - latest_depth_imported).total_seconds() / 86400.0)
        if pd.notna(latest_depth_imported)
        else math.inf
    )
    add_integrity_check(
        details,
        "DEPTH",
        "weekly_depth_snapshot_freshness",
        -1.0 <= depth_age_days <= 8.0,
        "latest depth build no more than 8 days old",
        {
            "latest_date_imported": json_value(latest_depth_imported),
            "age_days": depth_age_days,
        },
        "QB/OL starter identities must be refreshed during the current weekly cycle.",
    )
    qb_starters = depth_2026[
        pd.to_numeric(
            depth_2026.get("qb_authoritative_starter"), errors="coerce"
        ).eq(1)
    ].copy() if not depth_2026.empty else depth_2026
    qb_grades = pd.to_numeric(
        qb_starters.get("performance_grade"), errors="coerce"
    )
    qb_depth_ok = (
        len(qb_starters) == 32
        and qb_starters.get("team", pd.Series(dtype=str)).nunique() == 32
        and pd.to_numeric(
            qb_starters.get("likely_unavailable"), errors="coerce"
        ).eq(0).all()
        and pd.to_numeric(
            qb_starters.get("history_available"), errors="coerce"
        ).eq(1).all()
        and pd.to_numeric(
            qb_starters.get("usable_performance_grade"), errors="coerce"
        ).eq(1).all()
        and qb_grades.notna().all()
        and float(qb_grades.std(ddof=0)) > 1e-6
    )
    add_integrity_check(
        details,
        "QB",
        "authoritative_prior_qb_starters",
        qb_depth_ok,
        {
            "starters": 32,
            "teams": 32,
            "history_available": 1,
            "usable_performance_grade": 1,
            "grade_std": ">1e-6",
        },
        {
            "starters": len(qb_starters),
            "teams": qb_starters.get("team", pd.Series(dtype=str)).nunique(),
            "grade_coverage": float(qb_grades.notna().mean()) if len(qb_grades) else 0.0,
            "grade_std": float(qb_grades.std(ddof=0)) if len(qb_grades) else 0.0,
        },
        "Week 1 and all later weeks retain one non-degenerate authoritative prior QB per team.",
    )

    structural = table_or_empty(
        args.db_path,
        STRUCTURAL_TABLE,
        "LEARNED_CONSENSUS",
        details,
    )
    structural_week = frame_for_season_week(structural, week)
    structural_ids = set(
        structural_week.get("game_id", pd.Series(dtype=str)).astype(str)
    )
    structural_run_ids = {
        value
        for value in structural_week.get(
            "run_id", pd.Series(dtype=str)
        ).astype(str).str.strip()
        if value
    }
    predictor_learned_lineage = {
        "run_id": str(predictor_audit.get("structural_run_id", "")),
        "build_id": str(predictor_audit.get("structural_build_id", "")),
        "version": str(predictor_audit.get("structural_version", "")),
        "model_variant": str(
            predictor_audit.get("structural_model_variant", "")
        ),
        "bundle_build_id": str(
            predictor_audit.get("structural_learned_bundle_build_id", "")
        ),
        "bundle_version": str(
            predictor_audit.get("structural_learned_bundle_version", "")
        ),
        "bundle_sha256": str(
            predictor_audit.get("structural_learned_bundle_sha256", "")
        ),
        "snapshot_hash": str(
            predictor_audit.get("structural_learned_snapshot_hash", "")
        ),
    }
    expected_predictor_learned_lineage = {
        "run_id": next(iter(structural_run_ids))
        if len(structural_run_ids) == 1
        else "",
        "build_id": EXPECTED_STRUCTURAL_BUILD,
        "version": EXPECTED_STRUCTURAL_VERSION,
        "model_variant": EXPECTED_STRUCTURAL_MODEL_VARIANT,
        "bundle_build_id": EXPECTED_LEARNED_BUNDLE_BUILD,
        "bundle_version": EXPECTED_LEARNED_BUNDLE_VERSION,
        "bundle_sha256": EXPECTED_LEARNED_BUNDLE_SHA256,
        "snapshot_hash": EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH,
    }
    home_cutoff = pd.to_numeric(
        structural_week.get("home_rating_through_week"), errors="coerce"
    )
    away_cutoff = pd.to_numeric(
        structural_week.get("away_rating_through_week"), errors="coerce"
    )
    form_cutoff = pd.to_numeric(
        structural_week.get("form_through_week"), errors="coerce"
    )
    projection_columns = (
        "projected_home_margin",
        "unit_projection",
        "slot_projection",
        "nonlinear_projection",
        "consensus_projection",
        "projection_range",
    )
    projections = {
        column: pd.to_numeric(
            structural_week.get(column, pd.Series(dtype=float)),
            errors="coerce",
        )
        for column in projection_columns
    }
    finite_projections = all(
        len(values) == len(structural_week)
        and values.notna().all()
        and np.isfinite(values).all()
        for values in projections.values()
    )
    consensus_mean = (
        projections["unit_projection"]
        + projections["nonlinear_projection"]
    ) / 2.0
    projection_reconciles = (
        finite_projections
        and np.allclose(
            projections["projected_home_margin"],
            consensus_mean,
            atol=1e-10,
            rtol=0.0,
        )
        and np.allclose(
            projections["consensus_projection"],
            consensus_mean,
            atol=1e-10,
            rtol=0.0,
        )
    )
    calculated_range = (
        pd.concat(
            [
                projections["unit_projection"],
                projections["slot_projection"],
                projections["nonlinear_projection"],
            ],
            axis=1,
        ).max(axis=1)
        - pd.concat(
            [
                projections["unit_projection"],
                projections["slot_projection"],
                projections["nonlinear_projection"],
            ],
            axis=1,
        ).min(axis=1)
    )
    range_reconciles = (
        finite_projections
        and np.allclose(
            projections["projection_range"],
            calculated_range,
            atol=1e-10,
            rtol=0.0,
        )
    )
    agreement_count = pd.to_numeric(
        structural_week.get("model_agreement_count"), errors="coerce"
    )
    agreement_ok = (
        len(agreement_count) == len(structural_week)
        and agreement_count.notna().all()
        and agreement_count.between(0, 3).all()
        and np.allclose(
            agreement_count,
            np.round(agreement_count),
            atol=1e-10,
            rtol=0.0,
        )
    )
    projection_hash = structural_week.get(
        "independent_projection_hash", pd.Series(dtype=str)
    ).astype(str)
    learned_consensus_ok = (
        len(structural_week) == len(target_schedule)
        and structural_ids == target_ids
        and len(structural_run_ids) == 1
        and predictor_learned_lineage
        == expected_predictor_learned_lineage
        and home_cutoff.eq(cutoff_week).all()
        and away_cutoff.eq(cutoff_week).all()
        and form_cutoff.eq(cutoff_week).all()
        and pd.to_numeric(
            structural_week.get("prediction_uses_market_inputs"), errors="coerce"
        ).eq(0).all()
        and structural_week.get("build_id", pd.Series(dtype=str)).astype(str).eq(
            EXPECTED_STRUCTURAL_BUILD
        ).all()
        and structural_week.get("version", pd.Series(dtype=str)).astype(str).eq(
            EXPECTED_STRUCTURAL_VERSION
        ).all()
        and structural_week.get(
            "model_variant", pd.Series(dtype=str)
        ).astype(str).eq(EXPECTED_STRUCTURAL_MODEL_VARIANT).all()
        and structural_week.get(
            "learned_bundle_build_id", pd.Series(dtype=str)
        ).astype(str).eq(EXPECTED_LEARNED_BUNDLE_BUILD).all()
        and structural_week.get(
            "learned_bundle_version", pd.Series(dtype=str)
        ).astype(str).eq(EXPECTED_LEARNED_BUNDLE_VERSION).all()
        and structural_week.get(
            "learned_bundle_sha256", pd.Series(dtype=str)
        ).astype(str).eq(EXPECTED_LEARNED_BUNDLE_SHA256).all()
        and structural_week.get(
            "learned_structural_snapshot_hash", pd.Series(dtype=str)
        ).astype(str).eq(EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH).all()
        and projection_reconciles
        and range_reconciles
        and agreement_ok
        and pd.to_numeric(
            structural_week.get(
                "legacy_additive_components_used_in_final_projection"
            ),
            errors="coerce",
        ).eq(0).all()
        and len(projection_hash) == len(structural_week)
        and projection_hash.str.fullmatch(r"[0-9a-f]{64}").all()
    )
    add_integrity_check(
        details,
        "LEARNED_CONSENSUS",
        "exact_prior_only_learned_consensus_projection",
        learned_consensus_ok,
        {
            "games": len(target_schedule),
            "game_ids": sorted(target_ids),
            "through_week": cutoff_week,
            "market_inputs": 0,
            "build_id": EXPECTED_STRUCTURAL_BUILD,
            "version": EXPECTED_STRUCTURAL_VERSION,
            "model_variant": EXPECTED_STRUCTURAL_MODEL_VARIANT,
            "learned_bundle_build_id": EXPECTED_LEARNED_BUNDLE_BUILD,
            "learned_bundle_version": EXPECTED_LEARNED_BUNDLE_VERSION,
            "learned_bundle_sha256": EXPECTED_LEARNED_BUNDLE_SHA256,
            "learned_structural_snapshot_hash": (
                EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH
            ),
            "legacy_additive_components_used": 0,
        },
        {
            "rows": len(structural_week),
            "game_ids": sorted(structural_ids),
            "structural_run_ids": sorted(structural_run_ids),
            "predictor_learned_lineage": predictor_learned_lineage,
            "home_cutoffs": sorted(home_cutoff.dropna().unique().tolist()),
            "away_cutoffs": sorted(away_cutoff.dropna().unique().tolist()),
            "form_cutoffs": sorted(form_cutoff.dropna().unique().tolist()),
            "build_ids": sorted(
                structural_week.get("build_id", pd.Series(dtype=str))
                .astype(str).unique().tolist()
            ),
            "versions": sorted(
                structural_week.get("version", pd.Series(dtype=str))
                .astype(str).unique().tolist()
            ),
            "model_variants": sorted(
                structural_week.get("model_variant", pd.Series(dtype=str))
                .astype(str).unique().tolist()
            ),
            "projection_reconciles": projection_reconciles,
            "range_reconciles": range_reconciles,
            "agreement_count_valid": agreement_ok,
        },
        "The confidence source must be the exact frozen learned consensus, "
        "exact-week, prior-only, internally reconciled, and market-independent.",
    )

    portfolio["entry_number"] = pd.to_numeric(
        portfolio.get("entry_number"), errors="coerce"
    ).astype("Int64")
    portfolio["contest_rank"] = pd.to_numeric(
        portfolio.get("contest_rank"), errors="coerce"
    ).astype("Int64")
    counts = {
        int(key): int(value)
        for key, value in portfolio.groupby("entry_number").size().to_dict().items()
        if not pd.isna(key)
    }
    policies = {
        int(entry): sorted(group["production_selection_policy"].astype(str).unique().tolist())
        for entry, group in portfolio.groupby("entry_number")
        if not pd.isna(entry)
    } if "production_selection_policy" in portfolio.columns else {}
    unique_games_by_entry = {
        int(entry): int(group["game_id"].astype(str).nunique())
        for entry, group in portfolio.groupby("entry_number")
        if not pd.isna(entry)
    }
    matchup_teams = {
        str(row.game_id): {str(row.home_team), str(row.away_team)}
        for row in target_schedule.itertuples(index=False)
    }
    selected_teams_valid = all(
        normalize_team(row.selected_team) in matchup_teams.get(str(row.game_id), set())
        for row in portfolio.itertuples(index=False)
    ) if "selected_team" in portfolio.columns else False
    card_statuses = {
        int(entry): sorted(group["card_status"].astype(str).unique().tolist())
        for entry, group in portfolio.groupby("entry_number")
        if not pd.isna(entry)
    } if "card_status" in portfolio.columns else {}
    entries_ok = (
        len(portfolio) == 10
        and counts == {1: 5, 2: 5}
        and set(portfolio[portfolio["entry_number"].eq(1)]["contest_rank"].dropna().astype(int)) == set(range(1, 6))
        and set(portfolio[portfolio["entry_number"].eq(2)]["contest_rank"].dropna().astype(int)) == set(range(1, 6))
        and policies == {1: [ENTRY_1_POLICY], 2: [ENTRY_2_POLICY]}
        and unique_games_by_entry == {1: 5, 2: 5}
        and selected_teams_valid
        and card_statuses == {1: ["OFFICIAL_TOP5"], 2: ["CEILING_TOP5"]}
        and set(portfolio.get("game_id", pd.Series(dtype=str)).astype(str)).issubset(target_ids)
        and pd.to_numeric(
            portfolio.get("contest_only_model"), errors="coerce"
        ).eq(1).all()
        and pd.to_numeric(
            portfolio.get("recommended_sportsbook_stake"), errors="coerce"
        ).fillna(0).eq(0).all()
    )
    add_integrity_check(
        details,
        "PORTFOLIO",
        "two_frozen_five_pick_entries",
        entries_ok,
        {
            "rows": 10,
            "counts": {1: 5, 2: 5},
            "ranks": [1, 2, 3, 4, 5],
            "policies": {1: ENTRY_1_POLICY, 2: ENTRY_2_POLICY},
            "sportsbook_stake": 0,
        },
        {
            "rows": len(portfolio),
            "counts": counts,
            "policies": policies,
            "unique_games_by_entry": unique_games_by_entry,
            "card_statuses": card_statuses,
            "selected_teams_valid": selected_teams_valid,
        },
        "The certified deliverable is exactly the approved five picks for each frozen entry.",
    )

    metrics: dict[str, Any] = {
        "schedule_source": schedule_source,
        "prediction_cutoff_week": cutoff_week,
        "scheduled_target_games": len(target_schedule),
        "expected_prior_games": len(prior_ids),
        "captured_prior_games": len(captured_ids),
        "captured_game_ids": sorted(captured_ids),
        "source_game_fingerprint": source_fingerprint,
        "official_line_csv_path": str(lines_path),
        "official_line_csv_sha256": board_sha,
        "official_line_audit_path": str(lines_audit_path),
        "predictor_run_id": predictor_run_id,
        "predictor_created_at": predictor_audit.get("created_at"),
        "feature_source": feature_source,
        "qb_identity_match_rate": identity_rate,
        **qb_metrics,
        **model_file_metrics,
    }
    required_failures = [
        row for row in details if row["required"] == 1 and row["status"] != "PASS"
    ]
    payload = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "audit_run_id": audit_run_id,
        "audit_mode": audit_mode,
        "season": SEASON,
        "week": week,
        "status": "PASS" if not required_failures else "FAIL",
        "required_checks": sum(row["required"] for row in details),
        "passed_required_checks": sum(
            row["required"] for row in details if row["status"] == "PASS"
        ),
        "failed_required_checks": len(required_failures),
        "created_at": created_at,
        "metrics": json_value(metrics),
        "checks": json_value(details),
    }
    detail_frame = pd.DataFrame(details)
    detail_frame.insert(0, "audit_run_id", audit_run_id)
    detail_frame.insert(1, "season", SEASON)
    detail_frame.insert(2, "week", week)
    detail_frame["created_at"] = created_at
    return payload, detail_frame


def sqlite_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def append_with_schema_evolution(
    connection: sqlite3.Connection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if not table_exists(connection, table_name):
        frame.to_sql(table_name, connection, if_exists="replace", index=False)
        return
    escaped = table_name.replace('"', '""')
    existing = {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
    }
    for column in frame.columns:
        if column in existing:
            continue
        column_escaped = str(column).replace('"', '""')
        connection.execute(
            f'ALTER TABLE "{escaped}" ADD COLUMN "{column_escaped}" '
            f'{sqlite_type_for_series(frame[column])}'
        )
    frame.to_sql(table_name, connection, if_exists="append", index=False)


def persist_integrity_certificate(
    args: argparse.Namespace,
    payload: dict[str, Any],
    detail_frame: pd.DataFrame,
) -> Path:
    output_path = (
        args.project_root
        / "outputs"
        / "audits"
        / f"nfl_circa_weekly_integrity_2026_week_{int(payload['week'])}.json"
    )
    write_json_atomic(output_path, payload)
    run_frame = pd.DataFrame(
        [
            {
                "audit_run_id": payload["audit_run_id"],
                "build_id": payload["build_id"],
                "version": payload["version"],
                "audit_mode": payload["audit_mode"],
                "season": payload["season"],
                "week": payload["week"],
                "prediction_cutoff_week": payload["metrics"]["prediction_cutoff_week"],
                "status": payload["status"],
                "required_checks": payload["required_checks"],
                "passed_required_checks": payload["passed_required_checks"],
                "failed_required_checks": payload["failed_required_checks"],
                "predictor_run_id": payload["metrics"]["predictor_run_id"],
                "feature_source": payload["metrics"]["feature_source"],
                "source_game_fingerprint": payload["metrics"]["source_game_fingerprint"],
                "official_line_csv_sha256": payload["metrics"]["official_line_csv_sha256"],
                "v1_model_sha256": payload["metrics"]["v1_model_sha256"],
                "ceiling_model_sha256": payload["metrics"]["ceiling_model_sha256"],
                "certificate_path": str(output_path),
                "created_at": payload["created_at"],
            }
        ]
    )
    connection = sqlite3.connect(args.db_path)
    try:
        connection.execute("BEGIN")
        append_with_schema_evolution(connection, INTEGRITY_RUN_TABLE, run_frame)
        append_with_schema_evolution(
            connection, INTEGRITY_DETAIL_TABLE, detail_frame
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return output_path


def certify_week(
    args: argparse.Namespace,
    schedule: pd.DataFrame,
    schedule_source: str,
    week: int,
    lines_path: Path,
    lines_audit_path: Path,
    predictor_started_at: Optional[str],
    audit_mode: str,
) -> Path:
    payload, details = build_integrity_certificate(
        args,
        schedule,
        schedule_source,
        week,
        lines_path,
        lines_audit_path,
        predictor_started_at,
        audit_mode,
    )
    certificate_path = persist_integrity_certificate(args, payload, details)
    print("=" * 120)
    print("[CIRCA_WEEKLY] WEEKLY DATA INTEGRITY CERTIFICATE")
    print("=" * 120)
    print(
        f"[CIRCA_WEEKLY] Status: {payload['status']} | "
        f"required checks: {payload['passed_required_checks']}/"
        f"{payload['required_checks']}"
    )
    print(
        f"[CIRCA_WEEKLY] 2026 games captured through Week {week - 1}: "
        f"{payload['metrics']['captured_prior_games']}/"
        f"{payload['metrics']['expected_prior_games']}"
    )
    print(
        f"[CIRCA_WEEKLY] QB identity: "
        f"{payload['metrics']['qb_identity_match_rate']:.1%}"
    )
    print(f"[CIRCA_WEEKLY] Certificate: {certificate_path}")
    print(
        f"[CIRCA_WEEKLY] SQLite audit tables: {INTEGRITY_RUN_TABLE}, "
        f"{INTEGRITY_DETAIL_TABLE}"
    )
    if payload["status"] != "PASS":
        failures = [
            row["check_name"]
            for row in payload["checks"]
            if row["required"] == 1 and row["status"] != "PASS"
        ]
        raise RuntimeError(
            "Weekly data integrity certificate FAILED; cards are not approved "
            "for submission. Failed checks: " + ", ".join(failures)
        )
    return certificate_path


def preflight(
    args: argparse.Namespace,
    historical: ModuleType,
    predictor_path: Path,
    schedule: pd.DataFrame,
    schedule_source: str,
    week: int,
) -> None:
    predictor_build, predictor_version = verify_predictor(predictor_path)
    schedule_week = schedule[schedule["week"].eq(week)]
    if schedule_week.empty:
        raise RuntimeError(f"The 2026 schedule contains no Week {week} games.")
    if schedule_week["gameday"].isna().any():
        raise RuntimeError(f"Week {week} schedule rows have missing game dates.")
    fitz_available = int(historical._module_available("fitz"))
    tesseract_available = int(
        historical._tesseract_available(args.tesseract_path)
    )
    if not fitz_available or not tesseract_available:
        raise RuntimeError(
            "Circa PDF automation requires PyMuPDF and Tesseract. "
            f"PyMuPDF={fitz_available}, Tesseract={tesseract_available}."
        )
    print("[CIRCA_WEEKLY] PREFLIGHT PASSED")
    print(f"[CIRCA_WEEKLY] Schedule: {schedule_source}")
    print(f"[CIRCA_WEEKLY] Week: {week} | scheduled games: {len(schedule_week)}")
    print(
        "[CIRCA_WEEKLY] Historical parser: "
        f"{historical.BUILD_ID} | {historical.VERSION}"
    )
    print(
        "[CIRCA_WEEKLY] Predictor: "
        f"{predictor_build} | {predictor_version}"
    )
    print("[CIRCA_WEEKLY] Official-site network request made: NO")
    print("[CIRCA_WEEKLY] Files modified: NO")


def run_self_test() -> int:
    raw = pd.DataFrame(
        {
            "week": [1] * 6,
            "home_team": ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI"],
            "away_team": ["SEA", "TB", "CLE", "NYJ", "NO", "GB"],
            "game_date": ["2026-09-10"] + ["2026-09-13"] * 5,
        }
    )
    schedule = standardize_database_schedule(raw)
    if len(schedule) != 6:
        raise AssertionError("Schedule standardization failed.")
    if detect_current_week(schedule, None, dt.date(2026, 9, 10)) != 1:
        raise AssertionError("Automatic current-week detection failed.")
    if revision_number("Contest-Point-Spreads-3.pdf") != 3:
        raise AssertionError("PDF revision parsing failed.")
    manifest = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "url": "https://www.circasports.com/wp-content/uploads/2026/09/Circa-Sports-Million-VIII-Week-1-Contest-Point-Spreads.pdf",
                "wordpress_date": "2026-09-10T17:00:00Z",
                "download_status": "DOWNLOADED",
            },
            {
                "season": 2026,
                "week": 1,
                "url": "https://www.circasports.com/wp-content/uploads/2026/09/Circa-Sports-Million-VIII-Week-1-Contest-Point-Spreads-2.pdf",
                "wordpress_date": "2026-09-10T18:00:00Z",
                "download_status": "DOWNLOADED",
            },
        ]
    )
    selected = select_latest_manifest(manifest, 1)
    if not str(selected.iloc[0]["url"]).endswith("-2.pdf"):
        raise AssertionError("Latest official PDF selection failed.")

    teams = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
    ]
    schedule_rows: list[dict[str, Any]] = []
    for target_week in (1, 2):
        rotation = 0 if target_week == 1 else 1
        away_teams = teams[::2]
        home_teams = teams[1::2]
        if rotation:
            home_teams = home_teams[rotation:] + home_teams[:rotation]
        for index, (away, home) in enumerate(zip(away_teams, home_teams)):
            schedule_rows.append(
                {
                    "season": SEASON,
                    "week": target_week,
                    "game_id": f"2026_{target_week}_{away}_{home}",
                    "game_type": "REG",
                    "away_team": away,
                    "home_team": home,
                    "gameday": f"2026-09-{6 + 7 * target_week:02d}",
                    "home_score": 24 if target_week == 1 else np.nan,
                    "away_score": 20 if target_week == 1 else np.nan,
                }
            )
    integrity_schedule = standardize_database_schedule(pd.DataFrame(schedule_rows))

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        inputs = root / "inputs"
        inputs.mkdir()
        database = root / "identifier.sqlite"
        feature_database = root / "live.sqlite"
        v1_model = root / "v1.joblib"
        ceiling_model = root / "ceiling.joblib"
        v1_model.write_bytes(b"frozen-v1-self-test")
        ceiling_model.write_bytes(b"frozen-ceiling-self-test")

        week_two = integrity_schedule[integrity_schedule["week"].eq(2)].copy()
        lines_path = inputs / "nfl_circa_lines_2026_week_2.csv"
        board = week_two[
            ["week", "away_team", "home_team", "game_id"]
        ].copy()
        board["home_spread"] = -3.5
        board = board[
            ["week", "away_team", "home_team", "home_spread", "game_id"]
        ]
        board.to_csv(lines_path, index=False)
        official_pdf = root / "official_week_2.pdf"
        official_pdf.write_bytes(b"self-test-official-circa-pdf")
        line_audit_path = inputs / "nfl_circa_lines_2026_week_2_audit.json"
        write_json_atomic(
            line_audit_path,
            {
                "build_id": BUILD_ID,
                "version": VERSION,
                "season": SEASON,
                "week": 2,
                "complete_board": True,
                "manual_overrides": 0,
                "line_csv_sha256": sha256_file(lines_path),
                "official_pdf_url": (
                    "https://www.circasports.com/wp-content/uploads/2026/09/"
                    "Circa-Sports-Million-VIII-Week-2-Contest-Point-Spreads.pdf"
                ),
                "official_pdf_path": str(official_pdf),
                "official_pdf_sha256": sha256_file(official_pdf),
            },
        )

        game_matrix_rows: list[dict[str, Any]] = []
        for index, row in enumerate(week_two.itertuples(index=False)):
            game_matrix_rows.append(
                {
                    "season": SEASON,
                    "week": 2,
                    "game_id": row.game_id,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "rating_alpha": 10.0,
                    "personnel_complete": 1,
                    "home_qb_identity_matched": 1,
                    "away_qb_identity_matched": 1,
                    "home_qb_recent_epa": 0.01 * (index + 1),
                    "away_qb_recent_epa": -0.01 * (index + 1),
                    "home_qb_recent_cpoe": 0.20 * (index + 1),
                    "away_qb_recent_cpoe": -0.10 * (index + 1),
                    "qb_recent_epa_advantage": 0.02 * (index + 1),
                    "qb_recent_cpoe_advantage": 0.30 * (index + 1),
                    "ol_continuity_advantage": 0.01 * (index - 7),
                    "ol_stability_advantage": 0.02 * (index - 7),
                    "core_ol_health_advantage": 0.03 * (index - 7),
                }
            )
        game_matrix = pd.DataFrame(game_matrix_rows)
        week_one = integrity_schedule[integrity_schedule["week"].eq(1)]
        team_game_rows: list[dict[str, Any]] = []
        for row in week_one.itertuples(index=False):
            for offense, defense in (
                (row.home_team, row.away_team),
                (row.away_team, row.home_team),
            ):
                team_game_rows.append(
                    {
                        "season": SEASON,
                        "week": 1,
                        "game_id": row.game_id,
                        "offense_team": offense,
                        "defense_team": defense,
                    }
                )
        personnel = pd.DataFrame(
            {
                "season": SEASON,
                "prediction_week": 2,
                "team": teams,
                "history_games": 1,
            }
        )
        connection = sqlite3.connect(feature_database)
        try:
            game_matrix.to_sql(
                "nfl_matchup_game_matrix", connection, index=False
            )
            pd.DataFrame(team_game_rows).to_sql(
                TEAM_GAME_TABLE, connection, index=False
            )
            personnel.to_sql(PERSONNEL_TABLE, connection, index=False)
            connection.commit()
        finally:
            connection.close()

        predictor_run_id = "self_test_predictor_run"
        created_at = now_string()
        predictor_audit = pd.DataFrame(
            [
                {
                    "run_id": predictor_run_id,
                    "build_id": EXPECTED_PREDICTOR_BUILD,
                    "version": EXPECTED_PREDICTOR_VERSION,
                    "season": SEASON,
                    "week": 2,
                    "feature_source": (
                        f"{feature_database.resolve()}::nfl_matchup_game_matrix"
                    ),
                    "model_path": str(v1_model),
                    "model_build_id": EXPECTED_V1_MODEL_BUILD,
                    "model_version": EXPECTED_V1_MODEL_VERSION,
                    "ceiling_model_path": str(ceiling_model),
                    "ceiling_model_build_id": EXPECTED_CEILING_MODEL_BUILD,
                    "ceiling_model_version": EXPECTED_CEILING_MODEL_VERSION,
                    "rating_alpha": 10.0,
                    "contest_forward_test_override": 1,
                    "production_promoted": 1,
                    "immutable_model_card_preserved": 1,
                    "structural_used_to_change_model_card": 0,
                    "structural_run_id": "self_test_learned_run",
                    "structural_build_id": EXPECTED_STRUCTURAL_BUILD,
                    "structural_version": EXPECTED_STRUCTURAL_VERSION,
                    "structural_model_variant": (
                        EXPECTED_STRUCTURAL_MODEL_VARIANT
                    ),
                    "structural_learned_bundle_build_id": (
                        EXPECTED_LEARNED_BUNDLE_BUILD
                    ),
                    "structural_learned_bundle_version": (
                        EXPECTED_LEARNED_BUNDLE_VERSION
                    ),
                    "structural_learned_bundle_sha256": (
                        EXPECTED_LEARNED_BUNDLE_SHA256
                    ),
                    "structural_learned_snapshot_hash": (
                        EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH
                    ),
                    "sportsbook_staking_enabled": 0,
                    "created_at": created_at,
                }
            ]
        )
        portfolio_rows: list[dict[str, Any]] = []
        selected_games = list(week_two.head(5).itertuples(index=False))
        for entry_number, policy in (
            (1, ENTRY_1_POLICY),
            (2, ENTRY_2_POLICY),
        ):
            for rank, row in enumerate(selected_games, start=1):
                portfolio_rows.append(
                    {
                        "run_id": predictor_run_id,
                        "created_at": created_at,
                        "season": SEASON,
                        "week": 2,
                        "entry_number": entry_number,
                        "contest_rank": rank,
                        "game_id": row.game_id,
                        "selected_team": row.away_team,
                        "production_selection_policy": policy,
                        "card_status": (
                            "OFFICIAL_TOP5" if entry_number == 1 else "CEILING_TOP5"
                        ),
                        "contest_only_model": 1,
                        "recommended_sportsbook_stake": 0.0,
                    }
                )
        form = pd.DataFrame(
            {
                "season": SEASON,
                "through_week": 1,
                "team": teams,
                "games_played": 1,
                "build_id": EXPECTED_FORM_BUILD,
                "no_lookahead_filter_applied_flag": 1,
            }
        )
        power = pd.DataFrame(
            {
                "season": SEASON,
                "team": teams,
                "qb_context_adjustment": np.linspace(-1.0, 1.0, 32),
                "ol_continuity_strength_adjustment": np.linspace(-0.5, 0.5, 32),
                "build_id": EXPECTED_POWER_BUILD,
                "rating_excludes_home_field_flag": 1,
                "rating_excludes_injuries_flag": 1,
                "rating_excludes_rest_travel_flag": 1,
                "rating_excludes_weather_flag": 1,
            }
        )
        depth_rows: list[dict[str, Any]] = []
        for team_index, team in enumerate(teams):
            depth_rows.append(
                {
                    "season": SEASON,
                    "team": team,
                    "player_id": f"{team}_QB",
                    "starter_slot": "QB1",
                    "is_projected_starter": 1,
                    "qb_authoritative_starter": 1,
                    "likely_unavailable": 0,
                    "history_available": 1,
                    "usable_performance_grade": 1,
                    "performance_grade": 40.0 + team_index,
                    "unit_quality_grade": 40.0 + team_index,
                    "ol_authoritative_starter": 0,
                    "ol_quality_discount_applied": 0,
                    "build_id": EXPECTED_DEPTH_BUILD,
                    "date_imported": created_at,
                }
            )
            for slot in OL_STARTER_SLOTS:
                grade = float(50 + OL_STARTER_SLOTS.index(slot))
                depth_rows.append(
                    {
                        "season": SEASON,
                        "team": team,
                        "player_id": f"{team}_{slot}",
                        "starter_slot": slot,
                        "is_projected_starter": 1,
                        "ol_authoritative_starter": 1,
                        "qb_authoritative_starter": 0,
                        "likely_unavailable": 0,
                        "history_available": 1,
                        "usable_performance_grade": 1,
                        "ol_quality_discount_applied": 0,
                        "unit_quality_grade": grade,
                        "performance_grade": grade,
                        "build_id": EXPECTED_DEPTH_BUILD,
                        "date_imported": created_at,
                    }
                )
        structural = week_two[
            ["season", "week", "game_id", "away_team", "home_team"]
        ].copy()
        structural["home_rating_through_week"] = 1
        structural["away_rating_through_week"] = 1
        structural["run_id"] = "self_test_learned_run"
        structural["form_through_week"] = 1
        structural["prediction_uses_market_inputs"] = 0
        structural["build_id"] = EXPECTED_STRUCTURAL_BUILD
        structural["version"] = EXPECTED_STRUCTURAL_VERSION
        structural["model_variant"] = EXPECTED_STRUCTURAL_MODEL_VARIANT
        structural["learned_bundle_build_id"] = (
            EXPECTED_LEARNED_BUNDLE_BUILD
        )
        structural["learned_bundle_version"] = (
            EXPECTED_LEARNED_BUNDLE_VERSION
        )
        structural["learned_bundle_sha256"] = (
            EXPECTED_LEARNED_BUNDLE_SHA256
        )
        structural["learned_structural_snapshot_hash"] = (
            EXPECTED_LEARNED_STRUCTURAL_SNAPSHOT_HASH
        )
        structural["unit_projection"] = np.linspace(-3.0, 3.0, len(structural))
        structural["slot_projection"] = structural["unit_projection"]
        structural["nonlinear_projection"] = structural["unit_projection"]
        structural["consensus_projection"] = structural["unit_projection"]
        structural["projected_home_margin"] = structural["unit_projection"]
        structural["projection_range"] = 0.0
        structural["model_agreement_count"] = 3
        structural[
            "legacy_additive_components_used_in_final_projection"
        ] = 0
        structural["independent_projection_hash"] = [
            hashlib.sha256(str(game_id).encode("utf-8")).hexdigest()
            for game_id in structural["game_id"]
        ]
        connection = sqlite3.connect(database)
        try:
            predictor_audit.to_sql(
                PREDICTOR_AUDIT_TABLE, connection, index=False
            )
            pd.DataFrame(portfolio_rows).to_sql(
                PORTFOLIO_TABLE, connection, index=False
            )
            form.to_sql(FORM_TABLE, connection, index=False)
            power.to_sql(POWER_TABLE, connection, index=False)
            pd.DataFrame(depth_rows).to_sql(DEPTH_TABLE, connection, index=False)
            structural.to_sql(STRUCTURAL_TABLE, connection, index=False)
            connection.commit()
        finally:
            connection.close()

        self_test_args = SimpleNamespace(project_root=root, db_path=database)
        previous_certificate = (
            root
            / "outputs"
            / "audits"
            / "nfl_circa_weekly_integrity_2026_week_1.json"
        )
        write_json_atomic(
            previous_certificate,
            {
                "status": "PASS",
                "metrics": {
                    "captured_game_ids": [],
                    "source_game_fingerprint": hashlib.sha256(b"").hexdigest(),
                    "v1_model_sha256": sha256_file(v1_model),
                    "ceiling_model_sha256": sha256_file(ceiling_model),
                },
            },
        )
        payload, details = build_integrity_certificate(
            self_test_args,
            integrity_schedule,
            "SELF_TEST_SCHEDULE",
            2,
            lines_path,
            line_audit_path,
            None,
            "SELF_TEST",
        )
        if payload["status"] != "PASS":
            failed = details[details["status"].eq("FAIL")]
            raise AssertionError(
                "Passing integrity fixture failed:\n" + failed.to_string(index=False)
            )
        certificate = persist_integrity_certificate(
            self_test_args, payload, details
        )
        if not certificate.exists():
            raise AssertionError("Integrity certificate was not persisted.")
        connection = sqlite3.connect(database)
        try:
            if not table_exists(connection, INTEGRITY_RUN_TABLE):
                raise AssertionError("Integrity run table was not persisted.")
            if not table_exists(connection, INTEGRITY_DETAIL_TABLE):
                raise AssertionError("Integrity detail table was not persisted.")
        finally:
            connection.close()

        def assert_learned_source_rejected(
            candidate: pd.DataFrame,
            label: str,
        ) -> None:
            with sqlite3.connect(database) as connection:
                candidate.to_sql(
                    STRUCTURAL_TABLE,
                    connection,
                    if_exists="replace",
                    index=False,
                )
            rejected_payload, rejected_details = build_integrity_certificate(
                self_test_args,
                integrity_schedule,
                "SELF_TEST_SCHEDULE",
                2,
                lines_path,
                line_audit_path,
                None,
                label,
            )
            learned_check = rejected_details[
                rejected_details["check_name"].eq(
                    "exact_prior_only_learned_consensus_projection"
                )
            ]
            if (
                rejected_payload["status"] != "FAIL"
                or len(learned_check) != 1
                or str(learned_check.iloc[0]["status"]) != "FAIL"
            ):
                raise AssertionError(
                    f"Invalid learned-consensus source was not rejected: {label}."
                )

        legacy_structural = structural.copy()
        legacy_structural["model_variant"] = "STRUCTURAL_FORM_HFA"
        assert_learned_source_rejected(
            legacy_structural,
            "SELF_TEST_LEGACY_STRUCTURAL_REJECTION",
        )
        wrong_hash_structural = structural.copy()
        wrong_hash_structural["learned_bundle_sha256"] = "0" * 64
        assert_learned_source_rejected(
            wrong_hash_structural,
            "SELF_TEST_WRONG_LEARNED_SHA_REJECTION",
        )
        with sqlite3.connect(database) as connection:
            structural.to_sql(
                STRUCTURAL_TABLE,
                connection,
                if_exists="replace",
                index=False,
            )

        broken = game_matrix.copy()
        broken["qb_recent_epa_advantage"] = 0.0
        broken["qb_recent_cpoe_advantage"] = 0.0
        connection = sqlite3.connect(feature_database)
        try:
            broken.to_sql(
                "nfl_matchup_game_matrix",
                connection,
                if_exists="replace",
                index=False,
            )
            connection.commit()
        finally:
            connection.close()
        broken_payload, _ = build_integrity_certificate(
            self_test_args,
            integrity_schedule,
            "SELF_TEST_SCHEDULE",
            2,
            lines_path,
            line_audit_path,
            None,
            "SELF_TEST_BROKEN_QB",
        )
        if broken_payload["status"] != "FAIL":
            raise AssertionError("Zeroed Week 2 QB EPA/CPOE was not rejected.")
    print("[CIRCA_WEEKLY] Self-test passed.")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    builder_path = resolve_script(
        args.historical_builder_path,
        args.project_root,
        HISTORICAL_BUILDER_FILENAME,
    )
    predictor_path = resolve_script(
        args.predictor_path,
        args.project_root,
        PREDICTOR_FILENAME,
    )
    historical = load_historical_builder(builder_path)
    schedule, schedule_source = load_schedule(args.db_path, historical)
    week = detect_current_week(schedule, args.week)

    print("[CIRCA_WEEKLY] Automated official Circa board and production picks")
    print(f"[CIRCA_WEEKLY] Build ID: {BUILD_ID}")
    print(f"[CIRCA_WEEKLY] Version: {VERSION}")
    print(f"[CIRCA_WEEKLY] Season: {SEASON} | Week: {week}")
    if args.audit_only:
        predictor_build, predictor_version = verify_predictor(predictor_path)
        print("[CIRCA_WEEKLY] Mode: AUDIT ONLY")
        print(
            f"[CIRCA_WEEKLY] Predictor: {predictor_build} | "
            f"{predictor_version}"
        )
        lines_path, audit_path, _board, _line_audit = load_saved_board(
            args, week
        )
        certify_week(
            args,
            schedule,
            schedule_source,
            week,
            lines_path,
            audit_path,
            None,
            "AUDIT_ONLY",
        )
        print("[CIRCA_WEEKLY] AUDIT-ONLY VERIFICATION PASSED")
        return 0
    preflight(
        args,
        historical,
        predictor_path,
        schedule,
        schedule_source,
        week,
    )
    if args.preflight_only:
        return 0

    schedule_week = schedule[schedule["week"].eq(week)].copy()
    deadline = time.monotonic() + args.wait_minutes * 60.0
    attempt_number = 0
    while True:
        attempt_number += 1
        try:
            print(
                f"[CIRCA_WEEKLY] Official board discovery attempt "
                f"{attempt_number} at {now_string()}"
            )
            manifest, attempts, _team_lines, game_lines = fetch_board_once(
                args,
                historical,
                schedule_week,
                week,
            )
            break
        except BoardNotPublished as exc:
            remaining = deadline - time.monotonic()
            if args.wait_minutes <= 0 or remaining < args.poll_seconds:
                raise BoardNotPublished(
                    f"{exc} Circa normally publishes around Thursday "
                    "10:00 a.m. Pacific; no existing CSV was used."
                ) from exc
            print(
                f"[CIRCA_WEEKLY] {exc} Retrying in "
                f"{args.poll_seconds // 60} minutes."
            )
            time.sleep(args.poll_seconds)

    lines_path, audit_path = save_live_outputs(
        args,
        week,
        schedule_source,
        manifest,
        attempts,
        game_lines,
    )
    print(f"[CIRCA_WEEKLY] Complete official board: {lines_path}")
    print(f"[CIRCA_WEEKLY] Pull audit: {audit_path}")
    print(f"[CIRCA_WEEKLY] Games validated: {len(game_lines)}")
    if args.fetch_only:
        print("[CIRCA_WEEKLY] Fetch-only mode completed successfully.")
        return 0

    predictor_started_at = run_predictor(
        args, predictor_path, lines_path, week
    )
    certify_week(
        args,
        schedule,
        schedule_source,
        week,
        lines_path,
        audit_path,
        predictor_started_at,
        "POST_PREDICT",
    )
    print(
        "[CIRCA_WEEKLY] OFFICIAL BOARD + PRODUCTION PICKS + "
        "DATA CERTIFICATE COMPLETED"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CIRCA_WEEKLY] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CIRCA_WEEKLY] FAILED: {exc}", file=sys.stderr)
        raise
