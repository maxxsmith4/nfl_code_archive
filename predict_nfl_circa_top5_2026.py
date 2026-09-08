#!/usr/bin/env python
"""
Generate the production 2026 Circa Million two-entry portfolio.

Entry 1 uses the locked V1 residual model:
    models/nfl_circa_contest_model_v1.joblib

Its selection policy is the promoted adjusted-V1 rule:
    - raw residual ranks 1-4 are locked;
    - raw ranks 5-8 may compete for the fifth slot only when within 0.50
      residual points of raw rank 5;
    - the tiebreak uses prior-only conditional final-margin mass at keys 3/7.

Entry 2 uses the promoted Ceiling rule:
    Weeks 1-9: exact raw V1 predictions and ranks
    Weeks 10-18: full late-season BLENDED_SIGN ceiling residual model, with
                 selected home favorites of 7.5 points or more excluded before
                 residual ranking and replaced by the next eligible residual.

Both cards use the same point-in-time matchup matrix and official Circa board.
The predicted residual models remain unchanged; only the promoted, fully
audited contest-selection policies are applied.

Required line input
-------------------
CSV, XLSX, or Parquet with:
    week, away_team, home_team, circa_home_margin

Alternatively provide home_spread, where:
    home_spread = -3.5 -> circa_home_margin = +3.5

Default model bundles
---------------------
Locked V1:
    models/nfl_circa_contest_model_v1.joblib

Ceiling production model:
    models/nfl_circa_contest_season_phase_v3_1.joblib

If the V3.1 bundle is unavailable, the runner automatically checks the V3.2
bundle and extracts its unchanged A1000/W1/BLENDED_SIGN ceiling source.

Outputs
-------
Existing SQLite tables remain stable for downstream compatibility:
    nfl_circa_top5_predictions_2026
    nfl_circa_top5_prediction_history_2026

Production portfolio tables:
    nfl_circa_ceiling_predictions_2026
    nfl_circa_ceiling_prediction_history_2026
    nfl_circa_v1_vs_ceiling_comparison_2026
    nfl_circa_v1_vs_ceiling_comparison_history_2026
    nfl_circa_final_portfolio_predictions_2026
    nfl_circa_final_portfolio_prediction_history_2026
    nfl_circa_selection_confidence_2026
    nfl_circa_selection_confidence_history_2026
    nfl_circa_manual_submission_draft_2026

CSV snapshots:
    outputs/circa_2026/nfl_circa_top5_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_ceiling_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_v1_vs_ceiling_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_final_portfolio_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_confidence_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_manual_submission_week_<W>_<timestamp>.csv

The confidence score is an advisory selection-robustness score, not a cover
probability. The immutable model card remains separate from the editable
manual-submission draft. The market-independent STRUCTURAL_FORM_HFA fair line
is also joined as a confidence-only overlay. A contradiction of at least 1.5
points against within-card residual-strength ranks 4-5 raises manual-review
priority; it never changes either production card or invents a numeric score
penalty.

No sportsbook stake is recommended by this script.
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
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd


SEASON = 2026
BUILD_ID = "NFL_CIRCA_TOP5_2026_PRODUCTION_CONFIDENCE_BOARD"
VERSION = "v8_3_structural_lineage_live_feature_freshness"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)
DEFAULT_MODEL_FILENAME = "nfl_circa_contest_model_v1.joblib"
DEFAULT_CEILING_MODEL_FILENAMES = (
    "nfl_circa_contest_season_phase_v3_2.joblib",
)
DEFAULT_MATCHUP_BUILDER_FILENAME = (
    "build_backtest_nfl_weekly_matchup_residual.py"
)
LIVE_MATRIX_DATABASE_NAME = "nfl_weekly_matchup_residual_live_2026.sqlite"
DEFAULT_MARGIN_HISTORY_DATABASE_NAME = "nfl_circa_contest_lines.sqlite"
MARGIN_HISTORY_TABLE = "nfl_circa_backtest_matrix"

OUTPUT_TABLE = "nfl_circa_top5_predictions_2026"
OUTPUT_HISTORY_TABLE = "nfl_circa_top5_prediction_history_2026"
RUN_AUDIT_TABLE = "nfl_circa_top5_run_audit_2026"
RUN_AUDIT_HISTORY_TABLE = "nfl_circa_top5_run_audit_history_2026"
CEILING_OUTPUT_TABLE = "nfl_circa_ceiling_predictions_2026"
CEILING_OUTPUT_HISTORY_TABLE = (
    "nfl_circa_ceiling_prediction_history_2026"
)
COMPARISON_OUTPUT_TABLE = (
    "nfl_circa_v1_vs_ceiling_comparison_2026"
)
COMPARISON_OUTPUT_HISTORY_TABLE = (
    "nfl_circa_v1_vs_ceiling_comparison_history_2026"
)
PORTFOLIO_OUTPUT_TABLE = "nfl_circa_final_portfolio_predictions_2026"
PORTFOLIO_OUTPUT_HISTORY_TABLE = (
    "nfl_circa_final_portfolio_prediction_history_2026"
)
CONFIDENCE_OUTPUT_TABLE = "nfl_circa_selection_confidence_2026"
CONFIDENCE_OUTPUT_HISTORY_TABLE = (
    "nfl_circa_selection_confidence_history_2026"
)
MANUAL_SUBMISSION_DRAFT_TABLE = "nfl_circa_manual_submission_draft_2026"
HISTORICAL_CARD_DATABASE_NAME = "nfl_circa_joint_contest_sim.sqlite"
HISTORICAL_CARD_TABLE_CANDIDATES = (
    "nfl_circa_joint_weekly_refit_cards",
    "nfl_circa_joint_strategy_cards",
)

PRIMARY_KEYS = (3, 7)
PMF_MIN_MARGIN = -30
PMF_MAX_MARGIN = 30
PMF_SMOOTHING = 0.50
PMF_BANDWIDTH = 2.50
PMF_PRIOR_STRENGTH = 75.0
ADJUSTED_V1_TIEBREAK_TOLERANCE = 0.50
ADJUSTED_V1_CANDIDATE_RANK_MAX = 8
LATE_HOME_FAVORITE_VETO_MIN_MARGIN = 7.5
ADJUSTED_V1_POLICY = "TB_V1_P37_D050"
CEILING_POLICY = "CEILING_LATE_HOME_FAVORITE_7P5_VETO"
CONFIDENCE_METHODOLOGY = (
    "SELECTION_ROBUSTNESS_PLUS_STRUCTURAL_REVIEW_OVERLAY_NOT_COVER_PROBABILITY_V2"
)
CONFIDENCE_WEIGHTS = {
    "boundary_separation": 0.35,
    "cross_model_agreement": 0.25,
    "within_week_residual_strength": 0.20,
    "primary_key_path_support": 0.10,
    "selection_policy_stability": 0.10,
}
STRUCTURAL_LIVE_TABLE = "nfl_weekly_power_spread_predictions_2026"
STRUCTURAL_RUN_AUDIT_TABLE = "nfl_weekly_power_spread_run_audit_2026"
STRUCTURAL_MODEL_VARIANT = "STRUCTURAL_FORM_HFA"
EXPECTED_STRUCTURAL_BUILD_ID = (
    "NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4"
)
EXPECTED_STRUCTURAL_VERSION = (
    "v4_structural_form_hfa_fail_closed_lineage"
)
EXPECTED_FORM_BUILD_ID = "NFL_2026_FORM_RATING_CANONICAL_V3"
EXPECTED_FORM_VERSION = (
    "v3_current_structural_prior_asof_opponent_adjusted_form"
)
EXPECTED_STRUCTURAL_POWER_BUILD_ID = (
    "NFL_POWER_RATINGS_2026_CANONICAL_V2"
)
EXPECTED_LIVE_MATCHUP_BUILD_ID = (
    "NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V3"
)
EXPECTED_LIVE_MATCHUP_VERSION = (
    "v3_qb_gsis_identity_crosswalk_dead_feature_guard"
)
EXPECTED_V1_MODEL_BUILD_ID = "NFL_CIRCA_CONTEST_LINES_CANONICAL_V1"
EXPECTED_V1_MODEL_VERSION = "v1_5_qb_identity_integrity_guard"
EXPECTED_CEILING_MODEL_BUILD_ID = (
    "NFL_CIRCA_CONTEST_SEASON_PHASE_V3_2_QB_REBUILT"
)
EXPECTED_CEILING_MODEL_VERSION = (
    "v3_2_1_fixed_architecture_qb_integrity_lineage"
)
STRUCTURAL_REVIEW_THRESHOLD_POINTS = 1.50
STRUCTURAL_LOW_CONFIDENCE_RANK_MIN = 4
STRUCTURAL_USED_TO_CHANGE_MODEL_CARD = 0

FEATURE_TABLE_CANDIDATES = (
    "nfl_circa_live_features_2026",
    "nfl_matchup_game_matrix_live_2026",
    "nfl_matchup_game_matrix_2026",
    "nfl_matchup_game_matrix",
)

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}


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
    parser.add_argument(
        "--margin-history-db-path",
        type=Path,
        default=None,
        help=(
            "SQLite database containing prior Circa lines and final margins. "
            "Defaults to backtests/nfl_circa_contest_lines.sqlite."
        ),
    )
    parser.add_argument(
        "--margin-history-table",
        type=str,
        default=MARGIN_HISTORY_TABLE,
    )
    parser.add_argument(
        "--structural-table",
        type=str,
        default=STRUCTURAL_LIVE_TABLE,
        help=(
            "Current market-independent STRUCTURAL_FORM_HFA weekly "
            "projection table used only by the confidence overlay."
        ),
    )
    parser.add_argument("--week", type=int, required=False)
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help=(
            "Required structural/feature freshness date. Defaults to today."
        ),
    )
    parser.add_argument("--circa-lines-path", type=Path, default=None)
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--feature-table", type=str, default=None)
    parser.add_argument("--schedule-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--ceiling-model-path",
        type=Path,
        default=None,
        help=(
            "Optional season-phase bundle containing the full ceiling "
            "late model. Defaults only to the repaired V3.2 bundle."
        ),
    )
    parser.add_argument("--matchup-builder-path", type=Path, default=None)
    parser.add_argument(
        "--approve-contest-forward-test",
        action="store_true",
        help=(
            "Acknowledge that the historical bundle may carry "
            "implementation_ready=False under its original CLV/ROI safety "
            "gates, while the user has separately approved forward use as a "
            "contest-only mechanical top-five model."
        ),
    )
    parser.add_argument(
        "--rebuild-live-features",
        action="store_true",
        help=(
            "Compatibility flag; live features are rebuilt by default."
        ),
    )
    parser.add_argument(
        "--reuse-live-features",
        action="store_true",
        help=(
            "Reuse the isolated live feature database only after exact "
            "build/version/date/QB-integrity validation."
        ),
    )
    parser.add_argument(
        "--skip-auto-feature-build",
        action="store_true",
        help="Fail instead of automatically reconstructing live features.",
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    if args.margin_history_db_path is None:
        args.margin_history_db_path = (
            args.project_root
            / "backtests"
            / DEFAULT_MARGIN_HISTORY_DATABASE_NAME
        ).resolve()
    else:
        args.margin_history_db_path = args.margin_history_db_path.resolve()

    if args.model_path is None:
        args.model_path = (
            args.project_root / "models" / DEFAULT_MODEL_FILENAME
        ).resolve()
    else:
        args.model_path = args.model_path.resolve()

    if args.ceiling_model_path is None:
        ceiling_candidates = [
            (args.project_root / "models" / filename).resolve()
            for filename in DEFAULT_CEILING_MODEL_FILENAMES
        ]
        args.ceiling_model_path = next(
            (
                candidate
                for candidate in ceiling_candidates
                if candidate.exists()
            ),
            ceiling_candidates[0],
        )
    else:
        args.ceiling_model_path = args.ceiling_model_path.resolve()

    if args.matchup_builder_path is None:
        builder_candidates = (
            args.project_root / DEFAULT_MATCHUP_BUILDER_FILENAME,
            args.project_root
            / "backtests"
            / DEFAULT_MATCHUP_BUILDER_FILENAME,
        )
        args.matchup_builder_path = next(
            (
                candidate.resolve()
                for candidate in builder_candidates
                if candidate.exists()
            ),
            builder_candidates[0].resolve(),
        )
    else:
        args.matchup_builder_path = args.matchup_builder_path.resolve()

    for name in ("circa_lines_path", "features_path", "schedule_path"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if not args.self_test:
        if args.week is None or not 1 <= args.week <= 18:
            parser.error("--week is required and must be between 1 and 18.")
        parsed_as_of = pd.to_datetime(
            args.as_of_date if args.as_of_date else dt.date.today(),
            errors="coerce",
        )
        if pd.isna(parsed_as_of):
            parser.error("--as-of-date must be YYYY-MM-DD.")
        args.as_of_date = pd.Timestamp(parsed_as_of).normalize()
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_team(value: Any) -> str:
    text = "" if value is None else str(value).upper().strip()
    return TEAM_ALIASES.get(text, text)


def first_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    lookup = {str(column).lower().strip(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name=?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone() is not None


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        return pd.DataFrame()
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(
        f'SELECT * FROM "{escaped}"',
        connection,
    )
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def numeric(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def prepare_margin_history(raw: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        raw,
        {"season", "circa_home_margin", "actual_home_margin"},
        "Circa final-margin history",
    )
    frame = raw.copy()
    frame["season"] = numeric(frame, "season")
    frame["circa_home_margin"] = numeric(frame, "circa_home_margin")
    frame["actual_home_margin"] = numeric(frame, "actual_home_margin")
    if "reference_fallback_used" in frame.columns:
        frame = frame[
            numeric(frame, "reference_fallback_used", 0).fillna(0).eq(0)
        ]
    if "strict_circa_line_valid" in frame.columns:
        frame = frame[
            numeric(frame, "strict_circa_line_valid", 0).fillna(0).eq(1)
        ]
    frame = frame[
        frame["season"].between(2020, SEASON - 1)
        & frame["circa_home_margin"].notna()
        & frame["actual_home_margin"].notna()
    ].copy()
    required_seasons = set(range(2020, SEASON))
    available_seasons = set(frame["season"].dropna().astype(int))
    missing_seasons = sorted(required_seasons - available_seasons)
    if frame.empty or missing_seasons:
        raise RuntimeError(
            "Strict Circa margin history must include every prior season "
            f"from 2020 through {SEASON - 1}; missing={missing_seasons}."
        )
    frame["season"] = frame["season"].astype(int)
    return frame.reset_index(drop=True)


def load_margin_history(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str]:
    candidates: list[Path] = []
    for path in (args.margin_history_db_path, args.db_path):
        if path not in candidates:
            candidates.append(path)
    checked: list[str] = []
    for database in candidates:
        checked.append(f"{database}::{args.margin_history_table}")
        if not database.exists():
            continue
        with sqlite3.connect(database) as connection:
            raw = read_table(connection, args.margin_history_table)
        if not raw.empty:
            return (
                prepare_margin_history(raw),
                f"{database}::{args.margin_history_table}",
            )
    raise RuntimeError(
        "The adjusted-V1 production policy requires the strict prior Circa "
        "margin-history table. Checked: " + "; ".join(checked)
    )


def one_required_text(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> str:
    values = {
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    }
    if len(values) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one nonempty {column}; "
            f"found={sorted(values)}."
        )
    return next(iter(values))


def prepare_structural_confidence_projection(
    raw: pd.DataFrame,
    run_audit: pd.DataFrame,
    week: int,
    as_of_date: pd.Timestamp,
    expected_games: pd.DataFrame,
) -> pd.DataFrame:
    frame = raw.copy()
    required = {
        "run_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "projected_home_margin",
        "model_variant",
        "build_id",
        "version",
        "prediction_as_of_date",
        "prediction_timestamp",
        "prediction_uses_market_inputs",
        "independent_projection_hash",
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
    }
    require_columns(frame, required, "Structural confidence source")
    frame["season"] = numeric(frame, "season")
    frame["week"] = numeric(frame, "week")
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    frame["structural_home_margin"] = pd.to_numeric(
        frame["projected_home_margin"], errors="coerce"
    )
    frame = frame[
        frame["season"].eq(SEASON)
        & frame["week"].eq(int(week))
        & frame["model_variant"].astype(str).str.upper().eq(
            STRUCTURAL_MODEL_VARIANT
        )
        & frame["structural_home_margin"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError(
            "No current STRUCTURAL_FORM_HFA rows match the requested week."
        )
    if not numeric(
        frame, "prediction_uses_market_inputs", 1
    ).fillna(1).eq(0).all():
        raise RuntimeError(
            "Structural confidence source contains a projection that used "
            "market inputs or lacks affirmative market-independence evidence."
        )

    exact_values = {
        "build_id": EXPECTED_STRUCTURAL_BUILD_ID,
        "version": EXPECTED_STRUCTURAL_VERSION,
        "model_variant": STRUCTURAL_MODEL_VARIANT,
        "form_build_id": EXPECTED_FORM_BUILD_ID,
        "form_version": EXPECTED_FORM_VERSION,
        "structural_power_build_id": EXPECTED_STRUCTURAL_POWER_BUILD_ID,
    }
    for column, expected_value in exact_values.items():
        found = one_required_text(
            frame, column, "Structural confidence source"
        )
        if found != expected_value:
            raise RuntimeError(
                f"Structural lineage mismatch for {column}: "
                f"expected={expected_value!r}, found={found!r}."
            )

    expected_through_week = max(0, int(week) - 1)
    through = numeric(frame, "form_through_week").dropna().unique()
    if len(through) != 1 or int(through[0]) != expected_through_week:
        raise RuntimeError(
            "Structural form through-week mismatch: "
            f"expected={expected_through_week}, found={through}."
        )
    for column in ("prediction_as_of_date", "form_as_of_date"):
        dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
        if dates.isna().any() or set(dates) != {as_of_date}:
            found = sorted(
                pd.Timestamp(value).date().isoformat()
                for value in dates.dropna().unique()
            )
            raise RuntimeError(
                f"Structural {column} mismatch: expected="
                f"{as_of_date.date().isoformat()}, found={found}."
            )
    timestamps = pd.to_datetime(
        frame["prediction_timestamp"], errors="coerce"
    )
    if timestamps.isna().any():
        raise RuntimeError("Structural prediction timestamp is missing.")
    if (timestamps.dt.normalize() != as_of_date).any():
        raise RuntimeError(
            "Structural prediction was not generated on the requested "
            "as-of date."
        )

    keys = ["season", "week", "home_team", "away_team"]
    if frame.duplicated(keys, keep=False).any():
        duplicates = frame.loc[
            frame.duplicated(keys, keep=False), keys + ["run_id"]
        ]
        raise RuntimeError(
            "Structural confidence source contains duplicate current-week "
            "rows:\n" + duplicates.head(20).to_string(index=False)
        )
    run_id = one_required_text(frame, "run_id", "Structural source")
    for column in (
        "independent_projection_hash",
        "form_date_imported",
        "preseason_snapshot_hash",
        "structural_power_version",
        "structural_power_date_imported",
        "structural_power_snapshot_hash",
    ):
        if frame[column].isna().any() or frame[column].astype(
            str
        ).str.strip().eq("").any():
            raise RuntimeError(
                f"Structural lineage field {column} is missing."
            )

    require_columns(
        run_audit,
        {
            "run_id", "season", "week", "build_id", "version",
            "model_variant", "status", "games",
            "prediction_uses_market_inputs", "form_build_id",
            "form_version", "form_as_of_date", "form_through_week",
            "form_date_imported", "preseason_snapshot_hash",
            "structural_power_build_id", "structural_power_version",
            "structural_power_date_imported",
            "structural_power_snapshot_hash", "completed_at",
        },
        "Structural run audit",
    )
    audit = run_audit[run_audit["run_id"].astype(str).eq(run_id)].copy()
    if len(audit) != 1:
        raise RuntimeError(
            "Structural current rows do not match exactly one run-audit row: "
            f"run_id={run_id!r}, audit_rows={len(audit)}."
        )
    audit_row = audit.iloc[0]
    audit_expected = {
        "season": SEASON,
        "week": int(week),
        "build_id": EXPECTED_STRUCTURAL_BUILD_ID,
        "version": EXPECTED_STRUCTURAL_VERSION,
        "model_variant": STRUCTURAL_MODEL_VARIANT,
        "status": "SUCCESS",
        "prediction_uses_market_inputs": 0,
        "form_build_id": EXPECTED_FORM_BUILD_ID,
        "form_version": EXPECTED_FORM_VERSION,
        "form_through_week": expected_through_week,
        "structural_power_build_id": EXPECTED_STRUCTURAL_POWER_BUILD_ID,
    }
    for column, expected_value in audit_expected.items():
        actual = audit_row[column]
        if str(actual) != str(expected_value):
            raise RuntimeError(
                f"Structural run-audit mismatch for {column}: "
                f"expected={expected_value!r}, found={actual!r}."
            )
    audit_form_date = pd.to_datetime(
        audit_row["form_as_of_date"], errors="coerce"
    )
    audit_completed_at = pd.to_datetime(
        audit_row["completed_at"], errors="coerce"
    )
    if (
        pd.isna(audit_form_date)
        or pd.Timestamp(audit_form_date).normalize() != as_of_date
        or pd.isna(audit_completed_at)
        or pd.Timestamp(audit_completed_at).normalize() != as_of_date
    ):
        raise RuntimeError(
            "Structural run-audit is not fresh for the requested as-of date."
        )
    row_to_audit_fields = {
        "form_date_imported": "form_date_imported",
        "preseason_snapshot_hash": "preseason_snapshot_hash",
        "structural_power_version": "structural_power_version",
        "structural_power_date_imported": (
            "structural_power_date_imported"
        ),
    }
    for row_column, audit_column in row_to_audit_fields.items():
        row_value = one_required_text(
            frame, row_column, "Structural confidence source"
        )
        if str(audit_row[audit_column]) != row_value:
            raise RuntimeError(
                "Structural row/run-audit lineage mismatch for "
                f"{row_column}."
            )
    if int(float(audit_row["games"])) != len(frame):
        raise RuntimeError(
            "Structural run-audit game count does not reconcile to current "
            "prediction rows."
        )
    if str(audit_row["structural_power_snapshot_hash"]) != one_required_text(
        frame,
        "structural_power_snapshot_hash",
        "Structural confidence source",
    ):
        raise RuntimeError(
            "Structural run-audit snapshot hash does not match its rows."
        )

    expected = expected_games[["home_team", "away_team"]].copy()
    expected["home_team"] = expected["home_team"].map(normalize_team)
    expected["away_team"] = expected["away_team"].map(normalize_team)
    lineage_columns = [
        "run_id",
        "home_team",
        "away_team",
        "structural_home_margin",
        "model_variant",
        "build_id",
        "version",
        "prediction_as_of_date",
        "prediction_timestamp",
        "prediction_uses_market_inputs",
        "independent_projection_hash",
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
    ]
    matched = expected.merge(
        frame[lineage_columns],
        on=["home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    missing = matched[matched["structural_home_margin"].isna()]
    if not missing.empty:
        raise RuntimeError(
            "Current-week STRUCTURAL_FORM_HFA fair lines are missing for "
            "Circa matchups:\n" + missing.to_string(index=False)
        )
    matched["season"] = SEASON
    matched["week"] = int(week)
    matched["structural_prediction_uses_market_inputs"] = numeric(
        matched, "prediction_uses_market_inputs", 1
    ).astype(int)
    return matched.reset_index(drop=True)


def load_structural_confidence_projection(
    args: argparse.Namespace,
    expected_games: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if not args.db_path.exists():
        raise FileNotFoundError(args.db_path)
    with sqlite3.connect(args.db_path) as connection:
        raw = read_table(connection, args.structural_table)
        run_audit = read_table(connection, STRUCTURAL_RUN_AUDIT_TABLE)
    if raw.empty:
        raise RuntimeError(
            "Production confidence requires the current structural table "
            f"{args.db_path}::{args.structural_table}. Run the weekly "
            "STRUCTURAL_FORM_HFA projection step first."
        )
    return (
        prepare_structural_confidence_projection(
            raw,
            run_audit,
            int(args.week),
            pd.Timestamp(args.as_of_date).normalize(),
            expected_games,
        ),
        f"{args.db_path}::{args.structural_table}",
    )


def margin_probability_mass(
    history: pd.DataFrame,
    minimum: int = PMF_MIN_MARGIN,
    maximum: int = PMF_MAX_MARGIN,
    smoothing: float = PMF_SMOOTHING,
) -> dict[int, float]:
    support = np.arange(minimum, maximum + 1, dtype=int)
    margins = np.rint(
        numeric(history, "actual_home_margin").dropna().to_numpy()
    ).astype(int)
    margins = margins[(margins >= minimum) & (margins <= maximum)]
    counts = pd.Series(margins).value_counts().to_dict()
    denominator = float(len(margins) + smoothing * len(support))
    if denominator <= 0:
        raise RuntimeError("Cannot estimate final-margin probability mass.")
    return {
        int(value): float(counts.get(int(value), 0) + smoothing) / denominator
        for value in support
    }


def conditional_margin_probability_mass(
    history: pd.DataFrame,
    market: float,
    bandwidth: float = PMF_BANDWIDTH,
    prior_strength: float = PMF_PRIOR_STRENGTH,
) -> tuple[dict[int, float], dict[str, float]]:
    support = np.arange(PMF_MIN_MARGIN, PMF_MAX_MARGIN + 1, dtype=int)
    lines = numeric(history, "circa_home_margin").to_numpy()
    margins = np.rint(numeric(history, "actual_home_margin").to_numpy())
    valid = (
        np.isfinite(lines)
        & np.isfinite(margins)
        & (margins >= PMF_MIN_MARGIN)
        & (margins <= PMF_MAX_MARGIN)
    )
    lines = lines[valid].astype(float)
    margins = margins[valid].astype(int)
    if not len(margins):
        raise RuntimeError("Conditional final-margin history is empty.")
    weights = np.exp(-0.5 * ((lines - float(market)) / bandwidth) ** 2)
    weight_sum = float(weights.sum())
    squared_weight_sum = float(np.square(weights).sum())
    effective_sample_size = (
        weight_sum**2 / squared_weight_sum if squared_weight_sum > 0 else 0.0
    )
    global_pmf = margin_probability_mass(history)
    denominator = weight_sum + prior_strength
    pmf = {
        int(value): (
            float(weights[margins == value].sum())
            + prior_strength * global_pmf[int(value)]
        )
        / denominator
        for value in support
    }
    return pmf, {
        "pmf_market_bandwidth": bandwidth,
        "pmf_prior_strength": prior_strength,
        "pmf_effective_sample_size": effective_sample_size,
        "pmf_prior_seasons": float(history["season"].nunique()),
        "pmf_prior_games": float(len(history)),
    }


def crossed_integer_margins(market: float, adjusted: float) -> list[int]:
    if not np.isfinite(market) or not np.isfinite(adjusted):
        return []
    if math.isclose(float(market), float(adjusted), abs_tol=1e-12):
        return []
    if adjusted > market:
        first = math.floor(market) + 1
        last = math.floor(adjusted + 1e-12)
    else:
        first = math.ceil(adjusted - 1e-12)
        last = math.ceil(market) - 1
    return list(range(first, last + 1)) if last >= first else []


def add_primary_key_features(
    card: pd.DataFrame,
    margin_history: pd.DataFrame,
) -> pd.DataFrame:
    featured_rows: list[dict[str, Any]] = []
    for row in card.itertuples(index=False):
        market = float(row.circa_home_margin)
        residual = float(row.predicted_circa_residual)
        adjusted = market + residual
        pmf, diagnostics = conditional_margin_probability_mass(
            margin_history,
            market,
        )
        crossed = crossed_integer_margins(market, adjusted)
        supported = [value for value in crossed if value in pmf]
        integer_market = int(round(market))
        market_is_integer = math.isclose(
            market, float(integer_market), abs_tol=1e-9
        )
        push_mass = (
            float(pmf.get(integer_market, 0.0)) if market_is_integer else 0.0
        )
        key_masses = {
            key: float(
                sum(pmf[value] for value in supported if abs(value) == key)
            )
            for key in PRIMARY_KEYS
        }
        push_credit = (
            0.5 * push_mass
            if market_is_integer and abs(integer_market) in PRIMARY_KEYS
            else 0.0
        )
        featured_rows.append(
            {
                "adjusted_v1_crossed_key_mass_3": key_masses[3],
                "adjusted_v1_crossed_key_mass_7": key_masses[7],
                "adjusted_v1_market_push_mass": push_mass,
                "adjusted_v1_key_tiebreak_score": (
                    key_masses[3] + key_masses[7] + push_credit
                ),
                "adjusted_v1_crossed_integer_margins": "|".join(
                    str(value) for value in crossed
                ),
                **diagnostics,
            }
        )
    return pd.concat(
        [card.reset_index(drop=True), pd.DataFrame(featured_rows)],
        axis=1,
    )


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise RuntimeError(
            f"Unsupported file type {suffix!r}; use CSV, XLSX, or Parquet."
        )
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def find_default_lines_path(project_root: Path, week: int) -> Optional[Path]:
    names = (
        f"nfl_circa_lines_2026_week_{week}.csv",
        f"nfl_circa_lines_week_{week}.csv",
        f"circa_lines_2026_week_{week}.csv",
        "nfl_circa_lines_2026.csv",
        "circa_lines_2026.csv",
    )
    folders = (
        project_root,
        project_root / "inputs",
        project_root / "outputs",
        project_root / "circa",
    )
    for folder in folders:
        for name in names:
            path = folder / name
            if path.exists():
                return path.resolve()
    return None


def prepare_lines(raw: pd.DataFrame, week: int) -> pd.DataFrame:
    frame = raw.copy()
    week_column = first_existing(
        frame.columns,
        ("week", "game_week", "week_number"),
    )
    home_column = first_existing(
        frame.columns,
        ("home_team", "home", "home_team_abbr"),
    )
    away_column = first_existing(
        frame.columns,
        ("away_team", "away", "away_team_abbr"),
    )
    game_id_column = first_existing(
        frame.columns,
        ("game_id", "id", "gsis_id"),
    )
    margin_column = first_existing(
        frame.columns,
        ("circa_home_margin", "home_market_margin", "market_home_margin"),
    )
    home_spread_column = first_existing(
        frame.columns,
        ("home_spread", "circa_home_spread", "home_line"),
    )

    missing = []
    if home_column is None:
        missing.append("home_team")
    if away_column is None:
        missing.append("away_team")
    if margin_column is None and home_spread_column is None:
        missing.append("circa_home_margin or home_spread")
    if missing:
        raise RuntimeError(
            f"Circa line file is missing required fields: {missing}"
        )

    output = pd.DataFrame(index=frame.index)
    output["season"] = SEASON
    output["week"] = (
        pd.to_numeric(frame[week_column], errors="coerce")
        if week_column is not None
        else week
    )
    output["home_team"] = frame[home_column].map(normalize_team)
    output["away_team"] = frame[away_column].map(normalize_team)
    output["circa_home_margin"] = (
        pd.to_numeric(frame[margin_column], errors="coerce")
        if margin_column is not None
        else -pd.to_numeric(frame[home_spread_column], errors="coerce")
    )
    if game_id_column is not None:
        output["game_id"] = frame[game_id_column].astype(str).str.strip()
    else:
        output["game_id"] = ""

    output = output[
        output["week"].eq(int(week))
        & output["home_team"].ne("")
        & output["away_team"].ne("")
        & output["circa_home_margin"].notna()
    ].copy()
    output["week"] = output["week"].astype(int)

    if output.empty:
        raise RuntimeError(f"No usable Circa lines were found for Week {week}.")
    if output.duplicated(["home_team", "away_team"]).any():
        duplicates = output[
            output.duplicated(["home_team", "away_team"], keep=False)
        ]
        raise RuntimeError(
            "Duplicate Circa matchups detected:\n"
            + duplicates.to_string(index=False)
        )
    if len(output) < 5:
        raise RuntimeError(
            f"Only {len(output)} Circa games were loaded; at least five are "
            "required for a contest card."
        )
    return output.reset_index(drop=True)


def assert_model_bundle_qb_integrity(
    bundle: dict[str, Any],
    label: str,
) -> None:
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
        raise RuntimeError(
            f"{label} predates the repaired QB feature pipeline and is blocked. "
            f"Missing integrity metadata: {missing}. Rebuild the model."
        )

    metrics = {
        "passed": int(bundle["qb_feature_integrity_passed"]),
        "epa_coverage": float(bundle["qb_side_epa_coverage"]),
        "cpoe_coverage": float(bundle["qb_side_cpoe_coverage"]),
        "epa_std": float(bundle["qb_epa_advantage_std"]),
        "cpoe_std": float(bundle["qb_cpoe_advantage_std"]),
        "epa_nonzero": int(bundle["qb_epa_advantage_nonzero_rows"]),
        "cpoe_nonzero": int(bundle["qb_cpoe_advantage_nonzero_rows"]),
    }
    valid = (
        metrics["passed"] == 1
        and metrics["epa_coverage"] >= 0.99
        and metrics["cpoe_coverage"] >= 0.99
        and metrics["epa_std"] > 1e-6
        and metrics["cpoe_std"] > 1e-6
        and metrics["epa_nonzero"] > 0
        and metrics["cpoe_nonzero"] > 0
    )
    if not valid:
        raise RuntimeError(
            f"{label} failed its saved QB feature-integrity gate: "
            + json.dumps(metrics, sort_keys=True)
        )


def load_bundle(path: Path, approve_forward_test: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    required = {
        "model",
        "feature_order",
        "rating_alpha",
        "feature_set",
        "model_alpha",
        "selection_policy",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"Circa model bundle is missing keys: {missing}")
    assert_model_bundle_qb_integrity(bundle, "Circa V1 model bundle")
    bundle_lineage = {
        "build_id": str(bundle.get("build_id", "")),
        "version": str(bundle.get("version", "")),
    }
    expected_lineage = {
        "build_id": EXPECTED_V1_MODEL_BUILD_ID,
        "version": EXPECTED_V1_MODEL_VERSION,
    }
    if bundle_lineage != expected_lineage:
        raise RuntimeError(
            "Circa V1 model bundle is stale or incompatible: "
            + json.dumps(bundle_lineage, sort_keys=True)
        )
    if abs(float(bundle.get("qb_epa_standardized_coefficient", 0.0))) <= 1e-12:
        raise RuntimeError(
            "Circa V1 model bundle has no nonzero repaired QB EPA coefficient."
        )

    expected_policy = "TOP5_WEEKLY_BY_ABSOLUTE_PREDICTED_CIRCA_RESIDUAL"
    if str(bundle["selection_policy"]) != expected_policy:
        raise RuntimeError(
            "Unexpected Circa selection policy: "
            f"{bundle['selection_policy']!r}"
        )

    if not bool(bundle.get("implementation_ready", False)) and not approve_forward_test:
        raise RuntimeError(
            "The saved bundle has implementation_ready=False under its original "
            "historical safety gates. The user has approved contest-only forward "
            "deployment, but the command must include "
            "--approve-contest-forward-test to record that override."
        )
    return bundle


def late_model_from_mapping(
    mapping: Any,
    alpha: float,
) -> Any:
    if not isinstance(mapping, dict):
        raise RuntimeError("Ceiling bundle late_models must be a dictionary.")

    candidate_keys = (
        alpha,
        float(alpha),
        int(alpha) if float(alpha).is_integer() else alpha,
        str(alpha),
        f"{float(alpha):g}",
    )
    for key in candidate_keys:
        if key in mapping:
            return mapping[key]

    for key, value in mapping.items():
        try:
            if math.isclose(
                float(key),
                float(alpha),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return value
        except (TypeError, ValueError):
            continue

    raise RuntimeError(
        f"Ceiling late model alpha={alpha:g} was not found. "
        f"Available keys={list(mapping)}"
    )


def load_ceiling_bundle(
    path: Path,
    v1_bundle: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        searched = ", ".join(DEFAULT_CEILING_MODEL_FILENAMES)
        raise FileNotFoundError(
            f"Ceiling model bundle not found: {path}. "
            f"Expected one of these files under models: {searched}"
        )

    raw = joblib.load(path)
    required = {"feature_order", "rating_alpha", "late_models"}
    missing = sorted(required - set(raw))
    if missing:
        raise RuntimeError(
            f"Ceiling bundle is missing required keys: {missing}"
        )
    assert_model_bundle_qb_integrity(raw, "Circa ceiling model bundle")
    ceiling_lineage = {
        "build_id": str(raw.get("build_id", "")),
        "version": str(raw.get("version", "")),
    }
    expected_ceiling_lineage = {
        "build_id": EXPECTED_CEILING_MODEL_BUILD_ID,
        "version": EXPECTED_CEILING_MODEL_VERSION,
    }
    if ceiling_lineage != expected_ceiling_lineage:
        raise RuntimeError(
            "Circa ceiling model bundle is stale or incompatible: "
            + json.dumps(ceiling_lineage, sort_keys=True)
        )
    if abs(
        float(raw.get("selected_late_qb_epa_standardized_coefficient", 0.0))
    ) <= 1e-12:
        raise RuntimeError(
            "Circa ceiling bundle has no nonzero repaired QB EPA coefficient."
        )

    lineage = {
        "source_v1_build_id": str(v1_bundle.get("build_id", "")),
        "source_v1_version": str(v1_bundle.get("version", "")),
        "source_v1_created_at": str(v1_bundle.get("created_at", "")),
    }
    lineage_mismatch = {
        key: {"ceiling": str(raw.get(key, "")), "v1": expected}
        for key, expected in lineage.items()
        if str(raw.get(key, "")) != expected
    }
    if lineage_mismatch:
        raise RuntimeError(
            "Ceiling model was not rebuilt from the loaded repaired V1 model; "
            "stale mixed-generation bundles are blocked: "
            + json.dumps(lineage_mismatch, sort_keys=True)
        )

    v1_features = [str(value) for value in v1_bundle["feature_order"]]
    ceiling_features = [str(value) for value in raw["feature_order"]]
    if ceiling_features != v1_features:
        raise RuntimeError(
            "Ceiling and locked V1 feature orders do not match exactly."
        )
    if not math.isclose(
        float(raw["rating_alpha"]),
        float(v1_bundle["rating_alpha"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Ceiling and locked V1 rating_alpha values do not match."
        )

    if isinstance(raw.get("ceiling_source"), dict):
        profile = dict(raw["ceiling_source"])
        late_alpha = float(profile["late_alpha"])
        blend_weight = float(profile["blend_weight"])
        direction_policy = str(profile["direction_policy"])
        late_model = late_model_from_mapping(
            raw["late_models"],
            late_alpha,
        )
        source_profile = "V3_2_CEILING_SOURCE"
    else:
        raise RuntimeError(
            "Unsupported repaired V3.2 ceiling bundle: ceiling_source is "
            "missing."
        )

    if direction_policy != "BLENDED_SIGN":
        raise RuntimeError(
            "The requested ceiling comparison must use BLENDED_SIGN; "
            f"bundle policy={direction_policy!r}."
        )
    if not 0.0 <= blend_weight <= 1.0:
        raise RuntimeError(
            f"Invalid ceiling blend_weight={blend_weight}."
        )

    return {
        "raw_bundle": raw,
        "model_path": path,
        "build_id": str(raw.get("build_id", "")),
        "version": str(raw.get("version", "")),
        "feature_order": ceiling_features,
        "rating_alpha": float(raw["rating_alpha"]),
        "feature_set": str(raw.get("feature_set", "")),
        "late_model": late_model,
        "late_alpha": late_alpha,
        "blend_weight": blend_weight,
        "direction_policy": direction_policy,
        "source_profile": source_profile,
        "shadow_only": int(bool(raw.get("shadow_only", True))),
    }


def load_schedule_package() -> pd.DataFrame:
    errors = []
    try:
        import nflreadpy  # type: ignore

        loader = getattr(nflreadpy, "load_schedules", None)
        if loader is not None:
            for call in (
                lambda: loader(list(range(2018, 2027))),
                lambda: loader(seasons=list(range(2018, 2027))),
            ):
                try:
                    value = call()
                    frame = (
                        value.to_pandas()
                        if hasattr(value, "to_pandas")
                        else pd.DataFrame(value)
                    )
                    if not frame.empty:
                        return frame
                except Exception as exc:
                    errors.append(f"nflreadpy: {exc}")
    except Exception as exc:
        errors.append(f"nflreadpy import: {exc}")

    try:
        import nfl_data_py  # type: ignore

        loader = getattr(nfl_data_py, "import_schedules", None)
        if loader is not None:
            frame = pd.DataFrame(loader(list(range(2018, 2027))))
            if not frame.empty:
                return frame
    except Exception as exc:
        errors.append(f"nfl_data_py: {exc}")

    raise RuntimeError("Unable to load NFL schedules: " + " | ".join(errors))


def augment_schedule_for_target_week(
    raw_schedule: pd.DataFrame,
    lines: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    frame = raw_schedule.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    season_column = first_existing(frame.columns, ("season", "season_year", "year"))
    week_column = first_existing(frame.columns, ("week", "game_week", "week_number"))
    home_column = first_existing(frame.columns, ("home_team", "home", "home_team_abbr"))
    away_column = first_existing(frame.columns, ("away_team", "away", "away_team_abbr"))
    game_id_column = first_existing(frame.columns, ("game_id", "id", "gsis_id"))
    game_type_column = first_existing(frame.columns, ("game_type", "season_type"))
    home_score_column = first_existing(
        frame.columns,
        ("home_score", "score_home", "home_points"),
    )
    away_score_column = first_existing(
        frame.columns,
        ("away_score", "score_away", "away_points"),
    )
    spread_column = first_existing(
        frame.columns,
        (
            "spread_line",
            "closing_spread",
            "market_spread",
            "spread",
            "current_spread",
        ),
    )

    required = {
        "season": season_column,
        "week": week_column,
        "home": home_column,
        "away": away_column,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise RuntimeError(f"Schedule lacks required columns: {missing}")

    if home_score_column is None:
        home_score_column = "home_score"
        frame[home_score_column] = np.nan
    if away_score_column is None:
        away_score_column = "away_score"
        frame[away_score_column] = np.nan
    if spread_column is None:
        spread_column = "spread_line"
        frame[spread_column] = np.nan
    if game_type_column is None:
        game_type_column = "game_type"
        frame[game_type_column] = "REG"
    if game_id_column is None:
        game_id_column = "game_id"
        frame[game_id_column] = ""

    normalized_home = frame[home_column].map(normalize_team)
    normalized_away = frame[away_column].map(normalize_team)
    season_values = pd.to_numeric(frame[season_column], errors="coerce")
    week_values = pd.to_numeric(frame[week_column], errors="coerce")

    matched_count = 0
    for line in lines.itertuples(index=False):
        mask = (
            season_values.eq(SEASON)
            & week_values.eq(int(week))
            & normalized_home.eq(line.home_team)
            & normalized_away.eq(line.away_team)
        )
        matches = frame.index[mask].tolist()
        if len(matches) != 1:
            raise RuntimeError(
                f"Schedule reconciliation failed for "
                f"{line.away_team} at {line.home_team}: matches={len(matches)}"
            )
        index = matches[0]
        frame.loc[index, home_score_column] = 0.0
        frame.loc[index, away_score_column] = 0.0
        frame.loc[index, spread_column] = float(line.circa_home_margin)
        frame.loc[index, game_type_column] = "REG"
        if not str(frame.loc[index, game_id_column]).strip():
            frame.loc[index, game_id_column] = (
                f"2026_{week:02d}_{line.away_team}_{line.home_team}"
            )
        matched_count += 1

    if matched_count != len(lines):
        raise RuntimeError(
            f"Only {matched_count} of {len(lines)} Circa games were matched."
        )
    return frame


def patched_builder_source(source: str, target_week: int) -> str:
    # Historical Week 1 carries no same-season personnel performance by design.
    # Beginning in Week 2, the live build must load 2026 PBP and snap counts so
    # QB EPA/CPOE cannot silently remain at zero for the entire season.
    source_end = 2026 if int(target_week) == 1 else 2027
    replacements = {
        r"PBP_SOURCE_SEASONS\s*=\s*tuple\(range\(2017,\s*202(?:6|7)\)\)":
            f"PBP_SOURCE_SEASONS = tuple(range(2017, {source_end}))",
        r"SNAP_SOURCE_SEASONS\s*=\s*tuple\(range\(2018,\s*202(?:6|7)\)\)":
            f"SNAP_SOURCE_SEASONS = tuple(range(2018, {source_end}))",
        r"PREDICTION_SEASONS\s*=\s*tuple\(range\(2018,\s*2026\)\)":
            "PREDICTION_SEASONS = tuple(range(2018, 2027))",
        r'CACHE_DATABASE_NAME\s*=\s*"nfl_weekly_matchup_source_cache_v2\.sqlite"':
            'CACHE_DATABASE_NAME = "nfl_weekly_matchup_source_cache_v2_live_2026.sqlite"',
        r'OUTPUT_DATABASE_NAME\s*=\s*"nfl_weekly_matchup_residual\.sqlite"':
            f'OUTPUT_DATABASE_NAME = "{LIVE_MATRIX_DATABASE_NAME}"',
        r'MODEL_FILENAME\s*=\s*"nfl_matchup_residual_model_v2\.joblib"':
            'MODEL_FILENAME = "nfl_matchup_residual_model_v2_live_2026.joblib"',
        r'METADATA_FILENAME\s*=\s*"nfl_matchup_residual_model_v2_metadata\.json"':
            'METADATA_FILENAME = "nfl_matchup_residual_model_v2_live_2026_metadata.json"',
    }
    patched = source
    for pattern, replacement in replacements.items():
        patched, count = re.subn(pattern, replacement, patched)
        if count != 1:
            raise RuntimeError(
                f"Unable to patch exact matchup-builder contract: {pattern}"
            )
    return patched


def build_live_feature_database(
    args: argparse.Namespace,
    lines: pd.DataFrame,
) -> Path:
    if not args.matchup_builder_path.exists():
        raise FileNotFoundError(args.matchup_builder_path)

    live_db = (
        args.project_root / "backtests" / LIVE_MATRIX_DATABASE_NAME
    )
    if (
        live_db.exists()
        and args.reuse_live_features
        and not args.rebuild_live_features
    ):
        return live_db

    if args.schedule_path is not None:
        raw_schedule = read_frame(args.schedule_path)
    else:
        raw_schedule = load_schedule_package()
    augmented = augment_schedule_for_target_week(
        raw_schedule,
        lines,
        int(args.week),
    )

    source = args.matchup_builder_path.read_text(encoding="utf-8")
    patched = patched_builder_source(source, int(args.week))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        patched_path = temp / "build_live_matchup_matrix_2026.py"
        schedule_path = temp / "augmented_schedule_2018_2026.csv"
        patched_path.write_text(patched, encoding="utf-8")
        augmented.to_csv(schedule_path, index=False)

        command = [
            sys.executable,
            str(patched_path),
            "--project-root",
            str(args.project_root),
            "--schedule-path",
            str(schedule_path),
            "--rebuild-cache",
            "--no-csv",
        ]
        print("[CIRCA_TOP5] Building isolated exact live matchup matrix:")
        print(subprocess.list2cmdline(command))
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
        exit_code = int(process.wait())
        if exit_code != 0:
            raise RuntimeError(
                f"Live matchup matrix builder failed with exit code {exit_code}."
            )

    if not live_db.exists():
        raise RuntimeError(f"Expected live matrix database was not created: {live_db}")
    return live_db


def load_validated_live_feature_matrix(
    database: Path,
    week: int,
    as_of_date: pd.Timestamp,
    rating_alpha: float,
) -> pd.DataFrame:
    if not database.exists():
        raise RuntimeError(f"Live feature database is missing: {database}")
    with sqlite3.connect(database) as connection:
        matrix = read_table(connection, "nfl_matchup_game_matrix")
        audit = read_table(connection, "nfl_matchup_run_audit")
        qb_audit = read_table(
            connection, "nfl_matchup_qb_feature_integrity_audit"
        )
    if len(audit) != 1:
        raise RuntimeError(
            "Live feature database must contain exactly one run-audit row."
        )
    require_columns(
        audit,
        {
            "build_id", "version", "selected_rating_alpha",
            "selected_feature_set", "qb_identity_match_rate",
            "qb_feature_integrity_passed", "created_at",
        },
        "Live feature run audit",
    )
    row = audit.iloc[0]
    if str(row["build_id"]) != EXPECTED_LIVE_MATCHUP_BUILD_ID:
        raise RuntimeError(
            "Live feature build ID is stale or incompatible: "
            f"{row['build_id']!r}."
        )
    if str(row["version"]) != EXPECTED_LIVE_MATCHUP_VERSION:
        raise RuntimeError(
            "Live feature version is stale or incompatible: "
            f"{row['version']!r}."
        )
    if not math.isclose(
        float(row["selected_rating_alpha"]),
        float(rating_alpha),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Live feature rating alpha does not match the Circa bundle."
        )
    if str(row["selected_feature_set"]) != "FULL_COMPACT":
        raise RuntimeError("Live feature set is not FULL_COMPACT.")
    if int(float(row["qb_feature_integrity_passed"])) != 1:
        raise RuntimeError("Live feature QB integrity audit did not pass.")
    if not math.isclose(
        float(row["qb_identity_match_rate"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Live feature QB identity match rate is not 100%.")
    created_at = pd.to_datetime(row["created_at"], errors="coerce")
    if pd.isna(created_at) or created_at.normalize() != as_of_date:
        raise RuntimeError(
            "Live feature database was not rebuilt on the requested as-of "
            f"date {as_of_date.date().isoformat()}."
        )
    if qb_audit.empty or "feature_integrity_passed" not in qb_audit.columns:
        raise RuntimeError("Live feature database lacks the QB integrity audit.")
    if not numeric(
        qb_audit, "feature_integrity_passed", 0
    ).fillna(0).eq(1).all():
        raise RuntimeError("A live feature QB integrity scope failed.")

    require_columns(
        matrix,
        {"season", "week", "rating_alpha", "home_team", "away_team"},
        "Live feature matrix",
    )
    season = numeric(matrix, "season")
    weeks = numeric(matrix, "week")
    alphas = numeric(matrix, "rating_alpha")
    subset = matrix[
        season.eq(SEASON)
        & weeks.eq(int(week))
        & alphas.eq(float(rating_alpha))
    ].copy()
    if subset.empty:
        raise RuntimeError(
            "Validated live feature database has no exact target-week rows "
            f"at rating_alpha={rating_alpha}."
        )
    if subset.duplicated(["home_team", "away_team"]).any():
        raise RuntimeError("Live feature matrix contains duplicate matchups.")
    subset.attrs["live_feature_lineage"] = {
        "build_id": str(row["build_id"]),
        "version": str(row["version"]),
        "created_at": pd.Timestamp(created_at).isoformat(),
        "rating_alpha": float(row["selected_rating_alpha"]),
        "feature_set": str(row["selected_feature_set"]),
    }
    return subset


def load_feature_matrix(
    args: argparse.Namespace,
    lines: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    if args.features_path is not None:
        raise RuntimeError(
            "Direct feature files are blocked in production because their "
            "build/version/date lineage cannot be verified. Use the isolated "
            "live feature database."
        )

    live_db = args.project_root / "backtests" / LIVE_MATRIX_DATABASE_NAME
    if args.skip_auto_feature_build:
        if not args.reuse_live_features:
            raise RuntimeError(
                "--skip-auto-feature-build requires --reuse-live-features; "
                "the existing database will still be validated."
            )
    else:
        live_db = build_live_feature_database(args, lines)
    subset = load_validated_live_feature_matrix(
        live_db,
        int(args.week),
        pd.Timestamp(args.as_of_date).normalize(),
        float(bundle["rating_alpha"]),
    )
    return subset, f"{live_db}::nfl_matchup_game_matrix"


def merge_features_and_lines(
    features: pd.DataFrame,
    lines: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    frame = features.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]

    if "week" not in frame.columns and "prediction_week" in frame.columns:
        frame["week"] = frame["prediction_week"]
    required_keys = {"home_team", "away_team"}
    missing = sorted(required_keys - set(frame.columns))
    if missing:
        raise RuntimeError(f"Live feature matrix lacks matchup keys: {missing}")

    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    if "game_id" not in frame.columns:
        frame["game_id"] = (
            "2026_"
            + pd.to_numeric(frame["week"], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.zfill(2)
            + "_"
            + frame["away_team"]
            + "_"
            + frame["home_team"]
        )

    line_columns = [
        "week",
        "home_team",
        "away_team",
        "circa_home_margin",
    ]
    if "game_id" in lines.columns:
        line_columns.append("game_id")

    merged = frame.merge(
        lines[line_columns],
        on=["week", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_line"),
    )
    if len(merged) != len(lines):
        missing_lines = lines.merge(
            merged[["week", "home_team", "away_team"]],
            on=["week", "home_team", "away_team"],
            how="left",
            indicator=True,
        )
        missing_lines = missing_lines[missing_lines["_merge"].eq("left_only")]
        raise RuntimeError(
            "Not every Circa matchup joined to the live feature matrix:\n"
            + missing_lines.to_string(index=False)
        )

    merged["market_home_margin"] = pd.to_numeric(
        merged["circa_home_margin"],
        errors="coerce",
    )
    merged["absolute_market_home_margin"] = merged[
        "market_home_margin"
    ].abs()

    feature_order = [str(feature) for feature in bundle["feature_order"]]
    missing_features = sorted(set(feature_order) - set(merged.columns))
    if missing_features:
        raise RuntimeError(
            "Exact Circa model features are unavailable: "
            f"{missing_features}. The runner will not substitute structural "
            "spread disagreement or differently defined proxy fields."
        )
    merged.attrs["qb_feature_integrity"] = assert_live_qb_feature_integrity(
        merged
    )
    merged.attrs["live_feature_lineage"] = dict(
        features.attrs.get("live_feature_lineage", {})
    )
    return merged


def assert_live_qb_feature_integrity(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    week_values = pd.to_numeric(frame["week"], errors="coerce").dropna().unique()
    if len(week_values) != 1:
        raise RuntimeError(
            f"Live QB integrity requires exactly one week; found {week_values}."
        )
    week = int(week_values[0])
    if week == 1:
        return {
            "week": week,
            "status": "WEEK1_HISTORICAL_NO_SAME_SEASON_QB_FEATURES",
            "passed": 1,
            "qb_epa_advantage_nonzero_rows": int(
                numeric(frame, "qb_recent_epa_advantage", 0)
                .fillna(0)
                .abs()
                .gt(1e-12)
                .sum()
            ),
            "qb_cpoe_advantage_nonzero_rows": int(
                numeric(frame, "qb_recent_cpoe_advantage", 0)
                .fillna(0)
                .abs()
                .gt(1e-12)
                .sum()
            ),
        }

    required = {
        "personnel_complete",
        "home_qb_recent_epa",
        "away_qb_recent_epa",
        "home_qb_recent_cpoe",
        "away_qb_recent_cpoe",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Live QB feature integrity fields are missing: " + str(missing)
        )
    personnel_complete = numeric(frame, "personnel_complete", 0).fillna(0)
    side_epa = pd.concat(
        [
            numeric(frame, "home_qb_recent_epa"),
            numeric(frame, "away_qb_recent_epa"),
        ],
        ignore_index=True,
    )
    side_cpoe = pd.concat(
        [
            numeric(frame, "home_qb_recent_cpoe"),
            numeric(frame, "away_qb_recent_cpoe"),
        ],
        ignore_index=True,
    )
    epa_advantage = numeric(frame, "qb_recent_epa_advantage")
    cpoe_advantage = numeric(frame, "qb_recent_cpoe_advantage")
    metrics = {
        "week": week,
        "status": "QB_FEATURE_INTEGRITY_CHECKED",
        "personnel_complete_rate": float(personnel_complete.eq(1).mean()),
        "qb_side_epa_coverage": float(side_epa.notna().mean()),
        "qb_side_cpoe_coverage": float(side_cpoe.notna().mean()),
        "qb_epa_advantage_nonzero_rows": int(
            epa_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
        "qb_epa_advantage_std": float(epa_advantage.std(ddof=0)),
        "qb_cpoe_advantage_nonzero_rows": int(
            cpoe_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
        "qb_cpoe_advantage_std": float(cpoe_advantage.std(ddof=0)),
    }
    passed = (
        metrics["personnel_complete_rate"] == 1.0
        and metrics["qb_side_epa_coverage"] >= 0.90
        and metrics["qb_side_cpoe_coverage"] >= 0.90
        and metrics["qb_epa_advantage_nonzero_rows"] > 0
        and metrics["qb_cpoe_advantage_nonzero_rows"] > 0
        and metrics["qb_epa_advantage_std"] > 1e-6
        and metrics["qb_cpoe_advantage_std"] > 1e-6
    )
    metrics["passed"] = int(passed)
    if not passed:
        raise RuntimeError(
            "Live QB EPA/CPOE integrity gate failed. The Circa card will not "
            "run with dead or incomplete QB features: "
            + json.dumps(metrics, sort_keys=True)
        )
    return metrics


def add_selection_fields(
    output: pd.DataFrame,
    residual_column: str,
    absolute_column: str,
    adjusted_margin_column: str,
    top5_label: str,
    alternate_label: str,
) -> pd.DataFrame:
    result = output.copy()
    result[absolute_column] = result[residual_column].abs()
    result[adjusted_margin_column] = (
        result["circa_home_margin"] + result[residual_column]
    )
    result["selected_side"] = np.where(
        result[residual_column].gt(0),
        "HOME",
        np.where(result[residual_column].lt(0), "AWAY", "NONE"),
    )
    result["selected_team"] = np.where(
        result["selected_side"].eq("HOME"),
        result["home_team"],
        np.where(
            result["selected_side"].eq("AWAY"),
            result["away_team"],
            "",
        ),
    )
    result["selected_circa_spread"] = np.where(
        result["selected_side"].eq("HOME"),
        -result["circa_home_margin"],
        result["circa_home_margin"],
    )
    result = result.sort_values(
        [absolute_column, "game_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    result["contest_rank"] = np.arange(1, len(result) + 1)
    result["card_status"] = np.where(
        result["contest_rank"].le(5),
        top5_label,
        np.where(
            result["contest_rank"].le(10),
            alternate_label,
            "UNRANKED",
        ),
    )
    result["recommended_sportsbook_stake"] = 0.0
    result["contest_only_model"] = 1
    return result


def apply_adjusted_v1_policy(
    raw_v1_card: pd.DataFrame,
    margin_history: pd.DataFrame,
) -> pd.DataFrame:
    """Lock raw ranks 1-4 and use prior-only 3/7 mass for the fifth slot."""
    if len(raw_v1_card) < 5:
        raise RuntimeError("Adjusted V1 requires at least five games.")
    working = add_primary_key_features(raw_v1_card, margin_history)
    working["raw_residual_rank"] = numeric(
        working, "contest_rank"
    ).astype(int)
    raw_fifth = working[working["raw_residual_rank"].eq(5)].iloc[0]
    fifth_residual = float(raw_fifth["absolute_predicted_circa_residual"])
    fifth_key_score = float(raw_fifth["adjusted_v1_key_tiebreak_score"])
    candidate_pool = working[
        working["raw_residual_rank"].between(
            5, ADJUSTED_V1_CANDIDATE_RANK_MAX
        )
        & working["absolute_predicted_circa_residual"].ge(
            fifth_residual - ADJUSTED_V1_TIEBREAK_TOLERANCE - 1e-12
        )
    ].copy()
    candidate_pool = candidate_pool.sort_values(
        [
            "adjusted_v1_key_tiebreak_score",
            "absolute_predicted_circa_residual",
            "raw_residual_rank",
            "game_id",
        ],
        ascending=[False, False, True, True],
    )
    winner_index = candidate_pool.index[0]
    frozen = working[working["raw_residual_rank"].le(4)].copy()
    winner = working.loc[[winner_index]].copy()
    remaining = working[
        ~working.index.isin([*frozen.index, winner_index])
    ].sort_values("raw_residual_rank")
    result = pd.concat([frozen, winner, remaining], ignore_index=True)
    result["contest_rank"] = np.arange(1, len(result) + 1)
    result["card_status"] = np.where(
        result["contest_rank"].le(5),
        "OFFICIAL_TOP5",
        np.where(
            result["contest_rank"].le(10),
            "ALTERNATE_6_10",
            "UNRANKED",
        ),
    )
    result["production_selection_policy"] = ADJUSTED_V1_POLICY
    result["adjusted_v1_tiebreak_tolerance"] = (
        ADJUSTED_V1_TIEBREAK_TOLERANCE
    )
    result["adjusted_v1_candidate_rank_max"] = (
        ADJUSTED_V1_CANDIDATE_RANK_MAX
    )
    result["adjusted_v1_candidate_pool_size"] = int(len(candidate_pool))
    result["promoted_by_key"] = (
        result["contest_rank"].eq(5)
        & result["raw_residual_rank"].gt(5)
    ).astype(int)
    result["residual_deficit_vs_raw_fifth"] = np.where(
        result["contest_rank"].eq(5),
        fifth_residual - result["absolute_predicted_circa_residual"],
        0.0,
    )
    result["key_score_advantage_vs_raw_fifth"] = np.where(
        result["contest_rank"].eq(5),
        result["adjusted_v1_key_tiebreak_score"] - fifth_key_score,
        0.0,
    )
    locked = result[result["contest_rank"].le(4)]
    if not locked["contest_rank"].eq(locked["raw_residual_rank"]).all():
        raise RuntimeError("Adjusted V1 changed a locked raw rank 1-4.")
    fifth = result[result["contest_rank"].eq(5)].iloc[0]
    if int(fifth["raw_residual_rank"]) > ADJUSTED_V1_CANDIDATE_RANK_MAX:
        raise RuntimeError("Adjusted V1 promoted a pick from outside ranks 5-8.")
    if float(fifth["residual_deficit_vs_raw_fifth"]) > (
        ADJUSTED_V1_TIEBREAK_TOLERANCE + 1e-12
    ):
        raise RuntimeError("Adjusted V1 exceeded the 0.50 residual gate.")
    return result


def apply_late_home_favorite_veto(
    raw_ceiling_card: pd.DataFrame,
    week: int,
) -> pd.DataFrame:
    """Exclude late selected home favorites >= 7.5 before Ceiling ranking."""
    result = raw_ceiling_card.copy()
    result["raw_ceiling_residual_rank"] = numeric(
        result, "contest_rank"
    ).astype(int)
    result["late_home_favorite_veto_candidate"] = (
        int(week) >= 10
        and False
    )
    if int(week) >= 10:
        result["late_home_favorite_veto_candidate"] = (
            result["selected_side"].eq("HOME")
            & numeric(result, "circa_home_margin").ge(
                LATE_HOME_FAVORITE_VETO_MIN_MARGIN - 1e-12
            )
        )
        eligible = result[
            ~result["late_home_favorite_veto_candidate"]
        ].sort_values("raw_ceiling_residual_rank")
        vetoed = result[
            result["late_home_favorite_veto_candidate"]
        ].sort_values("raw_ceiling_residual_rank")
        if len(eligible) < 5:
            raise RuntimeError(
                "The late home-favorite veto left fewer than five eligible games."
            )
        result = pd.concat([eligible, vetoed], ignore_index=True)
        result["contest_rank"] = np.arange(1, len(result) + 1)
    else:
        result["late_home_favorite_veto_candidate"] = False

    result["late_home_favorite_veto_candidate"] = result[
        "late_home_favorite_veto_candidate"
    ].astype(int)
    result["card_status"] = np.where(
        result["late_home_favorite_veto_candidate"].eq(1),
        "VETOED_LATE_HOME_FAVORITE_7P5",
        np.where(
            result["contest_rank"].le(5),
            "CEILING_TOP5",
            np.where(
                result["contest_rank"].le(10),
                "CEILING_ALTERNATE_6_10",
                "UNRANKED",
            ),
        ),
    )
    result["production_selection_policy"] = CEILING_POLICY
    result["home_favorite_veto_min_margin"] = (
        LATE_HOME_FAVORITE_VETO_MIN_MARGIN
    )
    result["home_favorite_veto_applied"] = int(int(week) >= 10)
    result["promoted_by_home_favorite_veto"] = (
        result["contest_rank"].le(5)
        & result["raw_ceiling_residual_rank"].gt(5)
    ).astype(int)
    prohibited_top5 = result[
        result["contest_rank"].le(5)
        & result["late_home_favorite_veto_candidate"].eq(1)
    ]
    if not prohibited_top5.empty:
        raise RuntimeError("A prohibited late home favorite survived in the top five.")
    if int(week) <= 9 and not result["contest_rank"].eq(
        result["raw_ceiling_residual_rank"]
    ).all():
        raise RuntimeError("The late-only veto changed an early Ceiling rank.")
    return result


def predict_raw_v1_card(
    matrix: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    feature_order = [str(feature) for feature in bundle["feature_order"]]
    output = matrix.copy()
    output["predicted_circa_residual"] = bundle["model"].predict(
        output[feature_order]
    )
    return add_selection_fields(
        output,
        residual_column="predicted_circa_residual",
        absolute_column="absolute_predicted_circa_residual",
        adjusted_margin_column="circa_adjusted_home_margin",
        top5_label="OFFICIAL_TOP5",
        alternate_label="ALTERNATE_6_10",
    )


def predict_adjusted_v1_card(
    matrix: pd.DataFrame,
    bundle: dict[str, Any],
    margin_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_card = predict_raw_v1_card(matrix, bundle)
    adjusted_card = apply_adjusted_v1_policy(raw_card, margin_history)
    return adjusted_card, raw_card


def predict_ceiling_card(
    matrix: pd.DataFrame,
    v1_bundle: dict[str, Any],
    ceiling_bundle: dict[str, Any],
    week: int,
) -> pd.DataFrame:
    feature_order = [str(feature) for feature in v1_bundle["feature_order"]]
    output = matrix.copy()
    output["v1_predicted_circa_residual"] = v1_bundle[
        "model"
    ].predict(output[feature_order])
    output["late_model_predicted_circa_residual"] = ceiling_bundle[
        "late_model"
    ].predict(output[feature_order])

    if int(week) <= 9:
        output["ceiling_predicted_circa_residual"] = output[
            "v1_predicted_circa_residual"
        ]
        output["ceiling_phase"] = "WEEKS_1_9_EXACT_V1"
    else:
        weight = float(ceiling_bundle["blend_weight"])
        output["ceiling_predicted_circa_residual"] = (
            (1.0 - weight)
            * output["v1_predicted_circa_residual"]
            + weight
            * output["late_model_predicted_circa_residual"]
        )
        output["ceiling_phase"] = "WEEKS_10_18_LATE_MODEL"

    raw_result = add_selection_fields(
        output,
        residual_column="ceiling_predicted_circa_residual",
        absolute_column="absolute_ceiling_predicted_circa_residual",
        adjusted_margin_column="ceiling_adjusted_home_margin",
        top5_label="CEILING_TOP5",
        alternate_label="CEILING_ALTERNATE_6_10",
    )
    result = apply_late_home_favorite_veto(raw_result, week)
    result["ceiling_late_alpha"] = float(
        ceiling_bundle["late_alpha"]
    )
    result["ceiling_blend_weight"] = float(
        ceiling_bundle["blend_weight"]
    )
    result["ceiling_direction_policy"] = str(
        ceiling_bundle["direction_policy"]
    )
    return result


def assert_early_ceiling_identity(
    v1_card: pd.DataFrame,
    ceiling_card: pd.DataFrame,
    week: int,
) -> None:
    if int(week) > 9:
        return

    columns = [
        "game_id",
        "selected_side",
        "selected_team",
        "selected_circa_spread",
        "contest_rank",
    ]
    left = v1_card[columns].sort_values("game_id").reset_index(drop=True)
    right = (
        ceiling_card[columns]
        .sort_values("game_id")
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise RuntimeError(
            "Ceiling card violated the exact Weeks 1-9 V1 identity guard."
        )


def build_card_comparison(
    v1_card: pd.DataFrame,
    ceiling_card: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "circa_home_margin",
    ]
    v1 = v1_card[
        keys
        + [
            "selected_team",
            "selected_side",
            "selected_circa_spread",
            "predicted_circa_residual",
            "absolute_predicted_circa_residual",
            "circa_adjusted_home_margin",
            "contest_rank",
            "card_status",
            "raw_residual_rank",
            "promoted_by_key",
            "adjusted_v1_key_tiebreak_score",
            "residual_deficit_vs_raw_fifth",
            "production_selection_policy",
        ]
    ].rename(
        columns={
            "selected_team": "v1_selected_team",
            "selected_side": "v1_selected_side",
            "selected_circa_spread": "v1_selected_circa_spread",
            "predicted_circa_residual": "v1_predicted_circa_residual",
            "absolute_predicted_circa_residual": (
                "v1_absolute_predicted_circa_residual"
            ),
            "circa_adjusted_home_margin": (
                "v1_circa_adjusted_home_margin"
            ),
            "contest_rank": "v1_contest_rank",
            "card_status": "v1_card_status",
            "production_selection_policy": "v1_selection_policy",
        }
    )
    ceiling = ceiling_card[
        keys
        + [
            "selected_team",
            "selected_side",
            "selected_circa_spread",
            "late_model_predicted_circa_residual",
            "ceiling_predicted_circa_residual",
            "absolute_ceiling_predicted_circa_residual",
            "ceiling_adjusted_home_margin",
            "contest_rank",
            "card_status",
            "ceiling_phase",
            "ceiling_late_alpha",
            "ceiling_blend_weight",
            "ceiling_direction_policy",
            "raw_ceiling_residual_rank",
            "late_home_favorite_veto_candidate",
            "promoted_by_home_favorite_veto",
            "home_favorite_veto_min_margin",
            "production_selection_policy",
        ]
    ].rename(
        columns={
            "selected_team": "ceiling_selected_team",
            "selected_side": "ceiling_selected_side",
            "selected_circa_spread": (
                "ceiling_selected_circa_spread"
            ),
            "contest_rank": "ceiling_contest_rank",
            "card_status": "ceiling_card_status",
            "production_selection_policy": "ceiling_selection_policy",
        }
    )

    comparison = v1.merge(
        ceiling,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    comparison["selected_side_changed"] = (
        comparison["v1_selected_side"]
        != comparison["ceiling_selected_side"]
    ).astype(int)
    comparison["selected_team_changed"] = (
        comparison["v1_selected_team"]
        != comparison["ceiling_selected_team"]
    ).astype(int)
    comparison["v1_top5"] = comparison[
        "v1_contest_rank"
    ].le(5).astype(int)
    comparison["ceiling_top5"] = comparison[
        "ceiling_contest_rank"
    ].le(5).astype(int)
    comparison["top5_membership"] = np.select(
        [
            comparison["v1_top5"].eq(1)
            & comparison["ceiling_top5"].eq(1),
            comparison["v1_top5"].eq(1)
            & comparison["ceiling_top5"].eq(0),
            comparison["v1_top5"].eq(0)
            & comparison["ceiling_top5"].eq(1),
        ],
        [
            "BOTH_TOP5",
            "V1_ONLY_TOP5",
            "CEILING_ONLY_TOP5",
        ],
        default="NEITHER_TOP5",
    )
    comparison["rank_change_ceiling_minus_v1"] = (
        comparison["ceiling_contest_rank"]
        - comparison["v1_contest_rank"]
    )
    comparison["comparison_sort_rank"] = comparison[
        ["v1_contest_rank", "ceiling_contest_rank"]
    ].min(axis=1)
    return comparison.sort_values(
        [
            "comparison_sort_rank",
            "v1_contest_rank",
            "ceiling_contest_rank",
            "game_id",
        ],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_final_portfolio(
    adjusted_v1_card: pd.DataFrame,
    ceiling_card: pd.DataFrame,
) -> pd.DataFrame:
    v1 = adjusted_v1_card[adjusted_v1_card["contest_rank"].le(5)].copy()
    v1["entry_number"] = 1
    v1["entry_strategy"] = "ADJUSTED_V1_P37_D050"
    v1["portfolio_predicted_circa_residual"] = v1[
        "predicted_circa_residual"
    ]
    v1["portfolio_absolute_residual"] = v1[
        "absolute_predicted_circa_residual"
    ]
    v1["source_raw_residual_rank"] = v1["raw_residual_rank"]
    v1["selection_adjustment_applied"] = v1["promoted_by_key"]

    ceiling = ceiling_card[ceiling_card["contest_rank"].le(5)].copy()
    ceiling["entry_number"] = 2
    ceiling["entry_strategy"] = "CEILING_LATE_HOME_FAVORITE_7P5_VETO"
    ceiling["portfolio_predicted_circa_residual"] = ceiling[
        "ceiling_predicted_circa_residual"
    ]
    ceiling["portfolio_absolute_residual"] = ceiling[
        "absolute_ceiling_predicted_circa_residual"
    ]
    ceiling["source_raw_residual_rank"] = ceiling[
        "raw_ceiling_residual_rank"
    ]
    ceiling["selection_adjustment_applied"] = ceiling[
        "promoted_by_home_favorite_veto"
    ]

    columns = [
        "run_id",
        "created_at",
        "season",
        "week",
        "entry_number",
        "entry_strategy",
        "contest_rank",
        "game_id",
        "away_team",
        "home_team",
        "circa_home_margin",
        "selected_team",
        "selected_side",
        "selected_circa_spread",
        "portfolio_predicted_circa_residual",
        "portfolio_absolute_residual",
        "source_raw_residual_rank",
        "selection_adjustment_applied",
        "production_selection_policy",
        "card_status",
        "feature_source",
        "recommended_sportsbook_stake",
        "contest_only_model",
    ]
    for frame in (v1, ceiling):
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
    portfolio = pd.concat(
        [v1[columns], ceiling[columns]],
        ignore_index=True,
    ).sort_values(["entry_number", "contest_rank"])
    counts = portfolio.groupby("entry_number").size().to_dict()
    if counts != {1: 5, 2: 5}:
        raise RuntimeError(
            f"Final production portfolio must contain five picks per entry: {counts}"
        )
    return portfolio.reset_index(drop=True)


def load_historical_rank_analogs(
    project_root: Path,
) -> tuple[pd.DataFrame, str]:
    """Load descriptive production-rank results without using them in score."""
    database = project_root / "backtests" / HISTORICAL_CARD_DATABASE_NAME
    columns = [
        "entry_number",
        "contest_rank",
        "historical_contest_rank_points_rate",
        "historical_contest_rank_sample_size",
        "historical_contest_rank_seasons",
    ]
    if not database.exists():
        return pd.DataFrame(columns=columns), "UNAVAILABLE"
    try:
        with sqlite3.connect(database) as connection:
            for table_name in HISTORICAL_CARD_TABLE_CANDIDATES:
                if not table_exists(connection, table_name):
                    continue
                raw = read_table(connection, table_name)
                required = {
                    "season",
                    "contest_rank",
                    "contest_points",
                    "card_strategy",
                }
                if not required.issubset(raw.columns):
                    continue
                frame = raw.copy()
                frame["season"] = numeric(frame, "season")
                frame["contest_rank"] = numeric(frame, "contest_rank")
                frame["contest_points"] = numeric(frame, "contest_points")
                frame = frame[
                    frame["season"].between(2021, 2025)
                    & frame["contest_rank"].between(1, 5)
                    & frame["contest_points"].notna()
                ].copy()
                frame["entry_number"] = np.select(
                    [
                        frame["card_strategy"].astype(str).eq(
                            ADJUSTED_V1_POLICY
                        ),
                        frame["card_strategy"].astype(str).eq(
                            CEILING_POLICY
                        ),
                    ],
                    [1, 2],
                    default=0,
                )
                frame = frame[frame["entry_number"].gt(0)].copy()
                if frame.empty:
                    continue
                # Some research tables can carry repeated copies of a card.
                dedupe = [
                    column
                    for column in (
                        "entry_number",
                        "season",
                        "week",
                        "game_id",
                        "contest_rank",
                    )
                    if column in frame.columns
                ]
                frame = frame.drop_duplicates(dedupe, keep="first")
                analog = (
                    frame.groupby(["entry_number", "contest_rank"], as_index=False)
                    .agg(
                        historical_contest_rank_points_rate=(
                            "contest_points",
                            "mean",
                        ),
                        historical_contest_rank_sample_size=(
                            "contest_points",
                            "size",
                        ),
                        historical_contest_rank_seasons=("season", "nunique"),
                    )
                )
                analog[["entry_number", "contest_rank"]] = analog[
                    ["entry_number", "contest_rank"]
                ].astype(int)
                return analog[columns], f"{database}::{table_name}"
    except (OSError, sqlite3.Error, ValueError, KeyError):
        pass
    return pd.DataFrame(columns=columns), "UNAVAILABLE"


def confidence_tier(score: float) -> str:
    if score >= 80.0:
        return "A_STRONG_HOLD"
    if score >= 65.0:
        return "B_HOLD"
    if score >= 50.0:
        return "C_REVIEW"
    return "D_FIRST_REPLACEMENT"


def boundary_component(gap: float) -> float:
    clipped = float(np.clip(gap / 0.50, -20.0, 20.0))
    return float(100.0 / (1.0 + math.exp(-clipped)))


def agreement_component(
    selected_direction: float,
    selected_signal: float,
    other_signal: float,
) -> tuple[float, float, int]:
    selected_support = float(selected_direction * selected_signal)
    other_support = float(selected_direction * other_signal)
    denominator = max(abs(selected_support), 0.50)
    relative_support = min(abs(other_support) / denominator, 1.0)
    if other_support >= 0:
        component = 70.0 + 30.0 * relative_support
        agreement = 1
    else:
        component = 30.0 * (1.0 - relative_support)
        agreement = 0
    return float(np.clip(component, 0.0, 100.0)), other_support, agreement


def build_confidence_board(
    v1_card: pd.DataFrame,
    ceiling_card: pd.DataFrame,
    margin_history: pd.DataFrame,
    historical_analogs: pd.DataFrame,
    historical_analog_source: str,
    structural_projection: pd.DataFrame,
    structural_source: str,
) -> pd.DataFrame:
    """Rank picks by robustness plus a non-substitutive structural review flag."""
    require_columns(
        structural_projection,
        {
            "home_team",
            "away_team",
            "structural_home_margin",
            "model_variant",
        },
        "Structural confidence projection",
    )
    structural_lookup = structural_projection.set_index(
        ["home_team", "away_team"]
    )
    ceiling_keys = ceiling_card.copy()
    ceiling_keys["predicted_circa_residual"] = ceiling_keys[
        "ceiling_predicted_circa_residual"
    ]
    ceiling_keys = add_primary_key_features(ceiling_keys, margin_history)
    ceiling_key_lookup = ceiling_keys.set_index("game_id")
    v1_lookup = v1_card.set_index("game_id")
    ceiling_lookup = ceiling_card.set_index("game_id")

    specifications = (
        {
            "entry_number": 1,
            "entry_strategy": "ADJUSTED_V1_P37_D050",
            "card": v1_card,
            "signal_column": "predicted_circa_residual",
            "absolute_column": "absolute_predicted_circa_residual",
            "raw_rank_column": "raw_residual_rank",
            "adjustment_column": "promoted_by_key",
            "other_lookup": ceiling_lookup,
            "other_signal_column": "ceiling_predicted_circa_residual",
        },
        {
            "entry_number": 2,
            "entry_strategy": "CEILING_LATE_HOME_FAVORITE_7P5_VETO",
            "card": ceiling_card,
            "signal_column": "ceiling_predicted_circa_residual",
            "absolute_column": "absolute_ceiling_predicted_circa_residual",
            "raw_rank_column": "raw_ceiling_residual_rank",
            "adjustment_column": "promoted_by_home_favorite_veto",
            "other_lookup": v1_lookup,
            "other_signal_column": "predicted_circa_residual",
        },
    )
    rows: list[dict[str, Any]] = []
    for specification in specifications:
        card = specification["card"]
        selected = card[card["contest_rank"].le(5)].copy()
        selected = selected.sort_values(
            [specification["absolute_column"], "contest_rank", "game_id"],
            ascending=[False, True, True],
        )
        selected["production_raw_confidence_rank"] = np.arange(
            1, len(selected) + 1, dtype=int
        )
        alternate_rows = card[
            card["contest_rank"].gt(5)
            & ~numeric(card, "late_home_favorite_veto_candidate", 0)
            .fillna(0)
            .eq(1)
        ].sort_values("contest_rank")
        if alternate_rows.empty:
            raise RuntimeError(
                f"Entry {specification['entry_number']} has no eligible alternate."
            )
        alternate = alternate_rows.iloc[0]
        alternate_signal = float(alternate[specification["signal_column"]])
        alternate_absolute = abs(alternate_signal)
        game_count = int(len(card))
        for selected_row in selected.itertuples(index=False):
            row = pd.Series(selected_row._asdict())
            game_id = str(row["game_id"])
            selected_signal = float(row[specification["signal_column"]])
            selected_absolute = float(row[specification["absolute_column"]])
            selected_direction = 1.0 if row["selected_side"] == "HOME" else -1.0
            raw_rank = int(row[specification["raw_rank_column"]])
            adjustment = int(row[specification["adjustment_column"]])
            other_row = specification["other_lookup"].loc[game_id]
            if isinstance(other_row, pd.DataFrame):
                other_row = other_row.iloc[0]
            other_signal = float(other_row[specification["other_signal_column"]])
            agreement_score, other_support, direction_agreement = (
                agreement_component(
                    selected_direction,
                    selected_signal,
                    other_signal,
                )
            )
            other_top5 = int(float(other_row["contest_rank"]) <= 5)
            other_same_pick = int(
                other_top5
                and str(other_row["selected_side"]) == str(row["selected_side"])
            )
            residual_gap = selected_absolute - alternate_absolute
            boundary_score = boundary_component(residual_gap)
            strength_score = float(
                100.0 * (game_count - raw_rank) / max(game_count - 1, 1)
            )

            structural_key = (
                normalize_team(row["home_team"]),
                normalize_team(row["away_team"]),
            )
            try:
                structural_row = structural_lookup.loc[structural_key]
            except KeyError as error:
                raise RuntimeError(
                    "Structural fair line is missing for selected game "
                    f"{game_id}: {row['away_team']} at {row['home_team']}."
                ) from error
            if isinstance(structural_row, pd.DataFrame):
                structural_row = structural_row.iloc[-1]
            structural_home_margin = float(
                structural_row["structural_home_margin"]
            )
            structural_gap_vs_circa = float(
                structural_home_margin - float(row["circa_home_margin"])
            )
            structural_confirmation_points = float(
                selected_direction * structural_gap_vs_circa
            )
            production_raw_confidence_rank = int(
                row["production_raw_confidence_rank"]
            )
            structural_review_eligible = int(
                production_raw_confidence_rank
                >= STRUCTURAL_LOW_CONFIDENCE_RANK_MIN
            )
            structural_review_flag = int(
                structural_review_eligible == 1
                and structural_confirmation_points
                <= -STRUCTURAL_REVIEW_THRESHOLD_POINTS
            )
            if structural_review_flag:
                structural_review_label = (
                    "STRUCTURAL_CONTRADICTION_REVIEW"
                )
            elif not structural_review_eligible:
                structural_review_label = (
                    "TOP_THREE_NOT_STRUCTURAL_REVIEW_ELIGIBLE"
                )
            elif structural_confirmation_points > 0.0:
                structural_review_label = "STRUCTURAL_CONFIRMS_SELECTED_SIDE"
            else:
                structural_review_label = (
                    "NO_STRONG_STRUCTURAL_CONTRADICTION"
                )

            if specification["entry_number"] == 1:
                key_score = float(row["adjusted_v1_key_tiebreak_score"])
                key_advantage = float(
                    row.get("key_score_advantage_vs_raw_fifth", 0.0)
                )
                policy_score = (
                    100.0
                    if adjustment == 0
                    else 55.0
                    + 35.0 * (1.0 - math.exp(-max(key_advantage, 0.0) / 0.05))
                )
                near_veto = 0
            else:
                key_row = ceiling_key_lookup.loc[game_id]
                if isinstance(key_row, pd.DataFrame):
                    key_row = key_row.iloc[0]
                key_score = float(key_row["adjusted_v1_key_tiebreak_score"])
                key_advantage = 0.0
                near_veto = int(
                    int(row["week"]) >= 10
                    and row["selected_side"] == "HOME"
                    and float(row["circa_home_margin"]) >= 6.5
                )
                policy_score = 60.0 if adjustment else 100.0
                if near_veto:
                    policy_score = min(policy_score, 65.0)

            key_component = float(
                100.0 * (1.0 - math.exp(-max(key_score, 0.0) / 0.05))
            )
            confidence_score = float(
                CONFIDENCE_WEIGHTS["boundary_separation"] * boundary_score
                + CONFIDENCE_WEIGHTS["cross_model_agreement"] * agreement_score
                + CONFIDENCE_WEIGHTS["within_week_residual_strength"]
                * strength_score
                + CONFIDENCE_WEIGHTS["primary_key_path_support"] * key_component
                + CONFIDENCE_WEIGHTS["selection_policy_stability"] * policy_score
            )
            fragility: list[str] = []
            if int(row["contest_rank"]) == 5:
                fragility.append("FIFTH_SELECTION_SLOT")
            if residual_gap < 0.50:
                fragility.append("LOW_BOUNDARY_SEPARATION")
            if not direction_agreement:
                fragility.append("MODEL_SIDE_DISAGREEMENT")
            if not other_top5:
                fragility.append("OTHER_ENTRY_NOT_TOP5")
            if specification["entry_number"] == 1 and adjustment:
                fragility.append("KEY_TIEBREAK_PROMOTION")
            if specification["entry_number"] == 2 and adjustment:
                fragility.append("HOME_FAVORITE_VETO_REPLACEMENT")
            if near_veto:
                fragility.append("NEAR_7P5_HOME_FAVORITE_VETO")
            if structural_review_flag:
                fragility.append("STRUCTURAL_HFA_CONTRADICTION_GE_1P5")
            base_tier = confidence_tier(confidence_score)
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "created_at": row.get("created_at", ""),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "entry_number": int(specification["entry_number"]),
                    "entry_strategy": specification["entry_strategy"],
                    "contest_rank": int(row["contest_rank"]),
                    "game_id": game_id,
                    "away_team": row["away_team"],
                    "home_team": row["home_team"],
                    "selected_team": row["selected_team"],
                    "selected_side": row["selected_side"],
                    "selected_circa_spread": float(row["selected_circa_spread"]),
                    "entry_model_predicted_residual": selected_signal,
                    "entry_model_absolute_residual": selected_absolute,
                    "source_raw_residual_rank": raw_rank,
                    "selection_adjustment_applied": adjustment,
                    "production_raw_confidence_rank": (
                        production_raw_confidence_rank
                    ),
                    "confidence_score": round(confidence_score, 2),
                    "base_confidence_score": round(confidence_score, 2),
                    "confidence_tier": base_tier,
                    "base_confidence_tier": base_tier,
                    "boundary_separation_component": round(boundary_score, 2),
                    "cross_model_agreement_component": round(agreement_score, 2),
                    "within_week_residual_strength_component": round(
                        strength_score, 2
                    ),
                    "primary_key_path_support_component": round(
                        key_component, 2
                    ),
                    "selection_policy_stability_component": round(
                        policy_score, 2
                    ),
                    "residual_gap_vs_first_alternate": round(residual_gap, 4),
                    "other_model_support_for_selected_side": round(
                        other_support, 4
                    ),
                    "model_direction_agreement": direction_agreement,
                    "other_entry_contest_rank": int(other_row["contest_rank"]),
                    "other_entry_top5": other_top5,
                    "same_pick_in_both_entries": other_same_pick,
                    "structural_model_variant": str(
                        structural_row["model_variant"]
                    ),
                    "structural_source": structural_source,
                    "structural_run_id": str(
                        structural_row["run_id"]
                    ),
                    "structural_build_id": str(
                        structural_row["build_id"]
                    ),
                    "structural_version": str(
                        structural_row["version"]
                    ),
                    "structural_prediction_as_of_date": str(
                        structural_row["prediction_as_of_date"]
                    ),
                    "structural_prediction_timestamp": str(
                        structural_row["prediction_timestamp"]
                    ),
                    "structural_projection_hash": str(
                        structural_row["independent_projection_hash"]
                    ),
                    "structural_form_build_id": str(
                        structural_row["form_build_id"]
                    ),
                    "structural_form_version": str(
                        structural_row["form_version"]
                    ),
                    "structural_form_through_week": int(
                        structural_row["form_through_week"]
                    ),
                    "structural_form_as_of_date": str(
                        structural_row["form_as_of_date"]
                    ),
                    "structural_form_date_imported": str(
                        structural_row["form_date_imported"]
                    ),
                    "structural_preseason_snapshot_hash": str(
                        structural_row["preseason_snapshot_hash"]
                    ),
                    "structural_power_build_id": str(
                        structural_row["structural_power_build_id"]
                    ),
                    "structural_power_version": str(
                        structural_row["structural_power_version"]
                    ),
                    "structural_power_date_imported": str(
                        structural_row["structural_power_date_imported"]
                    ),
                    "structural_power_snapshot_hash": str(
                        structural_row[
                            "structural_power_snapshot_hash"
                        ]
                    ),
                    "structural_home_margin": round(
                        structural_home_margin, 4
                    ),
                    "structural_gap_vs_circa": round(
                        structural_gap_vs_circa, 4
                    ),
                    "structural_confirmation_points": round(
                        structural_confirmation_points, 4
                    ),
                    "structural_agrees_with_selected_side": int(
                        structural_confirmation_points > 0.0
                    ),
                    "structural_contradicts_selected_side": int(
                        structural_confirmation_points < 0.0
                    ),
                    "structural_review_eligible": (
                        structural_review_eligible
                    ),
                    "structural_review_threshold_points": (
                        STRUCTURAL_REVIEW_THRESHOLD_POINTS
                    ),
                    "structural_review_flag": structural_review_flag,
                    "structural_review_label": structural_review_label,
                    "structural_used_in_confidence_ordering": 1,
                    "structural_used_to_change_model_card": (
                        STRUCTURAL_USED_TO_CHANGE_MODEL_CARD
                    ),
                    "structural_numeric_confidence_penalty": 0.0,
                    "primary_key_path_score": round(key_score, 6),
                    "key_score_advantage_vs_raw_fifth": round(
                        key_advantage, 6
                    ),
                    "fragility_reason": (
                        "|".join(fragility) if fragility else "NONE"
                    ),
                    "manual_review_flag": int(
                        confidence_score < 65.0
                        or adjustment == 1
                        or direction_agreement == 0
                        or int(row["contest_rank"]) == 5
                        or structural_review_flag == 1
                    ),
                    "first_alternate_game_id": alternate["game_id"],
                    "first_alternate_team": alternate["selected_team"],
                    "first_alternate_side": alternate["selected_side"],
                    "first_alternate_circa_spread": float(
                        alternate["selected_circa_spread"]
                    ),
                    "first_alternate_entry_model_residual": alternate_signal,
                    "first_alternate_source_raw_rank": int(
                        alternate[specification["raw_rank_column"]]
                    ),
                    "confidence_methodology": CONFIDENCE_METHODOLOGY,
                    "confidence_is_cover_probability": 0,
                    "model_card_locked": 1,
                    "historical_analog_source": historical_analog_source,
                }
            )

    board = pd.DataFrame(rows)
    board["base_confidence_rank"] = (
        board.sort_values(
            ["entry_number", "confidence_score", "contest_rank"],
            ascending=[True, False, True],
        )
        .groupby("entry_number")
        .cumcount()
        .add(1)
        .reindex(board.index)
        .astype(int)
    )
    board["confidence_rank"] = (
        board.sort_values(
            [
                "entry_number",
                "structural_review_flag",
                "confidence_score",
                "contest_rank",
            ],
            ascending=[True, True, False, True],
        )
        .groupby("entry_number")
        .cumcount()
        .add(1)
        .reindex(board.index)
        .astype(int)
    )
    board["manual_review_priority"] = (
        board.sort_values(
            [
                "entry_number",
                "structural_review_flag",
                "confidence_score",
                "contest_rank",
            ],
            ascending=[True, False, True, False],
        )
        .groupby("entry_number")
        .cumcount()
        .add(1)
        .reindex(board.index)
        .astype(int)
    )
    structural_review = board["structural_review_flag"].eq(1)
    board.loc[
        structural_review & ~board["confidence_tier"].eq("D_FIRST_REPLACEMENT"),
        "confidence_tier",
    ] = "C_STRUCTURAL_REVIEW"
    if not historical_analogs.empty:
        board = board.merge(
            historical_analogs,
            on=["entry_number", "contest_rank"],
            how="left",
            validate="many_to_one",
        )
    else:
        board["historical_contest_rank_points_rate"] = np.nan
        board["historical_contest_rank_sample_size"] = 0
        board["historical_contest_rank_seasons"] = 0
    board["historical_results_used_in_confidence_score"] = 0
    counts = board.groupby("entry_number").size().to_dict()
    if counts != {1: 5, 2: 5}:
        raise RuntimeError(f"Confidence board must contain five picks per entry: {counts}")
    if not board["confidence_score"].between(0.0, 100.0).all():
        raise RuntimeError("Confidence scores must remain between 0 and 100.")
    if not board.groupby("entry_number")["confidence_rank"].apply(
        lambda values: set(values) == set(range(1, 6))
    ).all():
        raise RuntimeError("Confidence ranks must be unique from 1 through 5.")
    if (
        board["structural_review_flag"].eq(1)
        & board["production_raw_confidence_rank"].lt(
            STRUCTURAL_LOW_CONFIDENCE_RANK_MIN
        )
    ).any():
        raise RuntimeError(
            "Structural review escaped the bottom-two production gate."
        )
    if not board["structural_used_to_change_model_card"].eq(0).all():
        raise RuntimeError("Structural review was allowed to alter a model card.")
    return board.sort_values(
        ["entry_number", "confidence_rank"]
    ).reset_index(drop=True)


def add_confidence_to_portfolio(
    portfolio: pd.DataFrame,
    confidence_board: pd.DataFrame,
) -> pd.DataFrame:
    fields = [
        "run_id",
        "entry_number",
        "game_id",
        "production_raw_confidence_rank",
        "base_confidence_rank",
        "confidence_rank",
        "confidence_score",
        "base_confidence_score",
        "confidence_tier",
        "base_confidence_tier",
        "manual_review_priority",
        "manual_review_flag",
        "fragility_reason",
        "first_alternate_team",
        "first_alternate_circa_spread",
        "residual_gap_vs_first_alternate",
        "model_direction_agreement",
        "same_pick_in_both_entries",
        "structural_model_variant",
        "structural_source",
        "structural_run_id",
        "structural_build_id",
        "structural_version",
        "structural_prediction_as_of_date",
        "structural_prediction_timestamp",
        "structural_projection_hash",
        "structural_form_build_id",
        "structural_form_version",
        "structural_form_through_week",
        "structural_form_as_of_date",
        "structural_form_date_imported",
        "structural_preseason_snapshot_hash",
        "structural_power_build_id",
        "structural_power_version",
        "structural_power_date_imported",
        "structural_power_snapshot_hash",
        "structural_home_margin",
        "structural_gap_vs_circa",
        "structural_confirmation_points",
        "structural_agrees_with_selected_side",
        "structural_contradicts_selected_side",
        "structural_review_eligible",
        "structural_review_threshold_points",
        "structural_review_flag",
        "structural_review_label",
        "structural_used_in_confidence_ordering",
        "structural_used_to_change_model_card",
        "structural_numeric_confidence_penalty",
        "historical_contest_rank_points_rate",
        "historical_contest_rank_sample_size",
        "confidence_methodology",
        "confidence_is_cover_probability",
        "model_card_locked",
    ]
    return portfolio.merge(
        confidence_board[fields],
        on=["run_id", "entry_number", "game_id"],
        how="left",
        validate="one_to_one",
    ).sort_values(["entry_number", "contest_rank"]).reset_index(drop=True)


def build_manual_submission_draft(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    draft = portfolio[
        [
            "run_id",
            "created_at",
            "season",
            "week",
            "entry_number",
            "entry_strategy",
            "contest_rank",
            "game_id",
            "away_team",
            "home_team",
            "selected_team",
            "selected_side",
            "selected_circa_spread",
            "production_raw_confidence_rank",
            "base_confidence_rank",
            "confidence_rank",
            "confidence_score",
            "base_confidence_score",
            "confidence_tier",
            "base_confidence_tier",
            "manual_review_priority",
            "manual_review_flag",
            "fragility_reason",
            "structural_model_variant",
            "structural_home_margin",
            "structural_gap_vs_circa",
            "structural_confirmation_points",
            "structural_review_eligible",
            "structural_review_threshold_points",
            "structural_review_flag",
            "structural_review_label",
            "structural_used_to_change_model_card",
            "structural_numeric_confidence_penalty",
            "first_alternate_team",
            "first_alternate_circa_spread",
        ]
    ].copy()
    draft = draft.rename(
        columns={
            "selected_team": "model_selected_team",
            "selected_side": "model_selected_side",
            "selected_circa_spread": "model_selected_circa_spread",
        }
    )
    draft["submission_selected_team"] = draft["model_selected_team"]
    draft["submission_selected_side"] = draft["model_selected_side"]
    draft["submission_selected_circa_spread"] = draft[
        "model_selected_circa_spread"
    ]
    draft["manual_override"] = 0
    draft["replacement_team"] = ""
    draft["replacement_game_id"] = ""
    draft["override_reason"] = ""
    draft["new_information_type"] = ""
    draft["new_information_allowed_values"] = (
        "INJURY|QB_STATUS|WEATHER|LINE_CHANGE|FIELD_POPULARITY|OTHER"
    )
    draft["information_timestamp"] = ""
    draft["estimated_field_popularity"] = np.nan
    draft["manual_notes"] = ""
    draft["edited_at"] = ""
    draft["submission_status"] = "UNEDITED_MODEL_DEFAULT"
    draft["original_model_card_immutable"] = 1
    return draft


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
        frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
        )
        return

    escaped = table_name.replace('"', '""')
    existing = {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{escaped}")'
        ).fetchall()
    }
    for column in frame.columns:
        if column in existing:
            continue
        column_escaped = str(column).replace('"', '""')
        sql_type = sqlite_type_for_series(frame[column])
        connection.execute(
            f'ALTER TABLE "{escaped}" '
            f'ADD COLUMN "{column_escaped}" {sql_type}'
        )
    connection.commit()
    frame.to_sql(
        table_name,
        connection,
        if_exists="append",
        index=False,
    )

def save_outputs(
    args: argparse.Namespace,
    v1_card: pd.DataFrame,
    ceiling_card: pd.DataFrame,
    comparison: pd.DataFrame,
    portfolio: pd.DataFrame,
    confidence_board: pd.DataFrame,
    manual_submission_draft: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[
    Path,
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Optional[Path],
    Optional[Path],
]:
    v1_output_columns = [
        "run_id",
        "created_at",
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "circa_home_margin",
        "selected_team",
        "selected_side",
        "selected_circa_spread",
        "predicted_circa_residual",
        "absolute_predicted_circa_residual",
        "circa_adjusted_home_margin",
        "contest_rank",
        "card_status",
        "raw_residual_rank",
        "production_selection_policy",
        "adjusted_v1_tiebreak_tolerance",
        "adjusted_v1_candidate_rank_max",
        "adjusted_v1_candidate_pool_size",
        "promoted_by_key",
        "adjusted_v1_crossed_key_mass_3",
        "adjusted_v1_crossed_key_mass_7",
        "adjusted_v1_market_push_mass",
        "adjusted_v1_key_tiebreak_score",
        "adjusted_v1_crossed_integer_margins",
        "residual_deficit_vs_raw_fifth",
        "key_score_advantage_vs_raw_fifth",
        "pmf_market_bandwidth",
        "pmf_prior_strength",
        "pmf_effective_sample_size",
        "pmf_prior_seasons",
        "pmf_prior_games",
        "recommended_sportsbook_stake",
        "contest_only_model",
        "feature_source",
        "model_path",
        "model_build_id",
        "model_version",
        "model_feature_set",
        "model_alpha",
        "rating_alpha",
    ]
    v1_output = v1_card[
        [
            column
            for column in v1_output_columns
            if column in v1_card.columns
        ]
    ]

    ceiling_output_columns = [
        "run_id",
        "created_at",
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "circa_home_margin",
        "selected_team",
        "selected_side",
        "selected_circa_spread",
        "v1_predicted_circa_residual",
        "late_model_predicted_circa_residual",
        "ceiling_predicted_circa_residual",
        "absolute_ceiling_predicted_circa_residual",
        "ceiling_adjusted_home_margin",
        "contest_rank",
        "card_status",
        "ceiling_phase",
        "ceiling_late_alpha",
        "ceiling_blend_weight",
        "ceiling_direction_policy",
        "raw_ceiling_residual_rank",
        "production_selection_policy",
        "late_home_favorite_veto_candidate",
        "home_favorite_veto_min_margin",
        "home_favorite_veto_applied",
        "promoted_by_home_favorite_veto",
        "recommended_sportsbook_stake",
        "contest_only_model",
        "feature_source",
        "v1_model_path",
        "ceiling_model_path",
        "ceiling_model_build_id",
        "ceiling_model_version",
        "rating_alpha",
    ]
    ceiling_output = ceiling_card[
        [
            column
            for column in ceiling_output_columns
            if column in ceiling_card.columns
        ]
    ]

    comparison_output_columns = [
        "run_id",
        "created_at",
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "circa_home_margin",
        "v1_selected_team",
        "v1_selected_side",
        "v1_selected_circa_spread",
        "v1_predicted_circa_residual",
        "v1_absolute_predicted_circa_residual",
        "v1_contest_rank",
        "v1_card_status",
        "raw_residual_rank",
        "promoted_by_key",
        "adjusted_v1_key_tiebreak_score",
        "residual_deficit_vs_raw_fifth",
        "v1_selection_policy",
        "ceiling_selected_team",
        "ceiling_selected_side",
        "ceiling_selected_circa_spread",
        "late_model_predicted_circa_residual",
        "ceiling_predicted_circa_residual",
        "absolute_ceiling_predicted_circa_residual",
        "ceiling_contest_rank",
        "ceiling_card_status",
        "ceiling_phase",
        "ceiling_late_alpha",
        "ceiling_blend_weight",
        "ceiling_direction_policy",
        "raw_ceiling_residual_rank",
        "late_home_favorite_veto_candidate",
        "promoted_by_home_favorite_veto",
        "home_favorite_veto_min_margin",
        "ceiling_selection_policy",
        "selected_side_changed",
        "selected_team_changed",
        "v1_top5",
        "ceiling_top5",
        "top5_membership",
        "rank_change_ceiling_minus_v1",
        "feature_source",
        "v1_model_path",
        "ceiling_model_path",
    ]
    comparison_output = comparison[
        [
            column
            for column in comparison_output_columns
            if column in comparison.columns
        ]
    ]

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db_path) as connection:
        v1_output.to_sql(
            OUTPUT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            OUTPUT_HISTORY_TABLE,
            v1_output,
        )

        ceiling_output.to_sql(
            CEILING_OUTPUT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            CEILING_OUTPUT_HISTORY_TABLE,
            ceiling_output,
        )

        comparison_output.to_sql(
            COMPARISON_OUTPUT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            COMPARISON_OUTPUT_HISTORY_TABLE,
            comparison_output,
        )

        portfolio.to_sql(
            PORTFOLIO_OUTPUT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            PORTFOLIO_OUTPUT_HISTORY_TABLE,
            portfolio,
        )

        confidence_board.to_sql(
            CONFIDENCE_OUTPUT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            CONFIDENCE_OUTPUT_HISTORY_TABLE,
            confidence_board,
        )

        append_with_schema_evolution(
            connection,
            MANUAL_SUBMISSION_DRAFT_TABLE,
            manual_submission_draft,
        )

        audit.to_sql(
            RUN_AUDIT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(
            connection,
            RUN_AUDIT_HISTORY_TABLE,
            audit,
        )
    # sqlite3.Connection.__exit__ commits or rolls back but does not close the
    # handle.  Close it explicitly so Windows can release temporary/test DBs.
    connection.close()

    v1_csv_path = None
    ceiling_csv_path = None
    comparison_csv_path = None
    portfolio_csv_path = None
    confidence_csv_path = None
    manual_submission_csv_path = None
    if not args.no_csv:
        output_directory = args.project_root / "outputs" / "circa_2026"
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        v1_csv_path = (
            output_directory
            / f"nfl_circa_top5_week_{int(args.week):02d}_{timestamp}.csv"
        )
        ceiling_csv_path = (
            output_directory
            / f"nfl_circa_ceiling_week_{int(args.week):02d}_{timestamp}.csv"
        )
        comparison_csv_path = (
            output_directory
            / (
                "nfl_circa_v1_vs_ceiling_week_"
                f"{int(args.week):02d}_{timestamp}.csv"
            )
        )
        portfolio_csv_path = (
            output_directory
            / (
                "nfl_circa_final_portfolio_week_"
                f"{int(args.week):02d}_{timestamp}.csv"
            )
        )
        confidence_csv_path = (
            output_directory
            / f"nfl_circa_confidence_week_{int(args.week):02d}_{timestamp}.csv"
        )
        manual_submission_csv_path = (
            output_directory
            / (
                "nfl_circa_manual_submission_week_"
                f"{int(args.week):02d}_{timestamp}.csv"
            )
        )
        v1_output.to_csv(
            v1_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        ceiling_output.to_csv(
            ceiling_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        comparison_output.to_csv(
            comparison_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        portfolio.to_csv(
            portfolio_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        confidence_board.to_csv(
            confidence_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        manual_submission_draft.to_csv(
            manual_submission_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

    return (
        args.db_path,
        v1_csv_path,
        ceiling_csv_path,
        comparison_csv_path,
        portfolio_csv_path,
        confidence_csv_path,
        manual_submission_csv_path,
    )

def run_self_test() -> int:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features = [
        "pass_epa_advantage",
        "market_home_margin",
        "absolute_market_home_margin",
    ]
    training = pd.DataFrame(
        {
            "pass_epa_advantage": [-3, -2, -1, 0, 1, 2, 3, 4],
            "market_home_margin": [0, 1, -1, 2, -2, 3, -3, 4],
            "absolute_market_home_margin": [0, 1, 1, 2, 2, 3, 3, 4],
        }
    )
    v1_target = np.array(
        [-3, -2, -1, 0, 1, 2, 3, 4],
        dtype=float,
    )
    late_target = np.array(
        [4, 3, 2, 1, 0, -1, -2, -3],
        dtype=float,
    )

    def fit_model(target: np.ndarray) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        ).fit(training, target)

    v1_bundle = {
        "model": fit_model(v1_target),
        "feature_order": features,
        "rating_alpha": 10.0,
        "feature_set": "TEST",
        "model_alpha": 1.0,
        "selection_policy": (
            "TOP5_WEEKLY_BY_ABSOLUTE_PREDICTED_CIRCA_RESIDUAL"
        ),
        "implementation_ready": True,
    }
    ceiling_bundle = {
        "late_model": fit_model(late_target),
        "late_alpha": 1000.0,
        "blend_weight": 1.0,
        "direction_policy": "BLENDED_SIGN",
        "build_id": "TEST_CEILING",
        "version": "TEST",
        "feature_set": "TEST",
        "rating_alpha": 10.0,
    }
    matrix = pd.DataFrame(
        {
            "season": [2026] * 8,
            "week": [10] * 8,
            "game_id": [f"G{i}" for i in range(8)],
            "away_team": [f"A{i}" for i in range(8)],
            "home_team": [f"H{i}" for i in range(8)],
            "circa_home_margin": [0.0] * 8,
            "pass_epa_advantage": [-4, -3, -2, -1, 1, 2, 3, 4],
            "market_home_margin": [0.0] * 8,
            "absolute_market_home_margin": [0.0] * 8,
        }
    )

    history_rows: list[dict[str, Any]] = []
    for season in range(2020, 2026):
        for index in range(40):
            market = 2.0 if index < 20 else 11.0
            actual = 3.0 if index < 20 else 12.0
            history_rows.append(
                {
                    "season": season,
                    "circa_home_margin": market,
                    "actual_home_margin": actual,
                    "reference_fallback_used": 0,
                    "strict_circa_line_valid": 1,
                }
            )
    margin_history = prepare_margin_history(pd.DataFrame(history_rows))

    v1_card, raw_v1_card = predict_adjusted_v1_card(
        matrix,
        v1_bundle,
        margin_history,
    )
    ceiling_card = predict_ceiling_card(
        matrix,
        v1_bundle,
        ceiling_bundle,
        week=10,
    )
    for card in (v1_card, ceiling_card):
        card["run_id"] = "SELF_TEST_RUN"
        card["created_at"] = "2026-08-24T00:00:00+00:00"
        card["feature_source"] = "SELF_TEST"
    comparison = build_card_comparison(v1_card, ceiling_card)
    portfolio = build_final_portfolio(v1_card, ceiling_card)
    immutable_model_selections = portfolio[
        ["entry_number", "contest_rank", "game_id", "selected_team"]
    ].copy()
    structural_raw = matrix[
        ["season", "week", "home_team", "away_team"]
    ].copy()
    structural_run_id = "SELF_TEST_STRUCTURAL_RUN"
    structural_as_of_date = pd.Timestamp("2026-08-24")
    structural_raw["run_id"] = structural_run_id
    structural_raw["model_variant"] = STRUCTURAL_MODEL_VARIANT
    structural_raw["build_id"] = EXPECTED_STRUCTURAL_BUILD_ID
    structural_raw["version"] = EXPECTED_STRUCTURAL_VERSION
    structural_raw["prediction_as_of_date"] = "2026-08-24"
    structural_raw["prediction_timestamp"] = "2026-08-24T00:00:00"
    structural_raw["prediction_uses_market_inputs"] = 0
    structural_raw["projected_home_margin"] = 0.0
    structural_raw["independent_projection_hash"] = structural_raw.apply(
        lambda row: hashlib.sha256(
            f"{row['away_team']}@{row['home_team']}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    structural_raw["form_build_id"] = EXPECTED_FORM_BUILD_ID
    structural_raw["form_version"] = EXPECTED_FORM_VERSION
    structural_raw["form_as_of_date"] = "2026-08-24"
    structural_raw["form_through_week"] = 9
    structural_raw["form_date_imported"] = "2026-08-24T00:00:00"
    structural_raw["preseason_snapshot_hash"] = "SELF_TEST_PRESEASON_HASH"
    structural_raw["structural_power_build_id"] = (
        EXPECTED_STRUCTURAL_POWER_BUILD_ID
    )
    structural_raw["structural_power_version"] = "SELF_TEST_POWER_VERSION"
    structural_raw["structural_power_date_imported"] = (
        "2026-08-24T00:00:00"
    )
    structural_raw["structural_power_snapshot_hash"] = (
        "SELF_TEST_STRUCTURAL_POWER_HASH"
    )
    v1_low_confidence = (
        v1_card[v1_card["contest_rank"].le(5)]
        .sort_values(
            ["absolute_predicted_circa_residual", "contest_rank", "game_id"],
            ascending=[False, True, True],
        )
        .tail(2)
    )
    for low_row in v1_low_confidence.itertuples(index=False):
        selected_direction = 1.0 if low_row.selected_side == "HOME" else -1.0
        mask = (
            structural_raw["home_team"].eq(low_row.home_team)
            & structural_raw["away_team"].eq(low_row.away_team)
        )
        structural_raw.loc[mask, "projected_home_margin"] = (
            -2.0 * selected_direction
        )
    structural_run_audit = pd.DataFrame(
        [
            {
                "run_id": structural_run_id,
                "season": SEASON,
                "week": 10,
                "build_id": EXPECTED_STRUCTURAL_BUILD_ID,
                "version": EXPECTED_STRUCTURAL_VERSION,
                "model_variant": STRUCTURAL_MODEL_VARIANT,
                "status": "SUCCESS",
                "games": len(structural_raw),
                "prediction_uses_market_inputs": 0,
                "form_build_id": EXPECTED_FORM_BUILD_ID,
                "form_version": EXPECTED_FORM_VERSION,
                "form_as_of_date": "2026-08-24",
                "form_through_week": 9,
                "form_date_imported": "2026-08-24T00:00:00",
                "preseason_snapshot_hash": "SELF_TEST_PRESEASON_HASH",
                "structural_power_build_id": (
                    EXPECTED_STRUCTURAL_POWER_BUILD_ID
                ),
                "structural_power_version": "SELF_TEST_POWER_VERSION",
                "structural_power_date_imported": (
                    "2026-08-24T00:00:00"
                ),
                "structural_power_snapshot_hash": (
                    "SELF_TEST_STRUCTURAL_POWER_HASH"
                ),
                "completed_at": "2026-08-24T00:00:01",
            }
        ]
    )
    structural_projection = prepare_structural_confidence_projection(
        structural_raw,
        structural_run_audit,
        week=10,
        as_of_date=structural_as_of_date,
        expected_games=matrix,
    )
    confidence_board = build_confidence_board(
        v1_card,
        ceiling_card,
        margin_history,
        pd.DataFrame(),
        "UNAVAILABLE",
        structural_projection,
        "SELF_TEST_STRUCTURAL_SOURCE",
    )
    portfolio = add_confidence_to_portfolio(portfolio, confidence_board)
    manual_draft = build_manual_submission_draft(portfolio)

    if v1_card["card_status"].eq("OFFICIAL_TOP5").sum() != 5:
        raise AssertionError("Adjusted V1 top-five selection failed.")
    if ceiling_card["card_status"].eq("CEILING_TOP5").sum() != 5:
        raise AssertionError("Ceiling top-five selection failed.")
    if len(comparison) != len(matrix):
        raise AssertionError("V1/ceiling comparison merge failed.")
    if len(portfolio) != 10:
        raise AssertionError("Final two-entry portfolio did not contain 10 picks.")
    if len(confidence_board) != 10:
        raise AssertionError("Confidence board did not contain 10 picks.")
    if not confidence_board["confidence_score"].between(0, 100).all():
        raise AssertionError("Confidence score escaped the 0-100 range.")
    if not confidence_board["confidence_is_cover_probability"].eq(0).all():
        raise AssertionError("Confidence was incorrectly labeled a cover probability.")
    if not confidence_board.groupby("entry_number")["confidence_rank"].apply(
        lambda values: set(values) == set(range(1, 6))
    ).all():
        raise AssertionError("Confidence ranks are incomplete.")
    if not confidence_board["confidence_score"].eq(
        confidence_board["base_confidence_score"]
    ).all():
        raise AssertionError("Structural review changed a numeric confidence score.")
    if confidence_board["structural_review_flag"].sum() < 1:
        raise AssertionError("Structural confidence fixture produced no review flag.")
    if (
        confidence_board["structural_review_flag"].eq(1)
        & confidence_board["production_raw_confidence_rank"].lt(4)
    ).any():
        raise AssertionError("Structural review escaped the bottom-two gate.")
    if not confidence_board["structural_used_to_change_model_card"].eq(0).all():
        raise AssertionError("Structural review was labeled as a pick changer.")
    persisted_model_selections = portfolio[
        ["entry_number", "contest_rank", "game_id", "selected_team"]
    ]
    if not immutable_model_selections.equals(persisted_model_selections):
        raise AssertionError("Confidence integration changed a production pick.")
    if len(manual_draft) != 10 or not manual_draft[
        "original_model_card_immutable"
    ].eq(1).all():
        raise AssertionError("Manual draft did not preserve the original model card.")
    comparison["run_id"] = "SELF_TEST_RUN"
    comparison["created_at"] = "2026-08-24T00:00:00+00:00"
    audit = pd.DataFrame(
        [
            {
                "run_id": "SELF_TEST_RUN",
                "build_id": BUILD_ID,
                "version": VERSION,
                "confidence_methodology": CONFIDENCE_METHODOLOGY,
            }
        ]
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        test_args = argparse.Namespace(
            project_root=temporary_root,
            db_path=temporary_root / "self_test.sqlite",
            week=10,
            no_csv=False,
        )
        saved = save_outputs(
            test_args,
            v1_card,
            ceiling_card,
            comparison,
            portfolio,
            confidence_board,
            manual_draft,
            audit,
        )
        if not saved[0].exists() or any(path is None for path in saved[1:]):
            raise AssertionError("Confidence output persistence self-test failed.")
        connection = sqlite3.connect(saved[0])
        try:
            required_tables = {
                CONFIDENCE_OUTPUT_TABLE,
                CONFIDENCE_OUTPUT_HISTORY_TABLE,
                MANUAL_SUBMISSION_DRAFT_TABLE,
                PORTFOLIO_OUTPUT_TABLE,
            }
            present = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            persisted_confidence = pd.read_sql_query(
                f'SELECT * FROM "{CONFIDENCE_OUTPUT_TABLE}"',
                connection,
            )
            persisted_portfolio = pd.read_sql_query(
                f'SELECT * FROM "{PORTFOLIO_OUTPUT_TABLE}"',
                connection,
            )
        finally:
            connection.close()
        if not required_tables.issubset(present):
            raise AssertionError("Confidence SQLite tables are incomplete.")
        required_structural_columns = {
            "structural_home_margin",
            "structural_confirmation_points",
            "structural_review_flag",
            "structural_review_label",
            "structural_used_to_change_model_card",
        }
        if not required_structural_columns.issubset(
            persisted_confidence.columns
        ):
            raise AssertionError("Confidence output lost structural fields.")
        if not required_structural_columns.issubset(
            persisted_portfolio.columns
        ):
            raise AssertionError("Portfolio output lost structural fields.")
    if (
        v1_card["recommended_sportsbook_stake"].ne(0).any()
        or ceiling_card["recommended_sportsbook_stake"].ne(0).any()
    ):
        raise AssertionError("Contest-only stake guard failed.")

    early_matrix = matrix.copy()
    early_matrix["week"] = 1
    early_raw_v1 = predict_raw_v1_card(early_matrix, v1_bundle)
    early_ceiling = predict_ceiling_card(
        early_matrix,
        v1_bundle,
        ceiling_bundle,
        week=1,
    )
    assert_early_ceiling_identity(early_raw_v1, early_ceiling, week=1)

    gate_frame = pd.DataFrame(
        {
            "season": [2026] * 8,
            "week": [10] * 8,
            "game_id": [f"K{rank}" for rank in range(1, 9)],
            "away_team": [f"KA{rank}" for rank in range(1, 9)],
            "home_team": [f"KH{rank}" for rank in range(1, 9)],
            "circa_home_margin": [0.0, 0.0, 0.0, 0.0, 11.0, 2.0, 0.0, 0.0],
            "predicted_circa_residual": [4.0, 3.5, 3.0, 2.5, 2.0, 1.8, 1.6, 1.5],
        }
    )
    raw_gate = add_selection_fields(
        gate_frame,
        "predicted_circa_residual",
        "absolute_predicted_circa_residual",
        "circa_adjusted_home_margin",
        "OFFICIAL_TOP5",
        "ALTERNATE_6_10",
    )
    adjusted_gate = apply_adjusted_v1_policy(raw_gate, margin_history)
    gate_fifth = adjusted_gate[adjusted_gate["contest_rank"].eq(5)].iloc[0]
    if gate_fifth["game_id"] != "K6" or int(gate_fifth["promoted_by_key"]) != 1:
        raise AssertionError("The adjusted-V1 primary-key tiebreak did not promote K6.")

    veto_frame = gate_frame.copy()
    veto_frame["game_id"] = [f"V{rank}" for rank in range(1, 9)]
    veto_frame["circa_home_margin"] = [7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]
    veto_frame["ceiling_predicted_circa_residual"] = [
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
    ]
    raw_veto = add_selection_fields(
        veto_frame,
        "ceiling_predicted_circa_residual",
        "absolute_ceiling_predicted_circa_residual",
        "ceiling_adjusted_home_margin",
        "CEILING_TOP5",
        "CEILING_ALTERNATE_6_10",
    )
    veto_card = apply_late_home_favorite_veto(raw_veto, week=10)
    if "V1" in set(veto_card[veto_card["contest_rank"].le(5)]["game_id"]):
        raise AssertionError("The 7.5 late home-favorite veto did not remove V1.")
    if "V6" not in set(veto_card[veto_card["contest_rank"].le(5)]["game_id"]):
        raise AssertionError("The next eligible Ceiling residual did not fill.")

    qb_fixture = pd.DataFrame(
        {
            "week": [2] * 5,
            "personnel_complete": [1] * 5,
            "home_qb_recent_epa": [0.20, 0.10, 0.05, -0.05, -0.10],
            "away_qb_recent_epa": [-0.10, 0.00, 0.10, 0.05, -0.20],
            "home_qb_recent_cpoe": [5.0, 3.0, 1.0, -1.0, -3.0],
            "away_qb_recent_cpoe": [-2.0, 0.0, 2.0, 1.0, -4.0],
        }
    )
    qb_fixture["qb_recent_epa_advantage"] = (
        qb_fixture["home_qb_recent_epa"]
        - qb_fixture["away_qb_recent_epa"]
    )
    qb_fixture["qb_recent_cpoe_advantage"] = (
        qb_fixture["home_qb_recent_cpoe"]
        - qb_fixture["away_qb_recent_cpoe"]
    )
    if assert_live_qb_feature_integrity(qb_fixture)["passed"] != 1:
        raise AssertionError("Live QB feature-integrity positive fixture failed.")
    dead_qb_fixture = qb_fixture.copy()
    for column in (
        "home_qb_recent_epa",
        "away_qb_recent_epa",
        "home_qb_recent_cpoe",
        "away_qb_recent_cpoe",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
    ):
        dead_qb_fixture[column] = 0.0
    try:
        assert_live_qb_feature_integrity(dead_qb_fixture)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Live QB dead-feature fixture was not blocked.")

    print("[CIRCA_TOP5] Final production portfolio self-test passed.")
    return 0

def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    started = dt.datetime.now(dt.timezone.utc)
    run_id = str(uuid.uuid4())

    lines_path = args.circa_lines_path or find_default_lines_path(
        args.project_root,
        int(args.week),
    )
    if lines_path is None:
        raise RuntimeError(
            "No Circa lines file was supplied or found. Create a CSV with "
            "week, away_team, home_team, circa_home_margin and pass "
            "--circa-lines-path."
        )
    if not lines_path.exists():
        raise FileNotFoundError(lines_path)

    lines = prepare_lines(read_frame(lines_path), int(args.week))
    v1_bundle = load_bundle(
        args.model_path,
        args.approve_contest_forward_test,
    )
    ceiling_bundle = load_ceiling_bundle(
        args.ceiling_model_path,
        v1_bundle,
    )
    features, feature_source = load_feature_matrix(
        args,
        lines,
        v1_bundle,
    )
    matrix = merge_features_and_lines(
        features,
        lines,
        v1_bundle,
    )
    qb_live_integrity = dict(matrix.attrs.get("qb_feature_integrity", {}))
    live_feature_lineage = dict(
        matrix.attrs.get("live_feature_lineage", {})
    )
    circa_lines_sha256 = sha256_file(lines_path)
    circa_lines_modified_at = dt.datetime.fromtimestamp(
        lines_path.stat().st_mtime,
        tz=dt.timezone.utc,
    ).isoformat(timespec="seconds")
    margin_history, margin_history_source = load_margin_history(args)
    structural_projection, structural_source = (
        load_structural_confidence_projection(args, lines)
    )
    structural_lineage = {
        column: one_required_text(
            structural_projection,
            column,
            "Validated structural projection",
        )
        for column in (
            "run_id",
            "build_id",
            "version",
            "prediction_as_of_date",
            "prediction_timestamp",
            "form_build_id",
            "form_version",
            "form_as_of_date",
            "form_date_imported",
            "preseason_snapshot_hash",
            "structural_power_build_id",
            "structural_power_version",
            "structural_power_date_imported",
            "structural_power_snapshot_hash",
        )
    }
    historical_analogs, historical_analog_source = (
        load_historical_rank_analogs(args.project_root)
    )

    print("[CIRCA_TOP5] 2026 Circa Million final two-entry portfolio")
    print(f"[CIRCA_TOP5] Build ID: {BUILD_ID}")
    print(f"[CIRCA_TOP5] Version: {VERSION}")
    print(f"[CIRCA_TOP5] Week: {args.week}")
    print(f"[CIRCA_TOP5] Circa lines: {lines_path}")
    print(f"[CIRCA_TOP5] Feature source: {feature_source}")
    print(
        "[CIRCA_TOP5] QB feature integrity: "
        f"{qb_live_integrity.get('status', 'UNKNOWN')} | "
        f"passed={qb_live_integrity.get('passed', 0)} | "
        f"EPA nonzero rows="
        f"{qb_live_integrity.get('qb_epa_advantage_nonzero_rows', 0)} | "
        f"CPOE nonzero rows="
        f"{qb_live_integrity.get('qb_cpoe_advantage_nonzero_rows', 0)}"
    )
    print(f"[CIRCA_TOP5] V1 residual bundle: {args.model_path}")
    print(
        f"[CIRCA_TOP5] Ceiling bundle: "
        f"{args.ceiling_model_path}"
    )
    print(
        "[CIRCA_TOP5] Ceiling configuration: "
        f"alpha={ceiling_bundle['late_alpha']:g} | "
        f"weight={ceiling_bundle['blend_weight']:g} | "
        f"policy={ceiling_bundle['direction_policy']} | "
        f"source={ceiling_bundle['source_profile']}"
    )
    print(
        "[CIRCA_TOP5] Adjusted V1 selection: raw ranks 1-4 locked; "
        "ranks 5-8 within 0.50 compete on prior-only 3/7 mass"
    )
    print(
        "[CIRCA_TOP5] Ceiling selection: raw V1 in Weeks 1-9; late "
        "Ceiling with HOME >= 7.5 veto in Weeks 10-18"
    )
    print(
        f"[CIRCA_TOP5] Strict margin history: {margin_history_source} | "
        f"games={len(margin_history)} | "
        f"seasons={sorted(margin_history['season'].unique().tolist())}"
    )
    print(
        f"[CIRCA_TOP5] Structural confidence source: {structural_source} | "
        f"variant={STRUCTURAL_MODEL_VARIANT} | "
        f"games={len(structural_projection)}"
    )
    print(
        "[CIRCA_TOP5] Structural review overlay: within-card residual-strength "
        f"ranks {STRUCTURAL_LOW_CONFIDENCE_RANK_MIN}-5 only | "
        f"contradiction >= {STRUCTURAL_REVIEW_THRESHOLD_POINTS:g} points | "
        "weekly flag cap=NONE"
    )
    print(
        "[CIRCA_TOP5] Structural projection substituted for residual "
        "features or model picks: NO"
    )
    print("[CIRCA_TOP5] Entry 1 residual predictions influenced by Entry 2: NO")
    print("[CIRCA_TOP5] Sportsbook staking enabled: NO")
    print(
        "[CIRCA_TOP5] Confidence board: selection robustness only; "
        "not cover probability"
    )
    print(
        f"[CIRCA_TOP5] Historical rank analog source: "
        f"{historical_analog_source}"
    )

    if args.preflight_only:
        print(
            f"[CIRCA_TOP5] Preflight passed: games={len(matrix)} | "
            f"features={len(v1_bundle['feature_order'])}"
        )
        return 0

    v1_card, raw_v1_card = predict_adjusted_v1_card(
        matrix,
        v1_bundle,
        margin_history,
    )
    ceiling_card = predict_ceiling_card(
        matrix,
        v1_bundle,
        ceiling_bundle,
        int(args.week),
    )
    assert_early_ceiling_identity(
        raw_v1_card,
        ceiling_card,
        int(args.week),
    )
    comparison = build_card_comparison(v1_card, ceiling_card)

    created_at = now_string()
    for card in (v1_card, ceiling_card):
        card["run_id"] = run_id
        card["created_at"] = created_at
        card["season"] = SEASON
        card["week"] = int(args.week)
        card["feature_source"] = feature_source
        card["rating_alpha"] = float(
            v1_bundle.get("rating_alpha", np.nan)
        )

    v1_card["model_path"] = str(args.model_path)
    v1_card["model_build_id"] = str(
        v1_bundle.get("build_id", "")
    )
    v1_card["model_version"] = str(
        v1_bundle.get("version", "")
    )
    v1_card["model_feature_set"] = str(
        v1_bundle.get("feature_set", "")
    )
    v1_card["model_alpha"] = float(
        v1_bundle.get("model_alpha", np.nan)
    )

    ceiling_card["v1_model_path"] = str(args.model_path)
    ceiling_card["ceiling_model_path"] = str(
        args.ceiling_model_path
    )
    ceiling_card["ceiling_model_build_id"] = str(
        ceiling_bundle.get("build_id", "")
    )
    ceiling_card["ceiling_model_version"] = str(
        ceiling_bundle.get("version", "")
    )

    comparison["run_id"] = run_id
    comparison["created_at"] = created_at
    comparison["feature_source"] = feature_source
    comparison["v1_model_path"] = str(args.model_path)
    comparison["ceiling_model_path"] = str(
        args.ceiling_model_path
    )
    portfolio = build_final_portfolio(v1_card, ceiling_card)
    confidence_board = build_confidence_board(
        v1_card,
        ceiling_card,
        margin_history,
        historical_analogs,
        historical_analog_source,
        structural_projection,
        structural_source,
    )
    portfolio = add_confidence_to_portfolio(portfolio, confidence_board)
    manual_submission_draft = build_manual_submission_draft(portfolio)

    top5_overlap = int(
        comparison["top5_membership"].eq("BOTH_TOP5").sum()
    )
    v1_only_top5 = int(
        comparison["top5_membership"].eq("V1_ONLY_TOP5").sum()
    )
    ceiling_only_top5 = int(
        comparison["top5_membership"].eq(
            "CEILING_ONLY_TOP5"
        ).sum()
    )
    adjusted_v1_promotions = int(v1_card["promoted_by_key"].sum())
    ceiling_veto_promotions = int(
        ceiling_card["promoted_by_home_favorite_veto"].sum()
    )
    ceiling_vetoed_candidates = int(
        ceiling_card["late_home_favorite_veto_candidate"].sum()
    )

    audit = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "build_id": BUILD_ID,
                "version": VERSION,
                "season": SEASON,
                "week": int(args.week),
                "run_as_of_date": pd.Timestamp(
                    args.as_of_date
                ).date().isoformat(),
                "circa_lines_path": str(lines_path),
                "circa_lines_sha256": circa_lines_sha256,
                "circa_lines_modified_at": circa_lines_modified_at,
                "feature_source": feature_source,
                "live_feature_build_id": live_feature_lineage.get(
                    "build_id"
                ),
                "live_feature_version": live_feature_lineage.get(
                    "version"
                ),
                "live_feature_created_at": live_feature_lineage.get(
                    "created_at"
                ),
                "live_feature_rating_alpha": live_feature_lineage.get(
                    "rating_alpha"
                ),
                "live_feature_set": live_feature_lineage.get(
                    "feature_set"
                ),
                "qb_feature_integrity_status": qb_live_integrity.get(
                    "status", "UNKNOWN"
                ),
                "qb_feature_integrity_passed": qb_live_integrity.get(
                    "passed", 0
                ),
                "qb_epa_advantage_nonzero_rows": qb_live_integrity.get(
                    "qb_epa_advantage_nonzero_rows", 0
                ),
                "qb_cpoe_advantage_nonzero_rows": qb_live_integrity.get(
                    "qb_cpoe_advantage_nonzero_rows", 0
                ),
                "margin_history_source": margin_history_source,
                "margin_history_rows": len(margin_history),
                "margin_history_seasons": "|".join(
                    str(value)
                    for value in sorted(
                        margin_history["season"].unique().tolist()
                    )
                ),
                "model_path": str(args.model_path),
                "model_build_id": v1_bundle.get("build_id"),
                "model_version": v1_bundle.get("version"),
                "model_feature_set": v1_bundle.get("feature_set"),
                "model_alpha": v1_bundle.get("model_alpha"),
                "rating_alpha": v1_bundle.get("rating_alpha"),
                "bundle_implementation_ready": int(
                    bool(v1_bundle.get("implementation_ready", False))
                ),
                "contest_forward_test_override": int(
                    args.approve_contest_forward_test
                ),
                "games_ranked": len(v1_card),
                "official_top5_rows": int(
                    v1_card["card_status"].eq(
                        "OFFICIAL_TOP5"
                    ).sum()
                ),
                "alternate_rows": int(
                    v1_card["card_status"].eq(
                        "ALTERNATE_6_10"
                    ).sum()
                ),
                "adjusted_v1_selection_policy": ADJUSTED_V1_POLICY,
                "adjusted_v1_tiebreak_tolerance": (
                    ADJUSTED_V1_TIEBREAK_TOLERANCE
                ),
                "adjusted_v1_candidate_rank_max": (
                    ADJUSTED_V1_CANDIDATE_RANK_MAX
                ),
                "adjusted_v1_promotions": adjusted_v1_promotions,
                "sportsbook_staking_enabled": 0,
                "ceiling_model_path": str(
                    args.ceiling_model_path
                ),
                "ceiling_model_build_id": ceiling_bundle.get(
                    "build_id"
                ),
                "ceiling_model_version": ceiling_bundle.get(
                    "version"
                ),
                "ceiling_source_profile": ceiling_bundle.get(
                    "source_profile"
                ),
                "ceiling_late_alpha": ceiling_bundle.get(
                    "late_alpha"
                ),
                "ceiling_blend_weight": ceiling_bundle.get(
                    "blend_weight"
                ),
                "ceiling_direction_policy": ceiling_bundle.get(
                    "direction_policy"
                ),
                "ceiling_selection_policy": CEILING_POLICY,
                "ceiling_shadow_only": 0,
                "ceiling_production_entry": 1,
                "ceiling_home_favorite_veto_min_margin": (
                    LATE_HOME_FAVORITE_VETO_MIN_MARGIN
                ),
                "ceiling_home_favorite_veto_applied": int(
                    int(args.week) >= 10
                ),
                "ceiling_home_favorite_vetoed_candidates": (
                    ceiling_vetoed_candidates
                ),
                "ceiling_home_favorite_veto_promotions": (
                    ceiling_veto_promotions
                ),
                "ceiling_top5_rows": int(
                    ceiling_card["card_status"].eq(
                        "CEILING_TOP5"
                    ).sum()
                ),
                "top5_overlap_rows": top5_overlap,
                "v1_only_top5_rows": v1_only_top5,
                "ceiling_only_top5_rows": ceiling_only_top5,
                "all_game_side_changes": int(
                    comparison["selected_side_changed"].sum()
                ),
                "weeks_1_9_exact_identity_guard": int(
                    int(args.week) <= 9
                ),
                "final_portfolio_rows": len(portfolio),
                "final_portfolio_entries": int(
                    portfolio["entry_number"].nunique()
                ),
                "confidence_methodology": CONFIDENCE_METHODOLOGY,
                "confidence_board_rows": len(confidence_board),
                "confidence_is_cover_probability": 0,
                "confidence_score_uses_historical_outcomes": 0,
                "structural_confidence_source": structural_source,
                "structural_confidence_table": args.structural_table,
                "structural_model_variant": STRUCTURAL_MODEL_VARIANT,
                "structural_run_id": structural_lineage["run_id"],
                "structural_build_id": structural_lineage["build_id"],
                "structural_version": structural_lineage["version"],
                "structural_prediction_as_of_date": structural_lineage[
                    "prediction_as_of_date"
                ],
                "structural_prediction_timestamp": structural_lineage[
                    "prediction_timestamp"
                ],
                "structural_form_build_id": structural_lineage[
                    "form_build_id"
                ],
                "structural_form_version": structural_lineage[
                    "form_version"
                ],
                "structural_form_as_of_date": structural_lineage[
                    "form_as_of_date"
                ],
                "structural_form_date_imported": structural_lineage[
                    "form_date_imported"
                ],
                "structural_preseason_snapshot_hash": structural_lineage[
                    "preseason_snapshot_hash"
                ],
                "structural_power_build_id": structural_lineage[
                    "structural_power_build_id"
                ],
                "structural_power_version": structural_lineage[
                    "structural_power_version"
                ],
                "structural_power_date_imported": structural_lineage[
                    "structural_power_date_imported"
                ],
                "structural_power_snapshot_hash": structural_lineage[
                    "structural_power_snapshot_hash"
                ],
                "structural_projection_rows": len(structural_projection),
                "structural_all_current_games_matched": 1,
                "structural_review_lowest_rank_min": (
                    STRUCTURAL_LOW_CONFIDENCE_RANK_MIN
                ),
                "structural_review_threshold_points": (
                    STRUCTURAL_REVIEW_THRESHOLD_POINTS
                ),
                "structural_review_weekly_flag_cap": "NONE",
                "structural_review_flags": int(
                    confidence_board["structural_review_flag"].sum()
                ),
                "structural_numeric_confidence_penalty": 0.0,
                "structural_used_to_change_model_card": (
                    STRUCTURAL_USED_TO_CHANGE_MODEL_CARD
                ),
                "historical_rank_analog_source": historical_analog_source,
                "historical_rank_analog_available": int(
                    not historical_analogs.empty
                ),
                "manual_submission_draft_rows": len(
                    manual_submission_draft
                ),
                "immutable_model_card_preserved": 1,
                "production_promoted": 1,
                "created_at": created_at,
            }
        ]
    )

    (
        database,
        v1_csv_path,
        ceiling_csv_path,
        comparison_csv_path,
        portfolio_csv_path,
        confidence_csv_path,
        manual_submission_csv_path,
    ) = save_outputs(
        args,
        v1_card,
        ceiling_card,
        comparison,
        portfolio,
        confidence_board,
        manual_submission_draft,
        audit,
    )

    adjusted_v1_display_columns = [
        "contest_rank",
        "raw_residual_rank",
        "selected_team",
        "selected_circa_spread",
        "away_team",
        "home_team",
        "circa_home_margin",
        "predicted_circa_residual",
        "circa_adjusted_home_margin",
        "adjusted_v1_key_tiebreak_score",
        "promoted_by_key",
        "card_status",
    ]
    ceiling_display_columns = [
        "contest_rank",
        "selected_team",
        "selected_circa_spread",
        "away_team",
        "home_team",
        "circa_home_margin",
        "v1_predicted_circa_residual",
        "late_model_predicted_circa_residual",
        "ceiling_predicted_circa_residual",
        "ceiling_adjusted_home_margin",
        "raw_ceiling_residual_rank",
        "late_home_favorite_veto_candidate",
        "promoted_by_home_favorite_veto",
        "card_status",
    ]
    comparison_display_columns = [
        "away_team",
        "home_team",
        "v1_contest_rank",
        "v1_selected_team",
        "v1_selected_circa_spread",
        "v1_predicted_circa_residual",
        "ceiling_contest_rank",
        "ceiling_selected_team",
        "ceiling_selected_circa_spread",
        "ceiling_predicted_circa_residual",
        "selected_side_changed",
        "top5_membership",
    ]

    print("")
    print("=" * 126)
    print("[CIRCA_TOP5] ENTRY 1: ADJUSTED V1 TOP FIVE AND ALTERNATES")
    print("=" * 126)
    print(
        v1_card[
            v1_card["contest_rank"].le(10)
        ][adjusted_v1_display_columns].to_string(index=False)
    )

    print("")
    print("=" * 146)
    print("[CIRCA_TOP5] ENTRY 2: PRODUCTION CEILING TOP FIVE AND ALTERNATES")
    print("=" * 146)
    print(
        ceiling_card[
            ceiling_card["contest_rank"].le(10)
        ][ceiling_display_columns].to_string(index=False)
    )

    print("")
    print("=" * 166)
    print("[CIRCA_TOP5] ADJUSTED V1 VS PRODUCTION CEILING COMPARISON")
    print("=" * 166)
    comparison_display = comparison[
        comparison["v1_contest_rank"].le(10)
        | comparison["ceiling_contest_rank"].le(10)
    ]
    print(
        comparison_display[
            comparison_display_columns
        ].to_string(index=False)
    )

    print("")
    print("=" * 126)
    print("[CIRCA_TOP5] FINAL PRODUCTION PORTFOLIO (TWO ENTRIES)")
    print("=" * 126)
    print(
        portfolio[
            [
                "entry_number",
                "entry_strategy",
                "contest_rank",
                "selected_team",
                "selected_circa_spread",
                "away_team",
                "home_team",
                "portfolio_predicted_circa_residual",
                "source_raw_residual_rank",
                "selection_adjustment_applied",
                "production_raw_confidence_rank",
                "confidence_rank",
                "confidence_score",
                "confidence_tier",
                "manual_review_priority",
                "structural_confirmation_points",
                "structural_review_flag",
            ]
        ].to_string(index=False)
    )

    confidence_display_columns = [
        "entry_number",
        "production_raw_confidence_rank",
        "base_confidence_rank",
        "confidence_rank",
        "contest_rank",
        "selected_team",
        "selected_circa_spread",
        "confidence_score",
        "confidence_tier",
        "manual_review_priority",
        "structural_home_margin",
        "structural_gap_vs_circa",
        "structural_confirmation_points",
        "structural_review_flag",
        "structural_review_label",
        "model_direction_agreement",
        "same_pick_in_both_entries",
        "residual_gap_vs_first_alternate",
        "first_alternate_team",
        "first_alternate_circa_spread",
        "fragility_reason",
        "historical_contest_rank_points_rate",
        "historical_contest_rank_sample_size",
    ]
    print("")
    print("=" * 180)
    print(
        "[CIRCA_TOP5] ADVISORY CONFIDENCE BOARD "
        "(ROBUSTNESS + STRUCTURAL REVIEW — NOT COVER PROBABILITY)"
    )
    print("=" * 180)
    print(
        confidence_board[confidence_display_columns].to_string(index=False)
    )

    print(
        "\n[CIRCA_TOP5] Top-five overlap: "
        f"{top5_overlap}/5 | "
        f"V1-only={v1_only_top5} | "
        f"ceiling-only={ceiling_only_top5} | "
        "all-game side changes="
        f"{int(comparison['selected_side_changed'].sum())}"
    )
    print(f"[CIRCA_TOP5] Output database: {database}")
    if v1_csv_path is not None:
        print(f"[CIRCA_TOP5] Adjusted V1 CSV: {v1_csv_path}")
    if ceiling_csv_path is not None:
        print(
            f"[CIRCA_TOP5] Ceiling CSV: {ceiling_csv_path}"
        )
    if comparison_csv_path is not None:
        print(
            f"[CIRCA_TOP5] Comparison CSV: "
            f"{comparison_csv_path}"
        )
    if portfolio_csv_path is not None:
        print(
            f"[CIRCA_TOP5] Final portfolio CSV: "
            f"{portfolio_csv_path}"
        )
    if confidence_csv_path is not None:
        print(
            f"[CIRCA_TOP5] Confidence board CSV: "
            f"{confidence_csv_path}"
        )
    if manual_submission_csv_path is not None:
        print(
            f"[CIRCA_TOP5] Manual submission draft CSV: "
            f"{manual_submission_csv_path}"
        )
    print(
        "[CIRCA_TOP5] Completed in "
        f"{(dt.datetime.now(dt.timezone.utc) - started).total_seconds():.2f} "
        "seconds"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CIRCA_TOP5] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CIRCA_TOP5] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
