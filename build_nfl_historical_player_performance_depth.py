#!/usr/bin/env python
"""
Build historical player-performance grades and projected depth charts.

This is Stage 4 of the true canonical NFL historical reconstruction.

For each isolated target-season database, this wrapper:

1. verifies the approved canonical source files:
       build_nfl_player_performance.py
       build_nfl_projected_depth_chart.py
2. creates temporary runtime copies;
3. changes only season, legal history window, recent season, and table names;
4. executes the canonical player-performance V4 model with prior-only PFF OL data;
5. reconstructs the Week 1 OL unit from observed Week 1 snaps, the frozen
   roster, and prior-only PFF position evidence;
6. executes the canonical projected-depth V5 model with frozen Week 1 OL/QB starters;
7. treats non-point-in-time roster statuses as non-vetoing only for those
   frozen authoritative starters and audits every such override;
8. validates complete player reconciliation, 32 QB1s, and 160 OL starters;
9. writes historical readiness and audit tables;
10. leaves production scripts and the production database unchanged.

Historical information policy
-----------------------------
- Target-season personnel comes from the frozen Week 1 roster snapshot.
- Player performance uses only the legal four-year window ending in S-1.
- The recent-season component is S-1, never target season S.
- QB1 comes from the Week 1 prior established by
  prepare_nfl_historical_reconstruction.py.
- OL personnel comes from Week 1 offensive participation. Position assignment
  combines the Week 1 snap position, frozen roster position, and prior-only PFF
  position; no target-season performance grade is used to choose starters.
- Historical roster status is retained for audit but is not an as-of-Week-1
  injury feed. It therefore cannot veto a frozen authoritative QB/OL starter.
- Graded game replay begins in Week 2.

Required prior stages
---------------------
1. prepare_nfl_historical_reconstruction.py
2. backfill_nfl_historical_advanced_stats.py
3. build_nfl_historical_qb_defense.py
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


BUILD_ID = "NFL_HISTORICAL_PLAYER_PERFORMANCE_DEPTH_CANONICAL_V3"
VERSION = "v3_3_week1_snap_ol_bootstrap_prior_only"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_TARGET_SEASONS = (2023, 2024, 2025)
HISTORY_WINDOW_YEARS = 4
PFF_FIRST_SEASON = 2017
PFF_BASE_WEIGHTS = (0.10, 0.20, 0.30, 0.40)

CONTEXT_TABLE = "nfl_historical_reconstruction_context"
MASTER_TABLE = "nfl_player_master_target"
ADVANCED_CURRENT_TABLE = (
    "nfl_player_advanced_stats_current_roster_target"
)
ADVANCED_HISTORY_TABLE = "nfl_player_advanced_stats_history"
QB_RATING_TABLE = "nfl_qb_rbsdm_ratings_target"
DEFENSE_TABLE = (
    "nfl_defensive_player_metrics_current_roster_target"
)
HISTORICAL_QB_PRIOR_TABLE = (
    "nfl_projected_qb_starters_target"
)
DEPTH_QB_INPUT_TABLE = (
    "nfl_projected_qb_starters_depth_input"
)
DEPTH_OL_INPUT_TABLE = "nfl_projected_ol_starters_depth_input"
OL_STARTER_SLOTS = ("LT", "LG", "C", "RG", "RT")
MATCHUP_SOURCE_CACHE_FILENAME = "nfl_weekly_matchup_source_cache_v2.sqlite"
MATCHUP_SNAP_TABLE = "snap_source"

# Explicit nflverse historical-name aliases only. These resolve known long-name
# versus nickname changes without fuzzy matching authoritative starters.
HISTORICAL_OL_NAME_ALIASES = {
    "michaeljordan": "mikejordan",
    "rickywagner": "rickwagner",
    "josephnoteboom": "joenoteboom",
    "michaelonwenu": "mikeonwenu",
    "chrishubbard": "christopherhubbard",
    "oliudoh": "olisaemekaudoh",
    "shaqmason": "shaquillemason",
    "trentbrown": "trentonbrown",
    "yoshnijman": "yosuahnijman",
    "yoshinijman": "yosuahnijman",
    "samuelcosmi": "samcosmi",
    "trentonscott": "trentscott",
    "delmarglaze": "djglaze",
}

PERFORMANCE_TABLE = "nfl_player_performance_inputs_target"
SEASONAL_PERFORMANCE_TABLE = (
    "nfl_player_performance_seasonal_history"
)
PERFORMANCE_POSITION_SUMMARY_TABLE = (
    "nfl_player_performance_position_summary_target"
)
PERFORMANCE_COMPONENT_AUDIT_TABLE = (
    "nfl_player_performance_component_audit_target"
)
PERFORMANCE_UNRESOLVED_TABLE = (
    "nfl_player_performance_unresolved_master_audit_target"
)
PFF_OL_HISTORY_TABLE = "nfl_ol_pff_player_season_history_target"
PFF_OL_IDENTITY_AUDIT_TABLE = "nfl_ol_pff_identity_audit_target"

DEPTH_TABLE = "nfl_projected_depth_chart_target"
DEPTH_AUDIT_TABLE = "nfl_projected_depth_chart_audit_target"

READINESS_TABLE = (
    "nfl_historical_player_performance_depth_readiness_audit"
)
STEP_AUDIT_TABLE = (
    "nfl_historical_player_performance_depth_step_audit"
)

EXPECTED_PERFORMANCE_BUILD_ID = (
    "NFL_PLAYER_PERFORMANCE_2026_CANONICAL_V4"
)
EXPECTED_PERFORMANCE_VERSION = (
    "v4_1_pff_row_identity_collision_safe"
)
EXPECTED_DEPTH_BUILD_ID = (
    "NFL_PROJECTED_DEPTH_CHART_2026_CANONICAL_V5"
)
EXPECTED_DEPTH_VERSION = (
    "v5_pff_ol_talent_no_second_shrink_authoritative_starters"
)

PFF_DEFAULT_FILENAMES = {2017: 'offense_blocking_2017.csv', 2018: 'offense_blocking_2018.csv', 2019: 'offense_blocking_2019.csv', 2020: 'offense_blocking_2020.csv', 2021: 'offense_blocking_2021.csv', 2022: 'offense_blocking (3).csv', 2023: 'offense_blocking (2).csv', 2024: 'offense_blocking (1).csv'}


@dataclass(frozen=True)
class RuntimeScript:
    key: str
    source_path: Path
    patched_path: Path
    source_sha256: str


def parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(
        sorted(
            {
                int(piece.strip())
                for piece in value.split(",")
                if piece.strip()
            }
        )
    )
    if not seasons:
        raise argparse.ArgumentTypeError(
            "At least one target season is required."
        )
    return seasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
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
    parser.add_argument(
        "--source-cache-db",
        type=Path,
        default=None,
        help=(
            "SQLite source cache containing Week 1 snap_source rows. "
            "Defaults to <database-root>/"
            f"{MATCHUP_SOURCE_CACHE_FILENAME}."
        ),
    )
    parser.add_argument(
        "--target-seasons",
        type=parse_seasons,
        default=DEFAULT_TARGET_SEASONS,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--keep-patched-scripts",
        action="store_true",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
    )
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
    args.source_cache_db = (
        args.source_cache_db.resolve()
        if args.source_cache_db is not None
        else (
            args.database_root / MATCHUP_SOURCE_CACHE_FILENAME
        ).resolve()
    )
    args.python = (
        args.python.resolve()
        if args.python is not None
        else Path(sys.executable).resolve()
    )
    return args


def now_string() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    return connection.execute(
        """
        SELECT 1
        FROM sqlite_master
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
        raise RuntimeError(
            f"Missing required table: {table_name}"
        )
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(
        f'SELECT * FROM "{escaped}"',
        connection,
    )
    frame.columns = [
        str(column).lower().strip()
        for column in frame.columns
    ]
    return frame


def add_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    sql_type: str,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    }
    if column_name in columns:
        return
    connection.execute(
        f'ALTER TABLE "{table_name}" '
        f'ADD COLUMN "{column_name}" {sql_type}'
    )


def stable_hash(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> str:
    usable = [
        column for column in columns
        if column in frame.columns
    ]
    if not usable or frame.empty:
        return hashlib.sha256(b"").hexdigest()

    work = frame[usable].copy()
    for column in usable:
        work[column] = work[column].astype(str)
    payload = (
        work.sort_values(usable)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def replace_assignment(
    text: str,
    variable: str,
    expression: str,
) -> str:
    pattern = re.compile(
        rf"(?m)^{re.escape(variable)}\s*=\s*.*$"
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one assignment for {variable}; "
            f"found {len(matches)}."
        )
    return pattern.sub(
        f"{variable} = {expression}",
        text,
        count=1,
    )


def replace_mapping_assignment(
    text: str,
    variable: str,
    expression: str,
) -> str:
    pattern = re.compile(
        rf"(?ms)^{re.escape(variable)}\s*=\s*\{{.*?^\}}\s*"
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mapping assignment for {variable}; found {len(matches)}."
        )
    return pattern.sub(
        f"{variable} = {expression}\n",
        text,
        count=1,
    )


def replace_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} patch point; found {count}.")
    return text.replace(old, new, 1)


def clean_pff_column(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def normalize_pff_position(value: object) -> str:
    text = re.sub(r"[^A-Z]", "", str(value).upper())
    if text in {"T", "OT", "LT", "RT"}:
        return "T"
    if text in {"G", "OG", "LG", "RG"}:
        return "G"
    if text == "C":
        return "C"
    return ""


def pff_history_seasons(target_season: int) -> list[int]:
    seasons = list(range(PFF_FIRST_SEASON, target_season))
    if not seasons:
        raise RuntimeError(
            f"Target {target_season} has no prior PFF OL season; "
            "the leakage-safe combined audit begins in 2023."
        )
    if len(seasons) > len(PFF_BASE_WEIGHTS):
        seasons = seasons[-len(PFF_BASE_WEIGHTS):]
    return seasons


def resolve_pff_input_paths(
    project_root: Path,
    seasons: list[int],
) -> dict[int, Path]:
    search_roots = [
        project_root / "data" / "pff",
        project_root / "data" / "raw",
        project_root,
        Path.home() / "Downloads",
    ]
    resolved: dict[int, Path] = {}
    missing: list[str] = []
    for season in seasons:
        canonical = f"nfl_pff_offense_blocking_{season}.csv"
        names = [canonical, PFF_DEFAULT_FILENAMES.get(season, canonical)]
        found = next(
            (
                root / name
                for root in search_roots
                for name in names
                if (root / name).is_file()
            ),
            None,
        )
        if found is None:
            missing.append(f"{season}: {canonical}")
        else:
            resolved[season] = found.resolve()
    if missing:
        raise RuntimeError(
            "Missing prior-only PFF OL exports required for the combined audit:\n"
            + "\n".join(missing)
        )
    return resolved


def derive_pff_replacement_anchors(paths: dict[int, Path]) -> dict[str, float]:
    parts: list[pd.DataFrame] = []
    for season, path in sorted(paths.items()):
        frame = pd.read_csv(path)
        frame.columns = [clean_pff_column(column) for column in frame.columns]
        required = {"position", "grades_offense", "snap_counts_offense"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"{path.name} is missing PFF anchor columns: {missing}")
        part = pd.DataFrame(
            {
                "season": season,
                "position": frame["position"].map(normalize_pff_position),
                "overall_grade": pd.to_numeric(frame["grades_offense"], errors="coerce"),
                "offense_snaps": pd.to_numeric(frame["snap_counts_offense"], errors="coerce"),
            }
        )
        parts.append(part)
    archive = pd.concat(parts, ignore_index=True)
    qualified = archive[
        archive["position"].isin({"T", "G", "C"})
        & archive["offense_snaps"].ge(200.0)
        & archive["overall_grade"].between(0.0, 100.0)
    ].copy()
    anchors: dict[str, float] = {}
    for position in ("T", "G", "C"):
        values = qualified.loc[qualified["position"].eq(position), "overall_grade"]
        if values.empty:
            raise RuntimeError(
                f"Prior-only PFF archive has no qualified {position} rows for a replacement anchor."
            )
        anchors[position] = round(float(values.quantile(0.20)), 4)
    return {
        "T": anchors["T"], "OT": anchors["T"], "LT": anchors["T"], "RT": anchors["T"],
        "G": anchors["G"], "OG": anchors["G"], "LG": anchors["G"], "RG": anchors["G"],
        "C": anchors["C"],
        "OL": round(float(np.mean(list(anchors.values()))), 4),
    }


def copy_pff_inputs(paths: dict[int, Path], runtime_root: Path) -> None:
    target_dir = runtime_root / "data" / "pff"
    target_dir.mkdir(parents=True, exist_ok=True)
    for season, source in paths.items():
        shutil.copy2(source, target_dir / f"nfl_pff_offense_blocking_{season}.csv")


def verify_source(
    path: Path,
    expected_build_id: str,
    expected_version: str,
    allow_mismatch: bool,
) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    )
    if not allow_mismatch:
        if expected_build_id not in text:
            raise RuntimeError(
                f"{path.name} does not contain approved "
                f"Build ID {expected_build_id}."
            )
        if expected_version not in text:
            raise RuntimeError(
                f"{path.name} does not contain approved "
                f"version {expected_version}."
            )
    return text


def patch_performance_script(
    source_path: Path,
    patched_path: Path,
    target_season: int,
    history_seasons: list[int],
    prior_pff_seasons: list[int],
    replacement_anchors: dict[str, float],
    allow_mismatch: bool,
) -> RuntimeScript:
    text = verify_source(
        source_path,
        EXPECTED_PERFORMANCE_BUILD_ID,
        EXPECTED_PERFORMANCE_VERSION,
        allow_mismatch,
    )
    replacements = {
        "SEASON": repr(target_season),
        "HISTORICAL_SEASONS": repr(
            tuple(history_seasons)
        ),
        "RECENT_SEASON": repr(history_seasons[-1]),
        "MASTER_TABLE": repr(MASTER_TABLE),
        "ADVANCED_CURRENT_TABLE": repr(
            ADVANCED_CURRENT_TABLE
        ),
        "ADVANCED_HISTORY_TABLE": repr(
            ADVANCED_HISTORY_TABLE
        ),
        "QB_TABLE": repr(QB_RATING_TABLE),
        "DEFENSE_TABLE": repr(DEFENSE_TABLE),
        "OUTPUT_TABLE": repr(PERFORMANCE_TABLE),
        "SEASONAL_OUTPUT_TABLE": repr(
            SEASONAL_PERFORMANCE_TABLE
        ),
        "POSITION_SUMMARY_TABLE": repr(
            PERFORMANCE_POSITION_SUMMARY_TABLE
        ),
        "COMPONENT_AUDIT_TABLE": repr(
            PERFORMANCE_COMPONENT_AUDIT_TABLE
        ),
        "UNRESOLVED_AUDIT_TABLE": repr(
            PERFORMANCE_UNRESOLVED_TABLE
        ),
        "PFF_OL_HISTORY_TABLE": repr(PFF_OL_HISTORY_TABLE),
        "PFF_OL_IDENTITY_AUDIT_TABLE": repr(PFF_OL_IDENTITY_AUDIT_TABLE),
        "OL_PFF_SEASON_WEIGHTS": repr(
            {
                season: PFF_BASE_WEIGHTS[-len(prior_pff_seasons):][index]
                for index, season in enumerate(prior_pff_seasons)
            }
        ),
    }
    for variable, expression in replacements.items():
        text = replace_assignment(
            text,
            variable,
            expression,
        )

    text = replace_mapping_assignment(
        text,
        "PFF_DEFAULT_FILENAMES",
        repr(
            {
                season: PFF_DEFAULT_FILENAMES.get(
                    season,
                    f"nfl_pff_offense_blocking_{season}.csv",
                )
                for season in prior_pff_seasons
            }
        ),
    )

    history_assignment = f"HISTORICAL_SEASONS = {repr(tuple(history_seasons))}"
    text = replace_exact_once(
        text,
        history_assignment,
        history_assignment
        + f"\nPFF_HISTORICAL_SEASONS = {repr(tuple(prior_pff_seasons))}",
        "PFF historical-season declaration",
    )
    text = replace_exact_once(
        text,
        '    for season in HISTORICAL_SEASONS\n}',
        '    for season in PFF_HISTORICAL_SEASONS\n}',
        "PFF canonical-filename comprehension",
    )
    text = replace_exact_once(
        text,
        '    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)\n'
        '    for season in HISTORICAL_SEASONS:',
        '    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)\n'
        '    for season in PFF_HISTORICAL_SEASONS:',
        "PFF CLI loop",
    )
    text = replace_exact_once(
        text,
        '    resolved: dict[int, Path] = {}\n    missing: list[str] = []\n'
        '    for season in HISTORICAL_SEASONS:',
        '    resolved: dict[int, Path] = {}\n    missing: list[str] = []\n'
        '    for season in PFF_HISTORICAL_SEASONS:',
        "PFF path-resolution loop",
    )
    text = replace_exact_once(
        text,
        '    if set(frame["season"].astype(int).unique()) != set(HISTORICAL_SEASONS):',
        '    if set(frame["season"].astype(int).unique()) != set(PFF_HISTORICAL_SEASONS):',
        "PFF season validation",
    )
    text = replace_exact_once(
        text,
        'last_three = hist[hist["season"].isin((2023, 2024, 2025))].copy()',
        'last_three = hist[hist["season"].isin(HISTORICAL_SEASONS[-3:])].copy()',
        "dynamic recent-three-season history",
    )
    text = replace_mapping_assignment(
        text,
        "OL_PFF_REPLACEMENT_BY_POSITION",
        repr(replacement_anchors),
    )

    patched_path.write_text(text, encoding="utf-8")
    return RuntimeScript(
        key="player_performance",
        source_path=source_path,
        patched_path=patched_path,
        source_sha256=hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    )


def patch_depth_script(
    source_path: Path,
    patched_path: Path,
    target_season: int,
    allow_mismatch: bool,
) -> RuntimeScript:
    text = verify_source(
        source_path,
        EXPECTED_DEPTH_BUILD_ID,
        EXPECTED_DEPTH_VERSION,
        allow_mismatch,
    )
    replacements = {
        "SEASON": repr(target_season),
        "PERFORMANCE_TABLE": repr(PERFORMANCE_TABLE),
        "MASTER_TABLE": repr(MASTER_TABLE),
        "QB_STARTER_TABLE": repr(
            DEPTH_QB_INPUT_TABLE
        ),
        "OL_STARTER_TABLE": repr(DEPTH_OL_INPUT_TABLE),
        "OUTPUT_TABLE": repr(DEPTH_TABLE),
        "AUDIT_TABLE": repr(DEPTH_AUDIT_TABLE),
    }
    for variable, expression in replacements.items():
        text = replace_assignment(
            text,
            variable,
            expression,
        )

    text = replace_exact_once(
        text,
        '    out["likely_unavailable"] = out["status"].map(likely_unavailable)\n',
        '    out["likely_unavailable"] = out["status"].map(likely_unavailable)\n'
        '    out["historical_frozen_status_override"] = 0\n',
        "historical frozen-status audit initialization",
    )
    text = replace_exact_once(
        text,
        '    out["qb_authoritative_starter"] = (by_id | by_name).astype(int)\n\n'
        '    problems: list[str] = []',
        '    out["qb_authoritative_starter"] = (by_id | by_name).astype(int)\n'
        '    historical_qb_mask = out["qb_authoritative_starter"].eq(1)\n'
        '    historical_qb_override = historical_qb_mask & out["likely_unavailable"].eq(1)\n'
        '    out.loc[historical_qb_override, "historical_frozen_status_override"] = 1\n'
        '    out.loc[historical_qb_mask, "likely_unavailable"] = 0\n'
        '    out.loc[historical_qb_mask, "availability_score"] = out.loc[historical_qb_mask, "durability_score"]\n'
        '    out.loc[historical_qb_mask, "depth_order_score"] = (\n'
        '        0.85 * out.loc[historical_qb_mask, "unit_quality_grade"]\n'
        '        + 0.15 * out.loc[historical_qb_mask, "availability_score"]\n'
        '    )\n\n'
        '    problems: list[str] = []',
        "historical frozen QB status handling",
    )
    text = replace_exact_once(
        text,
        '    out["ol_authoritative_starter"] = out["configured_ol_starter_slot"].isin(OL_STARTER_SLOTS).astype(int)\n\n'
        '    matched_is_ol = out["position_group"].eq("OL") | out["canonical_role"].isin(OL_ROLES)',
        '    out["ol_authoritative_starter"] = out["configured_ol_starter_slot"].isin(OL_STARTER_SLOTS).astype(int)\n'
        '    historical_ol_mask = out["ol_authoritative_starter"].eq(1)\n'
        '    historical_ol_override = historical_ol_mask & out["likely_unavailable"].eq(1)\n'
        '    out.loc[historical_ol_override, "historical_frozen_status_override"] = 1\n'
        '    out.loc[historical_ol_mask, "likely_unavailable"] = 0\n'
        '    out.loc[historical_ol_mask, "availability_score"] = out.loc[historical_ol_mask, "durability_score"]\n'
        '    out.loc[historical_ol_mask, "depth_order_score"] = (\n'
        '        0.85 * out.loc[historical_ol_mask, "unit_quality_grade"]\n'
        '        + 0.15 * out.loc[historical_ol_mask, "availability_score"]\n'
        '    )\n\n'
        '    matched_is_ol = out["position_group"].eq("OL") | out["canonical_role"].isin(OL_ROLES)',
        "historical frozen OL status handling",
    )
    text = replace_exact_once(
        text,
        '        "likely_unavailable", "performance_grade", "replacement_baseline",',
        '        "likely_unavailable", "historical_frozen_status_override",\n'
        '        "performance_grade", "replacement_baseline",',
        "historical frozen-status audit output",
    )

    patched_path.write_text(text, encoding="utf-8")
    return RuntimeScript(
        key="projected_depth",
        source_path=source_path,
        patched_path=patched_path,
        source_sha256=hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
    )


def prepare_qb_depth_input(
    connection: sqlite3.Connection,
    target_season: int,
) -> pd.DataFrame:
    historical = read_table(
        connection,
        HISTORICAL_QB_PRIOR_TABLE,
    )
    required = {"team", "player_name"}
    missing = sorted(required - set(historical.columns))
    if missing:
        raise RuntimeError(
            f"{HISTORICAL_QB_PRIOR_TABLE} is missing "
            f"columns: {missing}"
        )

    source_values = (
        historical["starter_source"]
        if "starter_source" in historical.columns
        else pd.Series(
            "week_1_prior",
            index=historical.index,
        )
    )
    updated_values = (
        historical["date_imported"]
        if "date_imported" in historical.columns
        else pd.Series(
            now_string(),
            index=historical.index,
        )
    )

    out = pd.DataFrame(
        {
            "season": target_season,
            "team": (
                historical["team"]
                .astype(str)
                .str.upper()
                .str.strip()
            ),
            "player_name": (
                historical["player_name"]
                .astype(str)
                .str.strip()
            ),
            "source": source_values,
            "active": 1,
            "updated_at": updated_values,
        }
    )
    out = (
        out.drop_duplicates("team", keep="last")
        .sort_values("team")
        .reset_index(drop=True)
    )
    if len(out) != 32 or out["team"].nunique() != 32:
        raise RuntimeError(
            f"{target_season} historical QB prior must "
            f"contain 32 unique teams."
        )

    out.to_sql(
        DEPTH_QB_INPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{DEPTH_QB_INPUT_TABLE}_season_team
        ON {DEPTH_QB_INPUT_TABLE}(season, team)
        """
    )
    connection.commit()
    return out


def prepare_ol_depth_input(
    connection: sqlite3.Connection,
    target_season: int,
    source_cache_db: Path,
) -> pd.DataFrame:
    """Build the frozen Week 1 OL input without a pre-existing depth table.

    Replays begin in Week 2, so Week 1 offensive participation is legal as-of
    information. Personnel selection uses the five highest Week 1 offensive
    snap counts; slot assignment then maximizes position evidence. Target-season
    player quality is deliberately excluded.
    """
    if not source_cache_db.exists():
        raise FileNotFoundError(
            f"Missing Week 1 snap source cache: {source_cache_db}"
        )

    master = read_table(connection, MASTER_TABLE)
    performance = read_table(connection, PERFORMANCE_TABLE)
    master_required = {
        "team", "player_id", "player_name", "position",
        "position_group", "depth_chart_position",
    }
    missing_master = sorted(master_required.difference(master.columns))
    if missing_master:
        raise RuntimeError(
            f"{MASTER_TABLE} cannot identify Week 1 OL personnel; "
            f"missing {missing_master}."
        )
    performance_required = {
        "player_id", "position_group", "ol_pff_position",
    }
    missing_performance = sorted(
        performance_required.difference(performance.columns)
    )
    if missing_performance:
        raise RuntimeError(
            f"{PERFORMANCE_TABLE} cannot supply prior-only OL position "
            f"evidence; missing {missing_performance}."
        )

    with sqlite3.connect(
        f"{source_cache_db.as_uri()}?mode=ro",
        uri=True,
    ) as source_connection:
        if not table_exists(source_connection, MATCHUP_SNAP_TABLE):
            raise RuntimeError(
                f"Missing required table in {source_cache_db}: "
                f"{MATCHUP_SNAP_TABLE}"
            )
        snaps = pd.read_sql_query(
            f"""
            SELECT season, week, team, player_name, position,
                   position_group, offense_snaps
            FROM {MATCHUP_SNAP_TABLE}
            WHERE CAST(season AS INTEGER)=?
              AND CAST(week AS INTEGER)=1
              AND UPPER(COALESCE(position_group,''))='OL'
              AND COALESCE(offense_snaps,0)>0
            """,
            source_connection,
            params=(target_season,),
        )
    snaps.columns = [str(column).lower().strip() for column in snaps.columns]
    if snaps.empty:
        raise RuntimeError(
            f"{MATCHUP_SNAP_TABLE} contains no positive Week 1 OL snaps "
            f"for {target_season}."
        )

    master_ol = master[
        master["position_group"].fillna("").astype(str).str.upper().eq("OL")
    ].copy()
    master_ol["team"] = (
        master_ol["team"].fillna("").astype(str).str.upper().str.strip()
    )
    snaps["team"] = snaps["team"].fillna("").astype(str).str.upper().str.strip()
    master_ol["name_key"] = normalize_historical_ol_names(
        master_ol["player_name"]
    )
    snaps["name_key"] = normalize_historical_ol_names(snaps["player_name"])
    duplicate_master = master_ol[
        master_ol.duplicated(["team", "name_key"], keep=False)
    ]
    if not duplicate_master.empty:
        raise RuntimeError(
            "Frozen Week 1 OL master identity is not unique:\n"
            + duplicate_master[
                ["team", "player_id", "player_name"]
            ].to_string(index=False)
        )

    candidates = snaps.merge(
        master_ol[
            [
                "team", "name_key", "player_id", "player_name",
                "position", "depth_chart_position",
            ]
        ],
        on=["team", "name_key"],
        how="left",
        validate="many_to_one",
        suffixes=("_snap", "_master"),
    )
    unmatched = candidates[candidates["player_id"].fillna("").astype(str).eq("")]
    if not unmatched.empty:
        raise RuntimeError(
            f"{target_season} Week 1 OL snap identities did not reconcile "
            f"to {MASTER_TABLE}:\n"
            + unmatched[
                ["team", "player_name_snap", "position_snap", "offense_snaps"]
            ].to_string(index=False)
        )

    performance_ol = performance[
        performance["position_group"].fillna("").astype(str).str.upper().eq("OL")
    ][["player_id", "ol_pff_position"]].copy()
    if performance_ol["player_id"].duplicated().any():
        raise RuntimeError(
            f"{PERFORMANCE_TABLE} contains duplicate OL player_id rows."
        )
    candidates = candidates.merge(
        performance_ol,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    candidates["offense_snaps"] = pd.to_numeric(
        candidates["offense_snaps"], errors="coerce"
    ).fillna(0.0)
    candidates["snap_role"] = candidates["position_snap"].map(ol_role_family)
    candidates["master_role"] = candidates["depth_chart_position"].map(
        ol_role_family
    )
    candidates["pff_role"] = candidates["ol_pff_position"].map(ol_role_family)

    slot_families = (
        ("C", "C"), ("LG", "G"), ("RG", "G"),
        ("LT", "T"), ("RT", "T"),
    )
    selected_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for team, team_candidates in candidates.groupby("team", sort=True):
        if len(team_candidates) < 5:
            problems.append(
                f"{team}: only {len(team_candidates)} matched positive-snap OL players"
            )
            continue

        # The Week 2 as-of unit is the five OL players with the most Week 1
        # offensive participation. This handles in-game injuries and avoids
        # choosing personnel from a target-season performance grade.
        team_candidates = (
            team_candidates.sort_values(
                ["offense_snaps", "name_key", "player_id"],
                ascending=[False, True, True],
            )
            .head(5)
            .sort_values(["name_key", "player_id"])
            .reset_index(drop=True)
        )

        # Small dynamic program: each selected player fills exactly one
        # C/G/G/T/T slot. Week 1 snap, frozen-roster, and prior-only PFF roles
        # determine the best labels. A zero-evidence slot is retained and
        # audited rather than swapping in a lower-participation player.
        full_mask = (1 << len(slot_families)) - 1
        states: dict[
            int,
            tuple[
                tuple[float, int],
                tuple[tuple[int, int, int], ...],
            ],
        ] = {0: ((0.0, 0), tuple())}
        for row_index, row in team_candidates.iterrows():
            updated = dict(states)
            for mask, (metric, assignment) in states.items():
                for slot_index, (_slot, family) in enumerate(slot_families):
                    slot_bit = 1 << slot_index
                    if mask & slot_bit:
                        continue
                    evidence = ol_role_evidence(row, family)
                    candidate_metric = (
                        metric[0] + float(row["offense_snaps"]),
                        metric[1] + int(evidence),
                    )
                    candidate_mask = mask | slot_bit
                    current = updated.get(candidate_mask)
                    if current is None or candidate_metric > current[0]:
                        updated[candidate_mask] = (
                            candidate_metric,
                            assignment
                            + ((slot_index, int(row_index), int(evidence)),),
                        )
            states = updated

        best = states.get(full_mask)
        if best is None:
            role_view = team_candidates[
                [
                    "player_name_master", "position_snap",
                    "depth_chart_position", "ol_pff_position",
                    "offense_snaps",
                ]
            ]
            problems.append(
                f"{team}: unable to assign the five selected players to C/G/G/T/T; "
                f"candidates={role_view.to_dict(orient='records')}"
            )
            continue

        for slot_index, row_index, evidence in sorted(best[1]):
            slot, _family = slot_families[slot_index]
            row = team_candidates.iloc[row_index]
            selected_rows.append(
                {
                    "team": team,
                    "starter_slot": slot,
                    "player_name": str(row["player_name_master"]).strip(),
                    "player_id": str(row["player_id"]).strip(),
                    "offense_snaps": float(row["offense_snaps"]),
                    "snap_position": str(row["position_snap"] or "").strip(),
                    "master_position": str(
                        row["depth_chart_position"] or ""
                    ).strip(),
                    "pff_position": str(row["ol_pff_position"] or "").strip(),
                    "role_evidence_score": int(evidence),
                }
            )

    if problems:
        raise RuntimeError(
            f"{target_season} Week 1 OL reconstruction failed:\n"
            + "\n".join(problems)
        )
    selected = pd.DataFrame(selected_rows)
    expected_teams = set(master_ol["team"].dropna().astype(str).unique())
    if len(expected_teams) != 32:
        raise RuntimeError(
            f"{target_season} OL master contains {len(expected_teams)} teams; expected 32."
        )
    expected = {
        (team, slot)
        for team in expected_teams
        for slot in OL_STARTER_SLOTS
    }
    actual = set(zip(selected["team"], selected["starter_slot"]))
    if (
        len(selected) != 160
        or selected["team"].nunique() != 32
        or actual != expected
        or selected.duplicated(["team", "player_id"]).any()
    ):
        raise RuntimeError(
            f"{target_season} Week 1 OL freeze must contain exactly 160 "
            f"unique player/slot rows (32 teams x 5); found {len(selected)} "
            f"rows/{selected['team'].nunique()} teams."
        )

    out = pd.DataFrame(
        {
            "season": target_season,
            "team": selected["team"],
            "starter_slot": selected["starter_slot"],
            "player_name": selected["player_name"],
            "espn_athlete_id": "",
            "source": "week1_snap_top5_roster_prior_pff_position_reconstruction",
            "source_depth_rank": 1,
            "source_injury_status": "",
            "source_timestamp": f"{target_season}-W01",
            "active": 1,
            "updated_at": now_string(),
            "week1_offense_snaps": selected["offense_snaps"],
            "snap_position_evidence": selected["snap_position"],
            "master_position_evidence": selected["master_position"],
            "pff_position_evidence": selected["pff_position"],
            "role_evidence_score": selected["role_evidence_score"],
        }
    ).sort_values(["team", "starter_slot"]).reset_index(drop=True)
    out.to_sql(
        DEPTH_OL_INPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
    )
    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{DEPTH_OL_INPUT_TABLE}_season_team_slot
        ON {DEPTH_OL_INPUT_TABLE}(season, team, starter_slot)
        """
    )
    connection.commit()
    return out


def insert_step_audit(
    connection: sqlite3.Connection,
    record: dict[str, Any],
) -> None:
    pd.DataFrame([record]).to_sql(
        STEP_AUDIT_TABLE,
        connection,
        if_exists="append",
        index=False,
    )
    connection.commit()


def run_child(
    script: RuntimeScript,
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
    if script.key == "projected_depth":
        command.append("--no-refresh-ol-starters")

    print("")
    print("-" * 110)
    print(
        f"[HIST_PERF_DEPTH] Running {script.key}: "
        + subprocess.list2cmdline(command)
    )
    print("-" * 110)

    started = dt.datetime.now()
    started_perf = dt.datetime.now().timestamp()

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
            "NFL_HISTORICAL_PERFORMANCE_DEPTH_RUN_ID": (
                run_id
            ),
        },
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = int(process.wait())
    completed = dt.datetime.now()
    elapsed = completed.timestamp() - started_perf

    insert_step_audit(
        connection,
        {
            "run_id": run_id,
            "step_key": script.key,
            "source_script": str(script.source_path),
            "source_sha256": script.source_sha256,
            "patched_script": str(
                script.patched_path
            ),
            "status": (
                "SUCCESS"
                if return_code == 0
                else "FAILED"
            ),
            "return_code": return_code,
            "started_at": started.isoformat(
                timespec="seconds"
            ),
            "completed_at": completed.isoformat(
                timespec="seconds"
            ),
            "elapsed_seconds": elapsed,
            "command": subprocess.list2cmdline(
                command
            ),
            "wrapper_build_id": BUILD_ID,
            "wrapper_version": VERSION,
        },
    )

    if return_code != 0:
        raise RuntimeError(
            f"{script.key} failed with return code "
            f"{return_code}."
        )


def validate_numeric_range(
    frame: pd.DataFrame,
    column: str,
    minimum: float,
    maximum: float,
    label: str,
) -> None:
    if column not in frame.columns:
        raise RuntimeError(
            f"{label} is missing {column}."
        )
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    if values.isna().any():
        raise RuntimeError(
            f"{label}.{column} contains missing values."
        )
    invalid = values.lt(minimum) | values.gt(maximum)
    if invalid.any():
        raise RuntimeError(
            f"{label}.{column} contains values "
            f"outside {minimum}-{maximum}."
        )


def normalize_names(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(
            r"\b(jr|sr|ii|iii|iv|v)\b",
            "",
            regex=True,
        )
        .str.replace(
            r"[^a-z0-9]",
            "",
            regex=True,
        )
    )


def normalize_historical_ol_names(series: pd.Series) -> pd.Series:
    return normalize_names(series).replace(HISTORICAL_OL_NAME_ALIASES)


def ol_role_family(value: Any) -> str:
    role = str(value or "").upper().strip()
    if role == "C":
        return "C"
    if role in {"G", "OG", "LG", "RG"}:
        return "G"
    if role in {"T", "OT", "LT", "RT"}:
        return "T"
    if role == "OL":
        return "OL"
    return ""


def ol_role_evidence(row: pd.Series, family: str) -> int:
    score = 0
    for column, weight in (
        ("snap_role", 4),
        ("master_role", 3),
        ("pff_role", 3),
    ):
        observed = str(row.get(column, "") or "")
        if observed == family:
            score += weight
        elif observed == "OL":
            score += 1
    return score


def validate_target(
    connection: sqlite3.Connection,
    target_season: int,
    history_seasons: list[int],
    prior_pff_seasons: list[int],
) -> dict[str, Any]:
    context = read_table(
        connection,
        CONTEXT_TABLE,
    )
    master = read_table(
        connection,
        MASTER_TABLE,
    )
    performance = read_table(
        connection,
        PERFORMANCE_TABLE,
    )
    seasonal = read_table(
        connection,
        SEASONAL_PERFORMANCE_TABLE,
    )
    depth = read_table(
        connection,
        DEPTH_TABLE,
    )
    depth_audit = read_table(
        connection,
        DEPTH_AUDIT_TABLE,
    )
    qb_input = read_table(
        connection,
        DEPTH_QB_INPUT_TABLE,
    )
    ol_input = read_table(connection, DEPTH_OL_INPUT_TABLE)
    pff_history = read_table(connection, PFF_OL_HISTORY_TABLE)
    pff_identity = read_table(connection, PFF_OL_IDENTITY_AUDIT_TABLE)

    if len(context) != 1:
        raise RuntimeError(
            f"{target_season} context row count is not one."
        )
    if int(
        context.iloc[0].get(
            "qb_defense_ready_flag", 0
        )
    ) != 1:
        raise RuntimeError(
            f"{target_season} QB/defense stage "
            "was not ready."
        )

    for label, frame in {
        "master": master,
        "performance": performance,
        "depth": depth,
        "depth audit": depth_audit,
    }.items():
        if "player_id" not in frame.columns:
            raise RuntimeError(
                f"{label} is missing player_id."
            )
        if frame["player_id"].duplicated().any():
            raise RuntimeError(
                f"{label} contains duplicate player IDs."
            )

    if len(performance) != len(master):
        raise RuntimeError(
            f"{target_season} performance/master row "
            f"mismatch: performance={len(performance):,}, "
            f"master={len(master):,}."
        )
    if len(depth) != len(performance):
        raise RuntimeError(
            f"{target_season} depth/performance row "
            f"mismatch: depth={len(depth):,}, "
            f"performance={len(performance):,}."
        )
    if len(depth_audit) != len(depth):
        raise RuntimeError(
            f"{target_season} depth audit row mismatch."
        )

    validate_numeric_range(
        performance,
        "performance_input_score",
        0.0,
        100.0,
        "performance",
    )
    validate_numeric_range(
        performance,
        "position_confidence_score",
        0.0,
        1.0,
        "performance",
    )
    validate_numeric_range(
        depth,
        "performance_grade",
        0.0,
        100.0,
        "depth",
    )
    validate_numeric_range(
        depth,
        "unit_quality_grade",
        0.0,
        100.0,
        "depth",
    )

    seasonal_seasons = sorted(
        pd.to_numeric(
            seasonal["season"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if seasonal_seasons != history_seasons:
        raise RuntimeError(
            f"{target_season} seasonal performance "
            f"seasons mismatch. Expected "
            f"{history_seasons}; found "
            f"{seasonal_seasons}."
        )

    teams = sorted(
        depth["team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if len(teams) != 32:
        raise RuntimeError(
            f"{target_season} depth has "
            f"{len(teams)} teams; expected 32."
        )

    qb_authoritative = pd.to_numeric(
        depth["qb_authoritative_starter"],
        errors="coerce",
    ).fillna(0)
    if int(qb_authoritative.sum()) != 32:
        raise RuntimeError(
            f"{target_season} depth contains "
            f"{int(qb_authoritative.sum())} "
            "authoritative QB starters; expected 32."
        )

    ol_authoritative = pd.to_numeric(
        depth.get("ol_authoritative_starter"), errors="coerce"
    ).fillna(0)
    if len(ol_input) != 160 or int(ol_authoritative.sum()) != 160:
        raise RuntimeError(
            f"{target_season} must preserve 160 frozen authoritative OL starters."
        )
    frozen_status_override = pd.to_numeric(
        depth.get("historical_frozen_status_override"), errors="coerce"
    ).fillna(0).eq(1)
    authorized_frozen_starter = qb_authoritative.eq(1) | ol_authoritative.eq(1)
    unauthorized_override = frozen_status_override & ~authorized_frozen_starter
    if unauthorized_override.any():
        raise RuntimeError(
            f"{target_season} contains a historical status override outside "
            "the frozen authoritative QB/OL starter set."
        )
    actual_pff_seasons = sorted(
        pd.to_numeric(pff_history["season"], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    if actual_pff_seasons != prior_pff_seasons:
        raise RuntimeError(
            f"{target_season} PFF history mismatch: expected {prior_pff_seasons}; "
            f"found {actual_pff_seasons}."
        )
    if max(actual_pff_seasons) >= target_season:
        raise RuntimeError(f"{target_season} PFF history contains target/future data.")
    ol_mask = performance["position_group"].astype(str).str.upper().eq("OL")
    pff_available = pd.to_numeric(
        performance.get("ol_pff_history_available"), errors="coerce"
    ).fillna(0).gt(0)
    pff_ol_matches = int((ol_mask & pff_available).sum())
    if pff_ol_matches == 0:
        raise RuntimeError(f"{target_season} has no current-roster OL/PFF matches.")
    latest = pd.to_numeric(
        performance.loc[ol_mask & pff_available, "ol_pff_latest_season"],
        errors="coerce",
    )
    if latest.isna().any() or latest.ge(target_season).any():
        raise RuntimeError(f"{target_season} OL talent uses target/future PFF data.")

    projected_qb = depth[
        depth["canonical_role"].eq("QB")
        & pd.to_numeric(
            depth["is_projected_starter"],
            errors="coerce",
        ).fillna(0).eq(1)
    ].copy()
    if (
        len(projected_qb) != 32
        or projected_qb["team"].nunique() != 32
    ):
        raise RuntimeError(
            f"{target_season} must contain one "
            "projected QB starter per team."
        )

    qb_comparison = qb_input[
        ["team", "player_name"]
    ].rename(
        columns={
            "player_name": "expected_qb_name"
        }
    ).merge(
        projected_qb[
            ["team", "player_name"]
        ].rename(
            columns={
                "player_name": "actual_qb_name"
            }
        ),
        on="team",
        how="outer",
        validate="one_to_one",
    )
    qb_comparison["matched"] = (
        normalize_names(
            qb_comparison["expected_qb_name"]
        )
        == normalize_names(
            qb_comparison["actual_qb_name"]
        )
    )
    if not qb_comparison["matched"].all():
        raise RuntimeError(
            f"{target_season} projected QB mismatch:\n"
            + qb_comparison.loc[
                ~qb_comparison["matched"]
            ].to_string(index=False)
        )

    projected_starters = int(
        pd.to_numeric(
            depth["is_projected_starter"],
            errors="coerce",
        ).fillna(0).sum()
    )
    if projected_starters < 700:
        raise RuntimeError(
            f"{target_season} has only "
            f"{projected_starters} starter "
            "assignments; expected at least 700."
        )

    ol_discount = int(
        pd.to_numeric(
            depth["ol_quality_discount_applied"],
            errors="coerce",
        ).fillna(0).sum()
    )
    usable_history = int(
        pd.to_numeric(
            performance["history_available"],
            errors="coerce",
        ).fillna(0).gt(0).sum()
    )
    usable_grades = int(
        pd.to_numeric(
            performance["usable_performance_grade"],
            errors="coerce",
        ).fillna(0).gt(0).sum()
    )
    pff_collision_rows = int(
        pd.to_numeric(
            pff_history.get(
                "pff_id_collision_flag",
                pd.Series(0, index=pff_history.index),
            ),
            errors="coerce",
        ).fillna(0).eq(1).sum()
    )
    pff_ambiguous_identity_rows = int(
        pff_identity.get(
            "match_status",
            pd.Series("", index=pff_identity.index),
        ).astype(str).eq("ambiguous").sum()
    )
    pff_unmatched_identity_rows = int(
        pff_identity.get(
            "match_status",
            pd.Series("", index=pff_identity.index),
        ).astype(str).eq("unmatched").sum()
    )

    return {
        "target_season": target_season,
        "history_start_season": history_seasons[0],
        "history_end_season": history_seasons[-1],
        "history_seasons": ",".join(
            map(str, history_seasons)
        ),
        "master_rows": len(master),
        "performance_rows": len(performance),
        "seasonal_performance_rows": len(
            seasonal
        ),
        "players_with_history": usable_history,
        "players_with_usable_grade": usable_grades,
        "depth_rows": len(depth),
        "depth_teams": len(teams),
        "projected_starter_assignments": (
            projected_starters
        ),
        "authoritative_qb_starters": int(
            qb_authoritative.sum()
        ),
        "authoritative_ol_starters": int(ol_authoritative.sum()),
        "frozen_starter_status_overrides": int(frozen_status_override.sum()),
        "pff_history_seasons": ",".join(map(str, prior_pff_seasons)),
        "pff_history_rows": len(pff_history),
        "pff_identity_audit_rows": len(pff_identity),
        "pff_id_collision_rows": pff_collision_rows,
        "pff_ambiguous_identity_rows": pff_ambiguous_identity_rows,
        "pff_unmatched_identity_rows": pff_unmatched_identity_rows,
        "pff_current_ol_matches": pff_ol_matches,
        "pff_future_leakage_rows": int(
            pd.to_numeric(pff_history["season"], errors="coerce")
            .ge(target_season).sum()
        ),
        "projected_qb_starters": len(
            projected_qb
        ),
        "qb_starter_matches": int(
            qb_comparison["matched"].sum()
        ),
        "ol_discounted_rows": ol_discount,
        "performance_hash": stable_hash(
            performance,
            [
                "player_id",
                "team",
                "position_group",
                "performance_input_score",
                "position_confidence_score",
            ],
        ),
        "depth_hash": stable_hash(
            depth,
            [
                "player_id",
                "team",
                "canonical_role",
                "depth_rank",
                "is_projected_starter",
                "projected_snap_share",
                "unit_quality_grade",
            ],
        ),
    }


def write_mirror_tables(
    connection: sqlite3.Connection,
    target_season: int,
    history_seasons: list[int],
    prior_pff_seasons: list[int],
) -> None:
    mirrors = {
        PERFORMANCE_TABLE: (
            f"nfl_player_performance_inputs_"
            f"{target_season}"
        ),
        SEASONAL_PERFORMANCE_TABLE: (
            f"nfl_player_performance_seasonal_"
            f"{history_seasons[0]}_"
            f"{history_seasons[-1]}"
        ),
        PERFORMANCE_POSITION_SUMMARY_TABLE: (
            f"nfl_player_performance_position_summary_"
            f"{target_season}"
        ),
        PERFORMANCE_COMPONENT_AUDIT_TABLE: (
            f"nfl_player_performance_component_audit_"
            f"{target_season}"
        ),
        PERFORMANCE_UNRESOLVED_TABLE: (
            f"nfl_player_performance_unresolved_"
            f"master_audit_{target_season}"
        ),
        PFF_OL_HISTORY_TABLE: (
            f"nfl_ol_pff_player_season_{prior_pff_seasons[0]}_"
            f"{prior_pff_seasons[-1]}"
        ),
        PFF_OL_IDENTITY_AUDIT_TABLE: (
            f"nfl_ol_pff_identity_audit_{target_season}"
        ),
        DEPTH_OL_INPUT_TABLE: (
            f"nfl_projected_ol_starters_{target_season}"
        ),
        DEPTH_TABLE: (
            f"nfl_projected_depth_chart_"
            f"{target_season}"
        ),
        DEPTH_AUDIT_TABLE: (
            f"nfl_projected_depth_chart_audit_"
            f"{target_season}"
        ),
    }
    for source, target in mirrors.items():
        frame = read_table(
            connection,
            source,
        )
        frame.to_sql(
            target,
            connection,
            if_exists="replace",
            index=False,
        )


def update_context(
    connection: sqlite3.Connection,
    metrics: dict[str, Any],
) -> None:
    additions = {
        "player_performance_depth_ready_flag": (
            "INTEGER"
        ),
        "historical_performance_rows": "INTEGER",
        "historical_depth_rows": "INTEGER",
        "historical_projected_starters": (
            "INTEGER"
        ),
        "historical_qb_starter_matches": (
            "INTEGER"
        ),
        "historical_performance_hash": "TEXT",
        "historical_depth_hash": "TEXT",
        "player_performance_depth_build_id": (
            "TEXT"
        ),
        "player_performance_depth_version": (
            "TEXT"
        ),
        "player_performance_depth_completed_at": (
            "TEXT"
        ),
        "structural_inputs_pending": "TEXT",
    }
    for column, sql_type in additions.items():
        add_column(
            connection,
            CONTEXT_TABLE,
            column,
            sql_type,
        )

    connection.execute(
        f"""
        UPDATE {CONTEXT_TABLE}
        SET player_performance_depth_ready_flag = 1,
            historical_performance_rows = ?,
            historical_depth_rows = ?,
            historical_projected_starters = ?,
            historical_qb_starter_matches = ?,
            historical_performance_hash = ?,
            historical_depth_hash = ?,
            player_performance_depth_build_id = ?,
            player_performance_depth_version = ?,
            player_performance_depth_completed_at = ?,
            structural_inputs_ready_flag = 0,
            structural_inputs_pending = ?
        """,
        (
            int(metrics["performance_rows"]),
            int(metrics["depth_rows"]),
            int(
                metrics[
                    "projected_starter_assignments"
                ]
            ),
            int(metrics["qb_starter_matches"]),
            metrics["performance_hash"],
            metrics["depth_hash"],
            BUILD_ID,
            VERSION,
            now_string(),
            (
                "historical_ol_continuity,"
                "historical_team_units,"
                "historical_team_strength,"
                "historical_power_ratings"
            ),
        ),
    )


def export_outputs(
    connection: sqlite3.Connection,
    output_root: Path,
    target_season: int,
    no_csv: bool,
) -> None:
    if no_csv:
        return
    output_dir = (
        output_root
        / "historical_performance_depth"
        / str(target_season)
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    for table_name in (
        PERFORMANCE_TABLE,
        PERFORMANCE_POSITION_SUMMARY_TABLE,
        PFF_OL_HISTORY_TABLE,
        PFF_OL_IDENTITY_AUDIT_TABLE,
        DEPTH_TABLE,
        READINESS_TABLE,
    ):
        if table_exists(
            connection,
            table_name,
        ):
            read_table(
                connection,
                table_name,
            ).to_csv(
                output_dir
                / f"{table_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )


def main() -> int:
    args = parse_args()
    started = dt.datetime.now()
    run_id = str(uuid.uuid4())

    print(
        "[HIST_PERF_DEPTH] Building historical "
        "player performance and depth"
    )
    print(
        f"[HIST_PERF_DEPTH] Build ID: {BUILD_ID}"
    )
    print(
        f"[HIST_PERF_DEPTH] Version: {VERSION}"
    )
    print(
        "[HIST_PERF_DEPTH] Target seasons: "
        f"{list(args.target_seasons)}"
    )

    if not args.python.exists():
        raise FileNotFoundError(args.python)
    args.project_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    performance_source = (
        args.project_root
        / "build_nfl_player_performance.py"
    )
    depth_source = (
        args.project_root
        / "build_nfl_projected_depth_chart.py"
    )

    manifest_rows: list[dict[str, Any]] = []

    for target_season in args.target_seasons:
        history_seasons = list(
            range(
                target_season - HISTORY_WINDOW_YEARS,
                target_season,
            )
        )
        prior_pff_seasons = pff_history_seasons(target_season)
        pff_paths = resolve_pff_input_paths(
            args.project_root,
            prior_pff_seasons,
        )
        replacement_anchors = derive_pff_replacement_anchors(pff_paths)
        db_path = args.database_root / f"{target_season}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(db_path)

        print("")
        print("=" * 110)
        print(
            f"[HIST_PERF_DEPTH] Building "
            f"{target_season} from "
            f"{history_seasons} | PFF prior-only={prior_pff_seasons}"
        )
        print("=" * 110)

        runtime_root = (
            args.database_root
            / "runtime"
            / f"{target_season}_performance_depth"
        )
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        scripts_dir = runtime_root / "patched_scripts"
        scripts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (runtime_root / "logs").mkdir(
            parents=True,
            exist_ok=True,
        )
        (runtime_root / "outputs").mkdir(
            parents=True,
            exist_ok=True,
        )
        copy_pff_inputs(pff_paths, runtime_root)

        performance_script = patch_performance_script(
            performance_source,
            scripts_dir
            / "build_nfl_player_performance.py",
            target_season,
            history_seasons,
            prior_pff_seasons,
            replacement_anchors,
            args.allow_source_mismatch,
        )
        depth_script = patch_depth_script(
            depth_source,
            scripts_dir
            / "build_nfl_projected_depth_chart.py",
            target_season,
            args.allow_source_mismatch,
        )

        with sqlite3.connect(
            db_path,
            timeout=60,
        ) as connection:
            connection.execute(
                "PRAGMA busy_timeout = 60000"
            )
            context = read_table(
                connection,
                CONTEXT_TABLE,
            )
            if int(
                context.iloc[0].get(
                    "qb_defense_ready_flag", 0
                )
            ) != 1:
                raise RuntimeError(
                    f"{target_season} QB/defense "
                    "stage is not ready."
                )

            prepare_qb_depth_input(
                connection,
                target_season,
            )
            run_child(
                performance_script,
                args.python,
                runtime_root,
                db_path,
                run_id,
                connection,
            )
            prepare_ol_depth_input(
                connection,
                target_season,
                args.source_cache_db,
            )
            run_child(
                depth_script,
                args.python,
                runtime_root,
                db_path,
                run_id,
                connection,
            )

            metrics = validate_target(
                connection,
                target_season,
                history_seasons,
                prior_pff_seasons,
            )
            metrics.update(
                {
                    "run_id": run_id,
                    "player_performance_depth_ready_flag": 1,
                    "overall_structural_ready_flag": 0,
                    "wrapper_build_id": BUILD_ID,
                    "wrapper_version": VERSION,
                    "created_at": now_string(),
                }
            )
            pd.DataFrame([metrics]).to_sql(
                READINESS_TABLE,
                connection,
                if_exists="replace",
                index=False,
            )
            write_mirror_tables(
                connection,
                target_season,
                history_seasons,
                prior_pff_seasons,
            )
            update_context(
                connection,
                metrics,
            )
            connection.commit()

            export_outputs(
                connection,
                args.output_root,
                target_season,
                args.no_csv,
            )

        manifest_rows.append(metrics)
        print(
            f"[HIST_PERF_DEPTH] {target_season}: "
            f"performance="
            f"{metrics['performance_rows']:,} | "
            f"with_history="
            f"{metrics['players_with_history']:,} | "
            f"depth={metrics['depth_rows']:,} | "
            f"starters="
            f"{metrics['projected_starter_assignments']:,} | "
            f"QB={metrics['qb_starter_matches']}/32 | "
            f"OL={metrics['authoritative_ol_starters']}/160 | "
            f"status_overrides={metrics['frozen_starter_status_overrides']} | "
            f"PFF_matches={metrics['pff_current_ol_matches']} | "
            f"OL_discounted="
            f"{metrics['ol_discounted_rows']:,}"
        )

        if not args.keep_patched_scripts:
            shutil.rmtree(runtime_root)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = (
        args.output_root
        / "nfl_historical_player_performance_depth_manifest.csv"
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if not args.no_csv:
        manifest.to_csv(
            manifest_path,
            index=False,
            encoding="utf-8-sig",
        )

    print("")
    print("=" * 110)
    print("[HIST_PERF_DEPTH] SUMMARY")
    print("=" * 110)
    display = manifest[
        [
            "target_season",
            "history_start_season",
            "history_end_season",
            "performance_rows",
            "players_with_history",
            "depth_rows",
            "projected_starter_assignments",
            "qb_starter_matches",
            "ol_discounted_rows",
            "player_performance_depth_ready_flag",
            "overall_structural_ready_flag",
        ]
    ]
    print(display.to_string(index=False))
    print(
        "[HIST_PERF_DEPTH] Production scripts "
        "modified: NO"
    )
    print(
        "[HIST_PERF_DEPTH] Production database "
        "modified: NO"
    )
    print(
        "[HIST_PERF_DEPTH] Next stage: historical "
        "OL continuity, team units, team strength, "
        "and power ratings."
    )
    print(
        "[HIST_PERF_DEPTH] Completed in "
        f"{(dt.datetime.now() - started).total_seconds():.2f} "
        "seconds"
    )
    if not args.no_csv:
        print(
            f"[HIST_PERF_DEPTH] Manifest: "
            f"{manifest_path}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "[HIST_PERF_DEPTH] Cancelled.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[HIST_PERF_DEPTH] FAILED: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise SystemExit(1)
