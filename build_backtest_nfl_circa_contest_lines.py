#!/usr/bin/env python
"""
Discover, download, OCR, audit, and backtest official Circa Million weekly
contest point-spread boards for the 2020-2025 NFL seasons.

Why this stage exists
---------------------
The previous NFL reconstruction used one nflverse spread per completed game.
That did not establish whether the model could beat the static Circa Million
contest number posted around Thursday morning. This stage targets that exact
market.

Workflow
--------
    plan      Inspect dependencies and expected season/week coverage.
    discover  Find and permanently cache official Circa PDF boards.
    parse     OCR the cached boards, reconcile every line to the NFL schedule,
              and refuse silent partial weeks.
    backtest  Refit the locked canonical matchup-residual architecture against
              the Circa number, enforce QB EPA/CPOE integrity, and evaluate
              both threshold bets and mandatory top-five weekly selections.
    all       Run discover, parse, and backtest.

No paid odds API is used.

Official-source policy
----------------------
Only URLs on www.circasports.com under /wp-content/uploads/ are accepted.
Every downloaded PDF is retained permanently. When multiple official versions
exist, all are parsed and the earliest valid "updated" timestamp is selected.

OCR policy
----------
Circa's historical boards are generally image-only PDFs. The script renders
page 1 with PyMuPDF and runs local Tesseract OCR. It then uses the known weekly
NFL schedule to match team names, locate each team's adjacent spread, enforce
opposite-line symmetry, and reject incomplete or ambiguous weeks.

Required local dependencies
---------------------------
    pip install pymupdf requests joblib scikit-learn pandas numpy

Tesseract must also be installed. On Anaconda Windows:
    conda install -c conda-forge tesseract

Inputs
------
- backtests/nfl_weekly_matchup_residual.sqlite
- nflreadpy/nfl_data_py schedules, or --schedule-path CSV
- optional --manifest-path CSV for manually supplied official PDF URLs
- optional --manual-lines-path CSV for audited line corrections

Outputs
-------
- backtests/nfl_circa_contest_lines.sqlite
- backtests/nfl_circa_contest_pdfs/
- backtests/nfl_circa_contest_ocr/
- outputs/historical_weekly_replay/circa_contest_lines/
- models/nfl_circa_contest_model_v1.joblib
- models/nfl_circa_contest_model_v1_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUILD_ID = "NFL_CIRCA_CONTEST_LINES_CANONICAL_V1"
VERSION = "v1_5_qb_identity_integrity_guard"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_SEASONS = tuple(range(2020, 2026))
DEVELOPMENT_SEASON = 2020
VALIDATION_SEASONS = (2021, 2022, 2023)
BENCHMARK_SEASONS = (2024, 2025)

MATCHUP_DATABASE_NAME = "nfl_weekly_matchup_residual.sqlite"
MATCHUP_MATRIX_TABLE = "nfl_matchup_game_matrix"
MATCHUP_AUDIT_TABLE = "nfl_matchup_run_audit"
MATCHUP_QB_AUDIT_TABLE = "nfl_matchup_qb_feature_integrity_audit"
OUTPUT_DATABASE_NAME = "nfl_circa_contest_lines.sqlite"
PDF_DIRECTORY_NAME = "nfl_circa_contest_pdfs"
OCR_DIRECTORY_NAME = "nfl_circa_contest_ocr"
MODEL_FILENAME = "nfl_circa_contest_model_v1.joblib"
METADATA_FILENAME = "nfl_circa_contest_model_v1_metadata.json"
OUTPUT_DIRECTORY = Path(
    "outputs/historical_weekly_replay/circa_contest_lines"
)

MANIFEST_TABLE = "nfl_circa_pdf_manifest"
OCR_ATTEMPT_TABLE = "nfl_circa_ocr_attempts"
TEAM_LINE_TABLE = "nfl_circa_team_lines"
GAME_LINE_TABLE = "nfl_circa_game_lines"
WEEK_AUDIT_TABLE = "nfl_circa_week_audit"
BACKTEST_MATRIX_TABLE = "nfl_circa_backtest_matrix"
OOF_PREDICTION_TABLE = "nfl_circa_oof_predictions"
THRESHOLD_VALIDATION_TABLE = "nfl_circa_threshold_validation"
TOP5_VALIDATION_TABLE = "nfl_circa_top5_validation"
BENCHMARK_PREDICTION_TABLE = "nfl_circa_benchmark_predictions"
BENCHMARK_SUMMARY_TABLE = "nfl_circa_benchmark_summary"
COEFFICIENT_TABLE = "nfl_circa_model_coefficients"
RUN_AUDIT_TABLE = "nfl_circa_run_audit"

CIRCA_HOST = "www.circasports.com"
CIRCA_UPLOAD_PREFIX = "https://www.circasports.com/wp-content/uploads/"
CIRCA_MEDIA_API = "https://www.circasports.com/wp-json/wp/v2/media"
CIRCA_SITEMAP = "https://www.circasports.com/wp-sitemap.xml"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/150 Safari/537.36"
)

VERIFIED_SEED_URLS = (
    "https://www.circasports.com/wp-content/uploads/2020/10/Circa-Sports-Million-II-Week-7-Contest-Point-Spreads-2.pdf",
    "https://www.circasports.com/wp-content/uploads/2021/09/Circa-Sports-Million-III-Week-1-Contest-Point-Spreads.pdf",
    "https://www.circasports.com/wp-content/uploads/2022/09/Circa-Sports-Million-IV-Contest-Point-Spreads-Week-1.pdf",
    "https://www.circasports.com/wp-content/uploads/2023/09/Circa-Sports-Million-V-Contest-Point-Spreads-Week-1.pdf",
    "https://www.circasports.com/wp-content/uploads/2024/09/Circa-Sports-Million-VI-Contest-Point-Spreads-Week-1.pdf",
    "https://www.circasports.com/wp-content/uploads/2025/10/Circa-Sports-Million-VII-Contest-Point-Spreads-Week-7.pdf",
)

SEASON_ROMAN = {
    2020: "II",
    2021: "III",
    2022: "IV",
    2023: "V",
    2024: "VI",
    2025: "VII",
}

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}

CIRCA_TEAM_NAMES = {
    "ARI": ("CARDINALS", "CARDS"),
    "ATL": ("FALCONS",),
    "BAL": ("RAVENS",),
    "BUF": ("BILLS",),
    "CAR": ("PANTHERS",),
    "CHI": ("BEARS",),
    "CIN": ("BENGALS",),
    "CLE": ("BROWNS",),
    "DAL": ("COWBOYS",),
    "DEN": ("BRONCOS",),
    "DET": ("LIONS",),
    "GB": ("PACKERS",),
    "HOU": ("TEXANS",),
    "IND": ("COLTS",),
    "JAX": ("JAGUARS", "JAGS"),
    "KC": ("CHIEFS",),
    "LV": ("RAIDERS",),
    "LAC": ("CHARGERS",),
    "LAR": ("RAMS",),
    "MIA": ("DOLPHINS",),
    "MIN": ("VIKINGS",),
    "NE": ("PATRIOTS", "PATS"),
    "NO": ("SAINTS",),
    "NYG": ("GIANTS",),
    "NYJ": ("JETS",),
    "PHI": ("EAGLES",),
    "PIT": ("STEELERS",),
    "SF": ("49ERS", "NINERS", "A9ERS"),
    "SEA": ("SEAHAWKS", "HAWKS"),
    "TB": ("BUCS", "BUCCANEERS"),
    "TEN": ("TITANS",),
    "WAS": ("WASHINGTON", "COMMANDERS", "REDSKINS"),
}

MARKET_CONTROL_FEATURES = (
    "market_home_margin",
    "absolute_market_home_margin",
)

FEATURE_SETS = {
    "MARKET_CONTROL": MARKET_CONTROL_FEATURES,
    "PASS_COMPACT": (
        "pass_epa_advantage",
        "sack_advantage",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
        "qb_week1_stability_advantage",
        "qb_snap_share_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "RUSH_TRENCH": (
        "rush_epa_advantage",
        "early_down_epa_advantage",
        "ol_continuity_advantage",
        "ol_stability_advantage",
        "core_ol_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "PERSONNEL_QB": (
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
        "qb_week1_stability_advantage",
        "qb_last_game_stability_advantage",
        "qb_snap_share_advantage",
        "offense_continuity_advantage",
        "ol_continuity_advantage",
        "core_offense_health_advantage",
        "core_ol_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
    "FULL_COMPACT": (
        "pass_epa_advantage",
        "rush_epa_advantage",
        "sack_advantage",
        "turnover_advantage",
        "explosive_rate_advantage",
        "special_teams_advantage",
        "qb_recent_epa_advantage",
        "qb_week1_stability_advantage",
        "ol_continuity_advantage",
        "core_offense_health_advantage",
        "core_defense_health_advantage",
        *MARKET_CONTROL_FEATURES,
    ),
}

DEFAULT_THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
DEFAULT_ATS_PRICE = -110.0
DEFAULT_OCR_DPI = 300
DEFAULT_TESSERACT_PSMS = (3, 6, 11, 12)


# =============================================================================
# ARGUMENTS AND GENERIC HELPERS
# =============================================================================


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in value.split(",") if x.strip()}))
    if not values:
        raise argparse.ArgumentTypeError("At least one number is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "discover", "parse", "audit", "backtest", "all"),
        default="plan",
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--schedule-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--manual-lines-path", type=Path, default=None)
    parser.add_argument("--seasons", type=parse_int_list, default=DEFAULT_SEASONS)
    parser.add_argument("--tesseract-path", type=Path, default=None)
    parser.add_argument("--ocr-dpi", type=int, default=DEFAULT_OCR_DPI)
    parser.add_argument("--ocr-psms", type=parse_int_list, default=DEFAULT_TESSERACT_PSMS)
    parser.add_argument("--thresholds", type=parse_float_list, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--fallback-ats-price", type=float, default=DEFAULT_ATS_PRICE)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--request-pause", type=float, default=0.08)
    parser.add_argument("--probe-missing", action="store_true")
    parser.add_argument("--rebuild-downloads", action="store_true")
    parser.add_argument("--rebuild-ocr", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--backfill-missing-2025-with-reference",
        action="store_true",
        help=(
            "Fill only missing 2025 contest lines with the nflverse final "
            "spread. Fallback rows are explicitly labeled, excluded from "
            "strict Circa benchmark gates, and excluded from model fitting."
        ),
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    for name in ("schedule_path", "manifest_path", "manual_lines_path", "tesseract_path"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    unsupported = sorted(set(args.seasons) - set(SEASON_ROMAN))
    if unsupported:
        parser.error(f"Unsupported Circa archive seasons: {unsupported}")
    if args.ocr_dpi < 200 or args.ocr_dpi > 600:
        parser.error("--ocr-dpi must be between 200 and 600.")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive.")
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def normalize_team(value: Any) -> str:
    text = "" if value is None else str(value).upper().strip()
    return TEAM_ALIASES.get(text, text)


def normalize_token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", "" if value is None else str(value).upper())


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(column).lower().strip(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone() is not None


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        return pd.DataFrame()
    escaped = table_name.replace('"', '""')
    frame = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    return frame


def numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def frame_from_any(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def official_circa_pdf_url(url: str) -> bool:
    parsed = urlparse(str(url))
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() == CIRCA_HOST
        and parsed.path.startswith("/wp-content/uploads/")
        and parsed.path.lower().endswith(".pdf")
        and "million" in parsed.path.lower()
        and "contest-point-spreads" in parsed.path.lower()
    )


# =============================================================================
# SCHEDULE
# =============================================================================


def load_schedule_from_package(seasons: list[int]) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    try:
        import nflreadpy  # type: ignore
        loader = getattr(nflreadpy, "load_schedules", None)
        if loader is not None:
            for attempt in (lambda: loader(seasons), lambda: loader(seasons=seasons)):
                try:
                    frame = frame_from_any(attempt())
                    if not frame.empty:
                        return frame, "nflreadpy.load_schedules"
                except Exception as exc:
                    errors.append(f"nflreadpy.load_schedules: {exc}")
    except Exception as exc:
        errors.append(f"nflreadpy import: {exc}")

    try:
        import nfl_data_py  # type: ignore
        loader = getattr(nfl_data_py, "import_schedules", None)
        if loader is not None:
            for attempt in (lambda: loader(seasons), lambda: loader(seasons=seasons)):
                try:
                    frame = frame_from_any(attempt())
                    if not frame.empty:
                        return frame, "nfl_data_py.import_schedules"
                except Exception as exc:
                    errors.append(f"nfl_data_py.import_schedules: {exc}")
    except Exception as exc:
        errors.append(f"nfl_data_py import: {exc}")

    raise RuntimeError("Unable to load NFL schedules: " + " | ".join(errors))


def standardize_schedule(raw: pd.DataFrame, seasons: tuple[int, ...]) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    aliases = {
        "season": ("season",),
        "week": ("week",),
        "game_id": ("game_id",),
        "game_type": ("game_type", "season_type"),
        "home_team": ("home_team",),
        "away_team": ("away_team",),
        "gameday": ("gameday", "game_date", "date"),
        "home_score": ("home_score",),
        "away_score": ("away_score",),
        "spread_line": ("spread_line", "market_home_margin"),
    }
    output = pd.DataFrame(index=frame.index)
    for target, candidates in aliases.items():
        source = first_existing(frame.columns, candidates)
        output[target] = frame[source] if source is not None else np.nan

    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output["week"] = pd.to_numeric(output["week"], errors="coerce")
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output["spread_line"] = pd.to_numeric(output["spread_line"], errors="coerce")
    output["home_team"] = output["home_team"].map(normalize_team)
    output["away_team"] = output["away_team"].map(normalize_team)
    output["gameday"] = pd.to_datetime(output["gameday"], errors="coerce")

    regular = (
        output["game_type"].fillna("REG").astype(str).str.upper().str.strip().isin(
            {"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"}
        )
    )
    output = output[
        regular
        & output["season"].isin(seasons)
        & output["week"].between(1, 18)
        & output["game_id"].notna()
        & output["home_team"].ne("")
        & output["away_team"].ne("")
    ].copy()
    output[["season", "week"]] = output[["season", "week"]].astype(int)
    output["game_id"] = output["game_id"].astype(str)
    output["actual_home_margin"] = output["home_score"] - output["away_score"]
    return output.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def load_schedule(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.schedule_path is not None:
        raw = pd.read_csv(args.schedule_path, low_memory=False)
        source = str(args.schedule_path)
    else:
        raw, source = load_schedule_from_package(list(args.seasons))
    return standardize_schedule(raw, args.seasons), source


def expected_week_table(schedule: pd.DataFrame) -> pd.DataFrame:
    return (
        schedule.groupby(["season", "week"], as_index=False)
        .agg(
            scheduled_games=("game_id", "nunique"),
            first_game_date=("gameday", "min"),
            last_game_date=("gameday", "max"),
        )
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )


# =============================================================================
# PDF DISCOVERY AND DOWNLOAD
# =============================================================================


def infer_season_week_from_url(url: str) -> tuple[Optional[int], Optional[int]]:
    lower = url.lower()
    season = None
    for candidate, roman in SEASON_ROMAN.items():
        if re.search(rf"million[-_ ]*{roman.lower()}(?:[-_./]|$)", lower):
            season = candidate
            break
    week_match = re.search(r"week[-_ ]*(\d{1,2})", lower)
    week = int(week_match.group(1)) if week_match else None
    return season, week


def discover_via_media_api(session: requests.Session, timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search_term in (
        "Contest Point Spreads",
        "Million Contest Point Spreads",
        "Circa Sports Million",
    ):
        for page in range(1, 8):
            try:
                response = session.get(
                    CIRCA_MEDIA_API,
                    params={"search": search_term, "per_page": 100, "page": page},
                    timeout=timeout,
                )
                if response.status_code == 400 and page > 1:
                    break
                response.raise_for_status()
                payload = response.json()
                if not payload:
                    break
                for item in payload:
                    url = str(item.get("source_url") or "")
                    if not official_circa_pdf_url(url):
                        continue
                    season, week = infer_season_week_from_url(url)
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "url": url,
                            "discovery_method": "WORDPRESS_MEDIA_API",
                            "wordpress_media_id": item.get("id"),
                            "wordpress_date": item.get("date_gmt") or item.get("date"),
                        }
                    )
                total_pages = int(response.headers.get("X-WP-TotalPages", page))
                if page >= total_pages:
                    break
            except Exception:
                break
    return rows


def discover_via_sitemap(session: requests.Session, timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        root = session.get(CIRCA_SITEMAP, timeout=timeout)
        root.raise_for_status()
        root_xml = ET.fromstring(root.content)
    except Exception:
        return rows

    namespaces = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    sitemap_urls = [
        element.text.strip()
        for element in root_xml.findall(".//sm:loc", namespaces)
        if element.text
    ]
    for sitemap_url in sitemap_urls:
        if not any(token in sitemap_url.lower() for token in ("attachment", "media", "post")):
            continue
        try:
            response = session.get(sitemap_url, timeout=timeout)
            response.raise_for_status()
            xml = ET.fromstring(response.content)
        except Exception:
            continue
        for element in xml.findall(".//sm:loc", namespaces):
            url = (element.text or "").strip()
            if not official_circa_pdf_url(url):
                continue
            season, week = infer_season_week_from_url(url)
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "url": url,
                    "discovery_method": "WORDPRESS_SITEMAP",
                }
            )
    return rows


def candidate_months_for_week(week_schedule: pd.DataFrame) -> list[tuple[int, int]]:
    dates = pd.to_datetime(week_schedule["gameday"], errors="coerce").dropna()
    if dates.empty:
        return []
    representative = dates.median()
    candidates = {
        (int((representative + pd.Timedelta(days=offset)).year),
         int((representative + pd.Timedelta(days=offset)).month))
        for offset in (-10, -7, -4, 0, 4)
    }
    return sorted(candidates)


def candidate_filenames(season: int, week: int, week_schedule: pd.DataFrame) -> list[str]:
    roman = SEASON_ROMAN[season]
    bases = [
        f"Circa-Sports-Million-{roman}-Week-{week}-Contest-Point-Spreads",
        f"Circa-Sports-Million-{roman}-Contest-Point-Spreads-Week-{week}",
        f"Circa-Sports-Million-{roman}-Week-{week}-Contest-Spreads",
        f"Circa-Sports-Million-{roman}-Contest-Spreads-Week-{week}",
    ]
    filenames = []
    for base in bases:
        filenames.append(base + ".pdf")
        for suffix in ("-1", "-2", "-3"):
            filenames.append(base + suffix + ".pdf")

    dates = pd.to_datetime(week_schedule["gameday"], errors="coerce").dropna()
    if not dates.empty:
        main_date = dates.mode().iloc[0] if not dates.mode().empty else dates.min()
        for days_before in (3, 4, 5):
            posted = main_date - pd.Timedelta(days=days_before)
            stamp = posted.strftime("%Y-%m-%d")
            for base in bases:
                for suffix in ("", "a", "b", "c"):
                    filenames.append(f"{base}-{stamp}{suffix}.pdf")
    return list(dict.fromkeys(filenames))


def probe_candidate_urls(
    session: requests.Session,
    schedule: pd.DataFrame,
    missing_pairs: set[tuple[int, int]],
    timeout: float,
    pause: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for season, week in sorted(missing_pairs):
        week_schedule = schedule[
            schedule["season"].eq(season) & schedule["week"].eq(week)
        ]
        found = 0
        for year, month in candidate_months_for_week(week_schedule):
            for filename in candidate_filenames(season, week, week_schedule):
                url = f"{CIRCA_UPLOAD_PREFIX}{year:04d}/{month:02d}/{filename}"
                try:
                    response = session.get(
                        url,
                        headers={"Range": "bytes=0-2047"},
                        timeout=timeout,
                        allow_redirects=True,
                    )
                    content_type = response.headers.get("Content-Type", "").lower()
                    looks_pdf = (
                        response.status_code in {200, 206}
                        and (
                            "application/pdf" in content_type
                            or response.content.startswith(b"%PDF")
                        )
                    )
                    if looks_pdf:
                        rows.append(
                            {
                                "season": season,
                                "week": week,
                                "url": response.url,
                                "discovery_method": "OFFICIAL_URL_PATTERN_PROBE",
                            }
                        )
                        found += 1
                        if found >= 5:
                            break
                except Exception:
                    pass
                if pause:
                    time.sleep(pause)
            if found >= 5:
                break
    return rows


def load_manual_manifest(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    if "url" not in frame.columns:
        raise RuntimeError("Manual manifest must include a url column.")
    rows = []
    for row in frame.to_dict("records"):
        url = str(row.get("url") or "")
        if not official_circa_pdf_url(url):
            raise RuntimeError(f"Non-official or invalid Circa PDF URL: {url}")
        season, week = infer_season_week_from_url(url)
        rows.append(
            {
                "season": int(row.get("season") or season),
                "week": int(row.get("week") or week),
                "url": url,
                "discovery_method": "MANUAL_OFFICIAL_MANIFEST",
            }
        )
    return rows


def safe_pdf_filename(season: int, week: int, url: str, sequence: int) -> str:
    original = Path(urlparse(url).path).name
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", original)
    return f"{season}_W{week:02d}_{sequence:02d}_{stem}"


def download_manifest(
    session: requests.Session,
    manifest: pd.DataFrame,
    pdf_directory: Path,
    timeout: float,
    rebuild: bool,
) -> pd.DataFrame:
    pdf_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for (season, week), group in manifest.groupby(["season", "week"], sort=True):
        for sequence, source in enumerate(group.itertuples(index=False), start=1):
            local_path = pdf_directory / safe_pdf_filename(
                int(season), int(week), str(source.url), sequence
            )
            status = "CACHED"
            error = None
            if rebuild or not local_path.exists():
                try:
                    response = session.get(str(source.url), timeout=timeout)
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF"):
                        raise RuntimeError("Response was not a PDF.")
                    local_path.write_bytes(response.content)
                    status = "DOWNLOADED"
                except Exception as exc:
                    status = "FAILED"
                    error = str(exc)
            rows.append(
                {
                    **source._asdict(),
                    "local_path": str(local_path),
                    "download_status": status,
                    "download_error": error,
                    "file_size_bytes": local_path.stat().st_size if local_path.exists() else 0,
                    "sha256": sha256_file(local_path) if local_path.exists() else None,
                    "downloaded_at": now_string(),
                }
            )
    return pd.DataFrame(rows)


def discover_and_download(
    args: argparse.Namespace,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows = []
    rows.extend(load_manual_manifest(args.manifest_path))
    for url in VERIFIED_SEED_URLS:
        season, week = infer_season_week_from_url(url)
        rows.append(
            {
                "season": season,
                "week": week,
                "url": url,
                "discovery_method": "BUILTIN_VERIFIED_OFFICIAL_SEED",
            }
        )
    rows.extend(discover_via_media_api(session, args.request_timeout))
    rows.extend(discover_via_sitemap(session, args.request_timeout))

    discovered = pd.DataFrame(rows)
    if discovered.empty:
        discovered = pd.DataFrame(
            columns=["season", "week", "url", "discovery_method"]
        )
    discovered = discovered[
        discovered["season"].isin(args.seasons)
        & pd.to_numeric(discovered["week"], errors="coerce").between(1, 18)
    ].copy()
    discovered[["season", "week"]] = discovered[["season", "week"]].astype(int)
    discovered = discovered.drop_duplicates("url").reset_index(drop=True)

    expected_pairs = set(
        expected_week_table(schedule)[["season", "week"]].itertuples(index=False, name=None)
    )
    found_pairs = set(
        discovered[["season", "week"]].itertuples(index=False, name=None)
    )
    missing_pairs = expected_pairs - found_pairs

    if args.probe_missing and missing_pairs:
        probed = probe_candidate_urls(
            session,
            schedule,
            missing_pairs,
            args.request_timeout,
            args.request_pause,
        )
        if probed:
            discovered = pd.concat([discovered, pd.DataFrame(probed)], ignore_index=True)
            discovered = discovered.drop_duplicates("url").reset_index(drop=True)

    return download_manifest(
        session,
        discovered,
        args.project_root / "backtests" / PDF_DIRECTORY_NAME,
        args.request_timeout,
        args.rebuild_downloads,
    )


# =============================================================================
# OCR AND LINE RECONCILIATION
# =============================================================================


def find_tesseract(explicit: Optional[Path]) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    detected = shutil.which("tesseract")
    if detected:
        candidates.append(Path(detected))
    candidates.extend(
        [
            Path(sys.prefix) / "Library" / "bin" / "tesseract.exe",
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        "Tesseract was not found. Install it with "
        "`conda install -c conda-forge tesseract` or pass --tesseract-path."
    )


def render_pdf_first_page(pdf_path: Path, png_path: Path, dpi: int) -> None:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf") from exc
    document = fitz.open(str(pdf_path))
    if document.page_count < 1:
        raise RuntimeError(f"PDF has no pages: {pdf_path}")
    page = document.load_page(0)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(str(png_path))
    document.close()


def run_tesseract_tsv(
    tesseract_path: Path,
    image_path: Path,
    output_base: Path,
    psm: int,
) -> tuple[pd.DataFrame, str]:
    command = [
        str(tesseract_path),
        str(image_path),
        str(output_base),
        "--psm",
        str(psm),
        "tsv",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Tesseract failed ({completed.returncode}): {completed.stderr[-1000:]}"
        )
    tsv_path = output_base.with_suffix(".tsv")
    frame = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    frame["text"] = frame["text"].fillna("").astype(str)
    words = frame[frame["level"].eq(5) & frame["text"].str.strip().ne("")].copy()
    line_text = (
        words.sort_values(["block_num", "par_num", "line_num", "left"])
        .groupby(["block_num", "par_num", "line_num"])["text"]
        .apply(lambda series: " ".join(series.astype(str)))
        .tolist()
    )
    return words.reset_index(drop=True), "\n".join(line_text)


def alias_similarity(token: str, alias: str) -> float:
    token_norm = normalize_token(token)
    alias_norm = normalize_token(alias)
    if token_norm == alias_norm:
        return 1.0
    return difflib.SequenceMatcher(None, token_norm, alias_norm).ratio()


def team_word_candidates(
    words: pd.DataFrame,
    expected_teams: list[str],
) -> list[dict[str, Any]]:
    candidates = []
    for word_index, word in words.iterrows():
        token = str(word["text"])
        normalized = normalize_token(token)
        if len(normalized) < 3 or re.search(r"[+-]", token):
            continue
        for team in expected_teams:
            aliases = CIRCA_TEAM_NAMES.get(team, (team,))
            score = max(alias_similarity(token, alias) for alias in aliases)
            if score >= 0.68:
                candidates.append(
                    {
                        "team": team,
                        "word_index": int(word_index),
                        "team_ocr_text": token,
                        "team_match_score": float(score),
                        "left": float(word["left"]),
                        "top": float(word["top"]),
                        "width": float(word["width"]),
                        "height": float(word["height"]),
                        "conf": float(word.get("conf", np.nan)),
                    }
                )
    return sorted(candidates, key=lambda row: row["team_match_score"], reverse=True)


def assign_unique_team_words(
    words: pd.DataFrame,
    expected_teams: list[str],
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    used_word_indices: set[int] = set()
    for candidate in team_word_candidates(words, expected_teams):
        if candidate["team"] in assignments:
            continue
        if candidate["word_index"] in used_word_indices:
            continue
        assignments[candidate["team"]] = candidate
        used_word_indices.add(candidate["word_index"])
    return assignments


def looks_like_odds_token(text: str) -> bool:
    upper = str(text).upper().strip()
    return (
        "PK" in upper
        or "PICK" in upper
        or bool(re.search(r"[+\-−–—]", upper))
    )


def parse_odds_token(raw: Any) -> Optional[float]:
    text = "" if raw is None else str(raw).strip().upper()
    if not text:
        return None
    if "PK" in text or "PICK" in text:
        return 0.0

    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    sign_match = re.search(r"([+-])", text)
    if not sign_match:
        return None
    sign = 1.0 if sign_match.group(1) == "+" else -1.0
    body = text[sign_match.end():]
    body = body.replace("O", "0").replace("I", "1").replace("L", "1")
    body = re.sub(r"(?<=^)[A](?=[^0-9]*$)", "4", body)

    explicit_half = any(
        marker in body
        for marker in ("½", "1/2", ".5", "Y2", "V2")
    )
    digits = re.findall(r"\d+", body)
    if not digits:
        # Common OCR failure: 9½ becomes %% or %%.
        return None

    number_text = digits[0]
    value = float(number_text)
    if ".5" in body:
        value = float(re.search(r"\d+(?:\.\d+)?", body).group(0))
    else:
        clean_integer = re.fullmatch(r"\d+", body.strip()) is not None
        trailing_junk = re.sub(r"\d", "", body).strip()
        half_like_junk = bool(trailing_junk) and any(
            char in trailing_junk
            for char in ("%", "¥", ")", "(", "Y", "V", ",", "'", '"', "½", "/")
        )
        if explicit_half or (not clean_integer and half_like_junk):
            value += 0.5
    if value > 30:
        return None
    return sign * value


def nearest_odds_word(
    words: pd.DataFrame,
    team_assignment: dict[str, Any],
) -> tuple[Optional[str], Optional[float], Optional[int]]:
    team_center_y = team_assignment["top"] + team_assignment["height"] / 2.0
    team_right = team_assignment["left"] + team_assignment["width"]
    candidates = []
    for word_index, word in words.iterrows():
        text = str(word["text"])
        if not looks_like_odds_token(text):
            continue
        center_y = float(word["top"]) + float(word["height"]) / 2.0
        y_distance = abs(center_y - team_center_y)
        x_distance = float(word["left"]) - team_right
        if y_distance <= max(35.0, team_assignment["height"] * 0.85) and -20 <= x_distance <= 1200:
            parsed = parse_odds_token(text)
            candidates.append(
                (
                    y_distance * 4.0 + max(x_distance, 0.0),
                    x_distance,
                    int(word_index),
                    text,
                    parsed,
                )
            )
    if not candidates:
        return None, None, None
    chosen = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return chosen[3], chosen[4], chosen[2]


def extract_updated_timestamp(text: str, season: int) -> Optional[str]:
    normalized = text.replace("|", " ")
    match = re.search(
        r"updated\s+(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2})\s*([AP]M)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month, day, year, hour, minute, ampm = match.groups()
    year_value = int(year)
    if year_value < 100:
        year_value += 2000
    hour_value = int(hour) % 12 + (12 if ampm.upper() == "PM" else 0)
    try:
        timestamp = dt.datetime(
            year_value,
            int(month),
            int(day),
            hour_value,
            int(minute),
        )
    except ValueError:
        return None
    return timestamp.isoformat(timespec="minutes")


def parse_ocr_attempt(
    words: pd.DataFrame,
    full_text: str,
    season: int,
    week: int,
    week_schedule: pd.DataFrame,
    psm: int,
    source_pdf: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_teams = sorted(
        set(week_schedule["home_team"]) | set(week_schedule["away_team"])
    )
    assignments = assign_unique_team_words(words, expected_teams)
    team_rows = []
    for team in expected_teams:
        assignment = assignments.get(team)
        if assignment is None:
            team_rows.append(
                {
                    "season": season,
                    "week": week,
                    "team": team,
                    "team_found": 0,
                    "team_ocr_text": None,
                    "team_match_score": np.nan,
                    "spread_ocr_text": None,
                    "team_spread": np.nan,
                    "psm": psm,
                    "source_pdf": str(source_pdf),
                }
            )
            continue
        spread_text, spread_value, odds_word_index = nearest_odds_word(words, assignment)
        team_rows.append(
            {
                "season": season,
                "week": week,
                "team": team,
                "team_found": 1,
                "team_ocr_text": assignment["team_ocr_text"],
                "team_match_score": assignment["team_match_score"],
                "spread_ocr_text": spread_text,
                "team_spread": spread_value,
                "team_word_index": assignment["word_index"],
                "odds_word_index": odds_word_index,
                "team_left": assignment["left"],
                "team_top": assignment["top"],
                "psm": psm,
                "source_pdf": str(source_pdf),
            }
        )

    team_frame = pd.DataFrame(team_rows)
    lookup = team_frame.set_index("team")
    game_rows = []
    for game in week_schedule.itertuples(index=False):
        home_raw = lookup.loc[game.home_team, "team_spread"] if game.home_team in lookup.index else np.nan
        away_raw = lookup.loc[game.away_team, "team_spread"] if game.away_team in lookup.index else np.nan
        home_spread = float(home_raw) if pd.notna(home_raw) else np.nan
        away_spread = float(away_raw) if pd.notna(away_raw) else np.nan
        derived_side = None
        if not np.isfinite(home_spread) and np.isfinite(away_spread):
            home_spread = -away_spread
            derived_side = "HOME_FROM_AWAY"
        elif not np.isfinite(away_spread) and np.isfinite(home_spread):
            away_spread = -home_spread
            derived_side = "AWAY_FROM_HOME"
        symmetry_error = (
            abs(home_spread + away_spread)
            if np.isfinite(home_spread) and np.isfinite(away_spread)
            else np.nan
        )
        valid = int(
            np.isfinite(home_spread)
            and np.isfinite(away_spread)
            and symmetry_error <= 0.26
        )
        game_rows.append(
            {
                "season": season,
                "week": week,
                "game_id": str(game.game_id),
                "home_team": str(game.home_team),
                "away_team": str(game.away_team),
                "home_team_spread": home_spread,
                "away_team_spread": away_spread,
                "circa_home_margin": -home_spread if np.isfinite(home_spread) else np.nan,
                "symmetry_error": symmetry_error,
                "derived_side": derived_side,
                "line_valid": valid,
                "psm": psm,
                "source_pdf": str(source_pdf),
            }
        )

    game_frame = pd.DataFrame(game_rows)
    matched_games = int(game_frame["line_valid"].sum())
    missing_games = int(len(game_frame) - matched_games)
    quality = {
        "season": season,
        "week": week,
        "psm": psm,
        "source_pdf": str(source_pdf),
        "expected_games": int(len(game_frame)),
        "matched_games": matched_games,
        "missing_games": missing_games,
        "teams_found": int(team_frame["team_found"].sum()),
        "teams_with_spread": int(team_frame["team_spread"].notna().sum()),
        "average_team_match_score": float(team_frame["team_match_score"].mean()),
        "maximum_symmetry_error": float(game_frame["symmetry_error"].max()) if game_frame["symmetry_error"].notna().any() else np.nan,
        "updated_timestamp": extract_updated_timestamp(full_text, season),
        "full_text": full_text,
    }
    return team_frame, game_frame, quality


def ocr_pdf_versions(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tesseract = find_tesseract(args.tesseract_path)
    ocr_directory = args.project_root / "backtests" / OCR_DIRECTORY_NAME
    ocr_directory.mkdir(parents=True, exist_ok=True)

    all_attempts = []
    all_team_lines = []
    all_game_lines = []

    successful_manifest = manifest[
        manifest["download_status"].isin(["DOWNLOADED", "CACHED"])
    ].copy()

    # Discovery can encounter the same official PDF through both the sitemap
    # and a filename probe.  The SHA-256 identifies the actual file contents,
    # so OCR each season/week/file combination only once.  Without this guard,
    # the same attempt_id and every parsed game row could be appended twice.
    manifest_dedup_keys = ["season", "week", "sha256"]
    missing_manifest_keys = sorted(
        set(manifest_dedup_keys) - set(successful_manifest.columns)
    )
    if missing_manifest_keys:
        raise RuntimeError(
            "Downloaded PDF manifest is missing deduplication columns: "
            f"{missing_manifest_keys}"
        )
    successful_manifest = (
        successful_manifest
        .sort_values(
            ["season", "week", "sha256", "url"],
            na_position="last",
        )
        .drop_duplicates(manifest_dedup_keys, keep="first")
        .reset_index(drop=True)
    )

    for source in successful_manifest.itertuples(index=False):
        pdf_path = Path(str(source.local_path))
        if not pdf_path.exists():
            continue
        week_schedule = schedule[
            schedule["season"].eq(int(source.season))
            & schedule["week"].eq(int(source.week))
        ].copy()
        if week_schedule.empty:
            continue

        cache_stem = f"{int(source.season)}_W{int(source.week):02d}_{source.sha256[:12]}"
        image_path = ocr_directory / f"{cache_stem}_{args.ocr_dpi}dpi.png"
        if args.rebuild_ocr or not image_path.exists():
            render_pdf_first_page(pdf_path, image_path, args.ocr_dpi)

        for psm in args.ocr_psms:
            output_base = ocr_directory / f"{cache_stem}_psm{psm}"
            tsv_path = output_base.with_suffix(".tsv")
            if args.rebuild_ocr or not tsv_path.exists():
                words, full_text = run_tesseract_tsv(
                    tesseract, image_path, output_base, int(psm)
                )
            else:
                cached = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE)
                cached.columns = [str(column).lower().strip() for column in cached.columns]
                cached["text"] = cached["text"].fillna("").astype(str)
                words = cached[
                    cached["level"].eq(5) & cached["text"].str.strip().ne("")
                ].copy()
                full_text = "\n".join(
                    words.sort_values(["block_num", "par_num", "line_num", "left"])
                    .groupby(["block_num", "par_num", "line_num"])["text"]
                    .apply(lambda series: " ".join(series.astype(str)))
                    .tolist()
                )

            team_frame, game_frame, quality = parse_ocr_attempt(
                words,
                full_text,
                int(source.season),
                int(source.week),
                week_schedule,
                int(psm),
                pdf_path,
            )
            quality.update(
                {
                    "url": source.url,
                    "sha256": source.sha256,
                    "local_path": str(pdf_path),
                    "image_path": str(image_path),
                    "tsv_path": str(tsv_path),
                }
            )
            attempt_id = f"{source.sha256}_psm{psm}"
            quality["attempt_id"] = attempt_id
            team_frame["attempt_id"] = attempt_id
            game_frame["attempt_id"] = attempt_id
            all_attempts.append(quality)
            all_team_lines.append(team_frame)
            all_game_lines.append(game_frame)

    return (
        pd.DataFrame(all_attempts),
        pd.concat(all_team_lines, ignore_index=True) if all_team_lines else pd.DataFrame(),
        pd.concat(all_game_lines, ignore_index=True) if all_game_lines else pd.DataFrame(),
    )


def reconcile_ocr_sources(
    attempts: pd.DataFrame,
    raw_game_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconcile valid game lines across OCR modes for one source PDF.

    Earlier releases selected a single Tesseract PSM for an entire week.  A
    game read correctly by PSM 11 was therefore discarded whenever PSM 3 had
    the better aggregate week score.  This routine first reconciles each game
    across all OCR modes belonging to the same official PDF, and only then
    chooses the best official PDF version for the week.

    Exact agreement is accepted immediately.  When OCR modes disagree, a
    strict majority of at least two matching readings is required.  Ties or
    one-off disagreements remain unresolved and are excluded rather than
    guessed.
    """
    if attempts.empty or raw_game_lines.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_attempt_columns = {
        "attempt_id", "season", "week", "sha256", "url", "source_pdf",
        "updated_timestamp", "expected_games", "missing_games",
        "average_team_match_score", "psm",
    }
    required_line_columns = {
        "attempt_id", "season", "week", "game_id", "home_team", "away_team",
        "circa_home_margin", "home_team_spread", "away_team_spread",
        "symmetry_error", "derived_side", "line_valid", "psm", "source_pdf",
    }
    missing_attempt = sorted(required_attempt_columns - set(attempts.columns))
    missing_lines = sorted(required_line_columns - set(raw_game_lines.columns))
    if missing_attempt:
        raise RuntimeError(
            "OCR attempts are missing reconciliation columns: "
            f"{missing_attempt}"
        )
    if missing_lines:
        raise RuntimeError(
            "OCR game lines are missing reconciliation columns: "
            f"{missing_lines}"
        )

    metadata = attempts[
        [
            "attempt_id", "sha256", "url", "updated_timestamp",
            "expected_games", "missing_games", "average_team_match_score",
        ]
    ].drop_duplicates("attempt_id", keep="first")

    enriched = raw_game_lines.merge(
        metadata,
        on="attempt_id",
        how="left",
        validate="many_to_one",
    )
    enriched["circa_home_margin"] = pd.to_numeric(
        enriched["circa_home_margin"], errors="coerce"
    )
    enriched["line_valid"] = pd.to_numeric(
        enriched["line_valid"], errors="coerce"
    ).fillna(0).astype(int)
    enriched["psm"] = pd.to_numeric(enriched["psm"], errors="coerce")
    enriched["attempt_missing_games"] = pd.to_numeric(
        enriched["missing_games"], errors="coerce"
    )
    enriched["attempt_match_score"] = pd.to_numeric(
        enriched["average_team_match_score"], errors="coerce"
    )

    valid = enriched[
        enriched["line_valid"].eq(1)
        & enriched["circa_home_margin"].notna()
        & enriched["sha256"].notna()
    ].copy()
    valid["normalized_margin"] = valid["circa_home_margin"].round(3)

    game_keys = [
        "season", "week", "sha256", "game_id", "home_team", "away_team"
    ]
    resolved_rows: list[pd.Series] = []
    conflict_rows: list[dict[str, Any]] = []

    for keys, group in valid.groupby(game_keys, sort=True, dropna=False):
        counts = group["normalized_margin"].value_counts(dropna=False)
        if counts.empty:
            continue

        top_margin = float(counts.index[0])
        top_count = int(counts.iloc[0])
        second_count = int(counts.iloc[1]) if len(counts) > 1 else 0
        unique_count = int(len(counts))

        if unique_count == 1:
            status = "UNANIMOUS"
        elif top_count >= 2 and top_count > second_count:
            status = "STRICT_MAJORITY"
        else:
            season, week, sha256, game_id, home_team, away_team = keys
            conflict_rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "sha256": str(sha256),
                    "game_id": str(game_id),
                    "home_team": str(home_team),
                    "away_team": str(away_team),
                    "observed_margins": "|".join(
                        f"{float(value):g}:{int(count)}"
                        for value, count in counts.items()
                    ),
                    "attempts": int(len(group)),
                    "reason": "NO_STRICT_OCR_MAJORITY",
                }
            )
            continue

        candidates = group[group["normalized_margin"].eq(top_margin)].copy()
        candidates = candidates.sort_values(
            [
                "attempt_missing_games",
                "attempt_match_score",
                "psm",
                "attempt_id",
            ],
            ascending=[True, False, True, True],
            na_position="last",
        )
        chosen = candidates.iloc[0].copy()
        chosen["ocr_reconciliation"] = status
        chosen["ocr_support_count"] = top_count
        chosen["ocr_attempt_count"] = int(len(group))
        resolved_rows.append(chosen)

    resolved = (
        pd.DataFrame(resolved_rows).reset_index(drop=True)
        if resolved_rows
        else pd.DataFrame()
    )
    conflicts = pd.DataFrame(conflict_rows)

    source_base = (
        attempts.groupby(["season", "week", "sha256"], as_index=False)
        .agg(
            expected_games=("expected_games", "max"),
            source_best_single_attempt_games=("matched_games", "max"),
            source_best_attempt_missing=("missing_games", "min"),
            source_average_match_score=("average_team_match_score", "max"),
            updated_timestamp=("updated_timestamp", "first"),
            url=("url", "first"),
            source_pdf=("source_pdf", "first"),
            ocr_attempts=("attempt_id", "nunique"),
        )
    )

    if resolved.empty:
        resolved_count = pd.DataFrame(
            columns=["season", "week", "sha256", "resolved_games"]
        )
    else:
        resolved_count = (
            resolved.groupby(["season", "week", "sha256"], as_index=False)
            .agg(resolved_games=("game_id", "nunique"))
        )

    if conflicts.empty:
        conflict_count = pd.DataFrame(
            columns=["season", "week", "sha256", "conflicting_games"]
        )
    else:
        conflict_count = (
            conflicts.groupby(["season", "week", "sha256"], as_index=False)
            .agg(conflicting_games=("game_id", "nunique"))
        )

    source_summary = source_base.merge(
        resolved_count,
        on=["season", "week", "sha256"],
        how="left",
        validate="one_to_one",
    ).merge(
        conflict_count,
        on=["season", "week", "sha256"],
        how="left",
        validate="one_to_one",
    )
    source_summary["resolved_games"] = pd.to_numeric(
        source_summary["resolved_games"], errors="coerce"
    ).fillna(0).astype(int)
    source_summary["conflicting_games"] = pd.to_numeric(
        source_summary["conflicting_games"], errors="coerce"
    ).fillna(0).astype(int)
    source_summary["source_missing_games"] = (
        pd.to_numeric(source_summary["expected_games"], errors="coerce")
        - source_summary["resolved_games"]
    ).astype(int)
    source_summary["source_complete"] = source_summary[
        "source_missing_games"
    ].eq(0).astype(int)
    source_summary["updated_sort"] = pd.to_datetime(
        source_summary["updated_timestamp"], errors="coerce"
    )
    source_summary["updated_missing"] = source_summary[
        "updated_sort"
    ].isna().astype(int)

    selected_sources = []
    for (_, _), group in source_summary.groupby(["season", "week"], sort=True):
        complete = group[group["source_complete"].eq(1)].copy()
        candidate = complete if not complete.empty else group.copy()
        candidate = candidate.sort_values(
            [
                "source_missing_games",
                "conflicting_games",
                "updated_missing",
                "updated_sort",
                "source_average_match_score",
                "sha256",
            ],
            ascending=[True, True, True, True, False, True],
            na_position="last",
        )
        selected_sources.append(candidate.iloc[0])

    selected_source_frame = pd.DataFrame(selected_sources).drop(
        columns=["updated_sort", "updated_missing"], errors="ignore"
    )

    if resolved.empty or selected_source_frame.empty:
        selected_lines = pd.DataFrame()
    else:
        selected_lines = resolved.merge(
            selected_source_frame[
                [
                    "season", "week", "sha256", "source_missing_games",
                    "conflicting_games", "source_complete",
                ]
            ],
            on=["season", "week", "sha256"],
            how="inner",
            validate="many_to_one",
        )
        selected_lines = selected_lines.sort_values(
            ["season", "week", "game_id"]
        ).drop_duplicates(
            ["season", "week", "game_id", "home_team", "away_team"],
            keep="first",
        ).reset_index(drop=True)

    return selected_lines, selected_source_frame, conflicts


def load_manual_lines(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).lower().strip() for column in frame.columns]
    required = {"season", "week", "home_team", "away_team", "circa_home_margin"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Manual line file missing columns: {missing}")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    frame["circa_home_margin"] = pd.to_numeric(
        frame["circa_home_margin"], errors="raise"
    )
    return frame


def finalize_lines(
    schedule: pd.DataFrame,
    attempts: pd.DataFrame,
    raw_game_lines: pd.DataFrame,
    manual_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile OCR modes at game level and merge audited lines to schedule."""
    selected_lines, selected_sources, conflicts = reconcile_ocr_sources(
        attempts,
        raw_game_lines,
    )

    base_columns = [
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
        "gameday",
        "actual_home_margin",
        "spread_line",
    ]
    missing_schedule_columns = sorted(set(base_columns) - set(schedule.columns))
    if missing_schedule_columns:
        raise RuntimeError(
            "Schedule is missing required columns: "
            f"{missing_schedule_columns}"
        )
    base = schedule[base_columns].copy()

    if not selected_lines.empty:
        merge_keys = [
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
        ]
        merge_columns = [
            *merge_keys,
            "circa_home_margin",
            "home_team_spread",
            "away_team_spread",
            "symmetry_error",
            "derived_side",
            "line_valid",
            "attempt_id",
            "source_pdf",
            "updated_timestamp",
            "url",
            "sha256",
            "ocr_reconciliation",
            "ocr_support_count",
            "ocr_attempt_count",
            "source_missing_games",
            "conflicting_games",
            "source_complete",
        ]
        missing_line_columns = sorted(
            set(merge_columns) - set(selected_lines.columns)
        )
        if missing_line_columns:
            raise RuntimeError(
                "Reconciled OCR game lines are missing required columns: "
                f"{missing_line_columns}"
            )

        duplicate_keys = selected_lines.duplicated(merge_keys, keep=False)
        if duplicate_keys.any():
            duplicates = selected_lines.loc[
                duplicate_keys, merge_columns
            ].sort_values(merge_keys)
            raise RuntimeError(
                "Reconciled OCR lines still contain duplicate schedule keys. "
                "Review these rows:\n"
                + duplicates.to_string(index=False)
            )

        base = base.merge(
            selected_lines[merge_columns],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
    else:
        base["circa_home_margin"] = np.nan
        base["line_valid"] = 0
        base["ocr_reconciliation"] = None
        base["ocr_support_count"] = np.nan
        base["ocr_attempt_count"] = np.nan
        base["source_missing_games"] = np.nan
        base["conflicting_games"] = np.nan
        base["source_complete"] = np.nan

    if not manual_lines.empty:
        manual = manual_lines.copy()
        manual["manual_override"] = 1
        base = base.merge(
            manual,
            on=["season", "week", "home_team", "away_team"],
            how="left",
            suffixes=("", "_manual"),
            validate="one_to_one",
        )
        override = base["circa_home_margin_manual"].notna()
        base.loc[override, "circa_home_margin"] = base.loc[
            override, "circa_home_margin_manual"
        ]
        base.loc[override, "line_valid"] = 1
        base.loc[override, "source_pdf"] = "MANUAL_AUDITED_OVERRIDE"
        base.loc[override, "manual_override"] = 1
        base.loc[override, "ocr_reconciliation"] = "MANUAL_AUDITED_OVERRIDE"
        base["manual_override"] = base["manual_override"].fillna(0).astype(int)
    else:
        base["manual_override"] = 0

    base["line_valid"] = numeric(base, "line_valid", 0).fillna(0).astype(int)
    base["circa_home_spread"] = -numeric(base, "circa_home_margin")
    base["nflverse_reference_home_margin"] = numeric(base, "spread_line")

    audit = (
        base.groupby(["season", "week"], as_index=False)
        .agg(
            scheduled_games=("game_id", "nunique"),
            valid_games=("line_valid", "sum"),
            manual_overrides=("manual_override", "sum"),
            circa_lines_present=(
                "circa_home_margin",
                lambda series: int(series.notna().sum()),
            ),
            strict_majority_lines=(
                "ocr_reconciliation",
                lambda series: int(series.eq("STRICT_MAJORITY").sum()),
            ),
            unanimous_lines=(
                "ocr_reconciliation",
                lambda series: int(series.eq("UNANIMOUS").sum()),
            ),
        )
    )
    audit["missing_games"] = audit["scheduled_games"] - audit["valid_games"]
    audit["week_complete"] = audit["missing_games"].eq(0).astype(int)
    audit["coverage_rate"] = audit["valid_games"] / audit["scheduled_games"]

    if not selected_sources.empty:
        source_audit = selected_sources[
            [
                "season",
                "week",
                "sha256",
                "source_pdf",
                "url",
                "resolved_games",
                "conflicting_games",
                "source_missing_games",
                "source_complete",
                "ocr_attempts",
            ]
        ].rename(
            columns={
                "sha256": "selected_sha256",
                "source_pdf": "selected_source_pdf",
                "url": "selected_source_url",
            }
        )
        audit = audit.merge(
            source_audit,
            on=["season", "week"],
            how="left",
            validate="one_to_one",
        )

    if not conflicts.empty:
        conflict_weeks = (
            conflicts.groupby(["season", "week"], as_index=False)
            .agg(unresolved_ocr_conflicts=("game_id", "nunique"))
        )
        audit = audit.merge(
            conflict_weeks,
            on=["season", "week"],
            how="left",
            validate="one_to_one",
        )
    if "unresolved_ocr_conflicts" not in audit.columns:
        audit["unresolved_ocr_conflicts"] = 0
    audit["unresolved_ocr_conflicts"] = pd.to_numeric(
        audit["unresolved_ocr_conflicts"], errors="coerce"
    ).fillna(0).astype(int)

    return (
        base.sort_values(["season", "week", "game_id"]).reset_index(drop=True),
        audit.sort_values(["season", "week"]).reset_index(drop=True),
    )


def apply_2025_reference_backfill(
    game_lines: pd.DataFrame,
    week_audit: pd.DataFrame,
    enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only missing 2025 rows with nflverse final spreads when enabled.

    This is a diagnostic continuity fallback, not a Circa contest line. Every
    substituted row is labeled and remains excluded from the strict Circa
    benchmark gate and from final model fitting.
    """
    lines = game_lines.copy()
    audit = week_audit.copy()
    if lines.empty:
        return lines, audit

    for column, default in (
        ("reference_fallback_used", 0),
        ("strict_circa_line_valid", 0),
        ("line_source", "MISSING"),
    ):
        if column not in lines.columns:
            lines[column] = default
    if "manual_override" not in lines.columns:
        lines["manual_override"] = 0

    lines["reference_fallback_used"] = numeric(
        lines, "reference_fallback_used", 0
    ).fillna(0).astype(int)
    lines["strict_circa_line_valid"] = (
        numeric(lines, "line_valid", 0).fillna(0).astype(int)
        & lines["circa_home_margin"].notna().astype(int)
        & lines["reference_fallback_used"].eq(0).astype(int)
    )
    observed = lines["strict_circa_line_valid"].eq(1)
    lines.loc[observed & lines["manual_override"].eq(1), "line_source"] = (
        "MANUAL_AUDITED_CIRCA_OVERRIDE"
    )
    lines.loc[observed & lines["manual_override"].ne(1), "line_source"] = (
        "CIRCA_OFFICIAL_PDF_OCR"
    )

    if enabled:
        fallback_mask = (
            lines["season"].eq(2025)
            & lines["circa_home_margin"].isna()
            & lines["nflverse_reference_home_margin"].notna()
        )
        lines.loc[fallback_mask, "circa_home_margin"] = lines.loc[
            fallback_mask, "nflverse_reference_home_margin"
        ]
        lines.loc[fallback_mask, "circa_home_spread"] = -lines.loc[
            fallback_mask, "nflverse_reference_home_margin"
        ]
        lines.loc[fallback_mask, "line_valid"] = 1
        lines.loc[fallback_mask, "reference_fallback_used"] = 1
        lines.loc[fallback_mask, "strict_circa_line_valid"] = 0
        lines.loc[fallback_mask, "line_source"] = (
            "NFLVERSE_FINAL_SPREAD_FALLBACK_2025"
        )
        lines.loc[fallback_mask, "source_pdf"] = (
            "NFLVERSE_FINAL_SPREAD_FALLBACK_2025"
        )
        lines.loc[fallback_mask, "ocr_reconciliation"] = (
            "REFERENCE_CLOSE_FALLBACK"
        )

    lines["line_valid"] = numeric(lines, "line_valid", 0).fillna(0).astype(int)
    lines["reference_fallback_used"] = numeric(
        lines, "reference_fallback_used", 0
    ).fillna(0).astype(int)
    lines["strict_circa_line_valid"] = numeric(
        lines, "strict_circa_line_valid", 0
    ).fillna(0).astype(int)

    coverage = (
        lines.groupby(["season", "week"], as_index=False)
        .agg(
            scheduled_games_recalc=("game_id", "nunique"),
            valid_games_recalc=("line_valid", "sum"),
            strict_valid_games=("strict_circa_line_valid", "sum"),
            reference_fallback_lines=("reference_fallback_used", "sum"),
        )
    )
    coverage["missing_games_recalc"] = (
        coverage["scheduled_games_recalc"] - coverage["valid_games_recalc"]
    )
    coverage["strict_missing_games"] = (
        coverage["scheduled_games_recalc"] - coverage["strict_valid_games"]
    )
    coverage["week_complete_recalc"] = coverage[
        "missing_games_recalc"
    ].eq(0).astype(int)
    coverage["coverage_rate_recalc"] = (
        coverage["valid_games_recalc"] / coverage["scheduled_games_recalc"]
    )
    coverage["strict_coverage_rate"] = (
        coverage["strict_valid_games"] / coverage["scheduled_games_recalc"]
    )

    rename_map = {
        "scheduled_games_recalc": "scheduled_games",
        "valid_games_recalc": "valid_games",
        "missing_games_recalc": "missing_games",
        "week_complete_recalc": "week_complete",
        "coverage_rate_recalc": "coverage_rate",
    }
    if audit.empty:
        audit = coverage.rename(columns=rename_map)
    else:
        replace_columns = {
            "scheduled_games", "valid_games", "missing_games",
            "week_complete", "coverage_rate", "strict_valid_games",
            "strict_missing_games", "strict_coverage_rate",
            "reference_fallback_lines",
        }
        audit = audit.drop(
            columns=[column for column in replace_columns if column in audit.columns],
            errors="ignore",
        ).merge(
            coverage,
            on=["season", "week"],
            how="left",
            validate="one_to_one",
        ).rename(columns=rename_map)

    return (
        lines.sort_values(["season", "week", "game_id"]).reset_index(drop=True),
        audit.sort_values(["season", "week"]).reset_index(drop=True),
    )


# =============================================================================
# BACKTEST
# =============================================================================


def load_matchup_inputs(
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    database = project_root / "backtests" / MATCHUP_DATABASE_NAME
    if not database.exists():
        raise FileNotFoundError(database)
    with sqlite3.connect(database) as connection:
        matrix = read_table(connection, MATCHUP_MATRIX_TABLE)
        audit = read_table(connection, MATCHUP_AUDIT_TABLE)
        qb_audit = read_table(connection, MATCHUP_QB_AUDIT_TABLE)
    if matrix.empty or audit.empty or qb_audit.empty:
        raise RuntimeError(
            "Matchup residual inputs are incomplete. Rebuild with the canonical "
            "QB-identity-fixed matchup builder before running the Circa model."
        )
    source_audit = audit.iloc[0].to_dict()
    if int(source_audit.get("qb_feature_integrity_passed", 0)) != 1:
        raise RuntimeError(
            "Upstream matchup database did not pass its QB feature integrity gate."
        )
    if not pd.to_numeric(
        qb_audit["feature_integrity_passed"], errors="coerce"
    ).fillna(0).eq(1).all():
        raise RuntimeError(
            "At least one upstream QB feature-integrity scope failed."
        )
    return matrix, source_audit, qb_audit


def locked_configuration(audit: dict[str, Any]) -> tuple[float, str, float]:
    rating_alpha = float(audit.get("selected_rating_alpha", 10.0))
    feature_set = str(audit.get("selected_feature_set", "FULL_COMPACT"))
    model_alpha = float(audit.get("selected_model_alpha", 500.0))
    if feature_set not in FEATURE_SETS:
        raise RuntimeError(f"Unsupported selected feature set: {feature_set}")
    return rating_alpha, feature_set, model_alpha


def build_backtest_matrix(
    matchup_matrix: pd.DataFrame,
    game_lines: pd.DataFrame,
    rating_alpha: float,
) -> pd.DataFrame:
    matrix = matchup_matrix[
        pd.to_numeric(matchup_matrix["rating_alpha"], errors="coerce").eq(rating_alpha)
    ].copy()
    matrix["season"] = pd.to_numeric(matrix["season"], errors="coerce").astype(int)
    matrix["week"] = pd.to_numeric(matrix["week"], errors="coerce").astype(int)
    matrix["game_id"] = matrix["game_id"].astype(str)

    lines = game_lines[
        game_lines["line_valid"].eq(1)
        & game_lines["circa_home_margin"].notna()
    ][
        [
            "season", "week", "game_id", "home_team", "away_team",
            "circa_home_margin", "nflverse_reference_home_margin",
            "source_pdf", "updated_timestamp", "manual_override",
            "line_source", "reference_fallback_used",
            "strict_circa_line_valid",
        ]
    ].copy()

    frame = matrix.merge(
        lines,
        on=["season", "week", "game_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_circa"),
    )
    frame["market_home_margin"] = numeric(frame, "circa_home_margin")
    frame["absolute_market_home_margin"] = frame["market_home_margin"].abs()
    frame["actual_home_margin"] = numeric(frame, "actual_home_margin")
    frame["actual_circa_residual"] = (
        frame["actual_home_margin"] - frame["circa_home_margin"]
    )
    frame["reference_line_available"] = frame[
        "nflverse_reference_home_margin"
    ].notna().astype(int)
    frame["reference_fallback_used"] = numeric(
        frame, "reference_fallback_used", 0
    ).fillna(0).astype(int)
    frame["strict_circa_line_valid"] = numeric(
        frame, "strict_circa_line_valid", 0
    ).fillna(0).astype(int)
    return frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def assert_circa_qb_feature_integrity(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "home_qb_recent_epa",
        "away_qb_recent_epa",
        "home_qb_recent_cpoe",
        "away_qb_recent_cpoe",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Circa QB integrity fields are missing: {missing}")

    scope = frame[
        frame["season"].isin(DEFAULT_SEASONS)
        & frame["week"].between(2, 17)
    ].copy()
    if scope.empty:
        raise RuntimeError("Circa QB integrity audit has no eligible rows.")
    epa_advantage = pd.to_numeric(
        scope["qb_recent_epa_advantage"], errors="coerce"
    )
    cpoe_advantage = pd.to_numeric(
        scope["qb_recent_cpoe_advantage"], errors="coerce"
    )
    side_epa = pd.concat(
        [
            pd.to_numeric(scope["home_qb_recent_epa"], errors="coerce"),
            pd.to_numeric(scope["away_qb_recent_epa"], errors="coerce"),
        ],
        ignore_index=True,
    )
    side_cpoe = pd.concat(
        [
            pd.to_numeric(scope["home_qb_recent_cpoe"], errors="coerce"),
            pd.to_numeric(scope["away_qb_recent_cpoe"], errors="coerce"),
        ],
        ignore_index=True,
    )
    metrics = {
        "circa_qb_side_epa_coverage": float(side_epa.notna().mean()),
        "circa_qb_side_cpoe_coverage": float(side_cpoe.notna().mean()),
        "circa_qb_epa_advantage_nonzero_rows": int(
            epa_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
        "circa_qb_epa_advantage_std": float(epa_advantage.std(ddof=0)),
        "circa_qb_cpoe_advantage_nonzero_rows": int(
            cpoe_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
        "circa_qb_cpoe_advantage_std": float(cpoe_advantage.std(ddof=0)),
    }
    passed = (
        metrics["circa_qb_side_epa_coverage"] >= 0.99
        and metrics["circa_qb_side_cpoe_coverage"] >= 0.99
        and metrics["circa_qb_epa_advantage_nonzero_rows"] > 0
        and metrics["circa_qb_cpoe_advantage_nonzero_rows"] > 0
        and metrics["circa_qb_epa_advantage_std"] > 1e-6
        and metrics["circa_qb_cpoe_advantage_std"] > 1e-6
    )
    metrics["circa_qb_feature_integrity_passed"] = int(passed)
    if not passed:
        raise RuntimeError(
            "Circa QB feature integrity gate failed; training is blocked: "
            + json.dumps(metrics, sort_keys=True)
        )
    return metrics


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def rolling_oof_predictions(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_alpha: float,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    outputs = []
    for season in seasons:
        training = frame[
            frame["season"].lt(season) & frame["season"].ge(DEVELOPMENT_SEASON)
        ].copy()
        validation = frame[frame["season"].eq(season)].copy()
        if len(training) < 180 or len(validation) < 180:
            continue
        model = build_pipeline(model_alpha)
        model.fit(training[list(features)], training["actual_circa_residual"])
        validation["predicted_circa_residual"] = model.predict(
            validation[list(features)]
        )
        outputs.append(validation)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def grade_selected_rows(frame: pd.DataFrame, fallback_price: float) -> pd.DataFrame:
    output = frame.copy()
    output["selected_side"] = np.where(
        output["predicted_circa_residual"].gt(0),
        "HOME",
        np.where(output["predicted_circa_residual"].lt(0), "AWAY", "NONE"),
    )
    direction = np.where(
        output["selected_side"].eq("HOME"),
        1.0,
        np.where(output["selected_side"].eq("AWAY"), -1.0, np.nan),
    )
    output["absolute_model_edge"] = output["predicted_circa_residual"].abs()
    output["selected_cover_margin"] = direction * (
        output["actual_home_margin"] - output["circa_home_margin"]
    )
    output["ats_result"] = np.where(
        output["selected_cover_margin"].gt(0),
        "W",
        np.where(
            output["selected_cover_margin"].lt(0),
            "L",
            np.where(output["selected_cover_margin"].eq(0), "P", "N"),
        ),
    )
    output["contest_points"] = np.where(
        output["ats_result"].eq("W"),
        1.0,
        np.where(output["ats_result"].eq("P"), 0.5, 0.0),
    )
    win_profit = 100.0 / abs(fallback_price) if fallback_price < 0 else fallback_price / 100.0
    output["profit_units"] = np.where(
        output["ats_result"].eq("W"),
        win_profit,
        np.where(output["ats_result"].eq("L"), -1.0, 0.0),
    )
    output["selected_reference_clv"] = direction * (
        output["nflverse_reference_home_margin"] - output["circa_home_margin"]
    )
    return output


def select_top5_each_week(graded: pd.DataFrame) -> pd.DataFrame:
    return (
        graded.sort_values(
            ["season", "week", "absolute_model_edge", "game_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["season", "week"], as_index=False, group_keys=False)
        .head(5)
        .copy()
    )


def max_drawdown(profits: pd.Series) -> float:
    cumulative = pd.to_numeric(profits, errors="coerce").fillna(0.0).cumsum()
    if cumulative.empty:
        return 0.0
    return float((cumulative.cummax() - cumulative).max())


def strategy_summary(sample: pd.DataFrame, scope: str, strategy: str) -> dict[str, Any]:
    wins = int(sample["ats_result"].eq("W").sum())
    losses = int(sample["ats_result"].eq("L").sum())
    pushes = int(sample["ats_result"].eq("P").sum())
    picks = int(len(sample))
    profit = float(sample["profit_units"].sum())
    return {
        "scope": scope,
        "strategy": strategy,
        "picks": picks,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_win_rate": wins / (wins + losses) if wins + losses else np.nan,
        "contest_points": float(sample["contest_points"].sum()),
        "contest_points_per_pick": float(sample["contest_points"].mean()) if picks else np.nan,
        "profit_units_at_minus_110": profit,
        "roi_at_minus_110": profit / picks if picks else np.nan,
        "average_reference_clv": float(sample["selected_reference_clv"].mean()) if picks else np.nan,
        "positive_reference_clv_rate": float(sample["selected_reference_clv"].gt(0).mean()) if picks else np.nan,
        "edge_to_cover_correlation": (
            float(sample["absolute_model_edge"].corr(sample["selected_cover_margin"]))
            if picks >= 3
            and sample["absolute_model_edge"].std(ddof=0) > 0
            and sample["selected_cover_margin"].std(ddof=0) > 0
            else np.nan
        ),
        "maximum_drawdown_units": max_drawdown(sample["profit_units"]),
        "weeks": int(sample[["season", "week"]].drop_duplicates().shape[0]),
        "official_circa_picks": int(
            sample.get("reference_fallback_used", pd.Series(0, index=sample.index))
            .fillna(0).eq(0).sum()
        ),
        "reference_fallback_picks": int(
            sample.get("reference_fallback_used", pd.Series(0, index=sample.index))
            .fillna(0).eq(1).sum()
        ),
    }


def validation_outputs(
    oof: pd.DataFrame,
    thresholds: tuple[float, ...],
    fallback_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graded = grade_selected_rows(oof, fallback_price)
    top5 = select_top5_each_week(graded)
    top5_rows = [strategy_summary(top5, "VALIDATION_2021_2023", "TOP5_WEEKLY")]
    for season, group in top5.groupby("season", sort=True):
        top5_rows.append(
            strategy_summary(group, f"VALIDATION_{int(season)}", "TOP5_WEEKLY")
        )

    threshold_rows = []
    for threshold in thresholds:
        sample = graded[graded["absolute_model_edge"].ge(float(threshold))].copy()
        row = strategy_summary(
            sample,
            "VALIDATION_2021_2023",
            f"THRESHOLD_{threshold:g}",
        )
        row["threshold"] = float(threshold)
        for season in VALIDATION_SEASONS:
            season_sample = sample[sample["season"].eq(season)]
            season_summary = strategy_summary(
                season_sample,
                f"VALIDATION_{season}",
                f"THRESHOLD_{threshold:g}",
            )
            row[f"picks_{season}"] = season_summary["picks"]
            row[f"roi_{season}"] = season_summary["roi_at_minus_110"]
            row[f"clv_{season}"] = season_summary["average_reference_clv"]
        threshold_rows.append(row)
    return graded, pd.DataFrame(top5_rows), pd.DataFrame(threshold_rows)


def top5_validation_passed(summary_frame: pd.DataFrame) -> bool:
    if summary_frame.empty:
        return False
    overall = summary_frame[summary_frame["scope"].eq("VALIDATION_2021_2023")]
    seasons = summary_frame[summary_frame["scope"].isin([f"VALIDATION_{s}" for s in VALIDATION_SEASONS])]
    if overall.empty or len(seasons) != 3:
        return False
    overall_row = overall.iloc[0]
    profitable_seasons = int((seasons["roi_at_minus_110"] > 0).sum())
    positive_clv_seasons = int((seasons["average_reference_clv"] > 0).sum())
    return bool(
        overall_row["picks"] >= 200
        and np.isfinite(overall_row["roi_at_minus_110"])
        and overall_row["roi_at_minus_110"] > 0
        and profitable_seasons >= 2
        and seasons["roi_at_minus_110"].min() >= -0.08
        and np.isfinite(overall_row["average_reference_clv"])
        and overall_row["average_reference_clv"] > 0
        and positive_clv_seasons >= 2
        and seasons["average_reference_clv"].min() >= -0.15
    )


def benchmark_outputs(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_alpha: float,
    fallback_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = frame[
        frame["season"].isin((2020, 2021, 2022, 2023))
        & frame["reference_fallback_used"].eq(0)
    ].copy()
    benchmark = frame[frame["season"].isin(BENCHMARK_SEASONS)].copy()
    model = build_pipeline(model_alpha)
    model.fit(training[list(features)], training["actual_circa_residual"])
    benchmark["predicted_circa_residual"] = model.predict(benchmark[list(features)])
    graded = grade_selected_rows(benchmark, fallback_price)
    graded["strict_circa_eligible"] = graded["reference_fallback_used"].eq(0).astype(int)

    rows: list[dict[str, Any]] = []
    samples = (
        ("STRICT_CIRCA_ONLY", graded[graded["strict_circa_eligible"].eq(1)].copy()),
        ("HYBRID_WITH_2025_CLOSE_FALLBACK", graded.copy()),
    )
    for line_sample, sample in samples:
        top5 = select_top5_each_week(sample)
        overall = strategy_summary(
            top5, "BENCHMARK_2024_2025", "TOP5_WEEKLY"
        )
        overall["line_sample"] = line_sample
        rows.append(overall)
        for season, group in top5.groupby("season", sort=True):
            row = strategy_summary(
                group, f"BENCHMARK_{int(season)}", "TOP5_WEEKLY"
            )
            row["line_sample"] = line_sample
            rows.append(row)
    return graded, pd.DataFrame(rows)


def benchmark_passed(summary_frame: pd.DataFrame) -> bool:
    if summary_frame.empty:
        return False
    strict = summary_frame[
        summary_frame["line_sample"].eq("STRICT_CIRCA_ONLY")
    ].copy()
    overall = strict[strict["scope"].eq("BENCHMARK_2024_2025")]
    seasons = strict[
        strict["scope"].isin([f"BENCHMARK_{s}" for s in BENCHMARK_SEASONS])
    ]
    if overall.empty or len(seasons) != 2:
        return False
    overall_row = overall.iloc[0]
    return bool(
        overall_row["picks"] >= 150
        and np.isfinite(overall_row["roi_at_minus_110"])
        and overall_row["roi_at_minus_110"] >= 0
        and seasons["roi_at_minus_110"].min() >= -0.05
        and np.isfinite(overall_row["average_reference_clv"])
        and overall_row["average_reference_clv"] > 0
        and seasons["average_reference_clv"].min() >= -0.10
    )


def save_model_bundle(
    project_root: Path,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_alpha: float,
    rating_alpha: float,
    feature_set: str,
    implementation_ready: bool,
    qb_integrity: dict[str, Any],
) -> tuple[Path, Path, pd.DataFrame]:
    fit_frame = frame[frame["reference_fallback_used"].eq(0)].copy()
    if fit_frame.empty:
        raise RuntimeError("No strict Circa rows are available for final model fitting.")
    model = build_pipeline(model_alpha)
    model.fit(fit_frame[list(features)], fit_frame["actual_circa_residual"])
    transformed_names = list(
        model.named_steps["imputer"].get_feature_names_out(list(features))
    )
    transformed_coefficients = model.named_steps["ridge"].coef_
    qb_feature = "qb_recent_epa_advantage"
    if qb_feature not in transformed_names:
        raise RuntimeError(
            "Locked Circa feature set is missing qb_recent_epa_advantage."
        )
    qb_epa_coefficient = float(
        transformed_coefficients[transformed_names.index(qb_feature)]
    )
    if abs(qb_epa_coefficient) <= 1e-12:
        raise RuntimeError(
            "Circa V1 QB EPA coefficient is zero after the repaired-data fit."
        )
    model_directory = project_root / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / MODEL_FILENAME
    metadata_path = model_directory / METADATA_FILENAME
    bundle = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "model": model,
        "feature_order": list(features),
        "rating_alpha": rating_alpha,
        "feature_set": feature_set,
        "model_alpha": model_alpha,
        "selection_policy": "TOP5_WEEKLY_BY_ABSOLUTE_PREDICTED_CIRCA_RESIDUAL",
        "implementation_ready": implementation_ready,
        "strict_circa_training_rows": int(len(fit_frame)),
        "reference_fallback_rows_excluded_from_training": int(
            frame["reference_fallback_used"].eq(1).sum()
        ),
        "qb_feature_integrity_passed": int(
            qb_integrity["circa_qb_feature_integrity_passed"]
        ),
        "qb_side_epa_coverage": float(
            qb_integrity["circa_qb_side_epa_coverage"]
        ),
        "qb_side_cpoe_coverage": float(
            qb_integrity["circa_qb_side_cpoe_coverage"]
        ),
        "qb_epa_advantage_std": float(
            qb_integrity["circa_qb_epa_advantage_std"]
        ),
        "qb_epa_advantage_nonzero_rows": int(
            qb_integrity["circa_qb_epa_advantage_nonzero_rows"]
        ),
        "qb_cpoe_advantage_std": float(
            qb_integrity["circa_qb_cpoe_advantage_std"]
        ),
        "qb_cpoe_advantage_nonzero_rows": int(
            qb_integrity["circa_qb_cpoe_advantage_nonzero_rows"]
        ),
        "qb_epa_standardized_coefficient": qb_epa_coefficient,
        "created_at": now_string(),
    }
    joblib.dump(bundle, model_path)
    metadata = {key: value for key, value in bundle.items() if key != "model"}
    metadata["production_usage"] = (
        "enabled_for_circa_contest_top5"
        if implementation_ready
        else "disabled_until_validation_and_benchmark_gates_pass"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reloaded = joblib.load(model_path)
    if reloaded["feature_order"] != list(features):
        raise RuntimeError("Model bundle reload validation failed.")

    coefficients = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "standardized_coefficient": transformed_coefficients,
        }
    ).sort_values(
        "standardized_coefficient",
        key=lambda series: series.abs(),
        ascending=False,
    )
    return model_path, metadata_path, coefficients


# =============================================================================
# OUTPUT PERSISTENCE
# =============================================================================


def save_database_tables(
    database: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        for table_name, frame in tables.items():
            if frame is None or len(frame.columns) == 0:
                continue
            frame.to_sql(table_name, connection, if_exists="replace", index=False)


def load_existing_tables(database: Path) -> dict[str, pd.DataFrame]:
    if not database.exists():
        return {}
    names = (
        MANIFEST_TABLE,
        OCR_ATTEMPT_TABLE,
        TEAM_LINE_TABLE,
        GAME_LINE_TABLE,
        WEEK_AUDIT_TABLE,
    )
    output = {}
    with sqlite3.connect(database) as connection:
        for name in names:
            output[name] = read_table(connection, name)
    return output


def save_csv_outputs(project_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    directory = project_root / OUTPUT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    for table_name, frame in tables.items():
        if frame is not None and not frame.empty:
            frame.to_csv(
                directory / f"{table_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )


# =============================================================================
# SELF-TEST
# =============================================================================


def run_self_test() -> int:
    odds_cases = {
        "+4½": 4.5,
        "+4¥": 4.5,
        "-3)": -3.5,
        "+7%": 7.5,
        "-AY,": -4.5,
        "PK": 0.0,
        "+6": 6.0,
    }
    for raw, expected in odds_cases.items():
        actual = parse_odds_token(raw)
        if actual is None or not math.isclose(actual, expected):
            raise AssertionError(f"Odds parse failed: {raw} -> {actual}, expected {expected}")

    frame = pd.DataFrame(
        [
            {
                "season": 2023,
                "week": 1,
                "game_id": "g1",
                "predicted_circa_residual": 2.0,
                "actual_home_margin": 7.0,
                "circa_home_margin": 3.0,
                "nflverse_reference_home_margin": 3.5,
                "reference_fallback_used": 0,
                "strict_circa_line_valid": 1,
            },
            {
                "season": 2023,
                "week": 1,
                "game_id": "g2",
                "predicted_circa_residual": -2.0,
                "actual_home_margin": -5.0,
                "circa_home_margin": -3.0,
                "nflverse_reference_home_margin": -4.0,
                "reference_fallback_used": 0,
                "strict_circa_line_valid": 1,
            },
        ]
    )
    graded = grade_selected_rows(frame, -110)
    if graded["ats_result"].tolist() != ["W", "W"]:
        raise AssertionError("ATS direction test failed.")
    if not all(np.isclose(graded["selected_reference_clv"], [0.5, 1.0])):
        raise AssertionError("Reference CLV direction test failed.")

    fallback_lines = pd.DataFrame(
        [
            {
                "season": 2025, "week": 1, "game_id": "f1",
                "circa_home_margin": np.nan, "circa_home_spread": np.nan,
                "nflverse_reference_home_margin": 2.5, "line_valid": 0,
                "manual_override": 0,
            },
            {
                "season": 2024, "week": 1, "game_id": "f2",
                "circa_home_margin": np.nan, "circa_home_spread": np.nan,
                "nflverse_reference_home_margin": 3.0, "line_valid": 0,
                "manual_override": 0,
            },
        ]
    )
    fallback_audit = pd.DataFrame(
        [
            {"season": 2025, "week": 1},
            {"season": 2024, "week": 1},
        ]
    )
    filled, filled_audit = apply_2025_reference_backfill(
        fallback_lines, fallback_audit, enabled=True
    )
    if int(filled["reference_fallback_used"].sum()) != 1:
        raise AssertionError("2025-only reference fallback test failed.")
    if not math.isclose(
        float(filled.loc[filled["season"].eq(2025), "circa_home_margin"].iloc[0]),
        2.5,
    ):
        raise AssertionError("Reference fallback margin test failed.")
    if filled.loc[filled["season"].eq(2024), "circa_home_margin"].notna().any():
        raise AssertionError("Reference fallback leaked outside 2025.")
    if int(filled_audit["reference_fallback_lines"].sum()) != 1:
        raise AssertionError("Reference fallback audit test failed.")
    print("[CIRCA_LINES] Self-test passed.")
    return 0


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    started = dt.datetime.now(dt.timezone.utc)
    database = args.project_root / "backtests" / OUTPUT_DATABASE_NAME
    schedule, schedule_source = load_schedule(args)
    expected = expected_week_table(schedule)

    dependency_rows = [
        {
            "dependency": "PyMuPDF",
            "available": int(_module_available("fitz")),
            "required_for": "parse",
        },
        {
            "dependency": "Tesseract",
            "available": int(_tesseract_available(args.tesseract_path)),
            "required_for": "parse",
        },
        {
            "dependency": "Matchup Residual V2 DB",
            "available": int(
                (args.project_root / "backtests" / MATCHUP_DATABASE_NAME).exists()
            ),
            "required_for": "backtest",
        },
    ]
    dependencies = pd.DataFrame(dependency_rows)

    print("[CIRCA_LINES] Official Circa Million contest-line workflow")
    print(f"[CIRCA_LINES] Build ID: {BUILD_ID}")
    print(f"[CIRCA_LINES] Version: {VERSION}")
    print(f"[CIRCA_LINES] Schedule source: {schedule_source}")
    print(
        f"[CIRCA_LINES] Expected weeks: {len(expected):,} | "
        f"games={schedule['game_id'].nunique():,} | seasons={list(args.seasons)}"
    )

    if args.mode == "plan":
        print("\n[CIRCA_LINES] Dependency check:")
        print(dependencies.to_string(index=False))
        print("\n[CIRCA_LINES] Expected coverage:")
        print(
            expected.groupby("season", as_index=False)
            .agg(weeks=("week", "nunique"), games=("scheduled_games", "sum"))
            .to_string(index=False)
        )
        print(
            "\n[CIRCA_LINES] No PDFs downloaded and no existing database modified."
        )
        return 0

    existing = load_existing_tables(database)
    manifest = existing.get(MANIFEST_TABLE, pd.DataFrame())
    attempts = existing.get(OCR_ATTEMPT_TABLE, pd.DataFrame())
    team_lines = existing.get(TEAM_LINE_TABLE, pd.DataFrame())
    game_lines = existing.get(GAME_LINE_TABLE, pd.DataFrame())
    week_audit = existing.get(WEEK_AUDIT_TABLE, pd.DataFrame())

    if not game_lines.empty and args.mode in {"audit", "backtest"}:
        game_lines, week_audit = apply_2025_reference_backfill(
            game_lines,
            week_audit,
            enabled=args.backfill_missing_2025_with_reference,
        )

    if args.mode in {"discover", "all"}:
        manifest = discover_and_download(args, schedule)
        save_database_tables(
            database,
            {
                MANIFEST_TABLE: manifest,
                "nfl_circa_expected_weeks": expected,
                "nfl_circa_schedule": schedule,
            },
        )
        print(
            f"[CIRCA_LINES] Official PDFs cached: "
            f"{int(manifest['download_status'].isin(['DOWNLOADED', 'CACHED']).sum()) if not manifest.empty else 0}"
        )
        discovered_pairs = (
            manifest[manifest["download_status"].isin(["DOWNLOADED", "CACHED"])][
                ["season", "week"]
            ].drop_duplicates()
            if not manifest.empty
            else pd.DataFrame(columns=["season", "week"])
        )
        missing = expected.merge(
            discovered_pairs.assign(pdf_found=1),
            on=["season", "week"],
            how="left",
        )
        missing = missing[missing["pdf_found"].isna()]
        if not missing.empty:
            print("[CIRCA_LINES] Weeks still missing an official PDF:")
            print(missing[["season", "week", "scheduled_games"]].to_string(index=False))
            if not args.probe_missing:
                print(
                    "[CIRCA_LINES] Re-run discover with --probe-missing or provide "
                    "--manifest-path for the missing official URLs."
                )
        if args.mode == "discover":
            return 0

    if args.mode in {"parse", "audit", "backtest", "all"} and manifest.empty:
        raise RuntimeError(
            "No PDF manifest exists. Run --mode discover first."
        )

    if args.mode in {"parse", "all"}:
        attempts, team_lines, raw_game_lines = ocr_pdf_versions(
            args, manifest, schedule
        )
        manual_lines = load_manual_lines(args.manual_lines_path)
        game_lines, week_audit = finalize_lines(
            schedule, attempts, raw_game_lines, manual_lines
        )
        game_lines, week_audit = apply_2025_reference_backfill(
            game_lines,
            week_audit,
            enabled=args.backfill_missing_2025_with_reference,
        )
        save_database_tables(
            database,
            {
                MANIFEST_TABLE: manifest,
                OCR_ATTEMPT_TABLE: attempts,
                TEAM_LINE_TABLE: team_lines,
                GAME_LINE_TABLE: game_lines,
                WEEK_AUDIT_TABLE: week_audit,
                "nfl_circa_expected_weeks": expected,
                "nfl_circa_schedule": schedule,
            },
        )
        if not args.no_csv:
            save_csv_outputs(
                args.project_root,
                {
                    MANIFEST_TABLE: manifest,
                    OCR_ATTEMPT_TABLE: attempts.drop(columns=["full_text"], errors="ignore"),
                    GAME_LINE_TABLE: game_lines,
                    WEEK_AUDIT_TABLE: week_audit,
                },
            )
        print("\n[CIRCA_LINES] OCR coverage by season:")
        print(
            week_audit.groupby("season", as_index=False)
            .agg(
                weeks=("week", "nunique"),
                complete_weeks=("week_complete", "sum"),
                scheduled_games=("scheduled_games", "sum"),
                strict_valid_games=("strict_valid_games", "sum"),
                reference_fallback_lines=("reference_fallback_lines", "sum"),
                valid_games=("valid_games", "sum"),
                missing_games=("missing_games", "sum"),
            )
            .to_string(index=False)
        )
        if args.mode == "parse":
            return 0

    if args.mode == "audit":
        if week_audit.empty:
            raise RuntimeError("No parsed week audit exists. Run --mode parse first.")
        print(week_audit.to_string(index=False))
        incomplete = week_audit[week_audit["week_complete"].eq(0)]
        print(f"\n[CIRCA_LINES] Incomplete weeks: {len(incomplete)}")
        return 0

    if args.mode in {"backtest", "all"}:
        if game_lines.empty or week_audit.empty:
            raise RuntimeError("No parsed game lines exist. Run --mode parse first.")
        incomplete = week_audit[week_audit["week_complete"].eq(0)]
        if not incomplete.empty and not args.allow_incomplete:
            raise RuntimeError(
                f"Backtest blocked: {len(incomplete)} season/weeks are incomplete. "
                "Correct them with --manual-lines-path or use --allow-incomplete "
                "only for diagnostics."
            )

        matchup_matrix, matchup_audit, matchup_qb_audit = load_matchup_inputs(
            args.project_root
        )
        rating_alpha, feature_set, model_alpha = locked_configuration(matchup_audit)
        features = FEATURE_SETS[feature_set]
        backtest_matrix = build_backtest_matrix(
            matchup_matrix, game_lines, rating_alpha
        )
        missing_features = sorted(set(features) - set(backtest_matrix.columns))
        if missing_features:
            raise RuntimeError(f"Backtest matrix missing features: {missing_features}")
        circa_qb_integrity = assert_circa_qb_feature_integrity(backtest_matrix)
        print(
            "[CIRCA_LINES] QB feature integrity passed | "
            f"EPA coverage={circa_qb_integrity['circa_qb_side_epa_coverage']:.2%} | "
            f"CPOE coverage={circa_qb_integrity['circa_qb_side_cpoe_coverage']:.2%} | "
            f"EPA advantage std={circa_qb_integrity['circa_qb_epa_advantage_std']:.6f} | "
            f"CPOE advantage std={circa_qb_integrity['circa_qb_cpoe_advantage_std']:.6f}"
        )

        oof = rolling_oof_predictions(
            backtest_matrix,
            features,
            model_alpha,
            VALIDATION_SEASONS,
        )
        if oof.empty:
            raise RuntimeError("No rolling validation predictions were produced.")
        graded_validation, top5_validation, threshold_validation = validation_outputs(
            oof,
            args.thresholds,
            args.fallback_ats_price,
        )
        validation_passed = top5_validation_passed(top5_validation)

        benchmark_predictions, benchmark_summary = benchmark_outputs(
            backtest_matrix,
            features,
            model_alpha,
            args.fallback_ats_price,
        )
        safety_passed = benchmark_passed(benchmark_summary)
        implementation_ready = bool(validation_passed and safety_passed)

        model_path, metadata_path, coefficients = save_model_bundle(
            args.project_root,
            backtest_matrix,
            features,
            model_alpha,
            rating_alpha,
            feature_set,
            implementation_ready,
            circa_qb_integrity,
        )

        run_audit = pd.DataFrame(
            [
                {
                    "build_id": BUILD_ID,
                    "version": VERSION,
                    "schedule_source": schedule_source,
                    "matchup_build_id": matchup_audit.get("build_id"),
                    "matchup_version": matchup_audit.get("version"),
                    "matchup_qb_audit_scopes": len(matchup_qb_audit),
                    **circa_qb_integrity,
                    "locked_rating_alpha": rating_alpha,
                    "locked_feature_set": feature_set,
                    "locked_model_alpha": model_alpha,
                    "expected_weeks": len(expected),
                    "complete_weeks": int(week_audit["week_complete"].sum()),
                    "backtest_games": int(backtest_matrix["game_id"].nunique()),
                    "strict_circa_games": int(
                        backtest_matrix["reference_fallback_used"].eq(0).sum()
                    ),
                    "reference_fallback_games": int(
                        backtest_matrix["reference_fallback_used"].eq(1).sum()
                    ),
                    "reference_backfill_2025_enabled": int(
                        args.backfill_missing_2025_with_reference
                    ),
                    "validation_passed": int(validation_passed),
                    "benchmark_passed": int(safety_passed),
                    "implementation_ready": int(implementation_ready),
                    "created_at": now_string(),
                }
            ]
        )

        tables = {
            MANIFEST_TABLE: manifest,
            OCR_ATTEMPT_TABLE: attempts,
            TEAM_LINE_TABLE: team_lines,
            GAME_LINE_TABLE: game_lines,
            WEEK_AUDIT_TABLE: week_audit,
            BACKTEST_MATRIX_TABLE: backtest_matrix,
            OOF_PREDICTION_TABLE: graded_validation,
            THRESHOLD_VALIDATION_TABLE: threshold_validation,
            TOP5_VALIDATION_TABLE: top5_validation,
            BENCHMARK_PREDICTION_TABLE: benchmark_predictions,
            BENCHMARK_SUMMARY_TABLE: benchmark_summary,
            COEFFICIENT_TABLE: coefficients,
            RUN_AUDIT_TABLE: run_audit,
        }
        save_database_tables(database, tables)
        if not args.no_csv:
            save_csv_outputs(args.project_root, tables)

        print("\n" + "=" * 124)
        print("[CIRCA_LINES] 2021-2023 MANDATORY TOP-FIVE VALIDATION")
        print("=" * 124)
        print(top5_validation.to_string(index=False))
        print("\n" + "=" * 124)
        print("[CIRCA_LINES] 2021-2023 THRESHOLD DIAGNOSTICS")
        print("=" * 124)
        print(threshold_validation.to_string(index=False))
        print("\n" + "=" * 124)
        print("[CIRCA_LINES] 2024-2025 FIXED TOP-FIVE BENCHMARK")
        print("=" * 124)
        print(benchmark_summary.to_string(index=False))
        if int(backtest_matrix["reference_fallback_used"].sum()) > 0:
            print(
                "\n[CIRCA_LINES] NOTE: Hybrid rows use nflverse final spreads "
                "only where 2025 Circa lines were missing. Implementation "
                "gates use STRICT_CIRCA_ONLY results."
            )
        print("\n[CIRCA_LINES] Top coefficients:")
        print(coefficients.head(20).to_string(index=False))
        print("\n" + "=" * 124)
        print("[CIRCA_LINES] IMPLEMENTATION DECISION")
        print("=" * 124)
        print(f"Validation passed:       {validation_passed}")
        print(f"Benchmark safety passed: {safety_passed}")
        print(f"IMPLEMENTATION READY:    {implementation_ready}")
        print(f"[CIRCA_LINES] Output database: {database}")
        print(f"[CIRCA_LINES] Model bundle: {model_path}")
        print(f"[CIRCA_LINES] Metadata: {metadata_path}")
        print("[CIRCA_LINES] Paid odds API used: NO")
        print("[CIRCA_LINES] Existing model databases modified: NO")

    print(
        "[CIRCA_LINES] Completed in "
        f"{(dt.datetime.now(dt.timezone.utc) - started).total_seconds():.2f} seconds"
    )
    return 0


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _tesseract_available(explicit: Optional[Path]) -> bool:
    try:
        find_tesseract(explicit)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CIRCA_LINES] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CIRCA_LINES] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
