#!/usr/bin/env python
"""
Build a permanent NFL player identity crosswalk.

Outputs SQLite tables:
    nfl_player_crosswalk
    nfl_player_crosswalk_unmatched
    nfl_player_crosswalk_duplicate_audit
    nfl_player_crosswalk_conflict_audit

Outputs matching CSV files and a timestamped log.

The script is intentionally schema-tolerant:
- It inspects available SQLite columns before selecting data.
- It supports nflreadpy first and nfl_data_py as a fallback.
- It searches SQLite for snap-count tables containing pfr_player_id.
- It refuses ambiguous many-to-many identity mappings.
- It treats GSIS from nfl_player_master_2026 as the canonical working ID.
- It includes 2025 historical rosters and snap identities.
"""

# PRODUCTION FOUNDATION BUILD: 2026.2.1 - collision-safe aliases + vetted snap sources

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import traceback
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"

CURRENT_ROSTER_TABLE = "nfl_rosters_2026_raw"
PLAYER_MASTER_TABLE = "nfl_player_master_2026"

OUTPUT_TABLE = "nfl_player_crosswalk"
UNMATCHED_TABLE = "nfl_player_crosswalk_unmatched"
DUPLICATE_AUDIT_TABLE = "nfl_player_crosswalk_duplicate_audit"
CONFLICT_AUDIT_TABLE = "nfl_player_crosswalk_conflict_audit"

HISTORICAL_SEASONS = [2022, 2023, 2024, 2025]
CROSSWALK_VERSION = "2026.2.1"

ID_COLUMNS = [
    "gsis_id",
    "pfr_player_id",
    "espn_id",
    "sportradar_id",
    "yahoo_id",
    "rotowire_id",
    "pff_id",
    "fantasy_data_id",
    "sleeper_id",
    "esb_id",
    "gsis_it_id",
    "smart_id",
]

PROVIDER_ID_COLUMNS = [
    "pfr_player_id",
    "espn_id",
    "sportradar_id",
    "yahoo_id",
    "rotowire_id",
    "pff_id",
    "fantasy_data_id",
    "sleeper_id",
    "esb_id",
    "gsis_it_id",
    "smart_id",
]

NAME_COLUMNS = [
    "player_name",
    "full_name",
    "display_name",
    "player",
    "name",
    "first_name",
    "last_name",
    "football_name",
]

TEAM_ALIASES = {
    "ARZ": "ARI",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "HST": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "JAC": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "LV": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "LA": "LAR",
    "LAR": "LAR",
    "STL": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "SDG": "LAC",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
    "WFT": "WAS",
    "JAX": "JAX",
    "LA RAMS": "LAR",
    "LOS ANGELES RAMS": "LAR",
    "LOS ANGELES CHARGERS": "LAC",
    "WASHINGTON": "WAS",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
}

POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "HB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OT": "OL",
    "T": "OL",
    "LT": "OL",
    "RT": "OL",
    "OG": "OL",
    "G": "OL",
    "LG": "OL",
    "RG": "OL",
    "C": "OL",
    "OL": "OL",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "EDGE": "EDGE",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "CB": "DB",
    "DB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
    "KR": "ST",
    "PR": "ST",
}

SUFFIX_TOKENS = {
    "jr": "jr",
    "sr": "sr",
    "ii": "ii",
    "iii": "iii",
    "iv": "iv",
    "v": "v",
}


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging() -> tuple[logging.Logger, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"build_nfl_player_crosswalk_{stamp}.log"

    logger = logging.getLogger("nfl_player_crosswalk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger, log_path


LOGGER, LOG_PATH = configure_logging()


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def collapse_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate column labels without discarding non-null values.

    Duplicate labels can arise when a source already contains both a legacy
    provider-ID name (for example pfr_id) and its canonical compatibility
    name (pfr_player_id). Pandas returns a DataFrame for frame["column"] when
    labels are duplicated, so all source frames are made unique before any
    Series-level transformations are applied.
    """
    if frame.columns.is_unique:
        return frame.copy()

    ordered_names = list(dict.fromkeys(str(column) for column in frame.columns))
    collapsed: dict[str, pd.Series] = {}

    for name in ordered_names:
        positions = [index for index, column in enumerate(frame.columns) if str(column) == name]
        series = frame.iloc[:, positions[0]].copy()
        for position in positions[1:]:
            series = series.combine_first(frame.iloc[:, position])
        collapsed[name] = series

    return pd.DataFrame(collapsed, index=frame.index)


def apply_aliases_without_collisions(
    frame: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """Apply aliases by coalescing source values into existing targets.

    This avoids pandas duplicate-column creation from a direct rename such as
    pfr_id -> pfr_player_id when pfr_player_id is already present.
    """
    out = collapse_duplicate_columns(frame)

    for source, target in aliases.items():
        if source not in out.columns or source == target:
            continue

        source_values = out[source].copy()
        if target in out.columns:
            out[target] = out[target].combine_first(source_values)
            out = out.drop(columns=[source])
        else:
            out = out.rename(columns={source: target})

    out = collapse_duplicate_columns(out)
    if not out.columns.is_unique:
        duplicates = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns remain after alias normalization: {duplicates}")

    return out


def clean_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None

    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def normalize_id(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    return text.strip()


def normalize_gsis_id(value: Any) -> Optional[str]:
    text = normalize_id(value)
    if text is None:
        return None
    return text.upper()


def normalize_team(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    key = re.sub(r"\s+", " ", text.upper().replace(".", "").strip())
    return TEAM_ALIASES.get(key, key)


def normalize_position(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    value = text.upper().strip()
    value = re.sub(r"[^A-Z]", "", value)
    return value or None


def position_group(position: Any) -> Optional[str]:
    pos = normalize_position(position)
    if pos is None:
        return None
    return POSITION_GROUP_MAP.get(pos, pos)


def ascii_text(value: Any) -> str:
    text = clean_scalar(value) or ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def split_name_suffix(value: Any) -> tuple[str, Optional[str]]:
    text = ascii_text(value).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[']", "", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    suffix = None
    if tokens and tokens[-1] in SUFFIX_TOKENS:
        suffix = SUFFIX_TOKENS[tokens[-1]]
        tokens = tokens[:-1]
    return " ".join(tokens), suffix


def normalize_name(value: Any, preserve_suffix: bool = False) -> Optional[str]:
    base, suffix = split_name_suffix(value)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        return None
    if preserve_suffix and suffix:
        return f"{base} {suffix}"
    return base


def compact_name(value: Any, preserve_suffix: bool = False) -> Optional[str]:
    normalized = normalize_name(value, preserve_suffix=preserve_suffix)
    if normalized is None:
        return None
    return re.sub(r"[^a-z0-9]", "", normalized)


def initial_last_key(first_name: Any, last_name: Any, full_name: Any = None) -> Optional[str]:
    first = normalize_name(first_name)
    last = normalize_name(last_name)

    if not first or not last:
        full = normalize_name(full_name)
        if full:
            parts = full.split()
            if len(parts) >= 2:
                first = first or parts[0]
                last = last or parts[-1]

    if not first or not last:
        return None
    return f"{first[0]}{re.sub(r'[^a-z0-9]', '', last)}"


def normalize_date(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.strftime("%Y-%m-%d")


def numeric_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def integer_or_none(value: Any) -> Optional[int]:
    number = numeric_or_none(value)
    if number is None:
        return None
    return int(number)


def first_non_null(values: Iterable[Any]) -> Any:
    for value in values:
        if clean_scalar(value) is not None:
            return value
    return None


def most_common_non_null(values: Iterable[Any]) -> Any:
    cleaned = [clean_scalar(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = pd.Series(cleaned).value_counts(dropna=True)
    return counts.index[0]


def joined_unique(values: Iterable[Any], normalizer=None) -> Optional[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = normalizer(value) if normalizer else clean_scalar(value)
        if value is None:
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            output.append(value)
    if not output:
        return None
    return "|".join(sorted(output))


def dataframe_from_any(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def coalesce_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in candidates:
        if col in df.columns:
            mask = result.isna() | result.astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
            result.loc[mask] = df.loc[mask, col]
    return result


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# =============================================================================
# SQLITE HELPERS
# =============================================================================

def sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def sqlite_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    escaped = table_name.replace('"', '""')
    rows = conn.execute(f'PRAGMA table_info("{escaped}")').fetchall()
    return [row[1] for row in rows]


def read_sqlite_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    escaped = table_name.replace('"', '""')
    return pd.read_sql_query(f'SELECT * FROM "{escaped}"', conn)


def discover_snap_tables(conn: sqlite3.Connection) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for table in sqlite_tables(conn):
        columns = set(sqlite_columns(conn, table))
        if "pfr_player_id" not in columns:
            continue

        score = 0
        lowered = table.lower()
        if "snap" in lowered:
            score += 10
        if "advanced" in lowered:
            score += 3
        if {"offense_snaps", "defense_snaps"} & columns:
            score += 5
        if "season" in columns:
            score += 2
        if "player" in columns or "player_name" in columns:
            score += 2

        candidates.append((score, table))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [table for _, table in candidates]


# =============================================================================
# SOURCE LOADING
# =============================================================================

def load_historical_rosters() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    try:
        import nflreadpy as nfl  # type: ignore

        loaders = [
            ("load_rosters", lambda: nfl.load_rosters(HISTORICAL_SEASONS)),
            ("load_rosters", lambda: nfl.load_rosters(seasons=HISTORICAL_SEASONS)),
        ]
        for loader_name, loader in loaders:
            try:
                df = dataframe_from_any(loader())
                if not df.empty:
                    return df, f"nflreadpy.{loader_name}"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"nflreadpy.{loader_name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nflreadpy: {exc}")

    try:
        import nfl_data_py as nfl  # type: ignore

        loaders = [
            ("import_seasonal_rosters", lambda: nfl.import_seasonal_rosters(HISTORICAL_SEASONS)),
            ("import_rosters", lambda: nfl.import_rosters(HISTORICAL_SEASONS)),
        ]
        for loader_name, loader in loaders:
            if not hasattr(nfl, loader_name):
                continue
            try:
                df = dataframe_from_any(loader())
                if not df.empty:
                    return df, f"nfl_data_py.{loader_name}"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"nfl_data_py.{loader_name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nfl_data_py: {exc}")

    raise RuntimeError(
        "Unable to load historical rosters for 2022-2025. "
        "Install or upgrade nflreadpy, or retain nfl_data_py as a fallback. "
        f"Errors: {' | '.join(errors)}"
    )


def load_snap_records(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, list[str], str]:
    """Load raw snap-count identities from authoritative sources only.

    External nflverse snap counts are preferred. SQLite is a fallback limited
    to vetted raw/history tables; audit, unmatched, conflict and derived
    crosswalk tables are never treated as snap-count sources.
    """
    errors: list[str] = []

    try:
        import nflreadpy as nfl  # type: ignore

        loaders = [
            ("load_snap_counts", lambda: nfl.load_snap_counts(HISTORICAL_SEASONS)),
            ("load_snap_counts", lambda: nfl.load_snap_counts(seasons=HISTORICAL_SEASONS)),
        ]
        for loader_name, loader in loaders:
            try:
                df = dataframe_from_any(loader())
                if not df.empty:
                    return df, [f"nflreadpy.{loader_name}"], "nflreadpy"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"nflreadpy.{loader_name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nflreadpy: {exc}")

    try:
        import nfl_data_py as nfl  # type: ignore

        for loader_name in ["import_snap_counts", "import_snap_data"]:
            if not hasattr(nfl, loader_name):
                continue
            try:
                loader = getattr(nfl, loader_name)
                df = dataframe_from_any(loader(HISTORICAL_SEASONS))
                if not df.empty:
                    return df, [f"nfl_data_py.{loader_name}"], "nfl_data_py"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"nfl_data_py.{loader_name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"import nfl_data_py: {exc}")

    vetted_sqlite_tables = [
        "nfl_snap_counts_2022_2025",
        "nfl_player_snap_counts_2022_2025",
        "nfl_snap_counts",
        "nfl_player_snap_counts",
        "nfl_ol_player_snap_history",
    ]

    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    available_tables = set(sqlite_tables(conn))

    for table in vetted_sqlite_tables:
        if table not in available_tables:
            continue

        columns = set(sqlite_columns(conn, table))
        if "pfr_player_id" not in columns:
            continue

        df = read_sqlite_table(conn, table)
        if df.empty:
            continue

        if "season" in df.columns:
            seasons = pd.to_numeric(df["season"], errors="coerce")
            df = df[seasons.isin(HISTORICAL_SEASONS)].copy()

        if df.empty:
            continue

        frames.append(df)
        sources.append(f"sqlite:{table}")

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = collapse_duplicate_columns(combined).drop_duplicates()
        return combined, sources, "sqlite"

    LOGGER.warning(
        "[CROSSWALK] No authoritative snap-count source was found. "
        "Crosswalk will still build, but snap coverage cannot be audited. "
        "Errors: %s",
        " | ".join(errors),
    )
    return pd.DataFrame(), [], "none"


# =============================================================================
# SOURCE STANDARDIZATION
# =============================================================================

def standardize_source(
    df: pd.DataFrame,
    source_name: str,
    current_roster_flag: int,
    historical_roster_flag: int,
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    aliases = {
        "pfr_id": "pfr_player_id",
        "fantasydata_id": "fantasy_data_id",
        "fantasy_dataid": "fantasy_data_id",
        "sr_id": "sportradar_id",
        "sportradar_player_id": "sportradar_id",
        "player_id_gsis": "gsis_id",
        "gsis_player_id": "gsis_id",
        "birthdate": "birth_date",
        "birth_day": "birth_date",
        "draft_team": "draft_club",
        "draft_pick": "draft_number",
        "draft_pick_number": "draft_number",
        "experience": "years_exp",
        "years_experience": "years_exp",
        "rookie_season": "rookie_year",
        "season_entry": "entry_year",
    }
    out = apply_aliases_without_collisions(out, aliases)

    # Master player_id is canonical GSIS in this project.
    if source_name == PLAYER_MASTER_TABLE and "player_id" in out.columns:
        if "gsis_id" not in out.columns:
            out["gsis_id"] = out["player_id"]
        else:
            out["gsis_id"] = out["gsis_id"].where(out["gsis_id"].notna(), out["player_id"])

    out["player_name"] = coalesce_column(
        out,
        ["player_name", "full_name", "display_name", "football_name", "player", "name"],
    )
    out["first_name"] = coalesce_column(out, ["first_name", "first"])
    out["last_name"] = coalesce_column(out, ["last_name", "last"])
    out["football_name"] = coalesce_column(
        out, ["football_name", "short_name", "display_name"]
    )
    out["team"] = coalesce_column(
        out, ["team", "current_team", "club_code", "recent_team"]
    )
    out["position"] = coalesce_column(
        out, ["position", "depth_chart_position", "pos"]
    )
    out["season"] = coalesce_column(out, ["season", "roster_season"])
    out["college"] = coalesce_column(out, ["college", "college_name"])

    needed = [
        *ID_COLUMNS,
        "player_name",
        "first_name",
        "last_name",
        "football_name",
        "team",
        "position",
        "birth_date",
        "college",
        "years_exp",
        "entry_year",
        "rookie_year",
        "draft_club",
        "draft_number",
        "season",
    ]
    out = ensure_columns(out, needed)

    for col in ID_COLUMNS:
        if col == "gsis_id":
            out[col] = out[col].map(normalize_gsis_id)
        else:
            out[col] = out[col].map(normalize_id)

    out["player_name"] = out["player_name"].map(clean_scalar)
    out["first_name"] = out["first_name"].map(clean_scalar)
    out["last_name"] = out["last_name"].map(clean_scalar)
    out["football_name"] = out["football_name"].map(clean_scalar)
    out["team"] = out["team"].map(normalize_team)
    out["position"] = out["position"].map(normalize_position)
    out["birth_date"] = out["birth_date"].map(normalize_date)
    out["college"] = out["college"].map(clean_scalar)
    out["draft_club"] = out["draft_club"].map(normalize_team)

    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["years_exp"] = pd.to_numeric(out["years_exp"], errors="coerce").astype("Int64")
    out["entry_year"] = pd.to_numeric(out["entry_year"], errors="coerce").astype("Int64")
    out["rookie_year"] = pd.to_numeric(out["rookie_year"], errors="coerce").astype("Int64")
    out["draft_number"] = pd.to_numeric(out["draft_number"], errors="coerce").astype("Int64")

    out["clean_name"] = out["player_name"].map(normalize_name)
    out["clean_name_with_suffix"] = out["player_name"].map(
        lambda x: normalize_name(x, preserve_suffix=True)
    )
    out["compact_name"] = out["player_name"].map(compact_name)
    out["compact_name_with_suffix"] = out["player_name"].map(
        lambda x: compact_name(x, preserve_suffix=True)
    )
    out["initial_last_key"] = [
        initial_last_key(first, last, full)
        for first, last, full in zip(
            out["first_name"], out["last_name"], out["player_name"]
        )
    ]

    out["source_name"] = source_name
    out["current_roster_flag"] = int(current_roster_flag)
    out["historical_roster_flag"] = int(historical_roster_flag)

    return out[[
        *ID_COLUMNS,
        "player_name",
        "first_name",
        "last_name",
        "football_name",
        "clean_name",
        "clean_name_with_suffix",
        "compact_name",
        "compact_name_with_suffix",
        "initial_last_key",
        "team",
        "position",
        "birth_date",
        "college",
        "years_exp",
        "entry_year",
        "rookie_year",
        "draft_club",
        "draft_number",
        "season",
        "source_name",
        "current_roster_flag",
        "historical_roster_flag",
    ]].copy()


def standardize_snap_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "pfr_player_id",
            "player_name",
            "clean_name",
            "clean_name_with_suffix",
            "compact_name",
            "compact_name_with_suffix",
            "initial_last_key",
            "team",
            "position",
            "season",
        ])

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns={
        "pfr_id": "pfr_player_id",
        "name": "player_name",
        "player": "player_name",
        "pos": "position",
    })

    out = ensure_columns(
        out,
        ["pfr_player_id", "player_name", "team", "position", "season"],
    )
    out["pfr_player_id"] = out["pfr_player_id"].map(normalize_id)
    out["player_name"] = out["player_name"].map(clean_scalar)
    out["team"] = out["team"].map(normalize_team)
    out["position"] = out["position"].map(normalize_position)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["clean_name"] = out["player_name"].map(normalize_name)
    out["clean_name_with_suffix"] = out["player_name"].map(
        lambda x: normalize_name(x, preserve_suffix=True)
    )
    out["compact_name"] = out["player_name"].map(compact_name)
    out["compact_name_with_suffix"] = out["player_name"].map(
        lambda x: compact_name(x, preserve_suffix=True)
    )

    first_last = out["player_name"].fillna("").str.split()
    out["initial_last_key"] = [
        initial_last_key(
            parts[0] if len(parts) >= 1 else None,
            parts[-1] if len(parts) >= 2 else None,
            name,
        )
        for parts, name in zip(first_last, out["player_name"])
    ]

    return out[[
        "pfr_player_id",
        "player_name",
        "clean_name",
        "clean_name_with_suffix",
        "compact_name",
        "compact_name_with_suffix",
        "initial_last_key",
        "team",
        "position",
        "season",
    ]].copy()


# =============================================================================
# AUDIT AND MATCHING
# =============================================================================

@dataclass
class CandidateResult:
    canonical_player_id: Optional[str]
    match_method: str
    match_confidence: str
    conflict_reason: Optional[str]
    candidate_ids: list[str]
    evidence: dict[str, Any]


def duplicate_audit(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for id_column in ID_COLUMNS:
        subset = observations[observations[id_column].notna()].copy()
        if subset.empty:
            continue

        for identifier, group in subset.groupby(id_column, dropna=True):
            distinct_gsis = sorted(
                {x for x in group["gsis_id"].dropna().astype(str) if x}
            )
            distinct_names = sorted(
                {x for x in group["player_name"].dropna().astype(str) if x}
            )

            duplicate_rows = len(group) > 1
            conflicting_gsis = len(distinct_gsis) > 1
            if not duplicate_rows:
                continue

            rows.append({
                "audit_type": f"duplicate_{id_column}",
                "id_column": id_column,
                "identifier_value": identifier,
                "row_count": int(len(group)),
                "distinct_gsis_count": int(len(distinct_gsis)),
                "gsis_ids": "|".join(distinct_gsis) or None,
                "player_names": "|".join(distinct_names) or None,
                "source_names": joined_unique(group["source_name"]),
                "conflicting_gsis_flag": int(conflicting_gsis),
                "date_imported": datetime.now().isoformat(timespec="seconds"),
            })

    columns = [
        "audit_type",
        "id_column",
        "identifier_value",
        "row_count",
        "distinct_gsis_count",
        "gsis_ids",
        "player_names",
        "source_names",
        "conflicting_gsis_flag",
        "date_imported",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_conflicted_provider_values(
    observations: pd.DataFrame,
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    conflicted: dict[str, set[str]] = {column: set() for column in PROVIDER_ID_COLUMNS}
    rows: list[dict[str, Any]] = []

    for id_column in PROVIDER_ID_COLUMNS:
        subset = observations[
            observations[id_column].notna() & observations["gsis_id"].notna()
        ].copy()
        if subset.empty:
            continue

        for identifier, group in subset.groupby(id_column):
            gsis_ids = sorted(set(group["gsis_id"].dropna().astype(str)))
            if len(gsis_ids) <= 1:
                continue

            conflicted[id_column].add(str(identifier))
            rows.append({
                "conflict_type": f"one_{id_column}_to_multiple_gsis",
                "identifier_type": id_column,
                "identifier_value": identifier,
                "canonical_player_id": None,
                "candidate_canonical_ids": "|".join(gsis_ids),
                "candidate_count": len(gsis_ids),
                "player_name": most_common_non_null(group["player_name"]),
                "team": joined_unique(group["team"], normalize_team),
                "position": joined_unique(group["position"], normalize_position),
                "season": joined_unique(group["season"]),
                "source_names": joined_unique(group["source_name"]),
                "evidence": safe_json({
                    "gsis_ids": gsis_ids,
                    "names": sorted(set(group["player_name"].dropna().astype(str))),
                }),
                "resolution_status": "quarantined",
                "date_imported": datetime.now().isoformat(timespec="seconds"),
            })

    # One GSIS mapping to multiple PFR IDs is also a conflict.
    subset = observations[
        observations["gsis_id"].notna() & observations["pfr_player_id"].notna()
    ].copy()
    if not subset.empty:
        for gsis_id, group in subset.groupby("gsis_id"):
            pfr_ids = sorted(set(group["pfr_player_id"].dropna().astype(str)))
            if len(pfr_ids) <= 1:
                continue
            rows.append({
                "conflict_type": "one_gsis_to_multiple_pfr_player_id",
                "identifier_type": "gsis_id",
                "identifier_value": gsis_id,
                "canonical_player_id": gsis_id,
                "candidate_canonical_ids": gsis_id,
                "candidate_count": 1,
                "player_name": most_common_non_null(group["player_name"]),
                "team": joined_unique(group["team"], normalize_team),
                "position": joined_unique(group["position"], normalize_position),
                "season": joined_unique(group["season"]),
                "source_names": joined_unique(group["source_name"]),
                "evidence": safe_json({"pfr_player_ids": pfr_ids}),
                "resolution_status": "quarantined_pfr_ids",
                "date_imported": datetime.now().isoformat(timespec="seconds"),
            })
            conflicted["pfr_player_id"].update(pfr_ids)

    return conflicted, rows


def make_provider_maps(
    observations: pd.DataFrame,
    conflicted_values: dict[str, set[str]],
) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}

    for id_column in PROVIDER_ID_COLUMNS:
        mapping: dict[str, str] = {}
        subset = observations[
            observations[id_column].notna() & observations["gsis_id"].notna()
        ]
        for identifier, group in subset.groupby(id_column):
            identifier = str(identifier)
            if identifier in conflicted_values[id_column]:
                continue
            gsis_ids = sorted(set(group["gsis_id"].dropna().astype(str)))
            if len(gsis_ids) == 1:
                mapping[identifier] = gsis_ids[0]
        maps[id_column] = mapping

    return maps


def build_gsis_profiles(observations: pd.DataFrame) -> pd.DataFrame:
    current_priority = observations.sort_values(
        ["current_roster_flag", "historical_roster_flag", "season"],
        ascending=[False, False, False],
        na_position="last",
    )

    rows: list[dict[str, Any]] = []
    for gsis_id, group in current_priority[
        current_priority["gsis_id"].notna()
    ].groupby("gsis_id", sort=False):
        teams = [normalize_team(x) for x in group["team"]]
        teams = [x for x in teams if x]
        positions = [normalize_position(x) for x in group["position"]]
        positions = [x for x in positions if x]
        seasons = pd.to_numeric(group["season"], errors="coerce").dropna().astype(int)

        rows.append({
            "gsis_id": gsis_id,
            "player_name": first_non_null(group["player_name"]),
            "first_name": first_non_null(group["first_name"]),
            "last_name": first_non_null(group["last_name"]),
            "football_name": first_non_null(group["football_name"]),
            "clean_name": first_non_null(group["clean_name"]),
            "clean_name_with_suffix": first_non_null(group["clean_name_with_suffix"]),
            "compact_name": first_non_null(group["compact_name"]),
            "compact_name_with_suffix": first_non_null(group["compact_name_with_suffix"]),
            "initial_last_key": first_non_null(group["initial_last_key"]),
            "teams": sorted(set(teams)),
            "positions": sorted(set(positions)),
            "position_groups": sorted(set(position_group(x) for x in positions if x)),
            "birth_dates": sorted(set(group["birth_date"].dropna().astype(str))),
            "colleges": sorted(set(group["college"].dropna().astype(str))),
            "seasons": sorted(set(seasons.tolist())),
        })

    return pd.DataFrame(rows)


def profile_indexes(profiles: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    indexes: dict[str, dict[str, set[str]]] = {
        "compact_name_with_suffix": defaultdict(set),
        "compact_name": defaultdict(set),
        "initial_last_key": defaultdict(set),
    }

    for row in profiles.itertuples(index=False):
        gsis_id = row.gsis_id
        for key in indexes:
            value = clean_scalar(getattr(row, key))
            if value:
                indexes[key][value].add(gsis_id)
    return indexes


def score_name_candidate(
    snap: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    score = 0
    evidence: dict[str, Any] = {}

    exact_suffix_name = (
        snap.get("compact_name_with_suffix")
        and snap.get("compact_name_with_suffix") == profile.get("compact_name_with_suffix")
    )
    exact_name = (
        snap.get("compact_name")
        and snap.get("compact_name") == profile.get("compact_name")
    )

    if exact_suffix_name:
        score += 60
        evidence["exact_suffix_name"] = True
    elif exact_name:
        score += 45
        evidence["exact_name"] = True
    elif (
        snap.get("initial_last_key")
        and snap.get("initial_last_key") == profile.get("initial_last_key")
    ):
        score += 20
        evidence["initial_last_key"] = True
    else:
        return -999, evidence

    snap_team = normalize_team(snap.get("team"))
    teams = set(profile.get("teams") or [])
    if snap_team and snap_team in teams:
        score += 20
        evidence["team_overlap"] = snap_team

    snap_position = normalize_position(snap.get("position"))
    positions = set(profile.get("positions") or [])
    if snap_position and snap_position in positions:
        score += 15
        evidence["position_exact"] = snap_position
    elif snap_position and position_group(snap_position) in set(profile.get("position_groups") or []):
        score += 10
        evidence["position_group"] = position_group(snap_position)

    snap_season = integer_or_none(snap.get("season"))
    seasons = set(profile.get("seasons") or [])
    if snap_season is not None and snap_season in seasons:
        score += 10
        evidence["season_overlap"] = snap_season

    return score, evidence


def resolve_snap_pfr_id(
    pfr_id: str,
    group: pd.DataFrame,
    provider_maps: dict[str, dict[str, str]],
    profiles: pd.DataFrame,
    indexes: dict[str, dict[str, set[str]]],
    conflicted_values: dict[str, set[str]],
) -> CandidateResult:
    if pfr_id in conflicted_values["pfr_player_id"]:
        candidates = sorted(
            set(
                profiles[
                    profiles["gsis_id"].isin(
                        []
                    )
                ]["gsis_id"].astype(str)
            )
        )
        return CandidateResult(
            canonical_player_id=None,
            match_method="conflict_quarantined",
            match_confidence="none",
            conflict_reason="pfr_player_id_has_conflicting_gsis_mappings",
            candidate_ids=candidates,
            evidence={"pfr_player_id": pfr_id},
        )

    exact = provider_maps["pfr_player_id"].get(pfr_id)
    if exact:
        return CandidateResult(
            canonical_player_id=exact,
            match_method="exact_pfr_id",
            match_confidence="high",
            conflict_reason=None,
            candidate_ids=[exact],
            evidence={"pfr_player_id": pfr_id},
        )

    representative = {
        "player_name": most_common_non_null(group["player_name"]),
        "compact_name": most_common_non_null(group["compact_name"]),
        "compact_name_with_suffix": most_common_non_null(group["compact_name_with_suffix"]),
        "initial_last_key": most_common_non_null(group["initial_last_key"]),
        "team": most_common_non_null(group["team"]),
        "position": most_common_non_null(group["position"]),
        "season": most_common_non_null(group["season"]),
    }

    candidate_ids: set[str] = set()
    suffix_key = clean_scalar(representative["compact_name_with_suffix"])
    compact_key = clean_scalar(representative["compact_name"])
    initial_key = clean_scalar(representative["initial_last_key"])

    if suffix_key:
        candidate_ids.update(indexes["compact_name_with_suffix"].get(suffix_key, set()))
    if not candidate_ids and compact_key:
        candidate_ids.update(indexes["compact_name"].get(compact_key, set()))
    if not candidate_ids and initial_key:
        candidate_ids.update(indexes["initial_last_key"].get(initial_key, set()))

    if not candidate_ids:
        return CandidateResult(
            canonical_player_id=None,
            match_method="unmatched",
            match_confidence="none",
            conflict_reason=None,
            candidate_ids=[],
            evidence=representative,
        )

    profile_lookup = profiles.set_index("gsis_id").to_dict(orient="index")
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for candidate_id in sorted(candidate_ids):
        score, evidence = score_name_candidate(
            representative,
            profile_lookup[candidate_id],
        )
        scored.append((score, candidate_id, evidence))

    scored.sort(reverse=True)
    top_score = scored[0][0]
    top = [item for item in scored if item[0] == top_score]

    # Conservative fallback threshold:
    # exact normalized name plus at least one supporting field, or suffix-preserved
    # exact name with season/team/position corroboration.
    if top_score < 65:
        return CandidateResult(
            canonical_player_id=None,
            match_method="name_fallback_rejected",
            match_confidence="none",
            conflict_reason="insufficient_supporting_evidence",
            candidate_ids=[item[1] for item in scored],
            evidence={
                "representative": representative,
                "scores": [
                    {"canonical_player_id": cid, "score": score, "evidence": ev}
                    for score, cid, ev in scored
                ],
            },
        )

    if len(top) != 1:
        return CandidateResult(
            canonical_player_id=None,
            match_method="ambiguous_name_fallback",
            match_confidence="none",
            conflict_reason="multiple_candidates_tied",
            candidate_ids=[item[1] for item in top],
            evidence={
                "representative": representative,
                "scores": [
                    {"canonical_player_id": cid, "score": score, "evidence": ev}
                    for score, cid, ev in scored
                ],
            },
        )

    score, canonical_id, evidence = top[0]
    confidence = "medium" if score < 85 else "high"
    return CandidateResult(
        canonical_player_id=canonical_id,
        match_method="normalized_name_with_support",
        match_confidence=confidence,
        conflict_reason=None,
        candidate_ids=[canonical_id],
        evidence={
            "representative": representative,
            "score": score,
            "evidence": evidence,
        },
    )


# =============================================================================
# CROSSWALK CONSTRUCTION
# =============================================================================

def identity_quality(
    canonical_id: Optional[str],
    pfr_id: Optional[str],
    method: str,
    conflict_flag: int,
) -> str:
    if conflict_flag:
        return "conflict"
    if canonical_id and pfr_id and method in {"exact_pfr_id", "historical_roster_identifier"}:
        return "verified"
    if canonical_id and pfr_id and method == "normalized_name_with_support":
        return "supported"
    if canonical_id:
        return "canonical_only"
    return "unmatched"


def aggregate_canonical_rows(
    observations: pd.DataFrame,
    pfr_resolution: dict[str, CandidateResult],
    snap_summary: pd.DataFrame,
    conflicted_values: dict[str, set[str]],
) -> pd.DataFrame:
    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    observation_groups = observations[
        observations["gsis_id"].notna()
    ].groupby("gsis_id", sort=True)

    snap_by_pfr = (
        snap_summary.set_index("pfr_player_id").to_dict(orient="index")
        if not snap_summary.empty
        else {}
    )

    pfrs_by_canonical: dict[str, list[str]] = defaultdict(list)
    methods_by_canonical: dict[str, list[str]] = defaultdict(list)
    confidence_by_canonical: dict[str, list[str]] = defaultdict(list)

    for pfr_id, result in pfr_resolution.items():
        if result.canonical_player_id:
            pfrs_by_canonical[result.canonical_player_id].append(pfr_id)
            methods_by_canonical[result.canonical_player_id].append(result.match_method)
            confidence_by_canonical[result.canonical_player_id].append(result.match_confidence)

    for gsis_id, group in observation_groups:
        group = group.sort_values(
            ["current_roster_flag", "historical_roster_flag", "season"],
            ascending=[False, False, False],
            na_position="last",
        )

        current_rows = group[group["current_roster_flag"] == 1]
        preferred = current_rows if not current_rows.empty else group

        provider_values: dict[str, Optional[str]] = {}
        conflict_flag = 0

        for id_column in PROVIDER_ID_COLUMNS:
            values = sorted(set(group[id_column].dropna().astype(str)))
            valid_values = [
                value for value in values
                if value not in conflicted_values[id_column]
            ]
            if len(values) > 1:
                conflict_flag = 1

            if id_column == "pfr_player_id":
                resolved_values = sorted(set(pfrs_by_canonical.get(gsis_id, [])))
                all_values = sorted(set(valid_values + resolved_values))
                provider_values[id_column] = all_values[0] if len(all_values) == 1 else None
                if len(all_values) > 1:
                    conflict_flag = 1
            else:
                provider_values[id_column] = (
                    valid_values[0] if len(valid_values) == 1 else None
                )

        teams = [normalize_team(x) for x in group["team"]]
        teams = [x for x in teams if x]
        seasons = pd.to_numeric(group["season"], errors="coerce").dropna().astype(int)
        observed_seasons = sorted(set(seasons.tolist()))

        current_team = first_non_null(preferred["team"])
        current_position = first_non_null(preferred["position"])
        pfr_ids = [
            x for x in (provider_values["pfr_player_id"] or "").split("|") if x
        ]

        snap_flag = int(any(pfr_id in snap_by_pfr for pfr_id in pfr_ids))
        resolved_methods = methods_by_canonical.get(gsis_id, [])
        resolved_confidences = confidence_by_canonical.get(gsis_id, [])

        if "exact_pfr_id" in resolved_methods:
            match_method = "exact_pfr_id"
            match_confidence = "high"
        elif "normalized_name_with_support" in resolved_methods:
            match_method = "normalized_name_with_support"
            match_confidence = (
                "high" if "high" in resolved_confidences else "medium"
            )
        elif provider_values["pfr_player_id"]:
            match_method = "historical_roster_identifier"
            match_confidence = "high"
        else:
            match_method = "exact_gsis_id"
            match_confidence = "high"

        row = {
            "canonical_player_id": gsis_id,
            "gsis_id": gsis_id,
            **provider_values,
            "player_name": first_non_null(preferred["player_name"]),
            "first_name": first_non_null(preferred["first_name"]),
            "last_name": first_non_null(preferred["last_name"]),
            "football_name": first_non_null(preferred["football_name"]),
            "clean_name": first_non_null(preferred["clean_name"]),
            "compact_name": first_non_null(preferred["compact_name"]),
            "initial_last_key": first_non_null(preferred["initial_last_key"]),
            "current_team": normalize_team(current_team),
            "historical_teams": "|".join(sorted(set(teams))) if teams else None,
            "position": normalize_position(current_position),
            "position_group": position_group(current_position),
            "birth_date": first_non_null(preferred["birth_date"]),
            "college": first_non_null(preferred["college"]),
            "years_exp": integer_or_none(first_non_null(preferred["years_exp"])),
            "entry_year": integer_or_none(first_non_null(preferred["entry_year"])),
            "rookie_year": integer_or_none(first_non_null(preferred["rookie_year"])),
            "draft_club": normalize_team(first_non_null(preferred["draft_club"])),
            "draft_number": integer_or_none(first_non_null(preferred["draft_number"])),
            "current_roster_flag": int(group["current_roster_flag"].max()),
            "historical_roster_flag": int(group["historical_roster_flag"].max()),
            "snap_history_flag": snap_flag,
            "first_season": min(observed_seasons) if observed_seasons else None,
            "last_season": max(observed_seasons) if observed_seasons else None,
            "seasons_observed": "|".join(map(str, observed_seasons)) if observed_seasons else None,
            "match_method": match_method,
            "match_confidence": match_confidence,
            "identity_conflict_flag": conflict_flag,
            "identity_quality_flag": identity_quality(
                gsis_id,
                provider_values["pfr_player_id"],
                match_method,
                conflict_flag,
            ),
            "crosswalk_version": CROSSWALK_VERSION,
            "date_imported": now,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_snap_summary(snap_records: pd.DataFrame) -> pd.DataFrame:
    if snap_records.empty:
        return pd.DataFrame(columns=[
            "pfr_player_id",
            "player_name",
            "team",
            "position",
            "first_season",
            "last_season",
            "seasons_observed",
            "snap_row_count",
        ])

    rows: list[dict[str, Any]] = []
    for pfr_id, group in snap_records[
        snap_records["pfr_player_id"].notna()
    ].groupby("pfr_player_id"):
        seasons = pd.to_numeric(group["season"], errors="coerce").dropna().astype(int)
        observed = sorted(set(seasons.tolist()))
        rows.append({
            "pfr_player_id": pfr_id,
            "player_name": most_common_non_null(group["player_name"]),
            "team": joined_unique(group["team"], normalize_team),
            "position": joined_unique(group["position"], normalize_position),
            "first_season": min(observed) if observed else None,
            "last_season": max(observed) if observed else None,
            "seasons_observed": "|".join(map(str, observed)) if observed else None,
            "snap_row_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def append_unmatched_rows_to_crosswalk(
    crosswalk: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> pd.DataFrame:
    if unmatched.empty:
        return crosswalk

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    for row in unmatched.itertuples(index=False):
        player_name = clean_scalar(row.player_name)
        pos = clean_scalar(row.position)
        rows.append({
            "canonical_player_id": None,
            "gsis_id": None,
            "pfr_player_id": row.pfr_player_id,
            "espn_id": None,
            "sportradar_id": None,
            "yahoo_id": None,
            "rotowire_id": None,
            "pff_id": None,
            "fantasy_data_id": None,
            "sleeper_id": None,
            "esb_id": None,
            "gsis_it_id": None,
            "smart_id": None,
            "player_name": player_name,
            "first_name": None,
            "last_name": None,
            "football_name": None,
            "clean_name": normalize_name(player_name),
            "compact_name": compact_name(player_name),
            "initial_last_key": initial_last_key(None, None, player_name),
            "current_team": None,
            "historical_teams": row.team,
            "position": pos,
            "position_group": position_group(pos),
            "birth_date": None,
            "college": None,
            "years_exp": None,
            "entry_year": None,
            "rookie_year": None,
            "draft_club": None,
            "draft_number": None,
            "current_roster_flag": 0,
            "historical_roster_flag": 0,
            "snap_history_flag": 1,
            "first_season": row.first_season,
            "last_season": row.last_season,
            "seasons_observed": row.seasons_observed,
            "match_method": row.match_method,
            "match_confidence": "none",
            "identity_conflict_flag": int(row.conflict_reason is not None),
            "identity_quality_flag": (
                "conflict" if row.conflict_reason is not None else "unmatched"
            ),
            "crosswalk_version": CROSSWALK_VERSION,
            "date_imported": now,
        })

    return pd.concat([crosswalk, pd.DataFrame(rows)], ignore_index=True, sort=False)


def construct_outputs(
    observations: pd.DataFrame,
    snap_records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    duplicate_df = duplicate_audit(observations)
    conflicted_values, conflict_rows = build_conflicted_provider_values(observations)
    provider_maps = make_provider_maps(observations, conflicted_values)
    profiles = build_gsis_profiles(observations)
    indexes = profile_indexes(profiles)
    snap_summary = build_snap_summary(snap_records)

    resolutions: dict[str, CandidateResult] = {}

    # First pass: resolve every required snap PFR ID.
    if not snap_records.empty:
        for pfr_id, group in snap_records[
            snap_records["pfr_player_id"].notna()
        ].groupby("pfr_player_id"):
            resolutions[str(pfr_id)] = resolve_snap_pfr_id(
                str(pfr_id),
                group,
                provider_maps,
                profiles,
                indexes,
                conflicted_values,
            )

    # Safeguard: one canonical GSIS ID may not retain multiple PFR IDs.
    # Quarantine all involved PFR IDs rather than selecting one arbitrarily.
    pfrs_by_canonical: dict[str, list[str]] = defaultdict(list)
    for pfr_id, result in resolutions.items():
        if result.canonical_player_id:
            pfrs_by_canonical[result.canonical_player_id].append(pfr_id)

    for canonical_id, pfr_ids in pfrs_by_canonical.items():
        unique_pfr_ids = sorted(set(pfr_ids))
        if len(unique_pfr_ids) <= 1:
            continue

        for pfr_id in unique_pfr_ids:
            prior = resolutions[pfr_id]
            resolutions[pfr_id] = CandidateResult(
                canonical_player_id=None,
                match_method="conflict_quarantined",
                match_confidence="none",
                conflict_reason="one_gsis_id_resolved_to_multiple_pfr_player_ids",
                candidate_ids=[canonical_id],
                evidence={
                    "canonical_player_id": canonical_id,
                    "pfr_player_ids": unique_pfr_ids,
                    "prior_match_method": prior.match_method,
                    "prior_evidence": prior.evidence,
                },
            )

        profile_row = profiles[profiles["gsis_id"] == canonical_id]
        conflict_rows.append({
            "conflict_type": "one_gsis_to_multiple_pfr_player_id",
            "identifier_type": "gsis_id",
            "identifier_value": canonical_id,
            "canonical_player_id": canonical_id,
            "candidate_canonical_ids": canonical_id,
            "candidate_count": 1,
            "player_name": (
                profile_row.iloc[0]["player_name"] if not profile_row.empty else None
            ),
            "team": (
                "|".join(profile_row.iloc[0]["teams"])
                if not profile_row.empty and profile_row.iloc[0]["teams"]
                else None
            ),
            "position": (
                "|".join(profile_row.iloc[0]["positions"])
                if not profile_row.empty and profile_row.iloc[0]["positions"]
                else None
            ),
            "season": (
                "|".join(map(str, profile_row.iloc[0]["seasons"]))
                if not profile_row.empty and profile_row.iloc[0]["seasons"]
                else None
            ),
            "source_names": "resolved_snap_crosswalk",
            "evidence": safe_json({"pfr_player_ids": unique_pfr_ids}),
            "resolution_status": "quarantined_all_pfr_ids",
            "date_imported": datetime.now().isoformat(timespec="seconds"),
        })

    # Second pass: create unmatched and conflict audit rows after all
    # one-to-many safeguards have been applied.
    unmatched_rows: list[dict[str, Any]] = []
    if not snap_summary.empty:
        snap_summary_lookup = snap_summary.set_index("pfr_player_id").to_dict(
            orient="index"
        )
        for pfr_id, result in resolutions.items():
            if result.canonical_player_id is not None:
                continue

            summary_row = snap_summary_lookup[pfr_id]
            unmatched_rows.append({
                "pfr_player_id": pfr_id,
                "player_name": summary_row["player_name"],
                "team": summary_row["team"],
                "position": summary_row["position"],
                "first_season": summary_row["first_season"],
                "last_season": summary_row["last_season"],
                "seasons_observed": summary_row["seasons_observed"],
                "snap_row_count": summary_row["snap_row_count"],
                "match_method": result.match_method,
                "conflict_reason": result.conflict_reason,
                "candidate_canonical_ids": "|".join(result.candidate_ids) or None,
                "candidate_count": len(result.candidate_ids),
                "evidence": safe_json(result.evidence),
                "crosswalk_version": CROSSWALK_VERSION,
                "date_imported": datetime.now().isoformat(timespec="seconds"),
            })

            if result.conflict_reason:
                conflict_rows.append({
                    "conflict_type": result.match_method,
                    "identifier_type": "pfr_player_id",
                    "identifier_value": pfr_id,
                    "canonical_player_id": None,
                    "candidate_canonical_ids": "|".join(result.candidate_ids) or None,
                    "candidate_count": len(result.candidate_ids),
                    "player_name": summary_row["player_name"],
                    "team": summary_row["team"],
                    "position": summary_row["position"],
                    "season": summary_row["seasons_observed"],
                    "source_names": "snap_counts",
                    "evidence": safe_json(result.evidence),
                    "resolution_status": "unresolved",
                    "date_imported": datetime.now().isoformat(timespec="seconds"),
                })

    unmatched_columns = [
        "pfr_player_id",
        "player_name",
        "team",
        "position",
        "first_season",
        "last_season",
        "seasons_observed",
        "snap_row_count",
        "match_method",
        "conflict_reason",
        "candidate_canonical_ids",
        "candidate_count",
        "evidence",
        "crosswalk_version",
        "date_imported",
    ]
    unmatched_df = pd.DataFrame(unmatched_rows, columns=unmatched_columns)

    conflict_columns = [
        "conflict_type",
        "identifier_type",
        "identifier_value",
        "canonical_player_id",
        "candidate_canonical_ids",
        "candidate_count",
        "player_name",
        "team",
        "position",
        "season",
        "source_names",
        "evidence",
        "resolution_status",
        "date_imported",
    ]
    conflict_df = pd.DataFrame(conflict_rows, columns=conflict_columns)

    crosswalk = aggregate_canonical_rows(
        observations,
        resolutions,
        snap_summary,
        conflicted_values,
    )
    crosswalk = append_unmatched_rows_to_crosswalk(crosswalk, unmatched_df)

    # Final PFR accounting safeguard:
    # A snap PFR ID can initially resolve to a GSIS ID but later be removed from
    # that canonical row when the GSIS record contains multiple competing PFR
    # identifiers. Any such ID must be retained as a quarantined standalone row
    # and in the unmatched/conflict audits rather than being omitted.
    source_snap_pfr_ids = set(
        snap_records["pfr_player_id"].dropna().astype(str)
    )
    represented_pfr_ids: set[str] = set()
    for value in crosswalk["pfr_player_id"].dropna().astype(str):
        represented_pfr_ids.update(
            part for part in value.split("|") if part
        )

    missing_after_aggregation = sorted(
        source_snap_pfr_ids - represented_pfr_ids
    )

    if missing_after_aggregation:
        snap_summary_lookup = snap_summary.set_index(
            "pfr_player_id"
        ).to_dict(orient="index")

        existing_unmatched_ids = set(
            unmatched_df["pfr_player_id"].dropna().astype(str)
        ) if not unmatched_df.empty else set()

        supplemental_unmatched_rows: list[dict[str, Any]] = []
        supplemental_conflict_rows: list[dict[str, Any]] = []

        for pfr_id in missing_after_aggregation:
            summary_row = snap_summary_lookup.get(pfr_id, {})
            prior_result = resolutions.get(pfr_id)
            candidate_ids = (
                prior_result.candidate_ids
                if prior_result is not None
                else []
            )
            if (
                prior_result is not None
                and prior_result.canonical_player_id
                and prior_result.canonical_player_id not in candidate_ids
            ):
                candidate_ids = [
                    prior_result.canonical_player_id,
                    *candidate_ids,
                ]
            candidate_ids = sorted(set(candidate_ids))

            evidence = {
                "reason": (
                    "pfr_id_removed_from_canonical_row_due_to_multiple_"
                    "competing_pfr_identifiers"
                ),
                "prior_resolution": {
                    "canonical_player_id": (
                        prior_result.canonical_player_id
                        if prior_result is not None
                        else None
                    ),
                    "match_method": (
                        prior_result.match_method
                        if prior_result is not None
                        else None
                    ),
                    "match_confidence": (
                        prior_result.match_confidence
                        if prior_result is not None
                        else None
                    ),
                    "evidence": (
                        prior_result.evidence
                        if prior_result is not None
                        else None
                    ),
                },
            }

            if pfr_id not in existing_unmatched_ids:
                supplemental_unmatched_rows.append({
                    "pfr_player_id": pfr_id,
                    "player_name": summary_row.get("player_name"),
                    "team": summary_row.get("team"),
                    "position": summary_row.get("position"),
                    "first_season": summary_row.get("first_season"),
                    "last_season": summary_row.get("last_season"),
                    "seasons_observed": summary_row.get("seasons_observed"),
                    "snap_row_count": summary_row.get("snap_row_count"),
                    "match_method": "conflict_quarantined",
                    "conflict_reason": (
                        "canonical_player_has_multiple_pfr_player_ids"
                    ),
                    "candidate_canonical_ids": (
                        "|".join(candidate_ids) or None
                    ),
                    "candidate_count": len(candidate_ids),
                    "evidence": safe_json(evidence),
                    "crosswalk_version": CROSSWALK_VERSION,
                    "date_imported": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                })

            supplemental_conflict_rows.append({
                "conflict_type": "canonical_multiple_pfr_quarantine",
                "identifier_type": "pfr_player_id",
                "identifier_value": pfr_id,
                "canonical_player_id": None,
                "candidate_canonical_ids": (
                    "|".join(candidate_ids) or None
                ),
                "candidate_count": len(candidate_ids),
                "player_name": summary_row.get("player_name"),
                "team": summary_row.get("team"),
                "position": summary_row.get("position"),
                "season": summary_row.get("seasons_observed"),
                "source_names": "snap_counts",
                "evidence": safe_json(evidence),
                "resolution_status": "quarantined",
                "date_imported": datetime.now().isoformat(
                    timespec="seconds"
                ),
            })

        if supplemental_unmatched_rows:
            supplemental_unmatched_df = pd.DataFrame(
                supplemental_unmatched_rows,
                columns=unmatched_columns,
            )
            unmatched_df = pd.concat(
                [unmatched_df, supplemental_unmatched_df],
                ignore_index=True,
                sort=False,
            )
            crosswalk = append_unmatched_rows_to_crosswalk(
                crosswalk,
                supplemental_unmatched_df,
            )

        if supplemental_conflict_rows:
            conflict_df = pd.concat(
                [
                    conflict_df,
                    pd.DataFrame(
                        supplemental_conflict_rows,
                        columns=conflict_columns,
                    ),
                ],
                ignore_index=True,
                sort=False,
            )

    ordered_columns = [
        "canonical_player_id",
        "gsis_id",
        "pfr_player_id",
        "espn_id",
        "sportradar_id",
        "yahoo_id",
        "rotowire_id",
        "pff_id",
        "fantasy_data_id",
        "sleeper_id",
        "esb_id",
        "gsis_it_id",
        "smart_id",
        "player_name",
        "first_name",
        "last_name",
        "football_name",
        "clean_name",
        "compact_name",
        "initial_last_key",
        "current_team",
        "historical_teams",
        "position",
        "position_group",
        "birth_date",
        "college",
        "years_exp",
        "entry_year",
        "rookie_year",
        "draft_club",
        "draft_number",
        "current_roster_flag",
        "historical_roster_flag",
        "snap_history_flag",
        "first_season",
        "last_season",
        "seasons_observed",
        "match_method",
        "match_confidence",
        "identity_conflict_flag",
        "identity_quality_flag",
        "crosswalk_version",
        "date_imported",
    ]
    crosswalk = ensure_columns(crosswalk, ordered_columns)[ordered_columns]

    return crosswalk, unmatched_df, duplicate_df, conflict_df


# =============================================================================
# VALIDATION, PERSISTENCE, REPORTING
# =============================================================================

def validate_outputs(
    crosswalk: pd.DataFrame,
    unmatched: pd.DataFrame,
    snap_records: pd.DataFrame,
) -> None:
    if crosswalk.empty:
        raise RuntimeError("Crosswalk output is empty.")

    mapped = crosswalk[crosswalk["canonical_player_id"].notna()].copy()

    duplicate_gsis = mapped[
        mapped["gsis_id"].notna()
    ]["gsis_id"].duplicated(keep=False)
    if duplicate_gsis.any():
        examples = mapped.loc[duplicate_gsis, ["gsis_id", "player_name"]].head(20)
        raise RuntimeError(
            "Final output contains duplicate canonical GSIS rows:\n"
            + examples.to_string(index=False)
        )

    if not snap_records.empty:
        source_pfr = set(snap_records["pfr_player_id"].dropna().astype(str))
        output_pfr: set[str] = set()
        for value in crosswalk["pfr_player_id"].dropna().astype(str):
            output_pfr.update(x for x in value.split("|") if x)

        missing = sorted(source_pfr - output_pfr)
        if missing:
            raise RuntimeError(
                f"Final crosswalk omitted {len(missing)} snap PFR IDs. "
                f"Examples: {missing[:20]}"
            )

        unmatched_ids = set(unmatched["pfr_player_id"].dropna().astype(str))
        mapped_ids = source_pfr - unmatched_ids
        if len(mapped_ids) + len(unmatched_ids) != len(source_pfr):
            raise RuntimeError("Snap PFR accounting failed validation.")


def add_sqlite_indexes(conn: sqlite3.Connection) -> None:
    index_statements = [
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_canonical ON "{OUTPUT_TABLE}" (canonical_player_id)',
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_gsis ON "{OUTPUT_TABLE}" (gsis_id)',
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_pfr ON "{OUTPUT_TABLE}" (pfr_player_id)',
        f'CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_clean_name ON "{OUTPUT_TABLE}" (clean_name)',
        f'CREATE INDEX IF NOT EXISTS idx_{UNMATCHED_TABLE}_pfr ON "{UNMATCHED_TABLE}" (pfr_player_id)',
    ]
    for statement in index_statements:
        conn.execute(statement)
    conn.commit()


def persist_outputs(
    conn: sqlite3.Connection,
    crosswalk: pd.DataFrame,
    unmatched: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
) -> dict[str, Path]:
    frames = {
        OUTPUT_TABLE: crosswalk,
        UNMATCHED_TABLE: unmatched,
        DUPLICATE_AUDIT_TABLE: duplicate_df,
        CONFLICT_AUDIT_TABLE: conflict_df,
    }

    for table_name, frame in frames.items():
        if len(frame.columns) == 0:
            raise RuntimeError(
                f"Refusing to persist {table_name}: DataFrame has zero columns. "
                "Every output, including an empty audit, must have an explicit schema."
            )
        frame.to_sql(table_name, conn, if_exists="replace", index=False)
        LOGGER.info("[CROSSWALK] Saved %s rows to %s", len(frame), table_name)

    add_sqlite_indexes(conn)

    paths: dict[str, Path] = {}
    for table_name, frame in frames.items():
        path = OUTPUT_DIR / f"{table_name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[table_name] = path
        LOGGER.info("[CROSSWALK] CSV saved: %s", path)

    return paths


def explode_pfr_crosswalk(crosswalk: pd.DataFrame) -> pd.DataFrame:
    mapped = crosswalk[
        crosswalk["canonical_player_id"].notna()
        & crosswalk["pfr_player_id"].notna()
    ][["canonical_player_id", "pfr_player_id"]].copy()

    if mapped.empty:
        return mapped

    mapped["pfr_player_id"] = mapped["pfr_player_id"].astype(str).str.split("|")
    return mapped.explode("pfr_player_id").drop_duplicates()


def print_report(
    observations: pd.DataFrame,
    snap_records: pd.DataFrame,
    crosswalk: pd.DataFrame,
    unmatched: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
) -> None:
    LOGGER.info("")
    LOGGER.info("=" * 78)
    LOGGER.info("[CROSSWALK] BUILD SUMMARY")
    LOGGER.info("=" * 78)

    canonical = crosswalk[crosswalk["canonical_player_id"].notna()].copy()
    LOGGER.info("[CROSSWALK] Canonical players: %s", len(canonical))
    LOGGER.info(
        "[CROSSWALK] Canonical players with PFR ID: %s",
        canonical["pfr_player_id"].notna().sum(),
    )
    LOGGER.info(
        "[CROSSWALK] Duplicate-audit rows: %s", len(duplicate_df)
    )
    LOGGER.info(
        "[CROSSWALK] Conflict-audit rows: %s", len(conflict_df)
    )

    LOGGER.info("")
    LOGGER.info("[CROSSWALK] Match counts by method/confidence:")
    if crosswalk.empty:
        LOGGER.info("  none")
    else:
        counts = (
            crosswalk.groupby(["match_method", "match_confidence"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        for row in counts.itertuples(index=False):
            LOGGER.info(
                "  %-38s %-8s %s",
                str(row.match_method),
                str(row.match_confidence),
                int(row.rows),
            )

    current_ids = set(
        observations.loc[
            observations["current_roster_flag"] == 1, "gsis_id"
        ].dropna().astype(str)
    )
    historical_ids = set(
        observations.loc[
            observations["historical_roster_flag"] == 1, "gsis_id"
        ].dropna().astype(str)
    )
    canonical_ids = set(canonical["canonical_player_id"].dropna().astype(str))

    current_mapped = len(current_ids & canonical_ids)
    historical_mapped = len(historical_ids & canonical_ids)

    LOGGER.info("")
    LOGGER.info(
        "[CROSSWALK] Current-roster canonical coverage: %s/%s (%.2f%%)",
        current_mapped,
        len(current_ids),
        100.0 * current_mapped / len(current_ids) if current_ids else 0.0,
    )
    LOGGER.info(
        "[CROSSWALK] Historical-roster canonical coverage: %s/%s (%.2f%%)",
        historical_mapped,
        len(historical_ids),
        100.0 * historical_mapped / len(historical_ids) if historical_ids else 0.0,
    )

    if not snap_records.empty:
        snap_ids = set(snap_records["pfr_player_id"].dropna().astype(str))
        pfr_map = explode_pfr_crosswalk(crosswalk)
        mapped_snap_ids = set(pfr_map["pfr_player_id"].dropna().astype(str))
        mapped_count = len(snap_ids & mapped_snap_ids)
        unmatched_count = len(snap_ids - mapped_snap_ids)

        LOGGER.info("")
        LOGGER.info(
            "[CROSSWALK] Snap PFR IDs mapped: %s/%s (%.2f%%)",
            mapped_count,
            len(snap_ids),
            100.0 * mapped_count / len(snap_ids) if snap_ids else 0.0,
        )
        LOGGER.info(
            "[CROSSWALK] Snap PFR IDs unmatched/conflicted: %s",
            unmatched_count,
        )
        LOGGER.info(
            "[CROSSWALK] Snap rows represented: %s",
            len(snap_records),
        )
    else:
        LOGGER.info("[CROSSWALK] Snap coverage unavailable: no snap rows loaded.")

    LOGGER.info("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    started = datetime.now()
    LOGGER.info("[CROSSWALK] Building NFL player identity crosswalk")
    LOGGER.info("[CROSSWALK] Database: %s", DB_PATH)
    LOGGER.info("[CROSSWALK] Historical seasons: %s", HISTORICAL_SEASONS)
    LOGGER.info("[CROSSWALK] Version: %s", CROSSWALK_VERSION)

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        if not sqlite_table_exists(conn, CURRENT_ROSTER_TABLE):
            raise RuntimeError(
                f"Required table is missing: {CURRENT_ROSTER_TABLE}"
            )
        if not sqlite_table_exists(conn, PLAYER_MASTER_TABLE):
            raise RuntimeError(
                f"Required table is missing: {PLAYER_MASTER_TABLE}"
            )

        current_raw = read_sqlite_table(conn, CURRENT_ROSTER_TABLE)
        master_raw = read_sqlite_table(conn, PLAYER_MASTER_TABLE)

        LOGGER.info(
            "[CROSSWALK] Current raw roster rows: %s", len(current_raw)
        )
        LOGGER.info(
            "[CROSSWALK] Current player-master rows: %s", len(master_raw)
        )

        historical_raw, historical_source = load_historical_rosters()
        LOGGER.info(
            "[CROSSWALK] Historical roster rows: %s via %s",
            len(historical_raw),
            historical_source,
        )

        snap_raw, snap_sources, snap_source_type = load_snap_records(conn)
        LOGGER.info(
            "[CROSSWALK] Snap rows: %s via %s (%s)",
            len(snap_raw),
            ", ".join(snap_sources) if snap_sources else "none",
            snap_source_type,
        )

        current = standardize_source(
            current_raw,
            CURRENT_ROSTER_TABLE,
            current_roster_flag=1,
            historical_roster_flag=0,
        )
        master = standardize_source(
            master_raw,
            PLAYER_MASTER_TABLE,
            current_roster_flag=1,
            historical_roster_flag=0,
        )
        historical = standardize_source(
            historical_raw,
            historical_source,
            current_roster_flag=0,
            historical_roster_flag=1,
        )
        historical = historical[
            historical["season"].isin(HISTORICAL_SEASONS)
            | historical["season"].isna()
        ].copy()

        observations = pd.concat(
            [current, master, historical],
            ignore_index=True,
            sort=False,
        )
        observations = observations.drop_duplicates().reset_index(drop=True)

        LOGGER.info(
            "[CROSSWALK] Standardized identity observations: %s",
            len(observations),
        )
        LOGGER.info(
            "[CROSSWALK] Distinct GSIS IDs observed: %s",
            observations["gsis_id"].nunique(dropna=True),
        )
        LOGGER.info(
            "[CROSSWALK] Distinct roster PFR IDs observed: %s",
            observations["pfr_player_id"].nunique(dropna=True),
        )

        snap_records = standardize_snap_records(snap_raw)
        snap_records = snap_records[snap_records["pfr_player_id"].notna()].copy()
        snap_records = snap_records.drop_duplicates().reset_index(drop=True)

        crosswalk, unmatched, duplicate_df, conflict_df = construct_outputs(
            observations,
            snap_records,
        )

        validate_outputs(crosswalk, unmatched, snap_records)
        persist_outputs(
            conn,
            crosswalk,
            unmatched,
            duplicate_df,
            conflict_df,
        )
        print_report(
            observations,
            snap_records,
            crosswalk,
            unmatched,
            duplicate_df,
            conflict_df,
        )

    elapsed = (datetime.now() - started).total_seconds()
    LOGGER.info("[CROSSWALK] Completed successfully in %.2f seconds", elapsed)
    LOGGER.info("[CROSSWALK] Log saved: %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("[CROSSWALK] Cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("[CROSSWALK] FAILED: %s", exc)
        LOGGER.error(traceback.format_exc())
        LOGGER.error("[CROSSWALK] Log saved: %s", LOG_PATH)
        raise SystemExit(1)
