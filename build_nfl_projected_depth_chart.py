#!/usr/bin/env python
"""Build the canonical 2026 NFL projected depth chart.

Required SQLite inputs
----------------------
- nfl_player_performance_inputs_2026
- nfl_player_master_2026

Automatic availability inputs
-----------------------------
- ESPN current rosters, injury statuses and ordered depth charts for all teams
- QB and OL starter tables are refreshed from the first available source player
- No manual QB CSV is required or read

Outputs
-------
- nfl_projected_depth_chart_2026
- nfl_projected_depth_chart_audit_2026

Design rules
------------
1. Historical player-performance grades are retained; identified new players
   without a grade enter at replacement level and are flagged downstream.
2. Performance and roster/master fields are joined once by canonical player_id.
3. Player quality, availability and depth-order evidence remain separate.
4. OL performance consumes the PFF blocking-talent grade from the canonical
   player table without a second shrink or availability discount.
5. LT/LG/C/RG/RT assignments come from a position-specific current depth chart;
   player quality and historical durability never invent starter assignments.
6. Reserve/injured/developmental roster statuses cannot be projected starters.
7. The current source selects QB and OL starters automatically on every refresh.
8. No empirical mean/std normalization is performed in this file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import re
import sqlite3
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import nfl_live_depth_2026 as live_depth

SEASON = 2026
BUILD_ID = "NFL_PROJECTED_DEPTH_CHART_2026_CANONICAL_V5"
VERSION = "v5_4_unique_ol_starter_assignment"
SOURCE_UNAVAILABLE_EXIT_CODE = 20

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

PERFORMANCE_TABLE = "nfl_player_performance_inputs_2026"
MASTER_TABLE = "nfl_player_master_2026"
QB_STARTER_TABLE = "nfl_projected_qb_starters_2026"
OL_STARTER_TABLE = "nfl_projected_ol_starters_2026"
OUTPUT_TABLE = "nfl_projected_depth_chart_2026"
AUDIT_TABLE = "nfl_projected_depth_chart_audit_2026"

ESPN_DEPTH_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"
ESPN_TEAM_IDS = {
    "ARI": "22", "ATL": "1", "BAL": "33", "BUF": "2",
    "CAR": "29", "CHI": "3", "CIN": "4", "CLE": "5",
    "DAL": "6", "DEN": "7", "DET": "8", "GB": "9",
    "HOU": "34", "IND": "11", "JAX": "30", "KC": "12",
    "LAC": "24", "LAR": "14", "LV": "13", "MIA": "15",
    "MIN": "16", "NE": "17", "NO": "18", "NYG": "19",
    "NYJ": "20", "PHI": "21", "PIT": "23", "SEA": "26",
    "SF": "25", "TB": "27", "TEN": "10", "WAS": "28",
}
OL_STARTER_SLOTS = ("LT", "LG", "C", "RG", "RT")

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}

# Explicit source-to-canonical identity aliases only. Do not use fuzzy player
# matching for authoritative starter assignments.
PLAYER_NAME_MATCH_ALIASES = {
    ("NYJ", "olufashanu"): "olumuyiwafashanu",
}

POSITION_ALIASES = {
    "HB": "RB", "T": "OT", "G": "OG", "DE": "EDGE", "ED": "EDGE",
    "NT": "DT", "ILB": "LB", "MLB": "LB", "OLB": "LB_EDGE",
    "NB": "SLOT", "NCB": "SLOT", "PK": "K",
}

OL_ROLES = {"LT", "LG", "C", "RG", "RT", "OT", "OG", "OL"}
UNAVAILABLE_PATTERNS = (
    "INJURED RESERVE", "RESERVE/INJURED", "RESERVE INJURED", " IR ",
    "PHYSICALLY UNABLE", " PUP", "SUSPENDED", " NFI", "RESERVE/NFI",
    "RETIRED", "EXEMPT",
)
UNAVAILABLE_STATUS_CODES = {
    "RES", "IR", "PUP", "NFI", "SUS", "SUSP", "OUT",
    "RET", "EXE", "CUT", "DEV", "INA",
}

DEFAULT_QB_STARTERS = {
    "ARI": "Jacoby Brissett", "ATL": "Tua Tagovailoa",
    "BAL": "Lamar Jackson", "BUF": "Josh Allen", "CAR": "Bryce Young",
    "CHI": "Caleb Williams", "CIN": "Joe Burrow",
    "CLE": "Shedeur Sanders", "DAL": "Dak Prescott", "DEN": "Bo Nix",
    "DET": "Jared Goff", "GB": "Jordan Love", "HOU": "C.J. Stroud",
    "IND": "Daniel Jones", "JAX": "Trevor Lawrence",
    "KC": "Patrick Mahomes", "LAC": "Justin Herbert",
    "LAR": "Matthew Stafford", "LV": "Kirk Cousins",
    "MIA": "Malik Willis", "MIN": "Kyler Murray", "NE": "Drake Maye",
    "NO": "Tyler Shough", "NYG": "Jaxson Dart", "NYJ": "Geno Smith",
    "PHI": "Jalen Hurts", "PIT": "Aaron Rodgers",
    "SEA": "Sam Darnold", "SF": "Brock Purdy",
    "TB": "Baker Mayfield", "TEN": "Cam Ward",
    "WAS": "Jayden Daniels",
}

SNAP_SHARE_BY_ROLE = {
    "QB": {1: 1.00, 2: 0.05, 3: 0.00},
    "RB": {1: 0.58, 2: 0.30, 3: 0.10, 4: 0.02},
    "FB": {1: 0.18, 2: 0.04},
    "WR": {1: 0.88, 2: 0.80, 3: 0.65, 4: 0.38, 5: 0.20, 6: 0.08},
    "TE": {1: 0.78, 2: 0.36, 3: 0.14, 4: 0.05},
    "OT": {1: 0.98, 2: 0.95, 3: 0.15, 4: 0.08},
    "OG": {1: 0.98, 2: 0.95, 3: 0.15, 4: 0.08},
    "C": {1: 0.98, 2: 0.12},
    "OL": {1: 0.90, 2: 0.80, 3: 0.65, 4: 0.25, 5: 0.15, 6: 0.08},
    "EDGE": {1: 0.78, 2: 0.70, 3: 0.42, 4: 0.22},
    "DT": {1: 0.68, 2: 0.60, 3: 0.38, 4: 0.22},
    "LB": {1: 0.88, 2: 0.74, 3: 0.52, 4: 0.22},
    "CB": {1: 0.92, 2: 0.88, 3: 0.35, 4: 0.15},
    "SLOT": {1: 0.68, 2: 0.25},
    "S": {1: 0.90, 2: 0.88, 3: 0.30, 4: 0.15},
    "K": {1: 1.00, 2: 0.00}, "P": {1: 1.00, 2: 0.00},
    "LS": {1: 1.00, 2: 0.00},
}

STARTER_SLOTS = [
    ("QB1", ("QB",)), ("RB1", ("RB",)),
    ("WR1", ("WR",)), ("WR2", ("WR",)), ("WR3", ("WR",)),
    ("TE1", ("TE",)),
    ("LT", ("LT", "OT", "OL")), ("LG", ("LG", "OG", "OL")),
    ("C", ("C", "OL", "OG")), ("RG", ("RG", "OG", "OL")),
    ("RT", ("RT", "OT", "OL")),
    ("EDGE1", ("EDGE", "LB_EDGE", "DL")),
    ("EDGE2", ("EDGE", "LB_EDGE", "DL")),
    ("DT1", ("DT", "DL")), ("DT2", ("DT", "DL")),
    ("LB1", ("LB",)), ("LB2", ("LB",)), ("LB3", ("LB",)),
    ("CB1", ("CB", "DB")), ("CB2", ("CB", "DB")),
    ("SLOT", ("SLOT", "CB", "DB")),
    ("FS", ("FS", "S", "DB")), ("SS", ("SS", "S", "DB")),
    ("K", ("K",)), ("P", ("P",)), ("LS", ("LS",)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical 2026 NFL depth chart.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--no-refresh-ol-starters", "--no-refresh-depth-sources",
        action="store_true",
        help="Use the audited all-position source cache (maximum age 24 hours).",
    )
    parser.add_argument("--espn-timeout", type=float, default=30.0)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_depth_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().replace("\u200b", "").replace("\ufeff", "")
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def clean_id(value: Any) -> str:
    text = clean_scalar(value)
    return re.sub(r"\.0$", "", text)


def normalize_team(value: Any) -> str:
    team = clean_scalar(value).upper().replace(".", "")
    return TEAM_ALIASES.get(team, team)


def normalize_name(value: Any) -> str:
    text = clean_scalar(value).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def player_match_key(team: Any, player_name: Any) -> str:
    normalized_team = normalize_team(team)
    name_key = normalize_name(player_name)
    return PLAYER_NAME_MATCH_ALIASES.get((normalized_team, name_key), name_key)


def normalize_position(value: Any) -> str:
    pos = re.sub(r"[^A-Z]", "", clean_scalar(value).upper())
    return POSITION_ALIASES.get(pos, pos)


def canonical_role(position: str, position_group: str) -> str:
    pos = normalize_position(position)
    group = clean_scalar(position_group).upper()
    if pos in {"LT", "RT"}: return pos
    if pos in {"LG", "RG"}: return pos
    if pos in {"OT", "OG", "C", "OL"}: return pos
    if pos in {"QB", "RB", "FB", "WR", "TE", "EDGE", "DT", "LB", "CB", "SLOT", "FS", "SS", "K", "P", "LS"}: return pos
    if pos in {"S", "DB"}: return "S" if pos == "S" else "DB"
    if pos == "LB_EDGE" or group == "LB_EDGE": return "LB_EDGE"
    if group == "QB": return "QB"
    if group == "RB": return "RB"
    if group == "WR_TE": return "TE" if pos == "TE" else "WR"
    if group == "OL": return "OL"
    if group in {"EDGE", "DL"}: return "EDGE" if group == "EDGE" else "DL"
    if group == "LB": return "LB"
    if group == "DB": return "DB"
    if group == "ST": return pos if pos in {"K", "P", "LS"} else "ST"
    return pos or group or "OTHER"


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not table_exists(conn, table):
        raise RuntimeError(f"Missing required table: {table}")
    escaped = table.replace('"', '""')
    out = pd.read_sql_query(f'SELECT * FROM "{escaped}"', conn)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_inputs(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = read_table(conn, PERFORMANCE_TABLE)
    master = read_table(conn, MASTER_TABLE)
    for frame, name in [(perf, PERFORMANCE_TABLE), (master, MASTER_TABLE)]:
        if "player_id" not in frame.columns:
            raise RuntimeError(f"{name} is missing player_id")
        frame["player_id"] = frame["player_id"].map(clean_id)
    perf = perf[perf["player_id"].ne("")].drop_duplicates("player_id", keep="last")
    master = master[master["player_id"].ne("")].drop_duplicates("player_id", keep="last")
    return perf, master


def ensure_qb_starter_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {QB_STARTER_TABLE} (
            season INTEGER NOT NULL,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'manual_projection',
            active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT,
            PRIMARY KEY (season, team)
        )
    """)
    existing = conn.execute(f"SELECT COUNT(*) FROM {QB_STARTER_TABLE} WHERE season=?", (SEASON,)).fetchone()[0]
    if existing == 0:
        now = dt.datetime.now().isoformat(timespec="seconds")
        conn.executemany(
            f"INSERT INTO {QB_STARTER_TABLE}(season,team,player_name,source,active,updated_at) VALUES(?,?,?,?,1,?)",
            [(SEASON, team, name, "default_2026_projection", now) for team, name in DEFAULT_QB_STARTERS.items()],
        )
    conn.commit()


def load_qb_starters(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_qb_starter_table(conn)
    starters = pd.read_sql_query(
        f"SELECT * FROM {QB_STARTER_TABLE} WHERE CAST(season AS INTEGER)=? AND COALESCE(active,1)=1",
        conn, params=(SEASON,),
    )
    starters.columns = [str(c).strip().lower() for c in starters.columns]
    if "team" not in starters.columns or "player_name" not in starters.columns:
        raise RuntimeError(f"{QB_STARTER_TABLE} must contain team and player_name")
    starters["team"] = starters["team"].map(normalize_team)
    starters["player_name"] = starters["player_name"].map(clean_scalar)
    starters["name_key"] = starters["player_name"].map(normalize_name)
    if "player_id" in starters.columns:
        starters["starter_player_id"] = starters["player_id"].map(clean_id)
    else:
        starters["starter_player_id"] = ""
    if "source" not in starters.columns:
        starters["source"] = "manual_projection"
    duplicate = starters[starters.duplicated("team", keep=False)]
    if not duplicate.empty:
        raise RuntimeError("Duplicate active QB starter rows:\n" + duplicate[["team", "player_name"]].to_string(index=False))
    if len(starters) != 32 or starters["team"].nunique() != 32:
        raise RuntimeError(f"Expected 32 active QB starters, found {len(starters)} rows/{starters['team'].nunique()} teams")
    return starters


def ensure_ol_starter_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {OL_STARTER_TABLE} (
            season INTEGER NOT NULL,
            team TEXT NOT NULL,
            starter_slot TEXT NOT NULL,
            player_name TEXT NOT NULL,
            espn_athlete_id TEXT,
            source TEXT NOT NULL DEFAULT 'espn_current_depth_chart',
            source_depth_rank INTEGER NOT NULL DEFAULT 1,
            source_injury_status TEXT,
            source_timestamp TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT,
            PRIMARY KEY (season, team, starter_slot)
        )
    """)
    conn.commit()


def http_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "NFLStructuralModel/2026 depth-chart audit",
        },
    )
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object from {url}")
            return payload
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load {url}: {last_error}")


def fetch_espn_team_ol_starters(team: str, team_id: str, timeout: float) -> list[dict[str, Any]]:
    payload = http_json(ESPN_DEPTH_URL.format(team_id=team_id), timeout)
    if clean_scalar(payload.get("status")).lower() != "success":
        raise RuntimeError(f"{team}: ESPN depth-chart response was not successful")
    payload_season = payload.get("season", {})
    if int(payload_season.get("year", 0) or 0) != SEASON:
        raise RuntimeError(f"{team}: ESPN returned season {payload_season.get('year')} instead of {SEASON}")
    response_team = normalize_team(payload.get("team", {}).get("abbreviation"))
    if response_team != team:
        raise RuntimeError(f"{team}: ESPN response identified team as {response_team}")

    rows_by_slot: dict[str, dict[str, Any]] = {}
    for group in payload.get("depthchart", []):
        positions = group.get("positions", {}) if isinstance(group, dict) else {}
        nodes = positions.values() if isinstance(positions, dict) else positions
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            slot = normalize_position(node.get("position", {}).get("abbreviation"))
            if slot not in OL_STARTER_SLOTS:
                continue
            athletes = node.get("athletes") or []
            if not athletes:
                raise RuntimeError(f"{team} {slot}: ESPN returned no depth-chart athletes")
            athlete = athletes[0]
            player_name = clean_scalar(athlete.get("displayName"))
            if not player_name:
                raise RuntimeError(f"{team} {slot}: ESPN starter has no player name")
            injury_statuses = sorted({
                clean_scalar(injury.get("status"))
                for injury in athlete.get("injuries", [])
                if clean_scalar(injury.get("status"))
            })
            row = {
                "season": SEASON,
                "team": team,
                "starter_slot": slot,
                "player_name": player_name,
                "espn_athlete_id": clean_id(athlete.get("id")),
                "source": "espn_current_depth_chart",
                "source_depth_rank": 1,
                "source_injury_status": "|".join(injury_statuses),
                "source_timestamp": clean_scalar(payload.get("timestamp")),
                "active": 1,
            }
            prior = rows_by_slot.get(slot)
            if prior is not None and normalize_name(prior["player_name"]) != normalize_name(player_name):
                raise RuntimeError(
                    f"{team} {slot}: conflicting ESPN starters {prior['player_name']} and {player_name}"
                )
            rows_by_slot[slot] = row

    missing = [slot for slot in OL_STARTER_SLOTS if slot not in rows_by_slot]
    if missing:
        raise RuntimeError(f"{team}: ESPN depth chart missing OL slots {missing}")
    return [rows_by_slot[slot] for slot in OL_STARTER_SLOTS]


def fetch_espn_ol_starters(timeout: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_espn_team_ol_starters, team, team_id, timeout): team
            for team, team_id in ESPN_TEAM_IDS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            team = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                problems.append(f"{team}: {exc}")
    if problems:
        raise RuntimeError("ESPN OL starter refresh failed:\n" + "\n".join(sorted(problems)))
    return pd.DataFrame(rows)


def validate_ol_starters(starters: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season", "team", "starter_slot", "player_name", "source",
        "source_depth_rank", "source_injury_status", "source_timestamp", "active",
    }
    missing = sorted(required.difference(starters.columns))
    if missing:
        raise RuntimeError(f"{OL_STARTER_TABLE} missing columns: {missing}")
    out = starters.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").fillna(0).astype(int)
    out["team"] = out["team"].map(normalize_team)
    out["starter_slot"] = out["starter_slot"].map(normalize_position)
    out["player_name"] = out["player_name"].map(clean_scalar)
    out["name_key"] = [
        player_match_key(team, name)
        for team, name in zip(out["team"], out["player_name"])
    ]
    out["source"] = out["source"].map(clean_scalar)
    out["active"] = pd.to_numeric(out["active"], errors="coerce").fillna(1).astype(int)
    out = out[(out["season"] == SEASON) & (out["active"] == 1)].copy()

    expected_keys = {(team, slot) for team in ESPN_TEAM_IDS for slot in OL_STARTER_SLOTS}
    actual_keys = set(zip(out["team"], out["starter_slot"]))
    duplicate_keys = out[out.duplicated(["team", "starter_slot"], keep=False)]
    if not duplicate_keys.empty:
        raise RuntimeError(
            f"Duplicate active OL starter rows:\n{duplicate_keys[['team', 'starter_slot', 'player_name']].to_string(index=False)}"
        )
    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys.difference(actual_keys))
        extra_keys = sorted(actual_keys.difference(expected_keys))
        raise RuntimeError(f"Incomplete OL starter table; missing={missing_keys}; extra={extra_keys}")
    if out["name_key"].eq("").any():
        raise RuntimeError(f"{OL_STARTER_TABLE} contains blank starter names")
    duplicate_players = out[out.duplicated(["team", "name_key"], keep=False)]
    if not duplicate_players.empty:
        raise RuntimeError(
            "One player is assigned to multiple OL starter slots:\n"
            + duplicate_players[["team", "starter_slot", "player_name"]].to_string(index=False)
        )
    return out.sort_values(["team", "starter_slot"]).reset_index(drop=True)


def load_cached_ol_starters(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_ol_starter_table(conn)
    cached = pd.read_sql_query(
        f"SELECT * FROM {OL_STARTER_TABLE} WHERE CAST(season AS INTEGER)=? AND COALESCE(active,1)=1",
        conn,
        params=(SEASON,),
    )
    cached.columns = [str(c).strip().lower() for c in cached.columns]
    return validate_ol_starters(cached)


def load_ol_starters(
    conn: sqlite3.Connection,
    refresh: bool,
    timeout: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    ensure_ol_starter_table(conn)
    if not refresh:
        cached = load_cached_ol_starters(conn)
        logger.info("[DEPTH] OL starters: using complete cached authoritative table")
        return cached

    logger.info("[DEPTH] OL starters: refreshing 32 position-specific ESPN depth charts")
    try:
        fetched = validate_ol_starters(fetch_espn_ol_starters(timeout))
    except Exception as exc:
        try:
            cached = load_cached_ol_starters(conn)
        except Exception:
            raise RuntimeError(
                f"OL starter refresh failed and no complete cache is available: {exc}"
            ) from exc
        logger.warning("[DEPTH] OL refresh failed; using complete prior cache: %s", exc)
        return cached

    manual_rows = pd.read_sql_query(
        f"""
        SELECT team, starter_slot
        FROM {OL_STARTER_TABLE}
        WHERE CAST(season AS INTEGER)=?
          AND COALESCE(active,1)=1
          AND LOWER(COALESCE(source,'')) LIKE 'manual%'
        """,
        conn,
        params=(SEASON,),
    )
    manual_keys = {
        (normalize_team(row.team), normalize_position(row.starter_slot))
        for row in manual_rows.itertuples(index=False)
    }
    now = dt.datetime.now().isoformat(timespec="seconds")
    upsert_rows = []
    for row in fetched.itertuples(index=False):
        if (row.team, row.starter_slot) in manual_keys:
            continue
        upsert_rows.append(
            (
                int(row.season), row.team, row.starter_slot, row.player_name,
                clean_id(getattr(row, "espn_athlete_id", "")), row.source,
                int(row.source_depth_rank), clean_scalar(row.source_injury_status),
                clean_scalar(row.source_timestamp), 1, now,
            )
        )
    conn.executemany(
        f"""
        INSERT INTO {OL_STARTER_TABLE}(
            season, team, starter_slot, player_name, espn_athlete_id, source,
            source_depth_rank, source_injury_status, source_timestamp, active, updated_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(season,team,starter_slot) DO UPDATE SET
            player_name=excluded.player_name,
            espn_athlete_id=excluded.espn_athlete_id,
            source=excluded.source,
            source_depth_rank=excluded.source_depth_rank,
            source_injury_status=excluded.source_injury_status,
            source_timestamp=excluded.source_timestamp,
            active=excluded.active,
            updated_at=excluded.updated_at
        """,
        upsert_rows,
    )
    conn.commit()
    refreshed = load_cached_ol_starters(conn)
    logger.info(
        "[DEPTH] OL starters: %s authoritative rows (%s manual overrides)",
        len(refreshed),
        len(manual_keys),
    )
    return refreshed


def likely_unavailable(status: Any) -> int:
    normalized = clean_scalar(status).upper()
    if normalized in UNAVAILABLE_STATUS_CODES:
        return 1
    text = f" {normalized} "
    return int(any(pattern in text for pattern in UNAVAILABLE_PATTERNS))


def prepare_depth(
    perf: pd.DataFrame,
    master: pd.DataFrame,
    starters: pd.DataFrame,
    ol_starters: pd.DataFrame,
) -> pd.DataFrame:
    required = ["player_id", "player_name", "team", "position", "position_group", "performance_input_score", "position_confidence_score", "replacement_baseline"]
    missing = [c for c in required if c not in perf.columns]
    if missing:
        raise RuntimeError(f"{PERFORMANCE_TABLE} missing columns: {missing}")

    live_fields = ["original_roster_status", "live_depth_rank", "live_source_starter", "live_preferred_slot",
                   "live_injury_status", "live_uncertain", "live_injury_reported_at", "live_source_timestamp",
                   "live_fetched_at_utc", "identity_match_method"]
    roster_fields = [c for c in ["player_id", "status", "years_exp", "draft_number", "rookie_year", "birth_date", *live_fields] if c in master.columns]
    roster = master[roster_fields].copy()
    roster = roster.rename(columns={c: f"roster_{c}" for c in roster_fields if c != "player_id"})
    out = perf.merge(roster, on="player_id", how="left", validate="one_to_one")
    if any(c.endswith("_x") or c.endswith("_y") for c in out.columns):
        raise RuntimeError("Unexpected merge suffix columns detected in depth preparation")

    out["team"] = out["team"].map(normalize_team)
    out["position"] = out["position"].map(normalize_position)
    out["position_group"] = out["position_group"].map(lambda x: clean_scalar(x).upper())
    out["player_name"] = out["player_name"].map(clean_scalar)
    out["canonical_role"] = [canonical_role(p, g) for p, g in zip(out["position"], out["position_group"])]
    out["status"] = out.get("roster_status", pd.Series("", index=out.index)).map(clean_scalar)
    out["years_exp"] = numeric(out, "roster_years_exp", 0.0)
    out["draft_number"] = numeric(out, "roster_draft_number", np.nan)
    out["likely_unavailable"] = out["status"].map(likely_unavailable)
    for column in live_fields:
        original = f"roster_{column}"
        if column in {"live_depth_rank", "live_source_starter", "live_uncertain"}:
            out[column] = numeric(out, original, 999.0 if column == "live_depth_rank" else 0.0)
        else:
            out[column] = out.get(original, pd.Series("", index=out.index)).map(clean_scalar)

    out["performance_grade"] = numeric(out, "performance_input_score", 0.0).clip(0, 100)
    out["replacement_baseline"] = numeric(out, "replacement_baseline", 30.0).clip(0, 100)
    out["position_confidence_score"] = numeric(out, "position_confidence_score", 0.0).clip(0, 1)
    out["history_available"] = numeric(out, "history_available", 0.0).clip(0, 1).astype(int)
    out["usable_performance_grade"] = numeric(out, "usable_performance_grade", 0.0).clip(0, 1).astype(int)
    out["durability_score"] = numeric(out, "durability_score", 0.0).clip(0, 100)
    out["qb_starter_pool"] = numeric(out, "qb_starter_pool", 0.0).clip(0, 1).astype(int)

    is_ol = out["position_group"].eq("OL") | out["canonical_role"].isin(OL_ROLES)
    out["ol_quality_discount_applied"] = 0
    out["unit_quality_grade"] = out["performance_grade"].astype(float)
    out["unit_quality_grade"] = out["unit_quality_grade"].clip(0, 100)
    out["availability_score"] = out["durability_score"]
    out.loc[out["likely_unavailable"].eq(1), "availability_score"] = 0.0
    out["depth_order_score"] = 0.85 * out["unit_quality_grade"] + 0.15 * out["availability_score"]
    out.loc[out["likely_unavailable"].eq(1), "depth_order_score"] -= 1000.0

    out["player_name_key"] = [
        player_match_key(team, name)
        for team, name in zip(out["team"], out["player_name"])
    ]
    starter_lookup = starters[["team", "player_name", "name_key", "starter_player_id", "source"]].rename(
        columns={"player_name": "configured_qb_starter", "name_key": "configured_qb_name_key", "source": "qb_starter_source"}
    )
    out = out.merge(starter_lookup, on="team", how="left", validate="many_to_one")
    if any(c.endswith("_x") or c.endswith("_y") for c in out.columns):
        raise RuntimeError("Unexpected QB merge suffix columns detected")
    qb_mask = out["canonical_role"].eq("QB")
    by_id = qb_mask & out["starter_player_id"].ne("") & out["player_id"].eq(out["starter_player_id"])
    by_name = qb_mask & out["starter_player_id"].eq("") & out["player_name_key"].eq(out["configured_qb_name_key"])
    out["qb_authoritative_starter"] = (by_id | by_name).astype(int)

    problems: list[str] = []
    for team, team_starters in starters.groupby("team"):
        qbs = out[(out["team"] == team) & qb_mask]
        matches = qbs[qbs["qb_authoritative_starter"].eq(1)]
        if len(matches) != 1:
            names = ", ".join(sorted(qbs["player_name"].dropna().astype(str).unique()))
            configured = team_starters.iloc[0]["player_name"]
            problems.append(f"{team}: configured={configured}; matches={len(matches)}; QBs=[{names}]")
        elif int(matches["likely_unavailable"].iloc[0]) == 1:
            problems.append(f"{team}: configured starter {matches['player_name'].iloc[0]} is unavailable")
    if problems:
        raise RuntimeError("Authoritative QB starter validation failed:\n" + "\n".join(problems))

    out.loc[qb_mask, "qb_starter_pool"] = 0
    out.loc[out["qb_authoritative_starter"].eq(1), "qb_starter_pool"] = 1
    out.loc[out["qb_authoritative_starter"].eq(1), "depth_order_score"] += 10000.0

    ol_lookup = ol_starters[
        [
            "team", "player_name", "name_key", "starter_slot", "source",
            "source_depth_rank", "source_injury_status", "source_timestamp",
        ]
    ].rename(
        columns={
            "player_name": "configured_ol_starter",
            "name_key": "configured_ol_name_key",
            "starter_slot": "configured_ol_starter_slot",
            "source": "ol_starter_source",
            "source_depth_rank": "ol_source_depth_rank",
            "source_injury_status": "ol_source_injury_status",
            "source_timestamp": "ol_source_timestamp",
        }
    )
    out = out.merge(
        ol_lookup,
        how="left",
        left_on=["team", "player_name_key"],
        right_on=["team", "configured_ol_name_key"],
        validate="many_to_one",
    )
    if any(c.endswith("_x") or c.endswith("_y") for c in out.columns):
        raise RuntimeError("Unexpected OL starter merge suffix columns detected")
    out["configured_ol_starter"] = out["configured_ol_starter"].fillna("").map(clean_scalar)
    out["configured_ol_starter_slot"] = out["configured_ol_starter_slot"].fillna("").map(normalize_position)
    out["ol_starter_source"] = out["ol_starter_source"].fillna("").map(clean_scalar)
    out["ol_source_injury_status"] = out["ol_source_injury_status"].fillna("").map(clean_scalar)
    out["ol_source_timestamp"] = out["ol_source_timestamp"].fillna("").map(clean_scalar)
    out["ol_source_depth_rank"] = pd.to_numeric(out["ol_source_depth_rank"], errors="coerce").fillna(0).astype(int)
    out["ol_authoritative_starter"] = out["configured_ol_starter_slot"].isin(OL_STARTER_SLOTS).astype(int)

    matched_is_ol = out["position_group"].eq("OL") | out["canonical_role"].isin(OL_ROLES)
    ol_problems: list[str] = []
    for row in ol_starters.itertuples(index=False):
        matches = out[
            out["team"].eq(row.team)
            & out["player_name_key"].eq(row.name_key)
            & matched_is_ol
        ]
        if len(matches) != 1:
            team_ol = out[out["team"].eq(row.team) & matched_is_ol]
            names = ", ".join(sorted(team_ol["player_name"].dropna().astype(str).unique()))
            ol_problems.append(
                f"{row.team} {row.starter_slot}: configured={row.player_name}; "
                f"matches={len(matches)}; OL=[{names}]"
            )
        elif int(matches["likely_unavailable"].iloc[0]) == 1:
            ol_problems.append(
                f"{row.team} {row.starter_slot}: configured starter {row.player_name} "
                f"has unavailable roster status {matches['status'].iloc[0]}"
            )
    if ol_problems:
        raise RuntimeError("Authoritative OL starter validation failed:\n" + "\n".join(ol_problems))
    return out


def role_rank_key(role: str) -> str:
    if role in {"LT", "RT", "OT"}: return "OT"
    if role in {"LG", "RG", "OG"}: return "OG"
    if role in {"FS", "SS", "S"}: return "S"
    if role in {"EDGE", "LB_EDGE", "DL"}: return role
    return role


def compatible(role: str, allowed: Iterable[str]) -> bool:
    if role in allowed:
        return True
    if role == "OT" and any(x in allowed for x in ("LT", "RT", "OT")): return True
    if role == "OG" and any(x in allowed for x in ("LG", "RG", "OG")): return True
    if role == "OL" and any(x in allowed for x in OL_ROLES): return True
    if role == "S" and any(x in allowed for x in ("FS", "SS", "S", "DB")): return True
    if role == "DB" and any(x in allowed for x in ("CB", "SLOT", "FS", "SS", "S", "DB")): return True
    if role == "DL" and any(x in allowed for x in ("EDGE", "DT", "DL")): return True
    if role == "LB_EDGE" and any(x in allowed for x in ("EDGE", "LB_EDGE")): return True
    return False


def assign_starters_and_ranks(depth: pd.DataFrame) -> pd.DataFrame:
    out = depth.copy()
    out["role_rank_group"] = out["canonical_role"].map(role_rank_key)
    out = out.sort_values(["team", "role_rank_group", "likely_unavailable", "live_source_starter", "live_depth_rank", "depth_order_score", "player_id"],
                          ascending=[True, True, True, False, True, False, True])
    out["depth_rank"] = out.groupby(["team", "role_rank_group"]).cumcount() + 1
    out["is_projected_starter"] = 0
    out["starter_slot"] = ""

    for team, idx in out.groupby("team").groups.items():
        used: set[str] = set()
        team_df = out.loc[idx]
        for slot_name, allowed in STARTER_SLOTS:
            candidates = team_df[
                team_df["player_id"].map(lambda x: x not in used)
                & team_df["canonical_role"].map(lambda r: compatible(r, allowed))
                & team_df["likely_unavailable"].eq(0)
            ].copy()
            if candidates.empty:
                continue
            if slot_name == "QB1":
                auth = candidates[candidates["qb_authoritative_starter"].eq(1)]
                if not auth.empty:
                    candidates = auth
            elif slot_name in OL_STARTER_SLOTS:
                auth = candidates[candidates["configured_ol_starter_slot"].eq(slot_name)]
                if len(auth) != 1:
                    raise RuntimeError(
                        f"{team} {slot_name}: expected exactly one authoritative OL starter, found {len(auth)}"
                    )
                candidates = auth
            candidates["live_exact_slot"] = candidates["live_preferred_slot"].eq(slot_name).astype(int)
            candidates = candidates.sort_values(
                ["live_exact_slot", "live_source_starter", "live_depth_rank", "depth_order_score", "position_confidence_score", "player_id"],
                ascending=[False, False, True, False, False, True])
            chosen_idx = candidates.index[0]
            player_id = out.at[chosen_idx, "player_id"]
            used.add(player_id)
            out.at[chosen_idx, "is_projected_starter"] = 1
            out.at[chosen_idx, "starter_slot"] = slot_name

    out["projected_snap_share"] = 0.0
    for i, row in out.iterrows():
        role = role_rank_key(row["canonical_role"])
        rank = int(row["depth_rank"])
        share = SNAP_SHARE_BY_ROLE.get(role, {}).get(rank, 0.0)
        if int(row["is_projected_starter"]) == 1:
            if role in {"OT", "OG", "C", "OL", "QB"}:
                share = max(share, 0.95 if role != "QB" else 1.0)
            else:
                share = max(share, 0.60)
        if int(row["likely_unavailable"]) == 1:
            share = 0.0
        out.at[i, "projected_snap_share"] = float(np.clip(share, 0.0, 1.0))
    return out


def finalize(depth: pd.DataFrame) -> pd.DataFrame:
    out = depth.copy()
    out["season"] = SEASON
    out["depth_chart_position"] = out["canonical_role"]
    ol_slot_mask = out["ol_authoritative_starter"].eq(1)
    out.loc[ol_slot_mask, "depth_chart_position"] = out.loc[ol_slot_mask, "configured_ol_starter_slot"]
    is_ol = out["position_group"].eq("OL") | out["canonical_role"].isin(OL_ROLES)
    pff_available = pd.to_numeric(
        out.get("ol_pff_history_available", pd.Series(0, index=out.index)),
        errors="coerce",
    ).fillna(0).eq(1)
    out["quality_source"] = "player_performance_grade"
    out.loc[is_ol & pff_available, "quality_source"] = "pff_ol_talent_single_confidence_shrink"
    out.loc[is_ol & ~pff_available, "quality_source"] = "pff_ol_no_history_replacement"
    out["source"] = VERSION
    out["build_id"] = BUILD_ID
    out["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")
    columns = [
        "season", "team", "player_id", "player_name", "position", "position_group",
        "canonical_role", "role_rank_group", "depth_chart_position", "depth_rank",
        "is_projected_starter", "starter_slot", "projected_snap_share", "status",
        "likely_unavailable", "performance_grade", "replacement_baseline",
        "unit_quality_grade", "ol_quality_discount_applied", "quality_source",
        "ol_pff_history_available", "ol_pff_player_id", "ol_pff_position",
        "ol_pff_match_method", "ol_pff_talent_grade", "ol_pff_recent_grade",
        "ol_pff_latest_season", "ol_pff_seasons_observed", "ol_pff_total_snaps",
        "ol_pff_effective_sample", "ol_pff_confidence", "ol_pff_source",
        "availability_score", "position_confidence_score", "history_available",
        "usable_performance_grade", "qb_starter_pool", "qb_authoritative_starter",
        "configured_qb_starter", "qb_starter_source", "years_exp", "draft_number",
        "durability_score", "depth_order_score", "ol_authoritative_starter",
        "configured_ol_starter", "configured_ol_starter_slot", "ol_starter_source",
        "ol_source_depth_rank", "ol_source_injury_status", "ol_source_timestamp",
        "original_roster_status", "live_depth_rank", "live_source_starter", "live_preferred_slot",
        "live_injury_status", "live_uncertain", "live_injury_reported_at", "live_source_timestamp",
        "live_fetched_at_utc", "identity_match_method", "source", "build_id", "date_imported",
    ]
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns].sort_values(["team", "role_rank_group", "depth_rank", "player_id"]).reset_index(drop=True)


def validate(depth: pd.DataFrame, perf_rows: int) -> None:
    if len(depth) != perf_rows:
        raise RuntimeError(f"Depth rows {len(depth)} do not match canonical performance rows {perf_rows}")
    if depth["player_id"].duplicated().any():
        raise RuntimeError("Duplicate player_id rows in depth output")
    if depth["team"].nunique() != 32:
        raise RuntimeError(f"Expected 32 teams, found {depth['team'].nunique()}")
    qbs = depth[(depth["canonical_role"] == "QB") & (depth["qb_authoritative_starter"] == 1)]
    if len(qbs) != 32 or qbs["team"].nunique() != 32:
        raise RuntimeError(f"Expected one authoritative QB starter per team; found {len(qbs)}")
    required_numeric = ["performance_grade", "unit_quality_grade", "availability_score", "position_confidence_score", "depth_order_score", "projected_snap_share"]
    if depth[required_numeric].isna().any().any():
        raise RuntimeError("Null numeric values found in depth output")
    if any(c.endswith("_x") or c.endswith("_y") for c in depth.columns):
        raise RuntimeError("Merge collision suffix columns found in depth output")
    ol = depth[
        depth["position_group"].eq("OL")
        | depth["canonical_role"].isin(OL_ROLES)
    ]
    if not ol.empty:
        if not ol["ol_quality_discount_applied"].eq(0).all():
            raise RuntimeError("An obsolete OL quality discount is still active")
        if not np.allclose(ol["unit_quality_grade"], ol["performance_grade"], atol=1e-9):
            raise RuntimeError("OL quality received an unexpected second shrink")
    projected_ol = depth[
        depth["is_projected_starter"].eq(1)
        & depth["starter_slot"].isin(OL_STARTER_SLOTS)
    ].copy()
    expected_ol_keys = {(team, slot) for team in ESPN_TEAM_IDS for slot in OL_STARTER_SLOTS}
    actual_ol_keys = set(zip(projected_ol["team"], projected_ol["starter_slot"]))
    if actual_ol_keys != expected_ol_keys or len(projected_ol) != len(expected_ol_keys):
        raise RuntimeError("Projected OL starters are not exactly one LT/LG/C/RG/RT per team")
    if not projected_ol["ol_authoritative_starter"].eq(1).all():
        raise RuntimeError("A projected OL starter did not come from the authoritative OL table")
    if not projected_ol["starter_slot"].eq(projected_ol["configured_ol_starter_slot"]).all():
        raise RuntimeError("Projected OL starter slots do not match their authoritative positions")
    if projected_ol["likely_unavailable"].eq(1).any():
        bad = projected_ol.loc[
            projected_ol["likely_unavailable"].eq(1),
            ["team", "starter_slot", "player_name", "status"],
        ]
        raise RuntimeError("Unavailable players assigned as OL starters:\n" + bad.to_string(index=False))
    if likely_unavailable("RES") != 1 or likely_unavailable("DEV") != 1:
        raise RuntimeError("Reserve/developmental status regression")


def record_refresh_status(db_path: Path, status: str, attempted_at_utc: str,
                          error: str = "") -> None:
    """Record the latest attempt independently of the last successful snapshot."""
    with sqlite3.connect(db_path) as connection:
        pd.DataFrame([{
            "season": SEASON, "status": status,
            "attempted_at_utc": attempted_at_utc, "error": error,
            "depth_version": VERSION,
        }]).to_sql(live_depth.REFRESH_STATUS_TABLE, connection, if_exists="replace", index=False)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    db_path = args.db_path.resolve()
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir / "build_nfl_projected_depth_chart.log")
    logger.info("[DEPTH] Building canonical projected depth chart")
    logger.info("[DEPTH] Build ID: %s", BUILD_ID)
    logger.info("[DEPTH] Version: %s", VERSION)
    logger.info("[DEPTH] Database: %s", db_path)
    attempted_at_utc = pd.Timestamp.now(tz="UTC").isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            perf, master = load_inputs(conn)
            master = live_depth.attach_roster_provenance(conn, master)
            logger.info("[DEPTH] Refreshing automatic all-position roster, injuries and depth order; manual QB CSV is not used")
            sources, source_audit = live_depth.fetch_sources(
                project_root, ESPN_TEAM_IDS, max(float(args.espn_timeout), 1.0), args.no_refresh_ol_starters,
                master=master)
            perf, master, starters, ol_starters, availability = live_depth.build_live_inputs(
                perf, master, sources, source_audit, normalize_name)
            ol_starters = validate_ol_starters(ol_starters)
            prepared = prepare_depth(perf, master, starters, ol_starters)
            ranked = assign_starters_and_ranks(prepared)
            depth = finalize(ranked)
            validate(depth, len(perf))
            ensure_qb_starter_table(conn)
            ensure_ol_starter_table(conn)
            live_depth.persist_starters(conn, starters, ol_starters)
            live_depth.persist_sources(conn, project_root, availability, source_audit, not args.no_csv)

            audit_cols = [
                "team", "starter_slot", "canonical_role", "depth_rank", "player_id", "player_name",
                "is_projected_starter", "projected_snap_share", "performance_grade",
                "replacement_baseline", "unit_quality_grade", "ol_quality_discount_applied",
                "availability_score", "position_confidence_score", "qb_authoritative_starter",
                "configured_qb_starter", "qb_starter_source", "likely_unavailable", "depth_order_score",
                "ol_authoritative_starter", "configured_ol_starter",
                "configured_ol_starter_slot", "ol_starter_source", "ol_source_depth_rank",
                "ol_source_injury_status", "ol_source_timestamp",
                "live_injury_status", "live_uncertain", "live_depth_rank",
                "live_fetched_at_utc", "identity_match_method",
            ]
            audit = depth[audit_cols].copy()
            depth.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
            audit.to_sql(AUDIT_TABLE, conn, if_exists="replace", index=False)
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_player ON {OUTPUT_TABLE}(player_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_team_role ON {OUTPUT_TABLE}(team, canonical_role, depth_rank)")
            conn.commit()

        if not args.no_csv:
            depth.to_csv(output_dir / f"{OUTPUT_TABLE}.csv", index=False, encoding="utf-8-sig")
            audit.to_csv(output_dir / f"{AUDIT_TABLE}.csv", index=False, encoding="utf-8-sig")
            summary = depth.groupby(["team", "position_group"]).agg(
                players=("player_id", "count"), projected_starters=("is_projected_starter", "sum"),
                avg_player_quality=("unit_quality_grade", "mean"), avg_availability=("availability_score", "mean"),
                avg_confidence=("position_confidence_score", "mean"),
            ).reset_index()
            summary.to_csv(output_dir / "nfl_projected_depth_chart_2026_summary.csv", index=False, encoding="utf-8-sig")

        record_refresh_status(db_path, "SUCCESS", attempted_at_utc)
        logger.info("[DEPTH] Canonical players: %s", f"{len(depth):,}")
        logger.info("[DEPTH] Teams: %s", depth["team"].nunique())
        logger.info("[DEPTH] Authoritative QB starters: %s", int(depth["qb_authoritative_starter"].sum()))
        logger.info("[DEPTH] Authoritative OL starters: %s", int(depth["ol_authoritative_starter"].sum()))
        logger.info("[DEPTH] Projected starter assignments: %s", int(depth["is_projected_starter"].sum()))
        logger.info("[DEPTH] OL rows with obsolete second quality shrink: %s", int(depth["ol_quality_discount_applied"].sum()))
        logger.info("[DEPTH] OL quality source: PFF blocking talent; availability remains separate")
        logger.info("[DEPTH] Saved table: %s", OUTPUT_TABLE)
        logger.info("\n[DEPTH] Projected QB starters:\n%s", depth[depth["qb_authoritative_starter"].eq(1)][["team", "player_name", "performance_grade", "unit_quality_grade", "position_confidence_score"]].sort_values("team").to_string(index=False))
        return 0
    except live_depth.SourceUnavailable as exc:
        logger.error("[DEPTH] SOURCE_UNAVAILABLE: %s", exc)
        try:
            record_refresh_status(db_path, "SOURCE_UNAVAILABLE", attempted_at_utc, str(exc))
        except Exception as status_exc:
            logger.error("[DEPTH] Could not record source outage safely: %s", status_exc)
            return 1
        logger.warning("[DEPTH] Current availability was not refreshed. The weekly runner may continue the frozen baseline; adjusted lines will be withheld.")
        return SOURCE_UNAVAILABLE_EXIT_CODE
    except Exception as exc:
        logger.error("[DEPTH] FAILED: %s", exc)
        logger.error(traceback.format_exc())
        try:
            record_refresh_status(db_path, "FAILED", attempted_at_utc, f"{type(exc).__name__}: {exc}")
        except Exception as status_exc:
            logger.error("[DEPTH] Could not record failed refresh: %s", status_exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
