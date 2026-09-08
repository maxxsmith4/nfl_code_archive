#!/usr/bin/env python
"""Load one sharp reference spread per 2026 NFL matchup.

The loader uses The Odds API current NFL spreads endpoint and selects the first
available, non-stale bookmaker in an explicit priority order. It writes one
row per scheduled matchup for the requested week. It does not label any proxy
book as Circa; official Circa contest boards remain a separate manual input.

API-key setup (PowerShell session):
    $env:ODDS_API_KEY = "YOUR_KEY"

Typical run:
    python load_nfl_live_market_odds.py --week 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


SEASON = 2026
BUILD_ID = "NFL_LIVE_MARKET_ODDS_2026_CANONICAL_V1"
VERSION = "v1_odds_api_sharp_priority_audited_single_line_per_game"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)
DEFAULT_API_KEY_ENV = "ODDS_API_KEY"
DEFAULT_BOOKMAKER_PRIORITY = "pinnacle,lowvig,betonlineag"
DEFAULT_MAXIMUM_LINE_AGE_MINUTES = 1440.0

SPORT_KEY = "americanfootball_nfl"
API_ENDPOINT = (
    "https://api.the-odds-api.com/v4/sports/"
    f"{SPORT_KEY}/odds/"
)

CURRENT_TABLE = "nfl_live_market_odds_2026"
HISTORY_TABLE = "nfl_live_market_odds_history_2026"
AUDIT_TABLE = "nfl_live_market_odds_run_audit_2026"
AUDIT_HISTORY_TABLE = "nfl_live_market_odds_run_audit_history_2026"

SCHEDULE_TABLE_CANDIDATES = (
    "nfl_schedule_2026",
    "nfl_game_schedule_2026",
    "nfl_schedule_master",
    "nfl_game_schedule",
    "nfl_schedule",
)

TEAM_ALIASES = {
    "ARZ": "ARI",
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BLT": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "CLV": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "GNB": "GB",
    "HOUSTON TEXANS": "HOU",
    "HST": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "JAC": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "KAN": "KC",
    "KCC": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LAR",
    "LA": "LAR",
    "STL": "LAR",
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NWE": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NOR": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SFO": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TAM": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "WFT": "WAS",
    "WSH": "WAS",
}

SCHEDULE_ALIASES = {
    "season": ("season", "season_year", "year"),
    "week": ("week", "game_week", "week_number", "week_num"),
    "home_team": (
        "home_team",
        "home_team_abbr",
        "home",
        "home_abbr",
    ),
    "away_team": (
        "away_team",
        "away_team_abbr",
        "away",
        "away_abbr",
    ),
    "game_date": (
        "game_datetime_utc",
        "game_datetime_et",
        "game_datetime_ct",
        "game_date",
        "gameday",
        "date",
        "start_time",
        "kickoff",
    ),
    "game_id": ("game_id", "id", "gsis_id", "event_id"),
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
    parser.add_argument("--week", type=int, required=False)
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=DEFAULT_API_KEY_ENV,
    )
    parser.add_argument(
        "--bookmaker-priority",
        type=str,
        default=DEFAULT_BOOKMAKER_PRIORITY,
        help="Comma-separated The Odds API bookmaker keys.",
    )
    parser.add_argument(
        "--maximum-line-age-minutes",
        type=float,
        default=DEFAULT_MAXIMUM_LINE_AGE_MINUTES,
        help="Reject older bookmaker snapshots; use 0 to disable.",
    )
    parser.add_argument(
        "--require-primary-bookmaker",
        action="store_true",
        help="Reject games when the first-priority bookmaker is unavailable.",
    )
    parser.add_argument(
        "--allow-partial-week",
        action="store_true",
        help="Save available games instead of requiring the full schedule.",
    )
    parser.add_argument("--schedule-table", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--response-json-path",
        type=Path,
        default=None,
        help="Offline/replay input. When set, no API request is made.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--no-raw-json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.db_path = args.db_path.resolve()
    for name in ("output_path", "response_json_path"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())

    if not args.self_test and (args.week is None or not 1 <= args.week <= 18):
        parser.error("--week is required and must be between 1 and 18.")
    if args.maximum_line_age_minutes < 0:
        parser.error("--maximum-line-age-minutes cannot be negative.")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    return args


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


def parse_priority(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in str(raw).split(","):
        value = token.strip().lower()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise RuntimeError("At least one bookmaker key is required.")
    if "circa" in values or "circasports" in values:
        raise RuntimeError(
            "Circa is not a supported The Odds API bookmaker key. "
            "Do not label a proxy line as Circa."
        )
    return values


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone() is not None


def load_schedule(
    db_path: Path,
    week: int,
    requested_table: Optional[str],
) -> tuple[pd.DataFrame, str]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    with sqlite3.connect(db_path) as connection:
        candidates = (
            (requested_table,) if requested_table else SCHEDULE_TABLE_CANDIDATES
        )
        table_name = next(
            (name for name in candidates if name and table_exists(connection, name)),
            None,
        )
        if table_name is None:
            raise RuntimeError(
                "No recognized NFL schedule table was found in the database."
            )
        escaped = table_name.replace('"', '""')
        raw = pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)

    raw.columns = [str(column).lower().strip() for column in raw.columns]
    found = {
        key: first_existing(raw.columns, aliases)
        for key, aliases in SCHEDULE_ALIASES.items()
    }
    missing = [
        key for key in ("week", "home_team", "away_team")
        if found[key] is None
    ]
    if missing:
        raise RuntimeError(
            f"Schedule table {table_name} lacks required fields: {missing}"
        )

    out = pd.DataFrame(index=raw.index)
    out["season"] = (
        pd.to_numeric(raw[found["season"]], errors="coerce")
        if found["season"] is not None
        else SEASON
    )
    out["week"] = pd.to_numeric(raw[found["week"]], errors="coerce")
    out["home_team"] = raw[found["home_team"]].map(normalize_team)
    out["away_team"] = raw[found["away_team"]].map(normalize_team)
    out["game_date"] = (
        pd.to_datetime(raw[found["game_date"]], errors="coerce", utc=True)
        if found["game_date"] is not None
        else pd.NaT
    )
    out["schedule_game_id"] = (
        raw[found["game_id"]].astype(str).str.strip()
        if found["game_id"] is not None
        else ""
    )
    out = out[
        out["season"].fillna(SEASON).eq(SEASON)
        & out["week"].eq(int(week))
        & out["home_team"].ne("")
        & out["away_team"].ne("")
    ].copy()
    if out.empty:
        raise RuntimeError(f"No 2026 Week {week} schedule rows were found.")
    out["season"] = SEASON
    out["week"] = int(week)
    if out.duplicated(["home_team", "away_team"]).any():
        duplicates = out[
            out.duplicated(["home_team", "away_team"], keep=False)
        ]
        raise RuntimeError(
            "Duplicate Week schedule matchups:\n" + duplicates.to_string(index=False)
        )
    return out.reset_index(drop=True), str(table_name)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def parse_api_time(value: Any) -> Optional[dt.datetime]:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip().replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fetch_api_response(
    api_key: str,
    bookmakers: list[str],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Optional[str]]]:
    query = urllib.parse.urlencode(
        {
            "apiKey": api_key,
            "bookmakers": ",".join(bookmakers),
            "markets": "spreads",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
    )
    request = urllib.request.Request(
        f"{API_ENDPOINT}?{query}",
        headers={"User-Agent": f"{BUILD_ID}/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {
                "requests_remaining": response.headers.get(
                    "x-requests-remaining"
                ),
                "requests_used": response.headers.get("x-requests-used"),
                "requests_last": response.headers.get("x-requests-last"),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"The Odds API returned HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"The Odds API request failed: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("The Odds API response was not an event list.")
    return payload, headers


def load_response_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise RuntimeError("Offline odds JSON must contain an event list.")
    return payload


def bookmaker_spread_offer(
    event: dict[str, Any],
    bookmaker: dict[str, Any],
    retrieved_at: dt.datetime,
) -> Optional[dict[str, Any]]:
    home = normalize_team(event.get("home_team"))
    away = normalize_team(event.get("away_team"))
    if not home or not away:
        return None

    spread_market = next(
        (
            market
            for market in bookmaker.get("markets", [])
            if str(market.get("key", "")).lower() == "spreads"
        ),
        None,
    )
    if not spread_market:
        return None

    outcomes = {
        normalize_team(outcome.get("name")): outcome
        for outcome in spread_market.get("outcomes", [])
    }
    if home not in outcomes or away not in outcomes:
        return None

    home_outcome = outcomes[home]
    away_outcome = outcomes[away]
    try:
        home_spread = float(home_outcome["point"])
        away_spread = float(away_outcome["point"])
        home_price = float(home_outcome["price"])
        away_price = float(away_outcome["price"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(
        [home_spread, away_spread, home_price, away_price]
    ).all():
        return None
    if not np.isclose(home_spread + away_spread, 0.0, atol=0.01):
        return None

    last_update = parse_api_time(bookmaker.get("last_update"))
    age_minutes = (
        max(0.0, (retrieved_at - last_update).total_seconds() / 60.0)
        if last_update is not None
        else np.nan
    )
    return {
        "bookmaker_key": str(bookmaker.get("key", "")).lower().strip(),
        "sportsbook": str(bookmaker.get("title", "")).strip(),
        "home_spread": home_spread,
        "away_spread": away_spread,
        "home_spread_price": int(round(home_price)),
        "away_spread_price": int(round(away_price)),
        "market_last_update_utc": (
            iso_utc(last_update) if last_update is not None else ""
        ),
        "market_line_age_minutes": age_minutes,
    }


def select_week_lines(
    schedule: pd.DataFrame,
    events: list[dict[str, Any]],
    priority: list[str],
    maximum_age_minutes: float,
    require_primary: bool,
    retrieved_at: dt.datetime,
    run_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    event_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (
            normalize_team(event.get("away_team")),
            normalize_team(event.get("home_team")),
        )
        if all(key):
            event_lookup[key] = event

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for game in schedule.itertuples(index=False):
        key = (str(game.away_team), str(game.home_team))
        event = event_lookup.get(key)
        if event is None:
            missing.append(f"{game.away_team} at {game.home_team}: API_EVENT_MISSING")
            continue

        bookmaker_lookup = {
            str(book.get("key", "")).lower().strip(): book
            for book in event.get("bookmakers", [])
        }
        offers: list[tuple[int, dict[str, Any]]] = []
        for rank, bookmaker_key in enumerate(priority, start=1):
            book = bookmaker_lookup.get(bookmaker_key)
            if book is None:
                continue
            offer = bookmaker_spread_offer(event, book, retrieved_at)
            if offer is None:
                continue
            age = offer["market_line_age_minutes"]
            stale = bool(
                maximum_age_minutes > 0
                and pd.notna(age)
                and float(age) > maximum_age_minutes
            )
            offer["market_stale_flag"] = int(stale)
            if not stale:
                offers.append((rank, offer))

        selected: Optional[tuple[int, dict[str, Any]]] = None
        if require_primary:
            selected = next((item for item in offers if item[0] == 1), None)
        elif offers:
            selected = min(offers, key=lambda item: item[0])

        if selected is None:
            reason = (
                "PRIMARY_BOOK_MISSING_OR_STALE"
                if require_primary
                else "NO_PRIORITY_BOOK_SPREAD_AVAILABLE"
            )
            missing.append(f"{game.away_team} at {game.home_team}: {reason}")
            continue

        rank, offer = selected
        rows.append(
            {
                "run_id": run_id,
                "build_id": BUILD_ID,
                "version": VERSION,
                "season": SEASON,
                "week": int(game.week),
                "event_id": str(event.get("id", "")),
                "schedule_game_id": str(game.schedule_game_id),
                "game_date_utc": str(event.get("commence_time", "")),
                "away_team": str(game.away_team),
                "home_team": str(game.home_team),
                "sportsbook": offer["sportsbook"],
                "bookmaker_key": offer["bookmaker_key"],
                "source_priority_rank": rank,
                "home_spread": offer["home_spread"],
                "away_spread": offer["away_spread"],
                "current_market_home_margin": -float(offer["home_spread"]),
                "home_spread_price": offer["home_spread_price"],
                "away_spread_price": offer["away_spread_price"],
                "market_last_update_utc": offer["market_last_update_utc"],
                "market_retrieved_at_utc": iso_utc(retrieved_at),
                "market_line_age_minutes": offer["market_line_age_minutes"],
                "market_stale_flag": offer["market_stale_flag"],
                "market_key": "spreads",
                "odds_format": "american",
                "market_source": "THE_ODDS_API",
                "circa_line_flag": 0,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["week", "game_date_utc", "away_team", "home_team"]
        ).reset_index(drop=True)
    return frame, missing


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
        for row in connection.execute(f'PRAGMA table_info("{escaped}")')
    }
    for column in frame.columns:
        if column in existing:
            continue
        series = frame[column]
        if pd.api.types.is_integer_dtype(series.dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(series.dtype):
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        column_escaped = str(column).replace('"', '""')
        connection.execute(
            f'ALTER TABLE "{escaped}" ADD COLUMN "{column_escaped}" {sql_type}'
        )
    connection.commit()
    frame.to_sql(table_name, connection, if_exists="append", index=False)


def save_database(
    db_path: Path,
    lines: pd.DataFrame,
    audit: pd.DataFrame,
    week: int,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        if table_exists(connection, CURRENT_TABLE):
            current = pd.read_sql_query(
                f'SELECT * FROM "{CURRENT_TABLE}"',
                connection,
            )
            if "week" in current.columns:
                current = current[
                    pd.to_numeric(current["week"], errors="coerce").ne(week)
                ]
            current = pd.concat([current, lines], ignore_index=True, sort=False)
        else:
            current = lines.copy()
        current.to_sql(
            CURRENT_TABLE,
            connection,
            if_exists="replace",
            index=False,
        )
        append_with_schema_evolution(connection, HISTORY_TABLE, lines)
        audit.to_sql(AUDIT_TABLE, connection, if_exists="replace", index=False)
        append_with_schema_evolution(connection, AUDIT_HISTORY_TABLE, audit)


def save_files(
    project_root: Path,
    output_path: Optional[Path],
    lines: pd.DataFrame,
    events: list[dict[str, Any]],
    week: int,
    no_raw_json: bool,
) -> tuple[Path, Path, Optional[Path]]:
    output_directory = project_root / "outputs" / "nfl_market"
    output_directory.mkdir(parents=True, exist_ok=True)
    current_path = output_path or (
        output_directory / f"nfl_weekly_market_2026_week_{week:02d}.csv"
    )
    current_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = current_path.parent / (
        f"{current_path.stem}_{stamp}{current_path.suffix or '.csv'}"
    )
    lines.to_csv(current_path, index=False, encoding="utf-8-sig")
    lines.to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    raw_path: Optional[Path] = None
    if not no_raw_json:
        raw_directory = output_directory / "raw"
        raw_directory.mkdir(parents=True, exist_ok=True)
        raw_path = raw_directory / (
            f"the_odds_api_nfl_week_{week:02d}_{stamp}.json"
        )
        raw_path.write_text(
            json.dumps(events, indent=2),
            encoding="utf-8",
        )
    return current_path, snapshot_path, raw_path


def build_audit(
    run_id: str,
    week: int,
    schedule_source: str,
    schedule_games: int,
    lines: pd.DataFrame,
    missing: list[str],
    priority: list[str],
    maximum_age_minutes: float,
    require_primary: bool,
    response_source: str,
    headers: dict[str, Optional[str]],
    retrieved_at: dt.datetime,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "build_id": BUILD_ID,
                "version": VERSION,
                "season": SEASON,
                "week": week,
                "schedule_source": schedule_source,
                "schedule_games": schedule_games,
                "selected_lines": len(lines),
                "missing_games": len(missing),
                "missing_detail_json": json.dumps(missing),
                "bookmaker_priority": ",".join(priority),
                "bookmakers_selected_json": json.dumps(
                    lines["bookmaker_key"].value_counts().to_dict()
                    if not lines.empty
                    else {}
                ),
                "maximum_line_age_minutes": maximum_age_minutes,
                "require_primary_bookmaker": int(require_primary),
                "response_source": response_source,
                "requests_remaining": headers.get("requests_remaining"),
                "requests_used": headers.get("requests_used"),
                "requests_last": headers.get("requests_last"),
                "retrieved_at_utc": iso_utc(retrieved_at),
                "circa_supported_by_provider": 0,
            }
        ]
    )


def run_self_test() -> int:
    schedule = pd.DataFrame(
        {
            "season": [SEASON, SEASON],
            "week": [1, 1],
            "home_team": ["SEA", "LAR"],
            "away_team": ["NE", "SF"],
            "game_date": pd.to_datetime(
                ["2026-09-09T20:20:00Z", "2026-09-10T20:35:00Z"],
                utc=True,
            ),
            "schedule_game_id": ["G1", "G2"],
        }
    )
    retrieved = dt.datetime(2026, 8, 20, 20, 0, tzinfo=dt.timezone.utc)
    events = [
        {
            "id": "E1",
            "commence_time": "2026-09-09T20:20:00Z",
            "home_team": "Seattle Seahawks",
            "away_team": "New England Patriots",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": "2026-08-20T19:55:00Z",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Seattle Seahawks", "price": -108, "point": -3.0},
                                {"name": "New England Patriots", "price": -102, "point": 3.0},
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "id": "E2",
            "commence_time": "2026-09-10T20:35:00Z",
            "home_team": "Los Angeles Rams",
            "away_team": "San Francisco 49ers",
            "bookmakers": [
                {
                    "key": "lowvig",
                    "title": "LowVig.ag",
                    "last_update": "2026-08-20T19:50:00Z",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Los Angeles Rams", "price": -110, "point": -2.5},
                                {"name": "San Francisco 49ers", "price": -110, "point": 2.5},
                            ],
                        }
                    ],
                }
            ],
        },
    ]
    lines, missing = select_week_lines(
        schedule,
        events,
        ["pinnacle", "lowvig", "betonlineag"],
        maximum_age_minutes=1440.0,
        require_primary=False,
        retrieved_at=retrieved,
        run_id="SELF_TEST",
    )
    if missing or len(lines) != 2:
        raise AssertionError(f"Unexpected self-test match result: {missing}")
    first = lines.set_index(["away_team", "home_team"])
    if first.loc[("NE", "SEA"), "bookmaker_key"] != "pinnacle":
        raise AssertionError("Primary bookmaker selection failed.")
    if first.loc[("SF", "LAR"), "bookmaker_key"] != "lowvig":
        raise AssertionError("Fallback bookmaker selection failed.")
    if float(first.loc[("NE", "SEA"), "current_market_home_margin"]) != 3.0:
        raise AssertionError("Home-spread sign conversion failed.")
    if lines["circa_line_flag"].ne(0).any():
        raise AssertionError("Proxy line was incorrectly labeled as Circa.")
    try:
        parse_priority("circa,pinnacle")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Circa provider guard failed.")
    print("[NFL_LIVE_MARKET] Self-test passed.")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    started = utc_now()
    retrieved_at = utc_now()
    run_id = str(uuid.uuid4())
    priority = parse_priority(args.bookmaker_priority)

    schedule, schedule_source = load_schedule(
        args.db_path,
        int(args.week),
        args.schedule_table,
    )

    headers: dict[str, Optional[str]] = {
        "requests_remaining": None,
        "requests_used": None,
        "requests_last": None,
    }
    if args.response_json_path is not None:
        events = load_response_json(args.response_json_path)
        response_source = str(args.response_json_path)
    else:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {args.api_key_env} is not set. "
                "PowerShell example: $env:ODDS_API_KEY = 'YOUR_KEY'"
            )
        events, headers = fetch_api_response(
            api_key,
            priority,
            args.timeout_seconds,
        )
        response_source = "THE_ODDS_API_CURRENT_NFL_SPREADS"

    lines, missing = select_week_lines(
        schedule,
        events,
        priority,
        args.maximum_line_age_minutes,
        args.require_primary_bookmaker,
        retrieved_at,
        run_id,
    )
    if lines.empty:
        raise RuntimeError(
            "No Week market lines survived matching, bookmaker priority, "
            "and freshness checks. Missing detail: " + " | ".join(missing)
        )
    if missing and not args.allow_partial_week:
        raise RuntimeError(
            f"Only {len(lines)} of {len(schedule)} Week {args.week} games "
            "received an eligible line. Re-run with --allow-partial-week only "
            "if partial execution is intentional. Missing detail: "
            + " | ".join(missing)
        )

    audit = build_audit(
        run_id,
        int(args.week),
        schedule_source,
        len(schedule),
        lines,
        missing,
        priority,
        args.maximum_line_age_minutes,
        args.require_primary_bookmaker,
        response_source,
        headers,
        retrieved_at,
    )
    save_database(args.db_path, lines, audit, int(args.week))
    current_path, snapshot_path, raw_path = save_files(
        args.project_root,
        args.output_path,
        lines,
        events,
        int(args.week),
        args.no_raw_json,
    )

    print("=" * 118)
    print(f"[NFL_LIVE_MARKET] BUILD_ID={BUILD_ID}")
    print(f"[NFL_LIVE_MARKET] VERSION={VERSION}")
    print(f"[NFL_LIVE_MARKET] Week={args.week} | games={len(lines)}/{len(schedule)}")
    print(f"[NFL_LIVE_MARKET] Bookmaker priority={','.join(priority)}")
    print("[NFL_LIVE_MARKET] Circa line represented: NO")
    print("[NFL_LIVE_MARKET] One selected reference line per matchup: YES")
    display = lines[
        [
            "away_team",
            "home_team",
            "sportsbook",
            "home_spread",
            "home_spread_price",
            "away_spread",
            "away_spread_price",
            "market_line_age_minutes",
        ]
    ].copy()
    display["market_line_age_minutes"] = display[
        "market_line_age_minutes"
    ].round(1)
    print(display.to_string(index=False))
    if missing:
        print("[NFL_LIVE_MARKET] Missing games:")
        for item in missing:
            print(f"  - {item}")
    print(f"[NFL_LIVE_MARKET] Current CSV: {current_path}")
    print(f"[NFL_LIVE_MARKET] Snapshot CSV: {snapshot_path}")
    if raw_path is not None:
        print(f"[NFL_LIVE_MARKET] Raw audit JSON: {raw_path}")
    print(f"[NFL_LIVE_MARKET] Database: {args.db_path}")
    print(
        "[NFL_LIVE_MARKET] API quota: "
        f"remaining={headers.get('requests_remaining')} | "
        f"used={headers.get('requests_used')} | "
        f"last={headers.get('requests_last')}"
    )
    print(
        "[NFL_LIVE_MARKET] Completed in "
        f"{(utc_now() - started).total_seconds():.2f} seconds"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[NFL_LIVE_MARKET] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[NFL_LIVE_MARKET] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
