#!/usr/bin/env python
"""Load and standardize the current 2026 NFL roster while preserving identities.

Primary output
--------------
SQLite/CSV: nfl_rosters_2026_raw

Audit outputs
-------------
SQLite/CSV: nfl_rosters_2026_load_audit
SQLite/CSV: nfl_rosters_2026_duplicate_id_audit
SQLite/CSV: nfl_rosters_2026_identity_issues
SQLite/CSV: nfl_rosters_2026_refresh_status

Design rules
------------
1. GSIS is the canonical working identifier used by downstream nflverse data.
2. Every available provider ID is preserved; no provider ID is renamed away.
3. Text placeholders such as "nan" and "None" are converted to missing values.
4. Duplicate/conflicting IDs are audited rather than silently discarded.
5. The standardized raw roster remains a source table; canonical row selection
   belongs in build_nfl_player_master.py.
6. Only transport/dependency failures may reuse a complete roster imported
   within 24 hours. Reuse never changes the original import date or load audits.
"""

from __future__ import annotations

import argparse
import datetime as dt
from importlib import import_module
import logging
import re
import socket
import sys
import urllib.error
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import sqlalchemy as sql


SEASON = 2026
VERSION = "v2_1_validated_recent_roster_recovery"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

OUTPUT_TABLE = "nfl_rosters_2026_raw"
LOAD_AUDIT_TABLE = "nfl_rosters_2026_load_audit"
DUPLICATE_AUDIT_TABLE = "nfl_rosters_2026_duplicate_id_audit"
IDENTITY_ISSUE_TABLE = "nfl_rosters_2026_identity_issues"
REFRESH_STATUS_TABLE = "nfl_rosters_2026_refresh_status"
MAX_CACHE_AGE_HOURS = 24.0

ID_COLUMNS = [
    "gsis_id",
    "pfr_id",
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

TEAM_ALIASES = {
    "ARZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BLT": "BAL",
    "BUF": "BUF", "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "CLV": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "GNB": "GB", "HOU": "HOU", "HST": "HOU", "IND": "IND", "JAC": "JAX",
    "JAX": "JAX", "KC": "KC", "KAN": "KC", "KCC": "KC", "LA": "LAR",
    "LAR": "LAR", "STL": "LAR", "LAC": "LAC", "SD": "LAC", "SDG": "LAC",
    "LV": "LV", "LVR": "LV", "OAK": "LV", "MIA": "MIA", "MIN": "MIN",
    "NE": "NE", "NWE": "NE", "NO": "NO", "NOR": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "SF": "SF",
    "SFO": "SF", "TB": "TB", "TAM": "TB", "TEN": "TEN", "WAS": "WAS",
    "WSH": "WAS", "WFT": "WAS",
}

NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

NULL_STRINGS = {"", "nan", "none", "null", "<na>", "nat"}

COLUMN_ALIASES = {
    "gsis_player_id": "gsis_id",
    "player_gsis_id": "gsis_id",
    "nfl_id": "gsis_id",
    "pfr_player_id": "pfr_id",
    "sr_id": "sportradar_id",
    "sportradar_player_id": "sportradar_id",
    "fantasydata_id": "fantasy_data_id",
    "fantasy_dataid": "fantasy_data_id",
    "full_name": "player_name",
    "display_name": "player_name",
    "football_name": "football_name",
    "recent_team": "team",
    "club_code": "team",
    "team_abbr": "team",
    "pos": "position",
    "roster_position": "position",
    "birthdate": "birth_date",
    "experience": "years_exp",
    "years_experience": "years_exp",
    "jersey": "jersey_number",
}

FINAL_COLUMNS = [
    "season",
    "player_id",
    "gsis_id",
    "pfr_id",
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
    "team",
    "position",
    "depth_chart_position",
    "jersey_number",
    "height",
    "weight",
    "birth_date",
    "age",
    "years_exp",
    "college",
    "status",
    "rookie_year",
    "entry_year",
    "draft_club",
    "draft_number",
    "source",
    "source_row_number",
    "identity_issue_flag",
    "roster_version",
    "date_imported",
]


def clean_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return None
    return text


def clean_id(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    text = text.replace("\u200b", "").replace("\ufeff", "").strip()
    if text.endswith(".0") and re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text or None


def normalize_team(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    text = text.upper().replace(".", "").strip()
    return TEAM_ALIASES.get(text, text)


def normalize_position(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    return text.upper().replace(" ", "").strip()


def clean_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def coalesce_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in candidates:
        if column in frame.columns:
            values = frame[column].map(clean_scalar)
            result = result.where(result.notna(), values)
    return result


class RosterSourceUnavailable(RuntimeError):
    """An identified transport/dependency failure, eligible for recent-cache recovery."""


def is_transport_failure(exc: BaseException) -> bool:
    """Recognize network failures without treating arbitrary loader bugs as outages."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (urllib.error.URLError, socket.gaierror,
                                ConnectionError, TimeoutError)):
            return True
        module = type(current).__module__
        name = type(current).__name__
        if module.startswith(("requests.", "urllib3.", "httpx.")) and name in {
            "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
            "HTTPError", "HTTPStatusError", "SSLError", "ProxyError",
            "NewConnectionError", "MaxRetryError", "NameResolutionError",
            "ConnectError", "ReadError", "RemoteProtocolError",
        }:
            return True
        # nflreadpy releases may wrap the underlying requests exception in
        # RuntimeError. These narrow transport markers cover the reported
        # Windows DNS error, without classifying schema/parse errors as outages.
        message = str(current).lower()
        if isinstance(current, RuntimeError) and any(marker in message for marker in (
            "getaddrinfo failed", "name or service not known",
            "temporary failure in name resolution", "failed to establish a new connection",
            "connection timed out", "connection refused", "max retries exceeded",
            "http error 403", "http error 404", "403 client error", "404 client error",
            "429 client error", "500 server error", "502 server error", "503 server error",
        )):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_rosters(season: int, offline_csv: Optional[Path]) -> tuple[pd.DataFrame, str]:
    if offline_csv is not None:
        if not offline_csv.exists():
            raise FileNotFoundError(f"Roster CSV does not exist: {offline_csv}")
        return pd.read_csv(offline_csv, low_memory=False), f"csv:{offline_csv}"

    errors: list[str] = []
    # nfl_data_py documents import_seasonal_rosters; import_rosters does not
    # exist. nflreadpy remains primary; installing the deprecated fallback is
    # not required when it is absent.
    for module_name, function_name in (
        ("nflreadpy", "load_rosters"),
        ("nfl_data_py", "import_seasonal_rosters"),
    ):
        label = f"{module_name}.{function_name}"
        try:
            module = import_module(module_name)
        except ImportError as exc:
            errors.append(f"{label}: dependency unavailable: {exc}")
            continue
        loader = getattr(module, function_name, None)
        if not callable(loader):
            errors.append(f"{label}: installed module has no callable {function_name}")
            continue
        try:
            raw = loader([season])
            if hasattr(raw, "to_pandas"):
                raw = raw.to_pandas()
        except ImportError as exc:
            errors.append(f"{label}: dependency unavailable: {exc}")
            continue
        except Exception as exc:
            if not is_transport_failure(exc):
                raise
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        # Conversion/validation errors are programming or source-data errors,
        # and must not silently switch to an older cache.
        return pd.DataFrame(raw), label

    raise RosterSourceUnavailable("Unable to fetch NFL rosters. " + " | ".join(errors))


def timestamp_utc(value: Any) -> dt.datetime:
    """Interpret original timezone-naive imports in this machine's local zone."""
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("Missing roster import timestamp")
    stamp = parsed.to_pydatetime()
    # datetime.astimezone() applies the computer's timezone/DST at that date.
    return stamp.astimezone(dt.timezone.utc)


def validate_roster(roster: pd.DataFrame, season: int, *,
                    now: Optional[dt.datetime] = None,
                    require_recent: bool = False) -> tuple[dt.datetime, float]:
    """Validate normalized identities, complete team coverage, and import provenance."""
    missing = sorted(set(FINAL_COLUMNS) - set(roster.columns))
    if missing:
        raise ValueError(f"Roster schema missing canonical columns: {missing}")
    if roster.empty:
        raise ValueError("Roster contains zero rows")
    seasons = pd.to_numeric(roster["season"], errors="coerce")
    if seasons.isna().any() or not seasons.eq(season).all():
        raise ValueError(f"Roster season must be {season} on every row")
    teams = roster["team"].map(clean_scalar)
    if set(teams.dropna()) != NFL_TEAMS or teams.isna().any():
        raise ValueError(f"Roster must contain exactly all 32 canonical NFL teams; "
                         f"missing={sorted(NFL_TEAMS - set(teams.dropna()))}")
    for column in ("player_name", "position", "source", "roster_version"):
        if roster[column].map(clean_scalar).isna().any():
            raise ValueError(f"Roster has missing {column}")
    gsis = roster["gsis_id"].map(clean_id)
    player_ids = roster["player_id"].map(clean_id)
    if not gsis.fillna("").equals(player_ids.fillna("")):
        raise ValueError("Roster identity integrity failure: player_id differs from gsis_id")
    if not gsis.dropna().str.fullmatch(r"\d{2}-\d{7}").all():
        raise ValueError("Roster identity integrity failure: malformed GSIS identifier")
    # Preserve the original workflow's audited, occasional missing GSIS row.
    # A provider without usable canonical identities is not a usable cache.
    if gsis.notna().mean() < 0.95:
        raise ValueError("Roster is missing canonical GSIS identities for over 5% of rows")
    usable = pd.DataFrame({"team": teams, "gsis_id": gsis}).dropna()
    counts = usable.groupby("team")["gsis_id"].nunique()
    incomplete = counts[counts < 45].to_dict()
    if len(counts) != 32 or incomplete:
        raise ValueError(f"Roster coverage incomplete: require at least 45 unique GSIS IDs per team; {incomplete}")
    # Same player on multiple historical transaction rows is permitted. Reusing
    # a canonical ID for distinct names is a corrupt identity map.
    names = roster["player_name"].map(lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower()))
    mapping = pd.DataFrame({"gsis_id": gsis, "name": names}).dropna()
    conflicts = mapping.groupby("gsis_id")["name"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Roster identity integrity failure: a GSIS ID maps to conflicting names")
    row_numbers = pd.to_numeric(roster["source_row_number"], errors="coerce")
    if row_numbers.isna().any() or row_numbers.le(0).any() or row_numbers.duplicated().any():
        raise ValueError("Roster source_row_number is missing, nonpositive or duplicated")
    try:
        timestamps = roster["date_imported"].map(timestamp_utc)
    except Exception as exc:
        raise ValueError(f"Roster import timestamp is invalid: {exc}") from exc
    checked_at = now or dt.datetime.now(dt.timezone.utc)
    checked_at = checked_at.astimezone(dt.timezone.utc)
    if any(stamp > checked_at for stamp in timestamps):
        raise ValueError("Roster import timestamp is in the future")
    original_as_of = min(timestamps)
    age_hours = (checked_at - original_as_of).total_seconds() / 3600.0
    if require_recent and age_hours > MAX_CACHE_AGE_HOURS:
        raise ValueError(f"Roster cache is {age_hours:.2f} hours old; maximum is {MAX_CACHE_AGE_HOURS:g} hours")
    return original_as_of, age_hours


def read_recent_cache(engine, season: int, now: dt.datetime) -> tuple[pd.DataFrame, dt.datetime, float]:
    if not sql.inspect(engine).has_table(OUTPUT_TABLE):
        raise ValueError(f"No existing {OUTPUT_TABLE} table")
    with engine.connect() as connection:
        roster = pd.read_sql_query(sql.text(f'SELECT * FROM "{OUTPUT_TABLE}"'), connection)
    original_as_of, age_hours = validate_roster(roster, season, now=now, require_recent=True)
    return roster, original_as_of, age_hours


def refresh_status(season: int, status: str, attempted_at: dt.datetime, *,
                   original_as_of: Optional[dt.datetime] = None,
                   age_hours: Optional[float] = None, source: str = "",
                   error: str = "", roster_rows: int = 0) -> pd.DataFrame:
    return pd.DataFrame([{
        "season": season, "status": status,
        "roster_as_of_utc": original_as_of.isoformat() if original_as_of else "",
        "attempted_at_utc": attempted_at.isoformat(), "age_hours": age_hours,
        "source": source, "error": error, "roster_rows": roster_rows,
        "roster_version": VERSION,
    }])


def standardize_rosters(raw: pd.DataFrame, season: int, source_name: str) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Roster source returned zero rows.")

    frame = raw.copy().reset_index(drop=True)
    frame.columns = [clean_column_name(column) for column in frame.columns]
    frame = frame.rename(
        columns={
            old: new
            for old, new in COLUMN_ALIASES.items()
            if old in frame.columns and new not in frame.columns
        }
    )

    # Do not overwrite an existing canonical field; fill it from known aliases.
    for target, candidates in {
        "gsis_id": ["gsis_id", "player_id", "nflverse_id"],
        "pfr_id": ["pfr_id", "pfr_player_id"],
        "player_name": ["player_name", "full_name", "display_name", "name"],
        "team": ["team", "recent_team", "club_code", "team_abbr"],
        "position": ["position", "pos", "roster_position"],
        "first_name": ["first_name", "first"],
        "last_name": ["last_name", "last"],
        "football_name": ["football_name", "short_name"],
    }.items():
        frame[target] = coalesce_columns(frame, candidates)

    for column in ID_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = frame[column].map(clean_id)

    for column in [
        "depth_chart_position", "jersey_number", "height", "weight",
        "birth_date", "age", "years_exp", "college", "status",
        "rookie_year", "entry_year", "draft_club", "draft_number",
    ]:
        if column not in frame.columns:
            frame[column] = None

    out = pd.DataFrame(index=frame.index)
    out["season"] = pd.to_numeric(
        frame.get("season", pd.Series(season, index=frame.index)), errors="coerce"
    ).fillna(season).astype(int)
    out["gsis_id"] = frame["gsis_id"].map(clean_id)
    out["player_id"] = out["gsis_id"]
    for column in ID_COLUMNS:
        if column == "gsis_id":
            continue
        out[column] = frame[column].map(clean_id)

    out["player_name"] = frame["player_name"].map(clean_scalar)
    out["first_name"] = frame["first_name"].map(clean_scalar)
    out["last_name"] = frame["last_name"].map(clean_scalar)
    out["football_name"] = frame["football_name"].map(clean_scalar)
    out["team"] = frame["team"].map(normalize_team)
    out["position"] = frame["position"].map(normalize_position)
    out["depth_chart_position"] = frame["depth_chart_position"].map(normalize_position)

    for column in ["jersey_number", "height", "weight", "age", "years_exp", "rookie_year", "entry_year", "draft_number"]:
        out[column] = pd.to_numeric(frame[column], errors="coerce")

    out["birth_date"] = frame["birth_date"].map(clean_scalar)
    out["college"] = frame["college"].map(clean_scalar)
    out["status"] = frame["status"].map(clean_scalar)
    out["draft_club"] = frame["draft_club"].map(normalize_team)
    out["source"] = source_name
    out["source_row_number"] = out.index + 1

    issue_parts = []
    for row in out.itertuples(index=False):
        issues: list[str] = []
        if row.player_name is None:
            issues.append("missing_name")
        if row.team is None:
            issues.append("missing_team")
        if row.position is None:
            issues.append("missing_position")
        if row.gsis_id is None:
            issues.append("missing_gsis_id")
        issue_parts.append("|".join(issues) if issues else "ok")
    out["identity_issue_flag"] = issue_parts
    out["roster_version"] = VERSION
    out["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")

    for column in FINAL_COLUMNS:
        if column not in out.columns:
            out[column] = None

    out = out[FINAL_COLUMNS].drop_duplicates().reset_index(drop=True)
    return out


def build_duplicate_audit(roster: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for identifier in ID_COLUMNS:
        valid = roster[roster[identifier].notna()].copy()
        counts = valid.groupby(identifier).size()
        duplicate_values = set(counts[counts > 1].index.astype(str))
        if not duplicate_values:
            continue
        audit = valid[valid[identifier].astype(str).isin(duplicate_values)].copy()
        audit.insert(0, "identifier_type", identifier)
        audit.insert(1, "identifier_value", audit[identifier].astype(str))
        rows.append(audit)
    if not rows:
        return pd.DataFrame(columns=["identifier_type", "identifier_value", *FINAL_COLUMNS])
    return pd.concat(rows, ignore_index=True, sort=False)


def build_load_audit(roster: pd.DataFrame, source_name: str) -> pd.DataFrame:
    official = roster[roster["team"].isin(NFL_TEAMS)]
    return pd.DataFrame([
        {
            "season": SEASON,
            "source": source_name,
            "source_rows": len(roster),
            "official_team_rows": len(official),
            "official_teams": official["team"].nunique(),
            "unique_gsis_ids": roster["gsis_id"].nunique(dropna=True),
            "missing_gsis_ids": int(roster["gsis_id"].isna().sum()),
            "missing_names": int(roster["player_name"].isna().sum()),
            "identity_issue_rows": int(roster["identity_issue_flag"].ne("ok").sum()),
            "roster_version": VERSION,
            "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
    ])


def configure_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_rosters")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        log_dir / f"load_nfl_rosters_{dt.datetime.now():%Y%m%d_%H%M%S}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_frame(engine, output_dir: Path, table_name: str, frame: pd.DataFrame) -> None:
    if len(frame.columns) == 0:
        raise RuntimeError(
            f"Refusing to persist {table_name}: DataFrame has zero columns. "
            "Every output, including an empty audit, must have an explicit schema."
        )
    frame.to_sql(table_name, con=engine, if_exists="replace", index=False)
    frame.to_csv(output_dir / f"{table_name}.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the standardized 2026 NFL roster.")
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--input-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.project_root / "outputs"
    log_dir = args.project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(log_dir)

    logger.info("[ROSTERS] Season: %s", args.season)
    logger.info("[ROSTERS] Database: %s", args.db_path)
    if args.season != SEASON:
        raise ValueError(f"This canonical loader writes {SEASON} tables; --season must be {SEASON}")
    attempted_at = dt.datetime.now(dt.timezone.utc)
    engine = sql.create_engine(f"sqlite:///{args.db_path}", pool_pre_ping=True)
    try:
        try:
            raw, source_name = load_rosters(args.season, args.input_csv)
        except RosterSourceUnavailable as source_error:
            try:
                roster, original_as_of, age_hours = read_recent_cache(
                    engine, args.season, dt.datetime.now(dt.timezone.utc))
            except Exception as cache_error:
                raise RosterSourceUnavailable(
                    f"{source_error} | Existing roster cannot be reused: {cache_error}. "
                    "No current roster was invented or re-dated. Restore access to github.com "
                    "(DNS, internet connection, or proxy settings), then rerun. "
                    "A valid local --input-csv can also be imported explicitly."
                ) from source_error
            source_name = "|".join(sorted(set(roster["source"].dropna().astype(str))))
            status = refresh_status(args.season, "REUSED_RECENT_CACHE", attempted_at,
                                    original_as_of=original_as_of, age_hours=age_hours,
                                    source=source_name, error=str(source_error), roster_rows=len(roster))
            save_frame(engine, output_dir, REFRESH_STATUS_TABLE, status)
            logger.warning("[ROSTERS] REUSED_RECENT_CACHE: %s existing rows, originally imported %s, "
                           "age %.2f hours (limit %.0f). Original roster and load audits are unchanged.",
                           f"{len(roster):,}", original_as_of.isoformat(), age_hours, MAX_CACHE_AGE_HOURS)
            logger.warning("[ROSTERS] Refresh failed: %s", source_error)
            return 0

        logger.info("[ROSTERS] Loaded %s source rows via %s", f"{len(raw):,}", source_name)
        if "season" in raw.columns:
            source_seasons = pd.to_numeric(raw["season"], errors="coerce")
            if source_seasons.isna().any() or not source_seasons.eq(args.season).all():
                raise ValueError(f"Roster source includes missing/invalid/wrong season; expected {args.season}")
        roster = standardize_rosters(raw, args.season, source_name)
        original_as_of, age_hours = validate_roster(roster, args.season)
        duplicate_audit = build_duplicate_audit(roster)
        issues = roster[roster["identity_issue_flag"].ne("ok")].copy()
        load_audit = build_load_audit(roster, source_name)

        save_frame(engine, output_dir, OUTPUT_TABLE, roster)
        save_frame(engine, output_dir, LOAD_AUDIT_TABLE, load_audit)
        save_frame(engine, output_dir, DUPLICATE_AUDIT_TABLE, duplicate_audit)
        save_frame(engine, output_dir, IDENTITY_ISSUE_TABLE, issues)
        save_frame(engine, output_dir, REFRESH_STATUS_TABLE,
                   refresh_status(args.season, "LIVE", attempted_at, original_as_of=original_as_of,
                                  age_hours=age_hours, source=source_name, roster_rows=len(roster)))

        logger.info("[ROSTERS] LIVE: Saved %s rows to %s", f"{len(roster):,}", OUTPUT_TABLE)
        logger.info("[ROSTERS] Unique GSIS IDs: %s", f"{roster['gsis_id'].nunique(dropna=True):,}")
        logger.info("[ROSTERS] Missing GSIS IDs: %s", f"{roster['gsis_id'].isna().sum():,}")
        logger.info("[ROSTERS] Duplicate audit rows: %s", f"{len(duplicate_audit):,}")
        logger.info("[ROSTERS] Identity issue rows: %s", f"{len(issues):,}")
        return 0
    except Exception as exc:
        save_frame(engine, output_dir, REFRESH_STATUS_TABLE,
                   refresh_status(args.season, "FAILED", attempted_at,
                                  error=f"{type(exc).__name__}: {exc}"))
        logger.error("[ROSTERS] FAILED: %s", exc)
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
