"""Automatic current NFL depth/availability inputs; no manual starter file.

Player talent remains in the existing historical performance table. ESPN's
published roster, injury status and ordered depth chart supply availability
and depth order. All identities are matched by provider ID or a unique,
normalized team/name key without conflicting IDs. Unresolved identities block
the affected scenario; unmatched QB/OL starters stop the refresh.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEASON = 2026
VERSION = "v1_3_unique_ol_starter_assignment"
AVAILABILITY_TABLE = "nfl_live_player_availability_2026"
SOURCE_TABLE = "nfl_live_roster_source_audit_2026"
REFRESH_STATUS_TABLE = "nfl_live_roster_refresh_status_2026"
API_ROOT = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
UNAVAILABLE = {
    "out", "o", "injured reserve", "ir", "reserve injured", "reserve/injured",
    "pup", "physically unable to perform", "suspended", "suspension", "sus",
    "nfi", "non football injury", "inactive", "practice squad", "ps", "retired",
    "released", "waived", "cut", "reserve", "res", "dev", "ina",
}
UNCERTAIN = {"questionable", "q", "doubtful", "d", "day-to-day", "day to day"}
SLOT_KEYS = {
    "qb": "QB1", "rb": "RB1", "te": "TE1", "wr1": "WR1",
    "wr2": "WR2", "wr3": "WR3", "lt": "LT", "lg": "LG", "c": "C",
    "rg": "RG", "rt": "RT", "pk": "K", "k": "K", "p": "P", "ls": "LS",
    "lcb": "CB1", "rcb": "CB2", "nb": "SLOT", "fs": "FS", "ss": "SS",
}
IGNORED_KEYS = {"h", "kr", "pr"}
OL_SLOTS = ("lt", "lg", "c", "rg", "rt")
TEAM_ALIASES = {"WSH": "WAS", "LA": "LAR", "JAC": "JAX"}


def text(value: Any) -> str:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return str(value).strip()


def player_key(value: Any) -> str:
    value = unicodedata.normalize("NFKD", text(value)).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^a-z0-9 ]", "", value.lower()).split()
    if parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts.pop()
    key = "".join(parts)
    return {"olufashanu": "olumuyiwafashanu"}.get(key, key)


def utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing timestamp")
    if stamp.tzinfo is None:
        # Existing SQL builders write the computer's local time.
        stamp = pd.Timestamp(stamp.to_pydatetime().astimezone(dt.timezone.utc))
    return stamp.tz_convert("UTC")


def prediction_cutoff(value: str | None = None) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    if not value:
        return now
    if re.fullmatch(r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})", str(value)):
        tomorrow = dt.datetime.combine(pd.Timestamp(value).date() + dt.timedelta(days=1), dt.time.min)
        return min(now, pd.Timestamp(tomorrow.astimezone(dt.timezone.utc)) - pd.Timedelta(nanoseconds=1))
    return min(now, utc(value))


class SourceUnavailable(RuntimeError):
    """No usable current source: the frozen baseline can still be evaluated."""
    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


class SourceRequestError(SourceUnavailable):
    """Endpoint-specific transport failure, without bypassing access controls."""


def request_json(url: str, timeout: float) -> dict:
    attempted = pd.Timestamp.now(tz="UTC").isoformat()
    request = urllib.request.Request(url, headers={
        "User-Agent": "NFLStructuralModel/2026 availability audit",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        preview = " ".join(exc.read(512).decode("utf-8", errors="replace").split())[:400]
        headers = {key: value for key, value in (exc.headers or {}).items()
                   if key.lower() in {"server", "content-type", "via", "retry-after"}}
        attempt = {"attempted_at_utc": attempted, "url": url, "endpoint": url.rsplit("/", 1)[-1],
                   "http_status": exc.code, "error": str(exc), "response_preview": preview,
                   "response_headers": headers}
        raise SourceRequestError(f"{attempt['endpoint']} {url}: HTTP {exc.code} {exc.reason}", [attempt]) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        attempt = {"attempted_at_utc": attempted, "url": url, "endpoint": url.rsplit("/", 1)[-1],
                   "http_status": None, "error": f"{type(exc).__name__}: {exc}"}
        raise SourceRequestError(f"{attempt['endpoint']} {url}: {attempt['error']}", [attempt]) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Source response is not an object: {url}")
    return payload


def attach_roster_provenance(connection, master: pd.DataFrame) -> pd.DataFrame:
    """Keep the raw roster observation time distinct from a master rebuild time."""
    result = master.copy()
    names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result["roster_source_as_of_utc"] = ""
    result["roster_refresh_status"] = "UNVERIFIED"
    if "nfl_rosters_2026_raw" not in names:
        return result
    rows = pd.read_sql_query('SELECT season, date_imported FROM nfl_rosters_2026_raw', connection)
    if rows.empty or not pd.to_numeric(rows.season, errors="coerce").eq(SEASON).all():
        return result
    observed = rows.date_imported.map(utc).min()
    state = "EXISTING_RAW_ROSTER"
    if "nfl_rosters_2026_refresh_status" in names:
        status = pd.read_sql_query('SELECT * FROM nfl_rosters_2026_refresh_status', connection)
        if len(status) != 1 or int(status.iloc[0]["season"]) != SEASON:
            raise ValueError("Invalid roster refresh provenance")
        row = status.iloc[0]
        state = text(row["status"])
        if state in {"LIVE", "REUSED_RECENT_CACHE"}:
            # Use the older timestamp; metadata must never refresh old rows.
            observed = min(observed, utc(row["roster_as_of_utc"]))
    result["roster_source_as_of_utc"] = observed.isoformat()
    result["roster_refresh_status"] = state
    return result


def master_roster(team: str, master: pd.DataFrame | None, depth_payload: dict | None = None) -> dict:
    """Use the already-refreshed NFLverse master as an independent roster input.

    Current game injury designations still come from the ESPN depth feed.
    Freshness comes from the original raw roster observation, never its rebuild.
    """
    required = {"team", "player_id", "player_name", "position", "status", "season", "date_imported"}
    if master is None or required - set(master):
        raise SourceUnavailable(f"{team}: a refreshed canonical player master is unavailable")
    rows = master[master.team.astype(str).str.upper().eq(team)].copy()
    rows = rows[rows.player_id.map(text).ne("")]
    if len(rows) < 45 or rows.player_id.duplicated().any():
        raise SourceUnavailable(f"{team}: current master lacks a complete, unique roster")
    if not pd.to_numeric(rows.season, errors="coerce").eq(SEASON).all():
        raise ValueError(f"{team}: player master contains the wrong season")
    if "roster_source_as_of_utc" not in rows or "roster_refresh_status" not in rows:
        raise SourceUnavailable(f"{team}: original roster source freshness is unverified")
    if not rows.roster_refresh_status.isin({"LIVE", "REUSED_RECENT_CACHE", "EXISTING_RAW_ROSTER"}).all():
        raise SourceUnavailable(f"{team}: roster refresh did not produce verified source data")
    stamps = rows.roster_source_as_of_utc.map(utc)
    now = pd.Timestamp.now(tz="UTC")
    if (now - stamps.min()).total_seconds() > 86400 or (stamps.max() - now).total_seconds() > 300:
        raise SourceUnavailable(f"{team}: original roster source is older than 24 hours or future dated")
    chart_ids: dict[str, set[str]] = {}
    if depth_payload is not None:
        for _, node in depth_nodes(depth_payload):
            for athlete in node.get("athletes", []):
                chart_ids.setdefault(player_key(athlete.get("displayName")), set()).add(text(athlete.get("id")))
    athletes = []
    for row in rows.to_dict("records"):
        provider_id = text(row.get("espn_id")).removesuffix(".0")
        provider_method = "MASTER_ESPN_ID" if provider_id else "CANONICAL_ID_ONLY"
        matches = chart_ids.get(player_key(row["player_name"]), set()) - {""}
        if not provider_id and len(matches) == 1:
            provider_id = next(iter(matches))
            provider_method = "UNIQUE_DEPTH_NAME_WITH_NO_CONFLICTING_MASTER_ID"
        provider_id = provider_id or f"gsis:{text(row['player_id'])}"
        athletes.append({"id": provider_id, "canonical_player_id": text(row["player_id"]),
                         "provider_id_match_method": provider_method,
                         "displayName": text(row["player_name"]),
                         "position": {"abbreviation": text(row["position"])},
                         "status": {"name": text(row["status"])}, "injuries": []})
    if len({row["id"] for row in athletes}) != len(athletes):
        raise ValueError(f"{team}: duplicate provider IDs in master roster")
    return {"status": "success", "season": {"year": SEASON}, "team": {"abbreviation": team},
            "timestamp": stamps.min().isoformat(), "athletes": [{"items": athletes}],
            "roster_provider": "REFRESHED_NFLVERSE_PLAYER_MASTER",
            "timestamp_basis": "ORIGINAL_ROSTER_OBSERVATION_TIME",
            "roster_refresh_status": text(rows.roster_refresh_status.iloc[0])}


def validate_source_age(record: dict, team: str) -> None:
    if (record.get("roster_provider") == "REFRESHED_NFLVERSE_PLAYER_MASTER"
            and record["roster"].get("timestamp_basis") != "ORIGINAL_ROSTER_OBSERVATION_TIME"):
        raise SourceUnavailable(f"{team}: cached master roster lacks original source timestamp")
    now = pd.Timestamp.now(tz="UTC")
    stamps = [utc(record["fetched_at_utc"]), utc(record["roster"]["timestamp"]),
              utc(record["depthcharts"]["timestamp"])]
    if any((now - stamp).total_seconds() > 86400 or (stamp - now).total_seconds() > 300 for stamp in stamps):
        raise SourceUnavailable(f"{team}: live source/cache is older than 24 hours or future dated")


def validate_payload(payload: dict, team: str, kind: str) -> None:
    if text(payload.get("status")).lower() != "success":
        raise ValueError(f"{team}: {kind} source did not report success")
    if int(payload.get("season", {}).get("year", 0)) != SEASON:
        raise ValueError(f"{team}: wrong season in {kind} source")
    reported = text(payload.get("team", {}).get("abbreviation")).upper()
    if TEAM_ALIASES.get(reported, reported) != team:
        raise ValueError(f"{team}: wrong team {reported} in {kind} source")
    utc(payload.get("timestamp"))


def load_team(team: str, team_id: str, cache_dir: Path, timeout: float,
              use_cache: bool = False, master: pd.DataFrame | None = None) -> tuple[dict, dict]:
    path = cache_dir / f"{team}.json"
    errors, attempts = [], []
    record = None
    if not use_cache:
        payloads = {}
        # Fetch independently. A failure of the newly added roster endpoint must
        # not prevent the established depth endpoint from being tried.
        for kind in ("depthcharts", "roster"):
            url = f"{API_ROOT}/{team_id}/{kind}"
            try:
                payloads[kind] = request_json(url, timeout)
                validate_payload(payloads[kind], team, kind)
                attempts.append({"team": team, "attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                 "url": url, "endpoint": kind, "http_status": 200, "error": ""})
            except SourceUnavailable as exc:
                errors.append(str(exc))
                attempts.extend({"team": team, **item} for item in exc.attempts)
        if "depthcharts" in payloads and "roster" not in payloads:
            try:
                payloads["roster"] = master_roster(team, master, payloads["depthcharts"])
            except SourceUnavailable as exc:
                errors.append(str(exc))
        if set(payloads) == {"depthcharts", "roster"}:
            record = {"team": team, "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                      "roster_provider": payloads["roster"].get("roster_provider", "ESPN"), **payloads}
            try:
                validate_source_age(record, team)
            except SourceUnavailable as exc:
                errors.append(str(exc))
                record = None
            if record is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                temporary.replace(path)
                state = "LIVE" if record["roster_provider"] == "ESPN" else "LIVE_DEPTH_MASTER_ROSTER"
        if record is None:
            record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            state = "CACHED_AFTER_FETCH_ERROR"
    else:
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        state = "EXPLICIT_CACHE"
    if record is None:
        raise SourceUnavailable(f"{team}: current depth/roster unavailable; " + " | ".join(errors), attempts)
    for kind in ("roster", "depthcharts"):
        validate_payload(record[kind], team, kind)
    try:
        validate_source_age(record, team)
    except SourceUnavailable as exc:
        raise SourceUnavailable(str(exc) + (" | " + " | ".join(errors) if errors else ""), attempts) from exc
    provider = record.get("roster_provider", "ESPN")
    audit = {
        "season": SEASON, "team": team, "source_state": state,
        "fetched_at_utc": record["fetched_at_utc"],
        "roster_source_timestamp": record["roster"]["timestamp"],
        "depth_source_timestamp": record["depthcharts"]["timestamp"],
        "roster_provider": provider,
        "roster_timestamp_basis": record["roster"].get("timestamp_basis", "PROVIDER_TIMESTAMP"),
        "roster_refresh_status": record["roster"].get("roster_refresh_status", "LIVE_PROVIDER"),
        "roster_url": f"{API_ROOT}/{team_id}/roster" if provider == "ESPN" else "sqlite:nfl_player_master_2026",
        "depth_url": f"{API_ROOT}/{team_id}/depthcharts", "fetch_error": " | ".join(errors),
        "request_attempts_json": json.dumps(attempts, sort_keys=True),
        "source_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        "live_depth_version": VERSION,
    }
    return record, audit


def fetch_sources(root: Path, team_ids: dict[str, str], timeout: float,
                  use_cache: bool = False, master: pd.DataFrame | None = None) -> tuple[dict[str, dict], pd.DataFrame]:
    cache = root / "data" / "raw" / "nfl_live_depth_2026"
    sources, audits, errors, attempts, fatal = {}, [], [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(load_team, team, team_id, cache, timeout, use_cache, master): team
                   for team, team_id in team_ids.items()}
        for future in concurrent.futures.as_completed(futures):
            team = futures[future]
            try:
                sources[team], audit = future.result()
                audits.append(audit)
                attempts.extend(json.loads(audit["request_attempts_json"]))
            except SourceUnavailable as exc:
                errors.append(str(exc))
                attempts.extend(exc.attempts)
            except Exception as exc:
                fatal.append(f"{team}: {type(exc).__name__}: {exc}")
    output = root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {"attempted_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "version": VERSION,
                   "usable_teams": sorted(sources), "source_errors": sorted(errors),
                   "fatal_errors": sorted(fatal), "requests": attempts}
    (output / "nfl_live_roster_fetch_diagnostics_2026.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    if fatal:
        raise RuntimeError("Automatic roster source validation failed:\n" + "\n".join(sorted(fatal)))
    if errors:
        raise SourceUnavailable("Automatic roster refresh unavailable:\n" + "\n".join(sorted(errors)), attempts)
    return sources, pd.DataFrame(audits).sort_values("team").reset_index(drop=True)


def injury_state(roster_athlete: dict, chart_athletes: list[dict]) -> dict:
    statuses = []
    base = roster_athlete.get("status", {})
    if isinstance(base, dict):
        statuses.extend(text(base.get(key)) for key in ("name", "type", "abbreviation"))
    elif base:
        statuses.append(text(base))
    injuries = [injury for athlete in [roster_athlete, *chart_athletes]
                for injury in athlete.get("injuries", []) if isinstance(injury, dict)]
    # The newest dated injury record wins over an older conflicting designation.
    dated = [(utc(item["date"]), item) for item in injuries if item.get("date")]
    if dated:
        newest = max(stamp for stamp, _ in dated)
        injuries = [item for stamp, item in dated if stamp == newest]
    for item in injuries:
        statuses.append(text(item.get("status")))
        kind = item.get("type") or {}
        if isinstance(kind, dict):
            statuses.extend(text(kind.get(key)) for key in ("description", "abbreviation"))
    normalized = {item.lower().replace("_", " ") for item in statuses if item}
    unavailable = bool(normalized & UNAVAILABLE) or any(
        pattern in status for status in normalized
        for pattern in ("injured reserve", "physically unable", "suspend", "practice squad", "retired")
    )
    uncertain = not unavailable and bool(normalized & UNCERTAIN)
    chosen = "|".join(sorted({text(item.get("status")) for item in injuries if item.get("status")}))
    if not chosen:
        chosen = text(base.get("name")) if isinstance(base, dict) else text(base)
    return {"live_injury_status": chosen or "AVAILABLE",
            "live_unavailable": int(unavailable), "live_uncertain": int(uncertain),
            "live_injury_reported_at": max((stamp.isoformat() for stamp, _ in dated), default="")}


def depth_nodes(payload: dict) -> list[tuple[str, dict]]:
    nodes = []
    for group in payload.get("depthchart", []):
        positions = group.get("positions", {})
        if not isinstance(positions, dict):
            raise ValueError("Unsupported depth positions schema")
        for key, node in positions.items():
            if key.lower() not in IGNORED_KEYS and isinstance(node, dict):
                nodes.append((key.lower(), node))
    if not nodes:
        raise ValueError("Empty source depth chart")
    return nodes


def select_ol_starters(team: str, nodes: list[tuple[str, dict]],
                       mapped: dict[str, dict]) -> dict[str, tuple[dict, int]]:
    """Project five distinct OL players from position-specific published order.

    Retain available published rank-one starters first, then minimize the total
    source rank. Cross-listed backups cannot fill two positions. A tied best
    lineup is unresolved, not an invitation to choose by dictionary order.
    Unidentified candidates remain in the ranking; selected identities must
    still pass build_live_inputs' existing validation.
    """
    candidates: dict[str, list[tuple[dict, int]]] = {}
    for slot, node in nodes:
        if slot not in OL_SLOTS:
            continue
        if slot in candidates:
            raise RuntimeError(f"{team}: duplicate published OL position {slot.upper()}")
        choices, seen = [], set()
        for rank, athlete in enumerate(node.get("athletes", []), 1):
            provider_id = text(athlete.get("id"))
            row = mapped.get(provider_id)
            if row is None or row["live_unavailable"] or provider_id in seen:
                continue
            seen.add(provider_id)
            choices.append((row, rank))
        candidates[slot] = choices
    missing = [slot.upper() for slot in OL_SLOTS if not candidates.get(slot)]
    if missing:
        raise RuntimeError(f"{team}: no available player in published OL depth order for {', '.join(missing)}")

    def identity(choice: tuple[dict, int]) -> str:
        row = choice[0]
        return text(row["player_id"]) or "ESPN:" + text(row["espn_id"])

    first = {slot: candidates[slot][0] for slot in OL_SLOTS}
    if len({identity(choice) for choice in first.values()}) == len(OL_SLOTS):
        return first

    best_score: tuple[int, int] | None = None
    best_plans: list[dict[str, tuple[dict, int]]] = []

    def visit(index: int, plan: dict[str, tuple[dict, int]], used: set[str],
              primary: int, rank_sum: int) -> None:
        nonlocal best_score, best_plans
        if index == len(OL_SLOTS):
            score = (-primary, rank_sum)
            if best_score is None or score < best_score:
                best_score, best_plans = score, [plan.copy()]
            elif score == best_score and len(best_plans) < 2:
                best_plans.append(plan.copy())
            return
        slot = OL_SLOTS[index]
        for choice in candidates[slot]:
            player_id = identity(choice)
            if player_id in used:
                continue
            plan[slot] = choice
            visit(index + 1, plan, used | {player_id},
                  primary + int(choice[1] == 1), rank_sum + choice[1])
        plan.pop(slot, None)

    visit(0, {}, set(), 0, 0)
    if not best_plans:
        raise RuntimeError(f"{team}: published available OL candidates cannot fill five distinct starter slots")
    if len(best_plans) > 1:
        examples = [", ".join(f"{slot.upper()}={plan[slot][0]['player_name']} "
                              f"(rank {plan[slot][1]})" for slot in OL_SLOTS)
                    for plan in best_plans]
        raise RuntimeError(f"{team}: ambiguous OL assignment; equally ranked published lineups: "
                           + " | ".join(examples))
    return best_plans[0]


def build_live_inputs(perf: pd.DataFrame, master: pd.DataFrame,
                      sources: dict[str, dict], source_audit: pd.DataFrame,
                      normalize_name: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return current team/status inputs plus automatically ordered QB/OL starters."""
    master = master.copy(deep=True)
    perf = perf.copy(deep=True)
    master["player_id"] = master["player_id"].map(text)
    perf["player_id"] = perf["player_id"].map(text)
    extra_identities = perf[~perf.player_id.isin(master.player_id)]
    if not extra_identities.empty:
        master = pd.concat([master, extra_identities[[c for c in
            ("player_id", "player_name", "team", "position", "position_group") if c in extra_identities]]],
            ignore_index=True)
    if master.player_id.duplicated().any():
        raise ValueError("Player master has duplicate canonical IDs")
    identities = master.drop_duplicates("player_id").set_index("player_id")
    by_provider: dict[str, list[str]] = {}
    by_name: dict[tuple[str, str], list[str]] = {}
    for player_id, row in identities.iterrows():
        provider_id = text(row.get("espn_id")).removesuffix(".0")
        if provider_id:
            by_provider.setdefault(provider_id, []).append(player_id)
        by_name.setdefault((text(row.get("team")).upper(), player_key(row.get("player_name"))), []).append(player_id)
    # Performance rows can retain valid IDs for players omitted from a current master.
    for row in perf.to_dict("records"):
        key = (text(row.get("team")).upper(), player_key(row.get("player_name")))
        by_name.setdefault(key, [])
        if row["player_id"] not in by_name[key]:
            by_name[key].append(row["player_id"])
    availability, qb_rows, ol_rows, errors = [], [], [], []
    source_audit["identity_issues"] = ""
    source_audit["ol_assignment_note"] = ""
    assigned_ids: dict[str, str] = {}
    preferred: dict[str, str] = {}
    for team, record in sorted(sources.items()):
        nodes = depth_nodes(record["depthcharts"])
        chart_by_id: dict[str, list[dict]] = {}
        ranks: dict[str, int] = {}
        for key, node in nodes:
            for rank, athlete in enumerate(node.get("athletes", []), 1):
                provider_id = text(athlete.get("id"))
                if not provider_id:
                    raise ValueError(f"{team}: depth athlete without provider ID")
                chart_by_id.setdefault(provider_id, []).append(athlete)
                ranks[provider_id] = min(rank, ranks.get(provider_id, 999))
        roster = [athlete for group in record["roster"].get("athletes", [])
                  for athlete in group.get("items", [])]
        if len(roster) < 45:
            raise ValueError(f"{team}: incomplete source roster ({len(roster)} players)")
        roster_by_id = {text(a.get("id")): a for a in roster}
        if len(roster_by_id) != len(roster) or "" in roster_by_id:
            raise ValueError(f"{team}: duplicate or blank provider roster IDs")
        # A depth athlete can be newly promoted before the provider's roster updates.
        for provider_id, athletes in chart_by_id.items():
            roster_by_id.setdefault(provider_id, athletes[0])
        mapped: dict[str, dict] = {}
        for provider_id, athlete in roster_by_id.items():
            canonical_id = text(athlete.get("canonical_player_id"))
            if canonical_id:
                if canonical_id not in identities.index:
                    raise ValueError(f"{team}: master fallback refers to unknown canonical player {canonical_id}")
                candidates, method = [canonical_id], "REFRESHED_MASTER_CANONICAL_ID"
            else:
                candidates, method = by_provider.get(provider_id, []), "ESPN_ID"
            if not candidates:
                candidates = by_name.get((team, player_key(athlete.get("displayName"))), [])
                # A known, different provider ID is an identity conflict, even
                # when two teammates happen to share the same name.
                candidates = [candidate for candidate in candidates
                              if not text(identities.loc[candidate].get("espn_id")).removesuffix(".0")]
                method = "UNIQUE_TEAM_NAME"
            if len(candidates) > 1:
                errors.append(f"{team}: ambiguous identity for {athlete.get('displayName')}")
                continue
            player_id = candidates[0] if candidates else ""
            if player_id and player_id in assigned_ids and assigned_ids[player_id] != team:
                errors.append(f"{player_id}: present on two current source rosters")
                continue
            if player_id:
                assigned_ids[player_id] = team
            row = {"season": SEASON, "team": team, "player_id": player_id,
                   "espn_id": provider_id, "source_player_name": text(athlete.get("displayName")),
                   "player_name": text(identities.loc[player_id, "player_name"]) if player_id in identities.index else text(athlete.get("displayName")),
                   "identity_match_method": method if player_id else "UNMATCHED",
                   "source_position": text(athlete.get("position", {}).get("abbreviation")),
                   "live_depth_rank": ranks.get(provider_id, 999),
                   "live_preferred_slot": "", "live_source_starter": 0,
                   "live_identity_issue": "",
                   "live_source_timestamp": record["depthcharts"]["timestamp"],
                   "live_fetched_at_utc": record["fetched_at_utc"],
                   **injury_state(athlete, chart_by_id.get(provider_id, []))}
            mapped[provider_id] = row
            availability.append(row)
        selected_ol = select_ol_starters(team, nodes, mapped)
        first_ol = [next((text(a.get("id")) for a in node.get("athletes", [])
                          if text(a.get("id")) in mapped
                          and not mapped[text(a.get("id"))]["live_unavailable"]), "")
                    for key, node in nodes if key in OL_SLOTS]
        ol_collision = len(set(first_ol)) != len(first_ol)
        if ol_collision:
            source_audit.loc[source_audit.team.eq(team), "ol_assignment_note"] = (
                "PROJECTED_UNIQUE_OL_ASSIGNMENT: retain available published rank-one starters, "
                "then minimize position-specific depth ranks; "
                + "; ".join(f"{slot.upper()}={row['player_name']} (source rank {rank})"
                            for slot, (row, rank) in selected_ol.items()))
        for key, node in nodes:
            eligible = [mapped[text(a.get("id"))] for a in node.get("athletes", [])
                        if text(a.get("id")) in mapped and not mapped[text(a.get("id"))]["live_unavailable"]]
            if not eligible:
                if key == "qb" or key.upper() in {"LT", "LG", "C", "RG", "RT"}:
                    errors.append(f"{team} {key}: no available player in published depth order")
                continue
            top, slot_rank = selected_ol[key] if key in OL_SLOTS else (eligible[0], None)
            if not top["player_id"]:
                issue = f"{key}: listed starter {top['source_player_name']} (ESPN {top['espn_id']}) has no unique canonical identity"
                top["live_identity_issue"] = issue
                audit_mask = source_audit.team.eq(team)
                existing = str(source_audit.loc[audit_mask, "identity_issues"].iloc[0])
                source_audit.loc[audit_mask, "identity_issues"] = " | ".join(filter(None, [existing, issue]))
                if key == "qb" or key.upper() in {"LT", "LG", "C", "RG", "RT"}:
                    errors.append(f"{team} {issue}")
                continue
            top["live_source_starter"] = 1
            slot = SLOT_KEYS.get(key, "")
            if slot and (key in OL_SLOTS or top["player_id"] not in preferred):
                preferred[top["player_id"]] = slot
                top["live_preferred_slot"] = slot
            if key == "qb":
                qb_rows.append({"season": SEASON, "team": team, "player_name": top["player_name"],
                                "name_key": normalize_name(top["player_name"]),
                                "starter_player_id": top["player_id"], "source": "espn_live_depth_available_qb"})
            if key.upper() in {"LT", "LG", "C", "RG", "RT"}:
                ol_rows.append({"season": SEASON, "team": team, "starter_slot": key.upper(),
                                "player_name": top["player_name"], "espn_athlete_id": top["espn_id"],
                                "source": ("espn_live_depth_available_ol_unique_assignment" if ol_collision
                                           else "espn_live_depth_available_ol"),
                                "source_depth_rank": slot_rank,
                                "source_injury_status": top["live_injury_status"],
                                "source_timestamp": top["live_source_timestamp"], "active": 1})
    if errors:
        raise RuntimeError("Automatic depth identity/availability errors:\n" + "\n".join(sorted(set(errors))))
    live = pd.DataFrame(availability)
    matched = live[live.player_id.ne("")].copy()
    if matched.player_id.duplicated().any():
        duplicates = matched[matched.player_id.duplicated(keep=False)][
            ["team", "player_id", "espn_id", "source_player_name", "identity_match_method"]]
        raise ValueError("Duplicate canonical identities in automatic availability:\n" + duplicates.to_string(index=False))
    qb = pd.DataFrame(qb_rows)
    ol = pd.DataFrame(ol_rows)
    if len(qb) != len(sources) or qb.team.nunique() != len(sources):
        raise ValueError("Expected one automatic QB per team")
    # Refresh team assignments for existing talent rows; no talent re-estimation.
    locations = matched.set_index("player_id")["team"].to_dict()
    perf["team"] = perf.player_id.map(locations).fillna(perf["team"])
    master["team"] = master.player_id.map(locations).fillna(master["team"])
    # Add current, identified players without historical inputs at replacement level.
    missing = matched[~matched.player_id.isin(perf.player_id)]
    replacements = []
    for player in missing.to_dict("records"):
        identity = identities.loc[player["player_id"]]
        position = text(identity.get("position"))
        peers = perf[perf["position"].astype(str).eq(position)]
        baseline = pd.to_numeric(peers.get("replacement_baseline", pd.Series(dtype=float)), errors="coerce").median()
        if pd.isna(baseline):
            baseline = 30.0
        replacements.append({"player_id": player["player_id"], "player_name": player["player_name"],
                             "team": player["team"], "position": position,
                             "position_group": text(identity.get("position_group")),
                             "performance_input_score": baseline, "replacement_baseline": baseline,
                             "position_confidence_score": 0.0, "usable_performance_grade": 0,
                             "history_available": 0, "durability_score": 0.0, "qb_starter_pool": 0})
    if replacements:
        perf = pd.concat([perf, pd.DataFrame(replacements)], ignore_index=True)
    merged = master.merge(matched.drop(columns=["team", "player_name", "espn_id"], errors="ignore"),
                          on="player_id", how="left", validate="one_to_one")
    merged["original_roster_status"] = merged["status"].map(text)
    matched_ids = set(matched.player_id)
    for idx, row in merged.iterrows():
        if row["player_id"] not in matched_ids:
            merged.at[idx, "status"] = "INA"
        elif int(row["live_unavailable"]):
            merged.at[idx, "status"] = "OUT"
        else:
            merged.at[idx, "status"] = "ACT"
    return perf, merged, qb, ol, live


def persist_starters(connection: Any, qb: pd.DataFrame, ol: pd.DataFrame) -> None:
    """Replace old season-long starter selections with the current source choices."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    qb_columns = {row[1] for row in connection.execute('PRAGMA table_info("nfl_projected_qb_starters_2026")')}
    if "player_id" not in qb_columns:
        connection.execute('ALTER TABLE "nfl_projected_qb_starters_2026" ADD COLUMN player_id TEXT')
    for row in qb.to_dict("records"):
        connection.execute('''INSERT INTO nfl_projected_qb_starters_2026
            (season,team,player_name,player_id,source,active,updated_at) VALUES (?,?,?,?,?,1,?)
            ON CONFLICT(season,team) DO UPDATE SET player_name=excluded.player_name,
            player_id=excluded.player_id,source=excluded.source,active=1,updated_at=excluded.updated_at''',
            (SEASON,row["team"],row["player_name"],row["starter_player_id"],row["source"],now))
    for row in ol.to_dict("records"):
        connection.execute('''INSERT INTO nfl_projected_ol_starters_2026
            (season,team,starter_slot,player_name,espn_athlete_id,source,source_depth_rank,
             source_injury_status,source_timestamp,active,updated_at) VALUES (?,?,?,?,?,?,?,?,?,1,?)
            ON CONFLICT(season,team,starter_slot) DO UPDATE SET player_name=excluded.player_name,
            espn_athlete_id=excluded.espn_athlete_id,source=excluded.source,
            source_depth_rank=excluded.source_depth_rank,source_injury_status=excluded.source_injury_status,
            source_timestamp=excluded.source_timestamp,active=1,updated_at=excluded.updated_at''',
            (SEASON,row["team"],row["starter_slot"],row["player_name"],row["espn_athlete_id"],row["source"],
             row["source_depth_rank"],row["source_injury_status"],row["source_timestamp"],now))


def persist_sources(connection: Any, root: Path, live: pd.DataFrame, audit: pd.DataFrame,
                    write_csv: bool = True) -> None:
    for table, frame in ((AVAILABILITY_TABLE, live), (SOURCE_TABLE, audit)):
        frame.to_sql(table, connection, if_exists="replace", index=False)
        if write_csv:
            (root / "outputs").mkdir(parents=True, exist_ok=True)
            frame.to_csv(root / "outputs" / f"{table}.csv", index=False, encoding="utf-8-sig")
