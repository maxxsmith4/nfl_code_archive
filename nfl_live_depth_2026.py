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
import logging
import re
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEASON = 2026
VERSION = "v1_7_corroborated_owner_full_conflict_audit"
QUARANTINED_QB_OL_COLUMN = "quarantined_qb_ol_provider_ids_json"
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


def reconcile_depth_only_duplicates(master: pd.DataFrame, sources: dict[str, dict],
                                    source_audit: pd.DataFrame,
                                    name_candidates: dict[tuple[str, str], list[str]] | None = None) -> dict[str, set[str]]:
    """Audit all cross-team provider conflicts before returning any live inputs.

    A unique direct-provider roster plus a fresh exact-ID canonical owner can
    reject another team's contradicted depth-only entry. Non-QB/OL starter
    removals are provisional and withhold that team's adjusted projections.
    Disputed QB/OL starters, conflicting owners, stale evidence and ambiguous
    identities still block. Unresolved unused QB/OL backups retain v1_6 guards.
    Raw observations, talent grades and source depth ranks are never rewritten.
    """
    excluded = {team: set() for team in sources}
    if ("team" not in source_audit or source_audit.team.duplicated().any()
            or set(source_audit.team) != set(sources)):
        raise ValueError("Depth reconciliation requires one source audit row per team")
    source_audit[QUARANTINED_QB_OL_COLUMN] = "[]"
    source_audit["roster_assignment_note"] = ""
    if "identity_issues" not in source_audit:
        source_audit["identity_issues"] = ""
    source_audit.attrs["identity_conflict_audit"] = {
        "scan_complete": False, "scan_scope": "cross_team_provider_memberships",
        "conflicts": [], "blocking_errors": [],
    }
    rosters, charts, memberships = {}, {}, {}
    for team, record in sorted(sources.items()):
        athletes = [athlete for group in record["roster"].get("athletes", [])
                    for athlete in group.get("items", [])]
        roster = {text(athlete.get("id")): athlete for athlete in athletes}
        if len(roster) != len(athletes) or "" in roster:
            raise ValueError(f"{team}: duplicate or blank provider roster IDs")
        rosters[team] = roster
        charts[team] = depth_nodes(record["depthcharts"])
        ids = set(roster)
        for _, node in charts[team]:
            ids.update(text(athlete.get("id")) for athlete in node.get("athletes", []))
        for provider_id in ids - {""}:
            memberships.setdefault(provider_id, []).append(team)

    class ConflictRejected(RuntimeError):
        pass

    decisions: list[dict[str, Any]] = []
    blocking: list[str] = []
    for provider_id, teams in sorted(memberships.items()):
        if len(teams) < 2:
            continue
        decision: dict[str, Any] = {"espn_id": provider_id, "teams": list(teams),
                                    "decision": "UNCLASSIFIED", "observations": []}
        try:
            owners = [team for team in teams if provider_id in rosters[team]]
            evidence = []
            for team in teams:
                entries = [f"{slot.upper()}:{rank}" for slot, node in charts[team]
                           for rank, athlete in enumerate(node.get("athletes", []), 1)
                           if text(athlete.get("id")) == provider_id]
                status = injury_state(rosters[team].get(provider_id, {}), [])['live_injury_status']
                evidence.append(f"{team} roster={'yes' if team in owners else 'no'} "
                                f"status={status if team in owners else 'not-listed'} "
                                f"depth={','.join(entries) or 'none'}")
            matches = master[master.get("espn_id", pd.Series("", index=master.index)).map(
                lambda value: text(value).removesuffix(".0")).eq(provider_id)]
            player_id = text(matches.iloc[0].get("player_id")) if len(matches) == 1 else "unresolved"
            source_names = [text(athlete.get("displayName")) for team in teams
                            for athlete in [rosters[team].get(provider_id, {}),
                                *(athlete for _, node in charts[team] for athlete in node.get("athletes", [])
                                  if text(athlete.get("id")) == provider_id)] if text(athlete.get("displayName"))]
            name = text(matches.iloc[0].get("player_name")) if len(matches) == 1 else (source_names[0] if source_names else "unresolved player")
            prefix = f"{name} (GSIS {player_id}, ESPN {provider_id}): cross-team source conflict; " + " | ".join(evidence)

            def reject(reason: str) -> None:
                raise ConflictRejected(prefix + "; " + reason)

            decision.update({"roster_owners": owners, "canonical_player_id": player_id,
                             "player_name": name, "evidence_text": evidence})
            for observed_team in teams:
                observed_record = sources[observed_team]
                decision["observations"].append({
                    "team": observed_team,
                    "roster_provider": observed_record.get("roster_provider", "ESPN"),
                    "fetched_at_utc": text(observed_record.get("fetched_at_utc")),
                    "roster_timestamp": text(observed_record["roster"].get("timestamp")),
                    "depth_timestamp": text(observed_record["depthcharts"].get("timestamp")),
                    "roster_athlete": rosters[observed_team].get(provider_id),
                    "depth_entries": [{"slot": slot, "rank": rank, "athlete": athlete}
                        for slot, node in charts[observed_team]
                        for rank, athlete in enumerate(node.get("athletes", []), 1)
                        if text(athlete.get("id")) == provider_id],
                })
            if len(owners) != 1:
                reject("requires exactly one actual roster owner; no team selected")
            owner = owners[0]
            record = sources[owner]
            if any(marker != "ESPN" for marker in (
                    record.get("roster_provider", "ESPN"), record["roster"].get("roster_provider", "ESPN"))):
                reject("master-derived fallback roster is not independent ownership evidence")
            if len(matches) == 0:
                # Missing exact IDs do not justify assigning a player to either
                # team. A nonstarting backup can be retained for review rather than
                # aborting the league refresh. Keep it in the candidate order with
                # NO canonical identity: if it becomes the next available QB/OL or
                # is needed by the five-distinct-OL solver, the existing selection
                # guard must fail. Never drop it to let a later player pass instead.
                critical_backups = []
                for team in teams:
                    for slot, node in charts[team]:
                        if slot == "qb" or slot in OL_SLOTS:
                            ranks = [rank for rank, athlete in enumerate(node.get("athletes", []), 1)
                                     if text(athlete.get("id")) == provider_id]
                            if 1 in ranks:
                                reject(f"{team} {slot.upper()} conflict affects a published rank-one QB/OL starter; "
                                       "provider identity is unresolved")
                            critical_backups.extend(f"{team}:{slot.upper()}:{rank}" for rank in ranks)
                    try:
                        validate_source_age(sources[team], team)
                    except (ValueError, TypeError, SourceUnavailable) as exc:
                        reject(f"source freshness is not verified: {exc}")
                if len({player_key(value) for value in source_names}) != 1:
                    reject("conflicting source names for one provider ID")
                identities = master.copy()
                identities["player_id"] = identities.player_id.map(text)
                if identities.player_id.duplicated().any():
                    reject("canonical master identity is duplicated")
                identities = identities.set_index("player_id")
                names = name_candidates
                if names is None:
                    names = {}
                    for candidate, row in identities.iterrows():
                        names.setdefault((text(row.get("team")).upper(), player_key(row.get("player_name"))), []).append(candidate)
                linked = {}
                for team in teams:
                    athlete = rosters[team].get(provider_id) or next(
                        athlete for _, node in charts[team] for athlete in node.get("athletes", [])
                        if text(athlete.get("id")) == provider_id)
                    canonical = text(athlete.get("canonical_player_id"))
                    candidates = [canonical] if canonical else names.get((team, player_key(athlete.get("displayName"))), [])
                    candidates = sorted(set(candidates))
                    if any(candidate not in identities.index or not candidate for candidate in candidates):
                        reject("team/name fallback refers to an unknown canonical identity")
                    if any(text(identities.loc[candidate].get("espn_id")).removesuffix(".0") for candidate in candidates):
                        reject("team/name match has a conflicting known provider ID")
                    if len(candidates) > 1:
                        reject(f"{team}: ambiguous team/name canonical identity")
                    if candidates:
                        linked[team] = candidates[0]
                if len(linked) > 1:
                    reject("provider would map to canonical identities on multiple teams: " + str(linked))
                issue = ("UNRESOLVED_CROSS_TEAM_PROVIDER: " + prefix
                         + "; no exact canonical provider ID; team/name matches=" + json.dumps(linked, sort_keys=True)
                         + "; source observations retained; ownership not reassigned; adjusted lines withheld")
                if critical_backups:
                    issue += ("; QB_OL_BACKUP_REVIEW=" + ",".join(critical_backups)
                              + "; canonical mapping withheld on all observed teams; "
                              "published order retained; selected QB/OL conflict remains fatal")
                for team in teams:
                    mask = source_audit.team.eq(team)
                    for column in ("identity_issues", "roster_assignment_note"):
                        source_audit.loc[mask, column] = source_audit.loc[mask, column].map(
                            lambda prior: " || ".join(filter(None, [text(prior), issue])))
                    if critical_backups:
                        retained = set(json.loads(source_audit.loc[mask, QUARANTINED_QB_OL_COLUMN].iloc[0]))
                        retained.add(provider_id)
                        source_audit.loc[mask, QUARANTINED_QB_OL_COLUMN] = json.dumps(sorted(retained))
                decision.update({"decision": "REVIEW_UNRESOLVED_PROVIDER",
                                 "review_teams": list(teams), "detail": issue})
                if critical_backups:
                    logging.getLogger(__name__).warning(
                        "[LIVE_DEPTH] QB_OL_BACKUP_REVIEW: %s; canonical mapping withheld; "
                        "actual starter validation still required", prefix)
                continue
            if len(matches) != 1 or not player_id or player_id == "unresolved":
                reject("provider ID lacks one unique canonical master identity")
            identity = matches.iloc[0]
            expected_name = player_key(identity.get("player_name"))
            if (not expected_name or not source_names
                    or any(player_key(value) != expected_name for value in source_names)):
                reject("conflicting source names for one exact provider ID")
            if any(text(entry["athlete"].get("canonical_player_id")) not in {"", player_id}
                   for obs in decision["observations"] for entry in obs["depth_entries"]):
                reject("depth entry has a conflicting explicit canonical ID")
            if master.player_id.map(text).eq(player_id).sum() != 1:
                reject("canonical master identity is duplicated")
            master_team = TEAM_ALIASES.get(text(identity.get("team")).upper(), text(identity.get("team")).upper())
            if master_team != owner:
                reject(f"canonical master team {master_team or 'missing'} does not match roster owner {owner}")
            master_season = pd.to_numeric(identity.get("season"), errors="coerce")
            if pd.isna(master_season) or master_season != SEASON:
                reject("canonical master season is missing or incorrect")
            if text(identity.get("roster_refresh_status")) not in {"LIVE", "REUSED_RECENT_CACHE", "EXISTING_RAW_ROSTER"}:
                reject("canonical roster source provenance is unverified")
            try:
                master_stamp = utc(identity.get("roster_source_as_of_utc"))
                now = pd.Timestamp.now(tz="UTC")
                if now - master_stamp > pd.Timedelta(hours=24) or master_stamp - now > pd.Timedelta(minutes=5):
                    reject("canonical roster source is stale or future dated")
                for team in teams:
                    validate_source_age(sources[team], team)
            except (ValueError, TypeError, SourceUnavailable) as exc:
                reject(f"ownership source freshness is not verified: {exc}")
            canonical = text(rosters[owner][provider_id].get("canonical_player_id"))
            if canonical and canonical != player_id:
                reject("actual roster canonical ID conflicts with the master")
            stale_starter_roles: dict[str, list[str]] = {}
            for team in teams:
                if team == owner:
                    continue
                # Merge all chart injury records exactly as the normal mapper does;
                # raw rank two can be the next starter when rank one is unavailable.
                chart_by_id = {}
                for _, node in charts[team]:
                    for athlete in node.get("athletes", []):
                        chart_by_id.setdefault(text(athlete.get("id")), []).append(athlete)
                for slot, node in charts[team]:
                    athletes = node.get("athletes", [])
                    ranks = [rank for rank, athlete in enumerate(athletes, 1)
                             if text(athlete.get("id")) == provider_id]
                    if not ranks:
                        continue
                    if slot == "qb" or slot in OL_SLOTS:
                        reject(f"{team} {slot.upper()} conflict affects QB/OL depth; no automatic reassignment")
                    first_available = next((text(athlete.get("id")) for athlete in athletes
                        if not injury_state(rosters[team].get(text(athlete.get("id")), athlete),
                                            chart_by_id[text(athlete.get("id"))])["live_unavailable"]), "")
                    if 1 in ranks or first_available == provider_id:
                        # Ownership is proved, but the replacement starter is NOT.
                        # Only remove the impossible cross-team candidate. Ordinary
                        # published order is retained, and adjusted lines are held.
                        stale_starter_roles.setdefault(team, []).append(slot.upper())
            note = (f"CORROBORATED_ROSTER_OWNER: {name} (GSIS {player_id}, ESPN {provider_id}); "
                    f"kept={owner}; excluded_depth_only={','.join(team for team in teams if team != owner)}; "
                    f"basis=unique ESPN roster plus fresh exact-ID canonical master; "
                    f"master_source_as_of_utc={master_stamp.isoformat()}; " + " | ".join(evidence))
            for team in teams:
                if team != owner:
                    excluded[team].add(provider_id)
                mask = source_audit.team.eq(team)
                source_audit.loc[mask, "roster_assignment_note"] = source_audit.loc[mask, "roster_assignment_note"].map(
                    lambda prior: " || ".join(filter(None, [text(prior), note])))
            decision.update({"decision": "RESOLVED_CORROBORATED_OWNER",
                             "authoritative_owner": owner,
                             "canonical_master_source_as_of_utc": master_stamp.isoformat(),
                             "excluded_depth_teams": [team for team in teams if team != owner],
                             "detail": note})
            if stale_starter_roles:
                decision.update({"decision": "REVIEW_CORROBORATED_STALE_STARTER",
                                 "review_teams": sorted(stale_starter_roles),
                                 "affected_slots": stale_starter_roles})
                for team, slots in stale_starter_roles.items():
                    issue = ("STALE_STARTER_DEPTH_OWNER_CONFLICT: " + prefix
                             + f"; authoritative_owner={owner}; affected_slots={','.join(slots)}"
                             + "; contradicted depth-only candidate excluded; remaining depth order provisional;"
                             + " roster-adjusted projection withheld")
                    mask = source_audit.team.eq(team)
                    source_audit.loc[mask, "identity_issues"] = source_audit.loc[mask, "identity_issues"].map(
                        lambda prior: " || ".join(filter(None, [text(prior), issue])))
                    logging.getLogger(__name__).warning("[LIVE_DEPTH] STALE_STARTER_REVIEW: %s", issue)
        except ConflictRejected as exc:
            message = str(exc)
            decision.update({"decision": "BLOCKED", "detail": message})
            blocking.append(message)
            for team in teams:
                mask = source_audit.team.eq(team)
                issue = "BLOCKED_CROSS_TEAM_IDENTITY: " + message
                source_audit.loc[mask, "identity_issues"] = source_audit.loc[mask, "identity_issues"].map(
                    lambda prior: " || ".join(filter(None, [text(prior), issue])))
        finally:
            decisions.append(decision)
            # Preserve progress even when an unexpected schema/programming error
            # interrupts the scan. scan_complete remains false in that case.
            source_audit.attrs["identity_conflict_audit"]["conflicts"] = decisions

    source_audit.attrs["identity_conflict_audit"].update({
        "scan_complete": True, "conflicts": decisions, "blocking_errors": blocking,
    })
    logging.getLogger(__name__).warning(
        "[LIVE_DEPTH] IDENTITY_SCAN: cross_team_conflicts=%d reviewed=%d resolved=%d blocked=%d",
        len(decisions), sum(d["decision"].startswith("REVIEW_") for d in decisions),
        sum(d["decision"].startswith("RESOLVED_") for d in decisions), len(blocking))
    if blocking:
        raise RuntimeError("Automatic depth identity/availability errors (all cross-team conflicts scanned):\n"
                           + "\n".join(blocking))
    return excluded


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
    excluded_depth_ids = reconcile_depth_only_duplicates(master, sources, source_audit, by_name)
    quarantined_provider_ids = {
        row["team"]: set(json.loads(row[QUARANTINED_QB_OL_COLUMN]))
        for row in source_audit.to_dict("records")
    }
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
            if provider_id not in excluded_depth_ids[team]:
                roster_by_id.setdefault(provider_id, athletes[0])
        mapped: dict[str, dict] = {}
        for provider_id, athlete in roster_by_id.items():
            quarantined = provider_id in quarantined_provider_ids[team]
            canonical_id = text(athlete.get("canonical_player_id"))
            if quarantined:
                # Retain the unresolved candidate and real source availability;
                # do NOT create an ID, relocate talent, or skip it in selection.
                candidates, method = [], "UNRESOLVED_CROSS_TEAM_REVIEW"
            else:
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
                errors.append(f"{text(identities.loc[player_id].get('player_name'))} ({player_id}): "
                              f"present on source teams {assigned_ids[player_id]} and {team}; ESPN {provider_id}")
                continue
            if player_id:
                assigned_ids[player_id] = team
            row = {"season": SEASON, "team": team, "player_id": player_id,
                   "espn_id": provider_id, "source_player_name": text(athlete.get("displayName")),
                   "player_name": text(identities.loc[player_id, "player_name"]) if player_id in identities.index else text(athlete.get("displayName")),
                   "identity_match_method": method if player_id or quarantined else "UNMATCHED",
                   "source_position": text(athlete.get("position", {}).get("abbreviation")),
                   "live_depth_rank": ranks.get(provider_id, 999),
                   "live_preferred_slot": "", "live_source_starter": 0,
                   "live_identity_issue": ("UNRESOLVED_CROSS_TEAM_PROVIDER: " + provider_id
                                           + "; canonical mapping withheld; source candidate retained"
                                           if quarantined else ""),
                   "live_source_timestamp": record["depthcharts"]["timestamp"],
                   "live_fetched_at_utc": record["fetched_at_utc"],
                   **injury_state(athlete, chart_by_id.get(provider_id, []))}
            mapped[provider_id] = row
            availability.append(row)
        try:
            selected_ol = select_ol_starters(team, nodes, mapped)
        except RuntimeError as exc:
            errors.append(f"{team}: {exc}")
            source_audit.attrs.setdefault("starter_validation_errors", []).append(
                {"team": team, "stage": "OL_ASSIGNMENT", "error": str(exc)})
            selected_ol = {}
        first_ol = [next((text(a.get("id")) for a in node.get("athletes", [])
                          if text(a.get("id")) in mapped
                          and not mapped[text(a.get("id"))]["live_unavailable"]), "")
                    for key, node in nodes if key in OL_SLOTS]
        ol_collision = len(set(first_ol)) != len(first_ol)
        if ol_collision and selected_ol:
            source_audit.loc[source_audit.team.eq(team), "ol_assignment_note"] = (
                "PROJECTED_UNIQUE_OL_ASSIGNMENT: retain available published rank-one starters, "
                "then minimize position-specific depth ranks; "
                + "; ".join(f"{slot.upper()}={row['player_name']} (source rank {rank})"
                            for slot, (row, rank) in selected_ol.items()))
        for key, node in nodes:
            if key in OL_SLOTS and key not in selected_ol:
                continue  # This team's assignment error is already recorded.
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
        source_audit.attrs["input_validation_errors"] = sorted(set(errors))
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



def write_identity_report(root: Path, sources: dict[str, dict] | None,
                          source_audit: pd.DataFrame | None, *,
                          status: str, attempted_at_utc: str, error: str = "") -> dict[str, Any]:
    """Persist source evidence for successful and failed attempts; no DB writes.

    The canonical JSON/TXT refer to one report_id; timestamped JSON preserves
    prior attempts. Atomic replacements avoid partial JSON. Raw caches are not
    modified and missing observations are not presented as fresh evidence.
    """
    root = Path(root)
    sources = sources or {}
    audit = source_audit if source_audit is not None else pd.DataFrame()
    scan = audit.attrs.get("identity_conflict_audit", {})
    report_id = uuid.uuid4().hex
    team_rows = []
    for row in audit.to_dict("records"):
        team = text(row.get("team"))
        record = sources.get(team)
        team_rows.append({
            "team": team, "source_state": text(row.get("source_state")),
            "identity_issues": text(row.get("identity_issues")),
            "roster_assignment_note": text(row.get("roster_assignment_note")),
            "source_sha256": text(row.get("source_sha256")),
            "observed_payload_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                if record is not None else "",
            "fetched_at_utc": text(row.get("fetched_at_utc")),
            "roster_source_timestamp": text(row.get("roster_source_timestamp")),
            "depth_source_timestamp": text(row.get("depth_source_timestamp")),
            "roster_url": text(row.get("roster_url")), "depth_url": text(row.get("depth_url")),
            "raw_cache_path": str(root / "data" / "raw" / "nfl_live_depth_2026" / f"{team}.json"),
        })
    review = sorted(row["team"] for row in team_rows if row["identity_issues"])
    payload = {"report_id": report_id, "version": VERSION, "season": SEASON,
               "attempted_at_utc": attempted_at_utc,
               "written_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "status": status, "error": error,
               "source_team_count": len(sources), "scan_complete": bool(scan.get("scan_complete", False)),
               "scan_scope": scan.get("scan_scope", "not_reached"),
               "conflicts": scan.get("conflicts", []), "blocking_errors": scan.get("blocking_errors", []),
               "input_validation_errors": audit.attrs.get("input_validation_errors", []),
               "starter_validation_errors": audit.attrs.get("starter_validation_errors", []),
               "review_teams": review, "team_audit": team_rows,
               "notice": "Pipeline completion is not full roster certification. Team identity_issues withhold"
                         " supplemental roster-adjusted lines. No new player ID, grade or fitted coefficient is created."}
    output = root / "outputs"
    history = output / "audits" / "depth_identity"
    history.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    def atomic(path: Path, content: str) -> None:
        temporary = path.with_name(path.name + "." + report_id + ".tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
    atomic(history / f"{stamp}_{report_id}.json", encoded)
    atomic(output / "nfl_depth_identity_report_2026.json", encoded)
    lines = ["NFL DEPTH IDENTITY REPORT", f"Report ID: {report_id}", f"Attempt: {attempted_at_utc}",
             f"Status: {status} | Source teams: {len(sources)} | Full cross-team scan: {payload['scan_complete']}",
             f"Conflicts: {len(payload['conflicts'])} | Blocking: {len(payload['blocking_errors'])}",
             "Identity-review teams: " + (", ".join(review) or "none observed"), ""]
    for conflict in payload["conflicts"]:
        lines += [f"[{conflict.get('decision')}] {conflict.get('player_name', 'unresolved')} "
                  f"ESPN={conflict.get('espn_id')} GSIS={conflict.get('canonical_player_id', '')}",
                  *["  " + item for item in conflict.get("evidence_text", [])],
                  "  " + text(conflict.get("detail")), ""]
    for item in payload["input_validation_errors"]:
        lines += ["[INPUT_VALIDATION_ERROR] " + item]
    for row in team_rows:
        if row["identity_issues"]:
            lines += ["[TEAM_REVIEW] " + row["team"] + ": " + row["identity_issues"]]
    if error:
        lines += ["[ATTEMPT_ERROR] " + error]
    lines += ["", payload["notice"]]
    atomic(output / "nfl_depth_identity_report_2026.txt", "\n".join(lines) + "\n")
    return payload


def _run_self_tests() -> dict[str, Any]:
    """Offline synthetic regression tests. No network, database, or cache writes."""
    import copy

    results: list[dict[str, Any]] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def fixture(teams: tuple[str, ...] = ("ARI", "NE"), *, conflict: bool = True):
        stamp = pd.Timestamp.now(tz="UTC").isoformat()
        sources, master_rows, perf_rows, audits = {}, [], [], []
        for t, team in enumerate(teams):
            roster = []
            for i in range(50):
                provider = str(90000000 + 1000 * t + i)
                pid, name = f"TEST_{team}_{i}", f"Synthetic {team} Player {i}"
                position = "QB" if i == 0 else "OL" if i <= 5 else "WR"
                athlete = {"id": provider, "displayName": name,
                           "position": {"abbreviation": position},
                           "status": {"name": "Active"}, "injuries": []}
                roster.append(athlete)
                master_rows.append({"season": SEASON, "team": team, "player_id": pid,
                                    "espn_id": provider, "player_name": name, "position": position,
                                    "position_group": position, "status": "ACT",
                                    "roster_source_as_of_utc": stamp, "roster_refresh_status": "LIVE"})
                perf_rows.append({"player_id": pid, "player_name": name, "team": team,
                                  "position": position, "position_group": position,
                                  "performance_input_score": 50.0, "replacement_baseline": 30.0,
                                  "position_confidence_score": 1.0, "usable_performance_grade": 1,
                                  "history_available": 1, "durability_score": 1.0, "qb_starter_pool": 0})
            positions = {slot: {"athletes": [copy.deepcopy(roster[i])]}
                         for i, slot in enumerate(("qb", *OL_SLOTS))}
            envelope = {"status": "success", "season": {"year": SEASON},
                        "team": {"abbreviation": team}, "timestamp": stamp}
            sources[team] = {"team": team, "fetched_at_utc": stamp, "roster_provider": "ESPN",
                             "roster": {**copy.deepcopy(envelope), "athletes": [{"items": roster}]},
                             "depthcharts": {**copy.deepcopy(envelope), "depthchart": [{"positions": positions}]}}
            audits.append({"team": team, "season": SEASON, "identity_issues": ""})
        if conflict:
            # Mirrors the log's source structure, not a claim about live rosters.
            disputed = {"id": "3055886", "displayName": "Matt Pryor",
                        "position": {"abbreviation": "G"},
                        "status": {"name": "Practice Squad"}, "injuries": []}
            sources["ARI"]["roster"]["athletes"][0]["items"].append(disputed)
            backup = copy.deepcopy(disputed)
            backup["status"] = {"name": "Active"}
            sources["NE"]["depthcharts"]["depthchart"][0]["positions"]["lg"]["athletes"].append(backup)
        return pd.DataFrame(perf_rows), pd.DataFrame(master_rows), sources, pd.DataFrame(audits)

    def positions(data, team="NE"):
        return data[2][team]["depthcharts"]["depthchart"][0]["positions"]

    def roster(data, team="NE"):
        return data[2][team]["roster"]["athletes"][0]["items"]

    def run(data):
        return build_live_inputs(data[0], data[1], data[2], data[3], player_key)

    def expect_failure(data, text_part: str):
        try:
            run(data)
        except (RuntimeError, ValueError) as exc:
            require(text_part in str(exc), f"Wrong failure: {exc}; expected {text_part}")
        else:
            raise AssertionError("Unsafe input was accepted")

    def check(name, function):
        try:
            function()
            results.append({"name": name, "status": "PASS"})
        except Exception as exc:
            results.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    def nonstarter():
        data = fixture()
        untouched = copy.deepcopy(data[2])
        before_perf = data[0].copy(deep=True)
        output = run(data)
        pd.testing.assert_frame_equal(output[0], before_perf)
        require(data[2] == untouched, "Raw source payloads changed")
        live = output[4]
        rows = live[live.espn_id.eq("3055886")]
        require(len(rows) == 2, "Both source observations must remain")
        require(rows.player_id.eq("").all(), "An identity was fabricated")
        require(rows.live_source_starter.eq(0).all(), "Disputed player became a starter")
        require(rows.set_index("team").loc["ARI", "live_unavailable"] == 1, "Practice squad status lost")
        require(rows.set_index("team").loc["NE", "live_unavailable"] == 0, "Source availability was rewritten")
        require(data[3].identity_issues.str.contains("QB_OL_BACKUP_REVIEW").all(), "Review flags were not preserved")
        require(len(output[2]) == 2 and len(output[3]) == 10, "Complete starter selection not preserved")

    def full_league():
        teams = ("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
                 "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
                 "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS")
        data = fixture(teams)
        output = run(data)
        require(len(output[2]) == 32 and len(output[3]) == 160, "32 QB/160 OL check failed")
        require(output[3].groupby("team").espn_athlete_id.nunique().eq(5).all(), "Duplicate OL starters")
        require(set(data[3].loc[data[3].identity_issues.ne(""), "team"]) == {"ARI", "NE"}, "Unrelated teams flagged")
        clean = fixture(teams, conflict=False)
        clean_output = run(clean)
        pd.testing.assert_frame_equal(output[2], clean_output[2])
        # Timestamp differences between fixtures are not starter differences.
        cols = [col for col in output[3] if col != "source_timestamp"]
        pd.testing.assert_frame_equal(output[3][cols], clean_output[3][cols])

    def primary():
        data = fixture()
        positions(data)["lg"]["athletes"].reverse()
        expect_failure(data, "published rank-one QB/OL starter")

    def first_available():
        data = fixture()
        roster(data)[2]["status"] = {"name": "Out"}
        # Append a valid third choice. The disputed second choice must NOT be skipped.
        positions(data)["lg"]["athletes"].append(copy.deepcopy(roster(data)[6]))
        expect_failure(data, "listed starter Matt Pryor")

    def dated_injury():
        data = fixture()
        positions(data)["lg"]["athletes"][0]["injuries"] = [{"status": "Out", "date": pd.Timestamp.now(tz="UTC").isoformat()}]
        expect_failure(data, "listed starter Matt Pryor")

    def selected_by_ol_solver():
        data = fixture()
        # The same available player is first at LT and LG. Only the unresolved
        # backup can complete the unique OL. It must remain a fatal selection.
        positions(data)["lg"]["athletes"][0] = copy.deepcopy(positions(data)["lt"]["athletes"][0])
        expect_failure(data, "listed starter Matt Pryor")

    def tied_ol_solver():
        data = fixture()
        first = copy.deepcopy(positions(data)["lt"]["athletes"][0])
        disputed = copy.deepcopy(positions(data)["lg"]["athletes"][1])
        positions(data)["lt"]["athletes"] = [first, disputed]
        positions(data)["lg"]["athletes"] = [copy.deepcopy(first), copy.deepcopy(disputed)]
        expect_failure(data, "ambiguous OL assignment")

    def qb_backup(start: bool = False):
        data = fixture()
        disputed = positions(data)["lg"]["athletes"].pop()
        positions(data)["qb"]["athletes"].append(disputed)
        if start:
            roster(data)[0]["status"] = {"name": "Out"}
            expect_failure(data, "listed starter Matt Pryor")
        else:
            output = run(data)
            require(len(output[2]) == 2, "Backup QB review lost a valid starter")

    def change_time(kind):
        data = fixture()
        hours = -25 if kind == "stale" else 1
        data[2]["NE"]["depthcharts"]["timestamp"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours)).isoformat()
        expect_failure(data, "freshness is not verified")

    def roster_owner_error(count):
        data = fixture()
        if count == 2:
            roster(data).append(copy.deepcopy(positions(data)["lg"]["athletes"][-1]))
        else:
            roster(data,"ARI").pop()
            positions(data,"ARI")["lg"]["athletes"].append(copy.deepcopy(positions(data)["lg"]["athletes"][-1]))
        expect_failure(data, "exactly one actual roster owner")

    def fallback_owner():
        data = fixture()
        data[2]["ARI"]["roster_provider"] = "REFRESHED_NFLVERSE_PLAYER_MASTER"
        expect_failure(data, "not independent ownership evidence")

    def different_names():
        data = fixture()
        positions(data)["lg"]["athletes"][-1]["displayName"] = "Different Player"
        expect_failure(data, "conflicting source names")

    def canonical_link(team="NE", known_id="", ambiguous=False):
        data = fixture()
        rows = []
        for n in range(2 if ambiguous else 1):
            row = data[1].iloc[0].to_dict()
            row.update(player_id=f"TEST_DISPUTED_{n}", player_name="Matt Pryor", team=team,
                       espn_id=known_id, position="OL", position_group="OL")
            rows.append(row)
        data = (data[0], pd.concat([data[1], pd.DataFrame(rows)], ignore_index=True), data[2], data[3])
        if ambiguous:
            expect_failure(data, "ambiguous team/name canonical identity")
        elif known_id:
            expect_failure(data, "conflicting known provider ID")
        else:
            output = run(data)
            require(output[4].loc[output[4].espn_id.eq("3055886"), "player_id"].eq("").all(), "Name fallback bypassed quarantine")
            require(output[1].loc[output[1].player_id.eq("TEST_DISPUTED_0"), "status"].eq("INA").all(), "Unresolved talent assigned as active")

    def exact_id_conflict():
        data = fixture()
        row = data[1].iloc[0].to_dict()
        row.update(player_id="TEST_DISPUTED", player_name="Matt Pryor", espn_id="3055886", team="ARI")
        data = (data[0], pd.concat([data[1], pd.DataFrame([row])], ignore_index=True), data[2], data[3])
        expect_failure(data, "conflict affects QB/OL depth; no automatic reassignment")

    def duplicate_master():
        data = fixture()
        data = (data[0], pd.concat([data[1],data[1].iloc[[0]]],ignore_index=True), data[2], data[3])
        expect_failure(data, "duplicate canonical IDs")

    def audit_missing():
        data = fixture()
        data = (*data[:3], data[3].iloc[:1].copy())
        expect_failure(data, "one source audit row per team")

    def incomplete_ol():
        data = fixture()
        positions(data).pop("rt")
        expect_failure(data, "no available player in published OL depth order")

    def existing_unmatched_starter():
        data = fixture(conflict=False)
        positions(data)["lg"]["athletes"][0] = {"id":"88888888", "displayName":"Unknown Starter", "status":{"name":"Active"}}
        expect_failure(data, "listed starter Unknown Starter")

    def clean():
        data = fixture(conflict=False)
        output = run(data)
        require(len(output[2]) == 2 and len(output[3]) == 10, "Clean snapshot did not pass")
        require(data[3].identity_issues.eq("").all(), "Clean snapshot gained identity issues")

    def multiple_backups():
        data = fixture()
        disputed = positions(data)["lg"]["athletes"][-1]
        positions(data)["rg"]["athletes"].append(copy.deepcopy(disputed))
        output = run(data)
        require(len(output[3]) == 10, "Cross-listed nonstarters broke unique OL selection")
        require(all(json.loads(x)==["3055886"] for x in data[3][QUARANTINED_QB_OL_COLUMN]), "Duplicate quarantine entries")

    def duplicate_provider():
        data = fixture()
        roster(data,"ARI").append(copy.deepcopy(roster(data,"ARI")[-1]))
        expect_failure(data, "duplicate or blank provider roster IDs")

    def changed_personnel_rechecked():
        data = fixture()
        run(data)
        roster(data)[2]["status"]={"name":"Out"}
        expect_failure(data,"listed starter Matt Pryor")

    def noncritical_conflict():
        data = fixture()
        disputed = positions(data)["lg"]["athletes"].pop()
        positions(data)["wr1"]={"athletes":[copy.deepcopy(roster(data)[6]),disputed]}
        output = run(data)
        require(len(output[3]) == 10, "Existing non-QB/OL behavior regressed")
        require(data[3].identity_issues.str.contains("UNRESOLVED_CROSS_TEAM_PROVIDER").all(), "Existing conflict review lost")
        require(data[3][QUARANTINED_QB_OL_COLUMN].eq("[]").all(), "Noncritical branch unexpectedly changed")

    cases = [
        ("Nonstarter conflict preserves raw observations, availability, talent and starters", nonstarter),
        ("Synthetic 32-team integration retains 32 QB and 160 unique OL starters", full_league),
        ("Published rank-one QB/OL conflict still blocks", primary),
        ("First-available backup blocks rather than skipping to third choice", first_available),
        ("Current chart injury promotes disputed backup and still blocks", dated_injury),
        ("Unique-OL solver selecting a disputed backup still blocks", selected_by_ol_solver),
        ("Equally ranked OL assignments still block", tied_ol_solver),
        ("Unused unresolved QB backup retained for review", qb_backup),
        ("Disputed QB backup needed as starter still blocks", lambda: qb_backup(True)),
        ("Stale source still blocks", lambda: change_time("stale")),
        ("Future-dated source still blocks", lambda: change_time("future")),
        ("Two roster owners still block", lambda: roster_owner_error(2)),
        ("Zero roster owners still block", lambda: roster_owner_error(0)),
        ("Master-derived owner is not independent evidence", fallback_owner),
        ("Conflicting names for one provider still block", different_names),
        ("Nonowner name fallback cannot assign disputed talent", canonical_link),
        ("Owner name fallback cannot assign disputed talent", lambda: canonical_link("ARI")),
        ("Conflicting known provider ID still blocks", lambda: canonical_link(known_id="77777777")),
        ("Ambiguous name fallback still blocks", lambda: canonical_link(ambiguous=True)),
        ("Existing exact-ID QB/OL conflict guard is unchanged", exact_id_conflict),
        ("Duplicate canonical identities still block", duplicate_master),
        ("Incomplete source audit cannot drop review flags", audit_missing),
        ("Missing OL position still blocks", incomplete_ol),
        ("Ordinary unmatched QB/OL starter still blocks", existing_unmatched_starter),
        ("Clean data passes without identity review flags", clean),
        ("Cross-listed unused backups retain unique starter selection", multiple_backups),
        ("Duplicate provider roster IDs still block", duplicate_provider),
        ("Previously reviewed backup becomes blocking after starter injury", changed_personnel_rechecked),
        ("Existing unresolved non-QB/OL review behavior remains", noncritical_conflict),
    ]
    logger = logging.getLogger(__name__)
    prior_disabled = logger.disabled
    logger.disabled = True
    try:
        for name, function in cases:
            check(name, function)
    finally:
        logger.disabled = prior_disabled
    failed = [item for item in results if item["status"] != "PASS"]
    return {"version": VERSION, "test_scope": "offline synthetic fixtures; no live database or provider access",
            "total": len(results), "passed": len(results)-len(failed), "failed": len(failed),
            "tests": results}


def _run_reliability_tests() -> dict[str, Any]:
    """Offline regression tests for ownership reconciliation and full reporting.

    Synthetic fixtures only. Report tests use temporary directories; no network
    access, user database access, or user cache mutations occur.
    """
    import copy
    import tempfile
    results = []
    all_teams = ("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
                 "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
                 "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS")

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def fixture(teams=("ARI", "DAL", "NE", "PIT"), porter=True, pryor=True):
        stamp = pd.Timestamp.now(tz="UTC").isoformat()
        sources, perf_rows, masters, audits = {}, [], [], []
        for team_no, team in enumerate(teams):
            athletes = []
            for i in range(50):
                pid, provider = f"FIX_{team}_{i}", str(91000000 + team_no * 1000 + i)
                position = "QB" if i == 0 else "OL" if i <= 5 else "CB" if i in (6,7,8) else "WR"
                name = f"Fixture {team} Player {i}"
                athlete = {"id": provider, "displayName": name, "position": {"abbreviation": position},
                           "status": {"name": "Active"}, "injuries": []}
                athletes.append(athlete)
                masters.append({"season":SEASON,"team":team,"player_id":pid,"player_name":name,"espn_id":provider,
                                "position":position,"position_group":position,"status":"ACT",
                                "date_imported":stamp,"roster_source_as_of_utc":stamp,"roster_refresh_status":"LIVE"})
                perf_rows.append({"team":team,"player_id":pid,"player_name":name,"position":position,
                                  "position_group":position,"performance_input_score":50.0,"replacement_baseline":30.0,
                                  "position_confidence_score":1.0,"history_available":1,"usable_performance_grade":1,
                                  "durability_score":85.0,"qb_starter_pool":int(i==0)})
            positions = {slot: {"athletes":[copy.deepcopy(athletes[i])]}
                         for i,slot in enumerate(("qb",*OL_SLOTS))}
            positions["lcb"]={"athletes":[copy.deepcopy(athletes[6]),copy.deepcopy(athletes[7])]}
            envelope={"status":"success","season":{"year":SEASON},"team":{"abbreviation":team},"timestamp":stamp}
            record={"team":team,"fetched_at_utc":stamp,"roster_provider":"ESPN",
                    "roster":{**copy.deepcopy(envelope),"athletes":[{"items":athletes}]},
                    "depthcharts":{**copy.deepcopy(envelope),"depthchart":[{"positions":positions}]}}
            sources[team]=record
            audits.append({"team":team,"season":SEASON,"identity_issues":"","source_state":"SYNTHETIC_TEST",
                           "roster_source_timestamp":stamp,"depth_source_timestamp":stamp,"fetched_at_utc":stamp,
                           "roster_url":f"synthetic:{team}/roster","depth_url":f"synthetic:{team}/depthcharts"})
        data=[pd.DataFrame(perf_rows),pd.DataFrame(masters),sources,pd.DataFrame(audits)]
        if pryor:
            a={"id":"3055886","displayName":"Matt Pryor","position":{"abbreviation":"G"},
               "status":{"name":"Practice Squad"},"injuries":[]}
            roster(data,"ARI").append(copy.deepcopy(a))
            a["status"]={"name":"Active"}
            nodes(data,"NE")["lg"]["athletes"].append(a)
        if porter:
            add_owned(data,"4426506","00-0039167","Joey Porter Jr.","PIT","DAL","lcb",status="Out")
        return data

    def nodes(data, team):
        return data[2][team]["depthcharts"]["depthchart"][0]["positions"]

    def roster(data, team):
        return data[2][team]["roster"]["athletes"][0]["items"]

    def add_owned(data, provider, pid, name, owner, other, slot, status="Active"):
        a={"id":provider,"displayName":name,"position":{"abbreviation":"CB"},
           "status":{"name":status},"injuries":[]}
        roster(data,owner).append(copy.deepcopy(a))
        a["status"]={"name":"Active"}
        nodes(data,other).setdefault(slot,{"athletes":[]})["athletes"].insert(0,a)
        m=data[1].iloc[0].to_dict()
        m.update(player_id=pid,espn_id=provider,player_name=name,team=owner,position="CB",position_group="CB")
        p=data[0].iloc[0].to_dict()
        p.update(player_id=pid,player_name=name,team=owner,position="CB",position_group="CB",qb_starter_pool=0)
        data[0]=pd.concat([data[0],pd.DataFrame([p])],ignore_index=True)
        data[1]=pd.concat([data[1],pd.DataFrame([m])],ignore_index=True)

    def run(data):
        return build_live_inputs(*data, player_key)

    def failure(data, *needles):
        try:
            run(data)
        except (ValueError, RuntimeError) as exc:
            require(all(needle in str(exc) for needle in needles), f"Wrong rejection: {exc}")
            return str(exc)
        raise AssertionError("Expected blocker was accepted")

    def porter_recovery():
        d=fixture(pryor=False)
        before=copy.deepcopy(d[2]); orig=d[0].copy(deep=True)
        result=run(d)
        require(d[2]==before,"Raw source mutated")
        pd.testing.assert_frame_equal(result[0],orig)
        rows=result[4].loc[result[4].espn_id.eq("4426506")]
        require(len(rows)==1 and rows.iloc[0].team=="PIT","Player duplicated or assigned to Dallas")
        require(rows.iloc[0].live_unavailable==1 and rows.iloc[0].live_injury_status=="Out","Owner injury lost")
        require(result[1].set_index("player_id").loc["00-0039167","status"]=="OUT","Owner made active")
        require(d[3].loc[d[3].team.eq("DAL"),"identity_issues"].str.contains("STALE_STARTER").all(),"Dallas review absent")
        require(d[3].loc[d[3].team.eq("PIT"),"identity_issues"].eq("").all(),"Verified owner unnecessarily blocked")
        require(result[4].loc[result[4].team.eq("DAL") & result[4].live_preferred_slot.eq("CB1"),"espn_id"].tolist()
                ==[roster(d,"DAL")[6]["id"]],"Remaining published candidate not retained")

    def full_league():
        d=fixture(all_teams)
        result=run(d)
        require(len(result[2])==32 and len(result[3])==160,"Incomplete QB/OL set")
        require(result[3].groupby("team").espn_athlete_id.nunique().eq(5).all(),"Duplicate OL")
        require(result[4].loc[result[4].player_id.ne(""),"player_id"].is_unique,"Duplicate canonical identities")
        require(set(d[3].loc[d[3].identity_issues.ne(""),"team"])=={"ARI","NE","DAL"},"Wrong review scope")
        conflicts=d[3].attrs["identity_conflict_audit"]["conflicts"]
        require({c["decision"] for c in conflicts}=={"REVIEW_UNRESOLVED_PROVIDER","REVIEW_CORROBORATED_STALE_STARTER"},"Wrong decisions")

    def generic_role(slot):
        d=fixture(porter=False,pryor=False)
        add_owned(d,"7712345","SYNTHETIC_OWNED","Different Test Player","PIT","DAL",slot)
        run(d)
        c=d[3].attrs["identity_conflict_audit"]["conflicts"][0]
        require(c["decision"]=="REVIEW_CORROBORATED_STALE_STARTER","Non-generic recovery")
        require(c["affected_slots"]=={"DAL":[slot.upper()]},"Affected slot lost")

    def first_available():
        d=fixture(pryor=False)
        a=nodes(d,"DAL")["lcb"]["athletes"]
        a[0],a[1]=a[1],a[0]
        roster(d,"DAL")[6]["status"]={"name":"Out"}
        run(d)
        c=d[3].attrs["identity_conflict_audit"]["conflicts"][0]
        require(c["decision"]=="REVIEW_CORROBORATED_STALE_STARTER","First available not flagged")

    def resolved_backup():
        d=fixture(pryor=False)
        a=nodes(d,"DAL")["lcb"]["athletes"]
        a[0],a[1]=a[1],a[0]
        run(d)
        require(d[3].attrs["identity_conflict_audit"]["conflicts"][0]["decision"]=="RESOLVED_CORROBORATED_OWNER","Backup behavior regressed")
        require(d[3].identity_issues.eq("").all(),"Resolved nonstarter got a starter flag")

    def exact_guard(kind):
        d=fixture(pryor=False)
        if kind=="QB":
            a=nodes(d,"DAL")["lcb"]["athletes"].pop(0)
            nodes(d,"DAL")["qb"]["athletes"].insert(0,a)
            needle="QB/OL depth"
        elif kind=="OL":
            a=nodes(d,"DAL")["lcb"]["athletes"].pop(0)
            nodes(d,"DAL")["lg"]["athletes"].append(a)
            needle="QB/OL depth"
        elif kind=="team":
            d[1].loc[d[1].player_id.eq("00-0039167"),"team"]="DAL"; needle="does not match roster owner"
        elif kind=="name":
            nodes(d,"DAL")["lcb"]["athletes"][0]["displayName"]="Another Name"; needle="conflicting source names"
        elif kind=="stale_master":
            d[1].loc[d[1].player_id.eq("00-0039167"),"roster_source_as_of_utc"]=(pd.Timestamp.now(tz="UTC")-pd.Timedelta(hours=25)).isoformat()
            needle="canonical roster source is stale"
        elif kind=="stale_depth":
            d[2]["DAL"]["depthcharts"]["timestamp"]=(pd.Timestamp.now(tz="UTC")-pd.Timedelta(hours=25)).isoformat()
            needle="ownership source freshness"
        elif kind=="provenance":
            d[1].loc[d[1].player_id.eq("00-0039167"),"roster_refresh_status"]="UNVERIFIED"; needle="provenance is unverified"
        elif kind=="explicit_id":
            nodes(d,"DAL")["lcb"]["athletes"][0]["canonical_player_id"]="OTHER_CANONICAL"; needle="conflicting explicit canonical ID"
        elif kind=="owners":
            roster(d,"DAL").append(copy.deepcopy(roster(d,"PIT")[-1])); needle="exactly one actual roster owner"
        elif kind=="fallback":
            d[2]["PIT"]["roster_provider"]="REFRESHED_NFLVERSE_PLAYER_MASTER"; needle="not independent ownership evidence"
        else:
            raise AssertionError(kind)
        failure(d,needle)
        require(d[3].attrs["identity_conflict_audit"]["scan_complete"],"Expected blockers interrupted full scan")

    def all_conflicts():
        d=fixture()
        # First true error is before Porter alphabetically by provider ID;
        # last error is after him. Both must appear alongside both review cases.
        add_owned(d,"4010000","BAD_A","Early Conflict","PIT","DAL","wr1")
        roster(d,"DAL").append(copy.deepcopy(roster(d,"PIT")[-1]))
        add_owned(d,"7010000","BAD_B","Later Conflict","PIT","NE","wr2")
        roster(d,"NE").append(copy.deepcopy(roster(d,"PIT")[-1]))
        msg=failure(d,"4010000","7010000")
        scan=d[3].attrs["identity_conflict_audit"]
        require(scan["scan_complete"] and len(scan["conflicts"])==4 and len(scan["blocking_errors"])==2,"Not all conflicts collected")
        require(sum(x["decision"]=="REVIEW_CORROBORATED_STALE_STARTER" for x in scan["conflicts"])==1,"Recoverable issue omitted")
        with tempfile.TemporaryDirectory() as tmp:
            report=write_identity_report(Path(tmp),d[2],d[3],status="BLOCKED",attempted_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),error=msg)
            parsed=json.loads((Path(tmp)/"outputs/nfl_depth_identity_report_2026.json").read_text())
            text_report=(Path(tmp)/"outputs/nfl_depth_identity_report_2026.txt").read_text()
            require(parsed==report,"Report serialization changed content")
            require(all(pid in text_report for pid in ("3055886","4010000","4426506","7010000")),"Human report misses a conflict")
            require(len(list((Path(tmp)/"outputs/audits/depth_identity").glob('*.json')))==1,"History report absent")
            clean=fixture(porter=False,pryor=False); run(clean)
            write_identity_report(Path(tmp),clean[2],clean[3],status="COMPLETE",attempted_at_utc=pd.Timestamp.now(tz="UTC").isoformat())
            parsed=json.loads((Path(tmp)/"outputs/nfl_depth_identity_report_2026.json").read_text())
            require(parsed["conflicts"]==[] and parsed["review_teams"]==[],"Old error leaked into new successful report")
            require(len(list((Path(tmp)/"outputs/audits/depth_identity").glob('*.json')))==2,"Prior error history was lost")

    def multiple_ol_errors():
        d=fixture(porter=False,pryor=False)
        nodes(d,"DAL").pop("rt"); nodes(d,"NE").pop("lg")
        failure(d,"DAL","NE","RT","LG")
        require(len(d[3].attrs["starter_validation_errors"])==2,"OL failures not collected across teams")

    cases=[("Known owner stale LCB1 corrected; canonical ID/status/talent/raw data preserved",porter_recovery),
           ("Combined Pryor and Porter synthetic 32-team input build: 32 QB/160 OL",full_league),
           ("Generic receiver rank-one stale-owner conflict handled without player-specific rule",lambda:generic_role("wr1")),
           ("Generic defensive-line rank-one stale-owner conflict handled",lambda:generic_role("dt")),
           ("First-available stale defensive backup is reviewed",first_available),
           ("Previously safe exact-owner nonstarter exclusion remains resolved",resolved_backup),
           ("All blockers plus recoverable conflicts reported in one scan; history retained",all_conflicts),
           ("Independent OL assignment failures collected for multiple teams",multiple_ol_errors)]
    cases += [("Exact-owner guard remains blocking: "+kind,lambda kind=kind:exact_guard(kind))
              for kind in ("QB","OL","team","name","stale_master","stale_depth","provenance","explicit_id","owners","fallback")]
    logger=logging.getLogger(__name__); before=logger.disabled; logger.disabled=True
    try:
        for name,fn in cases:
            try:
                fn(); results.append({"name":name,"status":"PASS"})
            except Exception as exc:
                results.append({"name":name,"status":"FAIL","error":f"{type(exc).__name__}: {exc}"})
    finally:
        logger.disabled=before
    failed=sum(row["status"]!="PASS" for row in results)
    return {"version":VERSION,"test_scope":"offline synthetic fixtures; temporary report files only",
            "total":len(results),"passed":len(results)-failed,"failed":failed,"tests":results}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Shared NFL depth helper; offline self-test entry point.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic tests without network or database access.")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This is an imported helper. Use --self-test, or run the structural runner.")
    report = _run_self_tests()
    extra = _run_reliability_tests()
    report["tests"].extend(extra["tests"])
    for key in ("total", "passed", "failed"):
        report[key] += extra[key]
    for test in report["tests"]:
        print(f"[LIVE_DEPTH_SELF_TEST] {test['status']}: {test['name']}")
        if test.get("error"):
            print("  " + test["error"])
    print(f"[LIVE_DEPTH_SELF_TEST] {report['passed']}/{report['total']} PASS | {VERSION}")
    sys.exit(1 if report["failed"] else 0)
