"""Read-only NFL source check; writes only its diagnostic JSON report.

This probes transport and inventories local prerequisites. A successful small
HTTP request does not prove that a package download or the full pipeline works.
No roster loaders, model deserialization, network repairs, or retries are run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import socket
import sqlite3
import ssl
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

VERSION = "v1_0_read_only_source_diagnostics"
SEASON = 2026
HTTP_TIMEOUT_SECONDS = 8.0
PROBE_DEADLINE_SECONDS = 10.0
USER_AGENT = "NFLStructuralModel/2026 availability audit"
ENDPOINTS = {
    "nflverse_roster_parquet": "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.parquet",
    "nflverse_roster_csv": "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.csv",
    "espn_ne_depth": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/17/depthcharts",
    "espn_ne_roster": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/17/roster",
}
TABLES = (
    "nfl_rosters_2026_raw", "nfl_rosters_2026_refresh_status",
    "nfl_player_master_2026", "nfl_2026_form_ratings",
    "nfl_2026_form_process_model", "nfl_form_historical_team_game_features_2018_2025",
    "nfl_live_roster_refresh_status_2026", "nfl_live_roster_source_audit_2026",
    "pbp_source", "snap_source", "cache_audit",
)
METADATA_COLUMNS = (
    "season", "through_week", "as_of_date", "status", "roster_version",
    "build_id", "version", "current_structural_version",
    "source_state", "roster_timestamp_basis",
    "model_version", "feature_version", "calibration_seasons",
    "no_lookahead_filter_applied_flag",
)
TIME_COLUMNS = (
    "date_imported", "roster_as_of_utc", "attempted_at_utc", "fetched_at_utc",
    "current_structural_date_imported",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def public_url(url: str) -> str:
    """Drop credentials, query strings and fragments, including redirect tokens."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def error_summary(exc: BaseException) -> dict[str, Any]:
    """Use known error categories; never serialize arbitrary exception messages."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    result: dict[str, Any] = {"exception_type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        result.update(category="HTTP_ERROR", http_status=int(exc.code))
    elif isinstance(reason, socket.gaierror):
        result.update(category="DNS_LOOKUP_FAILED", errno=reason.errno)
    elif isinstance(reason, (TimeoutError, socket.timeout)):
        result["category"] = "TIMEOUT"
    elif isinstance(reason, ssl.SSLCertVerificationError):
        result["category"] = "TLS_CERTIFICATE_VERIFICATION_FAILED"
    elif isinstance(reason, ssl.SSLError):
        result["category"] = "TLS_ERROR"
    elif isinstance(reason, ConnectionError):
        result["category"] = "CONNECTION_ERROR"
    elif isinstance(reason, OSError):
        result.update(category="OS_ERROR", errno=reason.errno)
    else:
        result["category"] = "OTHER_ERROR"
    winerror = getattr(reason, "winerror", None)
    if isinstance(winerror, int):
        result["winerror"] = winerror
    return result


def dns_probe(host: str, resolver: Callable[..., Any] = socket.getaddrinfo) -> dict[str, Any]:
    try:
        addresses = resolver(host, 443, type=socket.SOCK_STREAM)
        return {"status": "OK", "host": host, "address_count": len({a[4][0] for a in addresses})}
    except Exception as exc:
        return {"status": "FAILED", "host": host, **error_summary(exc)}


def http_probe(url: str, opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json" if "espn.com/" in url else "*/*",
        "Range": "bytes=0-1023",
    })
    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            sample = response.read(1024)
            code = int(response.status)
            return {
                "status": "OK" if 200 <= code < 300 else "FAILED",
                "url": public_url(url), "http_status": code,
                "final_url": public_url(response.geturl()),
                "bytes_sampled": len(sample),
            }
    except Exception as exc:
        return {"status": "FAILED", "url": public_url(url), **error_summary(exc)}


def parallel_probes(tasks: dict[str, Callable[[], dict[str, Any]]], deadline: float = PROBE_DEADLINE_SECONDS) -> dict[str, Any]:
    """Bound total wait even when the operating system DNS resolver hangs."""
    results: dict[str, Any] = {}
    completed: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()

    def worker(name: str, task: Callable[[], dict[str, Any]]) -> None:
        start = time.monotonic()
        try:
            result = task()
            if not isinstance(result, dict):
                raise TypeError("Diagnostic probe did not return a mapping")
        except Exception as exc:
            result = {"status": "FAILED", **error_summary(exc)}
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        completed.put((name, result))

    expires = time.monotonic() + deadline
    for name, task in tasks.items():
        threading.Thread(target=worker, args=(name, task), daemon=True).start()
    while len(results) < len(tasks):
        remaining = expires - time.monotonic()
        if remaining <= 0:
            break
        try:
            name, result = completed.get(timeout=remaining)
        except queue.Empty:
            break
        results[name] = result
    for name in tasks:
        results.setdefault(name, {"status": "TIMED_OUT", "category": "PROBE_DEADLINE_EXCEEDED"})
    return results


def package_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for package in ("nflreadpy", "nfl_data_py", "pandas", "numpy", "scikit-learn", "joblib", "SQLAlchemy"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        except Exception as exc:
            result[package] = {"version": None, **error_summary(exc)}
            continue
        result[package] = {"version": version}
    for package, functions in {
        "nflreadpy": ("load_rosters", "load_pbp", "load_snap_counts"),
        "nfl_data_py": ("import_seasonal_rosters", "import_weekly_rosters", "import_pbp_data", "import_snap_counts"),
    }.items():
        try:
            module = importlib.import_module(package)
            result[package]["import_status"] = "OK"
            result[package]["callable_functions"] = {
                name: callable(getattr(module, name, None)) for name in functions
            }
        except Exception as exc:
            result[package].update(import_status="FAILED", **error_summary(exc))
    return result


def sqlite_inventory(path: Path, week: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "mode": "READ_ONLY"}
    if not path.is_file():
        return result
    connection = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
        connection.execute("PRAGMA query_only=ON")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        summaries: dict[str, Any] = {}
        for table in TABLES:
            if table not in tables:
                continue
            columns = {r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')}
            entry: dict[str, Any] = {"rows": connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]}
            for column in METADATA_COLUMNS:
                if column in columns:
                    values = [r[0] for r in connection.execute(f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT 13')]
                    entry[column] = values[:12]
                    if len(values) > 12:
                        entry[column + "_truncated"] = True
            for column in TIME_COLUMNS:
                if column in columns:
                    minimum, maximum = connection.execute(f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"').fetchone()
                    entry[column] = {"min_as_stored": minimum, "max_as_stored": maximum}
            for column in ("team", "team_abbr"):
                if column in columns:
                    entry["distinct_teams"] = connection.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()[0]
                    break
            if table == "nfl_2026_form_ratings":
                entry["required_through_week"] = week - 1
                entry["through_week_matches_requested_week"] = entry.get("through_week") == [week - 1]
                entry["local_today"] = dt.date.today().isoformat()
                try:
                    dates = {dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat() for value in entry.get("as_of_date", [])}
                    entry["as_of_date_matches_local_today"] = dates == {entry["local_today"]}
                except ValueError:
                    entry["as_of_date_matches_local_today"] = False
            summaries[table] = entry
        result["tables"] = summaries
        result["status"] = "OK"
    except Exception as exc:
        result.update(status="FAILED", **error_summary(exc))
    finally:
        if connection is not None:
            connection.close()
    return result


def file_inventory(root: Path, week: int) -> dict[str, Any]:
    expected = (
        "models/nfl_learned_consensus_2026.joblib",
        "models/nfl_learned_consensus_2026_metadata.json",
        "backtests/nfl_weekly_matchup_source_cache_v2.sqlite",
        f"outputs/nfl_market/nfl_weekly_market_2026_week_{week:02d}.csv",
    )

    def describe(path: Path) -> dict[str, Any]:
        row: dict[str, Any] = {"path": str(path.relative_to(root)), "exists": path.is_file()}
        if row["exists"]:
            stat = path.stat()
            row.update(size_bytes=stat.st_size, modified_at_utc=dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat())
        return row

    result: dict[str, Any] = {"expected": [describe(root / name) for name in expected], "candidate_data_files": []}
    scanned = 0
    for directory in (root / "data", root / "backtests", root / "outputs"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            scanned += 1
            if scanned > 5000 or len(result["candidate_data_files"]) >= 100:
                result["inventory_truncated"] = True
                return result
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".sqlite", ".db"} and any(word in path.name.lower() for word in ("pbp", "snap", "roster", "process", "form")):
                result["candidate_data_files"].append(describe(path))
    result["inventory_truncated"] = False
    return result


def build_report(project_root: Path, db_path: Path, week: int, *, probe_runner: Callable[..., Any] = parallel_probes) -> dict[str, Any]:
    tasks: dict[str, Callable[[], dict[str, Any]]] = {
        "dns_github": lambda: dns_probe("github.com"),
        "dns_espn": lambda: dns_probe("site.api.espn.com"),
        "dns_odds_api": lambda: dns_probe("api.the-odds-api.com"),
    }
    tasks.update({name: (lambda url=url: http_probe(url)) for name, url in ENDPOINTS.items()})
    return {
        "version": VERSION, "attempted_at_utc": utc_now(), "season": SEASON, "prediction_week": week,
        "project_root": str(project_root),
        "python": {"version": sys.version.split()[0], "executable": sys.executable, "platform": sys.platform},
        "proxy_environment_present": {name: bool(os.environ.get(name)) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")},
        "network": probe_runner(tasks),
        "packages": package_inventory(),
        "database": sqlite_inventory(db_path, week),
        "historical_source_cache": sqlite_inventory(project_root / "backtests/nfl_weekly_matchup_source_cache_v2.sqlite", week),
        "files": file_inventory(project_root, week),
        "limits": [
            "Small HTTP samples test transport only; they do not validate datasets or guarantee package/pipeline success.",
            "ESPN probes cover New England only; a successful probe does not validate all 32 teams.",
            "PBP and snap URLs are selected inside installed packages and were not guessed or independently probed.",
            "Market API was not probed because authentication would be required; no API keys or proxy values were read into this report.",
            "File presence and SQLite metadata do not establish source completeness, freshness, or model compatibility.",
            "Timestamp fields labeled as_stored retain their database representation; no timezone was inferred.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.week <= 18:
        parser.error("--week must be between 1 and 18")
    root = args.project_root.expanduser().resolve()
    database = args.db_path.expanduser().resolve()
    sys.dont_write_bytecode = True
    report = build_report(root, database, args.week)
    output = root / "outputs/nfl_source_diagnostics_2026.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    failed = [name for name, row in report["network"].items() if row["status"] != "OK"]
    print(f"[NFL_SOURCE_DIAGNOSTICS] Report: {output}")
    print(f"[NFL_SOURCE_DIAGNOSTICS] Network checks with failures/timeouts: {', '.join(failed) or 'none'}")
    print("[NFL_SOURCE_DIAGNOSTICS] Inspection complete. No model, roster, or database data was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
