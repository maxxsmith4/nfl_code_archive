#!/usr/bin/env python
"""Prepare isolated point-in-time personnel contexts for historical NFL reconstruction.

Stage 1 only: this script creates the season-specific roster, player-master,
QB-starter, schedule, context, and readiness tables required before the rebuilt
structural model can be replayed historically.

Default cutoff policy
---------------------
- Target seasons: 2022-2025.
- Personnel snapshot: weekly roster for Week 1.
- QB1: highest Week 1 offensive QB usage, observed from completed Week 1 PBP.
- First graded week: Week 2, so the QB seed is prior information.
- Allowed player history for target season S: S-4 through S-1.
- One isolated SQLite database per season under backtests/<season>.sqlite.
- The production/source database is opened read-only and never modified.

Optional manual QB override CSV columns:
    season,team,player_id,player_name,reason
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sqlite3
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

BUILD_ID = "NFL_HISTORICAL_RECONSTRUCTION_CONTEXT_CANONICAL_V1"
VERSION = "v1_weekly_roster_week1_qb_prior_isolated_databases"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_SOURCE_DB = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")
DEFAULT_TARGET_SEASONS = (2022, 2023, 2024, 2025)
HISTORY_WINDOW = 4

CONTEXT_TABLE = "nfl_historical_reconstruction_context"
ROSTER_TABLE = "nfl_weekly_roster_snapshot_target"
MASTER_TABLE = "nfl_player_master_target"
QB_TABLE = "nfl_projected_qb_starters_target"
SCHEDULE_TABLE = "nfl_schedule_target"
READINESS_TABLE = "nfl_historical_personnel_readiness_audit"
ADVANCED_ALIAS_TABLE = "nfl_player_advanced_stats_history_available"
CROSSWALK_TABLE = "nfl_player_crosswalk"
ADVANCED_CANDIDATES = (
    "nfl_player_advanced_stats_2018_2025",
    "nfl_player_advanced_stats_2019_2025",
    "nfl_player_advanced_stats_2020_2025",
    "nfl_player_advanced_stats_2021_2025",
    "nfl_player_advanced_stats_2022_2025",
)

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB",
    "HST": "HOU", "JAC": "JAX", "KAN": "KC", "KCC": "KC",
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO",
    "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}
POSITION_ALIASES = {
    "HB": "RB", "T": "OT", "G": "OG", "DE": "EDGE", "ED": "EDGE",
    "NT": "DT", "ILB": "LB", "MLB": "LB", "OLB": "LB_EDGE",
    "NB": "SLOT", "NCB": "SLOT", "PK": "K",
}
POSITION_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR_TE", "TE": "WR_TE",
    "LT": "OL", "RT": "OL", "OT": "OL", "LG": "OL", "RG": "OL",
    "OG": "OL", "C": "OL", "OL": "OL", "EDGE": "EDGE", "DT": "DL",
    "DL": "DL", "LB_EDGE": "LB_EDGE", "LB": "LB", "CB": "DB",
    "SLOT": "DB", "FS": "DB", "SS": "DB", "S": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST",
}
REG_TYPES = {"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"}


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def parse_seasons(value: str) -> tuple[int, ...]:
    seasons = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not seasons:
        raise argparse.ArgumentTypeError("At least one season is required")
    return seasons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    p.add_argument("--target-seasons", type=parse_seasons, default=DEFAULT_TARGET_SEASONS)
    p.add_argument("--snapshot-week", type=int, default=1)
    p.add_argument("--first-graded-week", type=int, default=2)
    p.add_argument("--manual-qb-overrides", type=Path, default=None)
    p.add_argument("--allow-season-roster-fallback", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--synthetic-test", action="store_true")
    a = p.parse_args()
    if not 1 <= a.snapshot_week <= 18:
        p.error("--snapshot-week must be 1-18")
    if not 2 <= a.first_graded_week <= 18 or a.first_graded_week <= a.snapshot_week:
        p.error("--first-graded-week must be after snapshot week and inside 2-18")
    a.project_root = a.project_root.resolve()
    a.source_db = a.source_db.resolve()
    if a.manual_qb_overrides:
        a.manual_qb_overrides = a.manual_qb_overrides.resolve()
    return a


def frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def clean(value: Any) -> Optional[str]:
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
    return text[:-2] if re.fullmatch(r"-?\d+\.0", text) else text


def clean_id(value: Any) -> Optional[str]:
    text = clean(value)
    return None if text is None else text.replace("\u200b", "").replace("\ufeff", "").strip()


def team(value: Any) -> Optional[str]:
    text = clean(value)
    if text is None:
        return None
    key = text.upper().replace(".", "").strip()
    return TEAM_ALIASES.get(key, key)


def position(value: Any) -> str:
    text = clean(value)
    if text is None:
        return "OTHER"
    key = re.sub(r"[^A-Z]", "", text.upper())
    return POSITION_ALIASES.get(key, key or "OTHER")


def pos_group(value: Any) -> str:
    return POSITION_GROUP.get(position(value), "OTHER")


def clean_name(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9 ]", " ", (clean(value) or "").upper())
    return re.sub(r"\s+", " ", text).strip()


def initial_last(value: Any) -> str:
    parts = clean_name(value).split()
    return "" if not parts else f"{parts[0][:1]}_{parts[-1]}"


def first(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower().strip(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone() is not None


def read_table(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    if not table_exists(conn, name):
        return pd.DataFrame()
    escaped = name.replace('"', '""')
    out = pd.read_sql_query(f'SELECT * FROM "{escaped}"', conn)
    out.columns = [str(c).lower().strip() for c in out.columns]
    return out


def hash_frame(df: pd.DataFrame, columns: list[str]) -> str:
    payload = df[columns].astype(str).sort_values(columns).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def try_loader(package: str, function_names: tuple[str, ...], seasons: list[int], pbp=False):
    module = __import__(package)
    errors = []
    for name in function_names:
        loader = getattr(module, name, None)
        if loader is None:
            continue
        attempts = [lambda l=loader: l(seasons), lambda l=loader: l(seasons=seasons)]
        if pbp and package == "nfl_data_py":
            attempts = [lambda l=loader: l(seasons)]
        for attempt in attempts:
            try:
                out = frame(attempt())
                if not out.empty:
                    return out, f"{package}.{name}"
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    raise RuntimeError(" | ".join(errors) or f"No supported loader in {package}")


def load_weekly_rosters(seasons: list[int]):
    errors = []
    for package, names in (
        ("nflreadpy", ("load_rosters_weekly",)),
        ("nfl_data_py", ("import_weekly_rosters", "import_weekly_roster_data")),
    ):
        try:
            return try_loader(package, names, seasons)
        except Exception as exc:
            errors.append(f"{package}: {exc}")
    raise RuntimeError("Unable to load weekly rosters. " + " | ".join(errors))


def load_season_rosters(seasons: list[int]):
    errors = []
    for package, names in (
        ("nflreadpy", ("load_rosters",)),
        ("nfl_data_py", ("import_rosters",)),
    ):
        try:
            return try_loader(package, names, seasons)
        except Exception as exc:
            errors.append(f"{package}: {exc}")
    raise RuntimeError("Unable to load season rosters. " + " | ".join(errors))


def load_schedules(seasons: list[int]):
    errors = []
    for package, names in (
        ("nflreadpy", ("load_schedules",)),
        ("nfl_data_py", ("import_schedules",)),
    ):
        try:
            return try_loader(package, names, seasons)
        except Exception as exc:
            errors.append(f"{package}: {exc}")
    raise RuntimeError("Unable to load schedules. " + " | ".join(errors))


def load_pbp(seasons: list[int]):
    errors = []
    for package, names in (
        ("nflreadpy", ("load_pbp",)),
        ("nfl_data_py", ("import_pbp_data",)),
    ):
        try:
            return try_loader(package, names, seasons, pbp=True)
        except Exception as exc:
            errors.append(f"{package}: {exc}")
    raise RuntimeError("Unable to load play-by-play. " + " | ".join(errors))


def standardize_rosters(raw: pd.DataFrame, fallback_week: Optional[int] = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    aliases = {
        "season": ("season", "season_year"),
        "week": ("week", "game_week", "week_num"),
        "team": ("team", "recent_team", "club_code", "team_abbr"),
        "player_id": ("gsis_id", "player_id", "nfl_id", "esb_id", "smart_id"),
        "player_name": ("full_name", "player_name", "football_name", "display_name", "name"),
        "position": ("position", "pos"),
        "depth_chart_position": ("depth_chart_position", "depth_position"),
        "status": ("status", "status_short_description", "roster_status"),
        "jersey_number": ("jersey_number", "jersey", "number"),
        "height": ("height",), "weight": ("weight",),
        "birth_date": ("birth_date", "birthdate"), "age": ("age",),
        "years_exp": ("years_exp", "years_of_experience", "experience"),
        "college": ("college", "college_name"),
        "draft_club": ("draft_club", "draft_team"),
        "draft_number": ("draft_number", "draft_pick", "pick"),
        "entry_year": ("entry_year",), "rookie_year": ("rookie_year",),
    }
    found = {k: first(df.columns, v) for k, v in aliases.items()}
    if found["week"] is None and fallback_week is not None:
        df["_synthetic_week"] = fallback_week
        found["week"] = "_synthetic_week"
    missing = [k for k in ("season", "week", "team", "player_id") if found[k] is None]
    if missing:
        raise RuntimeError(f"Roster source missing fields: {missing}")
    out = pd.DataFrame(index=df.index)
    for key in aliases:
        out[key] = df[found[key]] if found[key] is not None else None
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["team"] = out["team"].map(team)
    out["player_id"] = out["player_id"].map(clean_id)
    out["player_name"] = out["player_name"].map(lambda x: clean(x) or "")
    out["position"] = out["position"].map(position)
    out["depth_chart_position"] = out["depth_chart_position"].map(position)
    out["position_group"] = out["position"].map(pos_group)
    out = out.dropna(subset=["season", "week", "team", "player_id"])
    out[["season", "week"]] = out[["season", "week"]].astype(int)
    return out.sort_values(["season", "week", "team", "player_id"]).drop_duplicates(
        ["season", "week", "team", "player_id"], keep="last"
    ).reset_index(drop=True)


def standardize_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy(); df.columns = [str(c).lower().strip() for c in df.columns]
    aliases = {
        "season": ("season", "season_year"), "week": ("week", "game_week"),
        "game_type": ("game_type", "season_type", "type"),
        "game_id": ("game_id", "gsis_id", "id"),
        "game_date": ("gameday", "game_date", "date", "game_datetime", "start_time"),
        "home_team": ("home_team", "home_team_abbr", "home"),
        "away_team": ("away_team", "away_team_abbr", "away"),
        "home_score": ("home_score", "home_points", "score_home"),
        "away_score": ("away_score", "away_points", "score_away"),
        "neutral_site": ("neutral_site", "neutral", "is_neutral"),
        "spread_line": ("spread_line", "closing_spread", "market_spread", "spread"),
    }
    found = {k: first(df.columns, v) for k, v in aliases.items()}
    missing = [k for k in ("season", "week", "home_team", "away_team") if found[k] is None]
    if missing:
        raise RuntimeError(f"Schedule source missing fields: {missing}")
    out = pd.DataFrame(index=df.index)
    for key in aliases:
        out[key] = df[found[key]] if found[key] is not None else None
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["game_type"] = out["game_type"].fillna("REG").astype(str).str.upper().str.strip()
    out["home_team"] = out["home_team"].map(team); out["away_team"] = out["away_team"].map(team)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["home_score"] = pd.to_numeric(out["home_score"], errors="coerce")
    out["away_score"] = pd.to_numeric(out["away_score"], errors="coerce")
    out["spread_line"] = pd.to_numeric(out["spread_line"], errors="coerce")
    out["neutral_site"] = out["neutral_site"].astype(str).str.upper().isin({"1","TRUE","T","YES","Y"}).astype(int)
    out = out.dropna(subset=["season", "week", "home_team", "away_team"])
    out = out[out["game_type"].isin(REG_TYPES)].copy(); out[["season","week"]] = out[["season","week"]].astype(int)
    missing_id = out["game_id"].isna() | out["game_id"].astype(str).isin({"", "None", "nan"})
    out.loc[missing_id, "game_id"] = (
        out.loc[missing_id, "season"].astype(str) + "_" + out.loc[missing_id, "week"].astype(str)
        + "_" + out.loc[missing_id, "away_team"].astype(str) + "_" + out.loc[missing_id, "home_team"].astype(str)
    )
    out["completed"] = (out["home_score"].notna() & out["away_score"].notna()).astype(int)
    return out.sort_values(["season","week","game_id"]).drop_duplicates(
        ["season","week","home_team","away_team"], keep="last"
    ).reset_index(drop=True)


def standardize_pbp(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy(); df.columns = [str(c).lower().strip() for c in df.columns]
    aliases = {
        "season": ("season",), "week": ("week",), "season_type": ("season_type", "game_type"),
        "team": ("posteam", "offense_team"), "player_id": ("passer_player_id", "passer_id"),
        "player_name": ("passer_player_name", "passer_name"),
        "pass_attempt": ("pass_attempt",), "sack": ("sack",), "qb_scramble": ("qb_scramble",), "no_play": ("no_play",),
    }
    found = {k: first(df.columns, v) for k, v in aliases.items()}
    missing = [k for k in ("season","week","team","player_id") if found[k] is None]
    if missing:
        raise RuntimeError(f"PBP source missing fields: {missing}")
    out = pd.DataFrame(index=df.index)
    for key in aliases:
        out[key] = df[found[key]] if found[key] is not None else None
    out["season"] = pd.to_numeric(out["season"], errors="coerce"); out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out["season_type"] = out["season_type"].fillna("REG").astype(str).str.upper().str.strip()
    out["team"] = out["team"].map(team); out["player_id"] = out["player_id"].map(clean_id)
    out["player_name"] = out["player_name"].map(lambda x: clean(x) or "")
    for c in ("pass_attempt","sack","qb_scramble","no_play"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["season","week","team","player_id"])
    out = out[out["season_type"].isin(REG_TYPES) & out["no_play"].eq(0)].copy(); out[["season","week"]] = out[["season","week"]].astype(int)
    out["qb_usage_play"] = out["pass_attempt"] + out["sack"] + out["qb_scramble"]
    return out


def build_master(roster: pd.DataFrame, season: int, source: str) -> pd.DataFrame:
    out = roster[roster["season"].eq(season)].copy()
    out["current_player_name"] = out["player_name"]; out["current_team"] = out["team"]
    out["current_position"] = out["position"]; out["current_position_group"] = out["position_group"]
    out["clean_name"] = out["player_name"].map(clean_name); out["initial_last_key"] = out["player_name"].map(initial_last)
    out["canonical_key"] = "GSIS:" + out["player_id"].astype(str); out["aliases"] = out["player_name"]
    out["identity_quality_flag"] = "CANONICAL_GSIS"; out["identity_version"] = VERSION
    out["master_build_id"] = BUILD_ID; out["roster_snapshot_source"] = source; out["date_imported"] = now()
    columns = [
        "season","player_id","player_name","team","position","position_group","depth_chart_position","status",
        "jersey_number","height","weight","birth_date","age","years_exp","college","draft_club","draft_number",
        "entry_year","rookie_year","current_player_name","current_team","current_position","current_position_group",
        "clean_name","initial_last_key","canonical_key","aliases","identity_quality_flag","identity_version",
        "master_build_id","roster_snapshot_source","date_imported",
    ]
    for c in columns:
        if c not in out.columns: out[c] = None
    return out[columns].drop_duplicates("player_id", keep="last").sort_values(["team","position","player_name"]).reset_index(drop=True)


def infer_qbs(pbp: pd.DataFrame, master: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    w = pbp[pbp["season"].eq(season) & pbp["week"].eq(week) & pbp["qb_usage_play"].gt(0)].copy()
    q = w.groupby(["team","player_id","player_name"], as_index=False).agg(
        qb_usage_plays=("qb_usage_play","sum"), pass_attempts=("pass_attempt","sum"), sacks=("sack","sum"), scrambles=("qb_scramble","sum")
    ).sort_values(["team","qb_usage_plays","pass_attempts"], ascending=[True,False,False]).drop_duplicates("team")
    lookup = master[["player_id","player_name","team","position"]].rename(columns={"player_name":"master_name","team":"master_team","position":"master_position"})
    q = q.merge(lookup, on="player_id", how="left", validate="many_to_one")
    q["player_name"] = q["master_name"].combine_first(q["player_name"])
    q["season"] = season; q["active_flag"] = 1; q["starter_source"] = f"week_{week}_pbp_prior_observed"
    q["reason"] = f"Highest QB usage through completed Week {week}; graded replay starts later"
    q["inference_week"] = week; q["roster_team_match_flag"] = q["team"].eq(q["master_team"]).astype(int)
    q["position_qb_flag"] = q["master_position"].eq("QB").astype(int); q["date_imported"] = now()
    return q[["season","team","player_id","player_name","active_flag","starter_source","reason","inference_week",
              "qb_usage_plays","pass_attempts","sacks","scrambles","roster_team_match_flag","position_qb_flag","date_imported"]].sort_values("team").reset_index(drop=True)


def load_overrides(path: Optional[Path]) -> pd.DataFrame:
    cols = ["season","team","player_id","player_name","reason"]
    if path is None: return pd.DataFrame(columns=cols)
    df = pd.read_csv(path); df.columns = [str(c).lower().strip() for c in df.columns]
    if not {"season","team"}.issubset(df.columns): raise RuntimeError("QB override CSV requires season and team")
    for c in cols:
        if c not in df.columns: df[c] = None
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64"); df["team"] = df["team"].map(team)
    df["player_id"] = df["player_id"].map(clean_id); df["player_name"] = df["player_name"].map(lambda x: clean(x) or "")
    return df[cols].dropna(subset=["season","team"])


def apply_overrides(qbs: pd.DataFrame, overrides: pd.DataFrame, master: pd.DataFrame, season: int) -> pd.DataFrame:
    manual = overrides[overrides["season"].eq(season)]
    if manual.empty: return qbs
    out = qbs.set_index("team", drop=False); lookup = master.set_index("player_id", drop=False)
    for _, row in manual.iterrows():
        t = str(row["team"]); pid = clean_id(row["player_id"]); name = clean(row["player_name"]) or ""
        if pid is None and name:
            matches = master[master["player_name"].map(clean_name).eq(clean_name(name))]
            if len(matches) > 1: matches = matches[matches["team"].eq(t)]
            if len(matches) == 1: pid = str(matches.iloc[0]["player_id"])
        if pid is None or pid not in lookup.index: raise RuntimeError(f"Could not resolve QB override {season} {t}")
        m = lookup.loc[pid]; m = m.iloc[0] if isinstance(m, pd.DataFrame) else m
        out.loc[t] = pd.Series({
            "season":season,"team":t,"player_id":pid,"player_name":m["player_name"],"active_flag":1,
            "starter_source":"manual_override","reason":clean(row["reason"]) or "Manual override","inference_week":np.nan,
            "qb_usage_plays":np.nan,"pass_attempts":np.nan,"sacks":np.nan,"scrambles":np.nan,
            "roster_team_match_flag":int(m["team"]==t),"position_qb_flag":int(m["position"]=="QB"),"date_imported":now(),
        })
    return out.reset_index(drop=True).sort_values("team").reset_index(drop=True)


def copy_source(source: sqlite3.Connection, target: sqlite3.Connection, required_history: list[int]):
    if table_exists(source, CROSSWALK_TABLE):
        read_table(source, CROSSWALK_TABLE).to_sql(CROSSWALK_TABLE, target, if_exists="replace", index=False)
    chosen = None; adv = pd.DataFrame()
    for candidate in ADVANCED_CANDIDATES:
        if table_exists(source, candidate): chosen = candidate; adv = read_table(source, candidate); break
    if adv.empty or "season" not in adv.columns:
        pd.DataFrame(columns=["season"]).to_sql(ADVANCED_ALIAS_TABLE, target, if_exists="replace", index=False)
        return [], chosen, 0
    adv["season"] = pd.to_numeric(adv["season"], errors="coerce")
    available = sorted(set(adv["season"].dropna().astype(int)))
    overlap = adv[adv["season"].isin(required_history)].copy()
    overlap.to_sql(ADVANCED_ALIAS_TABLE, target, if_exists="replace", index=False)
    return available, chosen, len(overlap)


def synthetic(seasons: list[int]):
    teams = ["ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"]
    rosters=[]; schedules=[]; pbp=[]
    for s in seasons:
        for t in teams:
            for i in range(12):
                rosters.append({"season":s,"week":1,"team":t,"gsis_id":f"{s}_{t}_{i:02d}","full_name":f"Player {t} {i}","position":"QB" if i==0 else "WR" if i<4 else "OL","status":"ACT","years_exp":3})
            for _ in range(20): pbp.append({"season":s,"week":1,"season_type":"REG","posteam":t,"passer_player_id":f"{s}_{t}_00","passer_player_name":f"Player {t} 0","pass_attempt":1,"sack":0,"qb_scramble":0,"no_play":0})
        for w in (1,2,3):
            for i in range(0,32,2):
                a,h=teams[i],teams[i+1]; schedules.append({"season":s,"week":w,"game_type":"REG","game_id":f"{s}_{w}_{a}_{h}","gameday":f"{s}-09-{w+1:02d}","away_team":a,"home_team":h,"away_score":17 if w==1 else None,"home_score":20 if w==1 else None})
    return pd.DataFrame(rosters), pd.DataFrame(schedules), pd.DataFrame(pbp)


def build_season(season, rosters, roster_source, schedules, schedule_source, pbp, pbp_source, overrides, source_conn, args):
    root=args.project_root; backtests=root/"backtests"; backtests.mkdir(parents=True, exist_ok=True)
    db=backtests/f"{season}.sqlite"
    if db.exists():
        if not args.overwrite: raise RuntimeError(f"Target exists: {db}; use --overwrite")
        db.unlink()
    rs=rosters[rosters["season"].eq(season)&rosters["week"].eq(args.snapshot_week)].copy()
    sched=schedules[schedules["season"].eq(season)].copy()
    teams=sorted(set(sched["home_team"])|set(sched["away_team"]))
    if rs.empty: raise RuntimeError(f"No {season} Week {args.snapshot_week} roster")
    if len(teams)!=32: raise RuntimeError(f"{season} schedule teams={len(teams)}, expected 32")
    master=build_master(rs, season, roster_source); qbs=apply_overrides(infer_qbs(pbp, master, season, args.snapshot_week), overrides, master, season)
    missing_qbs=sorted(set(teams)-set(qbs["team"])); invalid=qbs[(qbs["roster_team_match_flag"]!=1)|(qbs["position_qb_flag"]!=1)]
    history=list(range(season-HISTORY_WINDOW, season))
    with sqlite3.connect(db) as target:
        rs.to_sql(ROSTER_TABLE,target,if_exists="replace",index=False); rs.to_sql(f"nfl_weekly_roster_snapshot_{season}",target,if_exists="replace",index=False)
        master.to_sql(MASTER_TABLE,target,if_exists="replace",index=False); master.to_sql(f"nfl_player_master_{season}",target,if_exists="replace",index=False)
        qbs.to_sql(QB_TABLE,target,if_exists="replace",index=False); qbs.to_sql(f"nfl_projected_qb_starters_{season}",target,if_exists="replace",index=False)
        sched.to_sql(SCHEDULE_TABLE,target,if_exists="replace",index=False); sched.to_sql(f"nfl_schedule_{season}",target,if_exists="replace",index=False)
        available, adv_source, copied=copy_source(source_conn,target,history); missing_hist=sorted(set(history)-set(available))
        strict=int(roster_source.endswith("weekly") or "rosters_weekly" in roster_source)
        personnel_ok=int(strict and not missing_qbs and invalid.empty); structural_ready=int(personnel_ok and not missing_hist)
        context=pd.DataFrame([{
            "target_season":season,"history_start_season":history[0],"history_end_season":history[-1],"history_seasons":",".join(map(str,history)),
            "roster_snapshot_week":args.snapshot_week,"first_graded_week":args.first_graded_week,"roster_source":roster_source,"schedule_source":schedule_source,
            "qb_inference_source":pbp_source,"weekly_roster_strict_flag":strict,"week1_qb_prior_flag":1,"week1_games_graded_flag":0,
            "future_personnel_leakage_allowed_flag":0,"roster_rows":len(rs),"master_rows":len(master),"teams":len(teams),"qb_starters":len(qbs),
            "missing_qb_teams":"|".join(missing_qbs),"invalid_qb_rows":len(invalid),"advanced_source_table":adv_source,"advanced_rows_copied":copied,
            "advanced_available_seasons":",".join(map(str,available)),"missing_required_history_seasons":",".join(map(str,missing_hist)),
            "personnel_context_strict_flag":personnel_ok,"structural_inputs_ready_flag":structural_ready,
            "roster_snapshot_hash":hash_frame(master,["season","team","player_id","position"]),
            "qb_starter_hash":hash_frame(qbs,["season","team","player_id","starter_source"]),
            "isolated_database":str(db),"build_id":BUILD_ID,"version":VERSION,"created_at":now(),
        }]); context.to_sql(CONTEXT_TABLE,target,if_exists="replace",index=False)
        readiness=pd.DataFrame([
            {"target_season":season,"check_name":"weekly_roster_snapshot","passed":strict,"detail":roster_source},
            {"target_season":season,"check_name":"schedule_32_teams","passed":int(len(teams)==32),"detail":f"teams={len(teams)}"},
            {"target_season":season,"check_name":"qb_prior_complete","passed":int(not missing_qbs and invalid.empty),"detail":f"qbs={len(qbs)}; missing={missing_qbs}; invalid={len(invalid)}"},
            {"target_season":season,"check_name":"four_year_advanced_history","passed":int(not missing_hist),"detail":f"required={history}; available={available}; missing={missing_hist}"},
            {"target_season":season,"check_name":"safe_first_graded_week","passed":int(args.first_graded_week>args.snapshot_week),"detail":f"snapshot={args.snapshot_week}; first_graded={args.first_graded_week}"},
        ]); readiness["build_id"]=BUILD_ID; readiness["version"]=VERSION; readiness["created_at"]=now(); readiness.to_sql(READINESS_TABLE,target,if_exists="replace",index=False)
    if not args.no_csv:
        out=root/"outputs"/"historical_context"/str(season); out.mkdir(parents=True,exist_ok=True)
        master.to_csv(out/f"nfl_player_master_{season}.csv",index=False,encoding="utf-8-sig")
        qbs.to_csv(out/f"nfl_projected_qb_starters_{season}.csv",index=False,encoding="utf-8-sig")
        context.to_csv(out/"nfl_historical_reconstruction_context.csv",index=False,encoding="utf-8-sig")
        readiness.to_csv(out/"nfl_historical_personnel_readiness_audit.csv",index=False,encoding="utf-8-sig")
    return context.iloc[0].to_dict()


def main() -> int:
    args=parse_args(); started=dt.datetime.now(); run_id=str(uuid.uuid4()); seasons=list(args.target_seasons)
    print("[HIST_CONTEXT] Preparing historical reconstruction contexts")
    print(f"[HIST_CONTEXT] Build ID: {BUILD_ID}"); print(f"[HIST_CONTEXT] Version: {VERSION}"); print(f"[HIST_CONTEXT] Target seasons: {seasons}")
    args.project_root.mkdir(parents=True,exist_ok=True)
    if args.synthetic_test:
        rr,ss,pp=synthetic(seasons); roster_source="synthetic_rosters_weekly"; schedule_source="synthetic_schedules"; pbp_source="synthetic_pbp"
    else:
        try: rr,roster_source=load_weekly_rosters(seasons); fallback_week=None
        except Exception:
            if not args.allow_season_roster_fallback: raise
            rr,roster_source=load_season_rosters(seasons); fallback_week=args.snapshot_week
        ss,schedule_source=load_schedules(seasons); pp,pbp_source=load_pbp(seasons)
    fallback_week=args.snapshot_week if ("load_rosters" in roster_source and "weekly" not in roster_source) else None
    rosters=standardize_rosters(rr,fallback_week=fallback_week); schedules=standardize_schedule(ss); pbp=standardize_pbp(pp); overrides=load_overrides(args.manual_qb_overrides)
    if args.synthetic_test:
        source=sqlite3.connect(":memory:")
        pd.DataFrame({"season":list(range(2018,2025)),"player_id":[f"H{x}" for x in range(2018,2025)]}).to_sql("nfl_player_advanced_stats_2018_2025",source,if_exists="replace",index=False)
        pd.DataFrame({"canonical_player_id":["TEST"],"gsis_id":["TEST"]}).to_sql(CROSSWALK_TABLE,source,if_exists="replace",index=False)
    else:
        if not args.source_db.exists(): raise FileNotFoundError(args.source_db)
        source=sqlite3.connect(f"file:{args.source_db}?mode=ro",uri=True)
    manifest=[]
    try:
        for s in seasons:
            print("\n"+"="*108); print(f"[HIST_CONTEXT] Building {s}"); print("="*108)
            row=build_season(s,rosters,roster_source,schedules,schedule_source,pbp,pbp_source,overrides,source,args); manifest.append(row)
            print(f"[HIST_CONTEXT] {s}: master={row['master_rows']:,} | QB={row['qb_starters']} | strict={row['personnel_context_strict_flag']} | structural_ready={row['structural_inputs_ready_flag']}")
            if row["missing_required_history_seasons"]: print(f"[HIST_CONTEXT] Missing history: {row['missing_required_history_seasons']}")
    finally: source.close()
    mf=pd.DataFrame(manifest); mf["run_id"]=run_id; path=args.project_root/"outputs"/"nfl_historical_reconstruction_manifest.csv"; path.parent.mkdir(parents=True,exist_ok=True)
    if not args.no_csv: mf.to_csv(path,index=False,encoding="utf-8-sig")
    print("\n"+"="*108); print("[HIST_CONTEXT] SUMMARY"); print("="*108)
    print(mf[["target_season","roster_rows","master_rows","qb_starters","personnel_context_strict_flag","missing_required_history_seasons","structural_inputs_ready_flag","isolated_database"]].to_string(index=False))
    print("[HIST_CONTEXT] Production database modified: NO"); print("[HIST_CONTEXT] Week 1 eligible for grading: NO")
    print("[HIST_CONTEXT] Next: backfill missing advanced-history seasons, then parameterize canonical structural builders")
    print(f"[HIST_CONTEXT] Completed in {(dt.datetime.now()-started).total_seconds():.2f} seconds")
    if not args.no_csv: print(f"[HIST_CONTEXT] Manifest: {path}")
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
    except Exception as exc:
        print(f"[HIST_CONTEXT] FAILED: {exc}",file=sys.stderr); traceback.print_exc(); raise SystemExit(1)
