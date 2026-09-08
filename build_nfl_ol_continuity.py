#!/usr/bin/env python
"""Build canonical historical and projected NFL offensive-line continuity.

Required SQLite inputs
----------------------
- nfl_ol_pff_player_season_2022_2025
- nfl_projected_depth_chart_2026

Outputs
-------
- nfl_ol_player_snap_history
- nfl_ol_continuity_historical
- nfl_ol_continuity_2026
- nfl_ol_continuity_player_audit
- nfl_ol_continuity_unmatched_audit

Design rules
------------
- Consume the validated PFF/GSIS OL player-season table produced by the player
  performance stage; do not create another crosswalk or reload external data.
- Continuity measures retained teammates and combinations only.
- Give partial continuity credit to an established same-team starter returning
  after missing or playing limited snaps in the immediately prior season.
- Retain pair overlap only for audit; all ten mathematical pairs are not a
  defensible proxy for five adjacent OL combinations and receive zero weight.
- Projected starter availability and projected starter quality are published as
  separate context fields and are not folded into continuity_score.
- Missing continuity history is neutral with zero confidence, not falsely poor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import re
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SEASON = 2026
HISTORICAL_SEASONS = (2022, 2023, 2024, 2025)
PRIOR_SEASON = 2025
BUILD_ID = "NFL_OL_CONTINUITY_2026_CANONICAL_V3"
VERSION = "v3_injury_returner_credit_bounded_secondary_context"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

PFF_HISTORY_TABLE = "nfl_ol_pff_player_season_2022_2025"
DEPTH_TABLE = "nfl_projected_depth_chart_2026"
PLAYER_HISTORY_TABLE = "nfl_ol_player_snap_history"
HISTORICAL_TABLE = "nfl_ol_continuity_historical"
PROJECTED_TABLE = "nfl_ol_continuity_2026"
PLAYER_AUDIT_TABLE = "nfl_ol_continuity_player_audit"
UNMATCHED_AUDIT_TABLE = "nfl_ol_continuity_unmatched_audit"

OL_POSITIONS = {"C", "G", "OG", "LG", "RG", "T", "OT", "LT", "RT", "OL"}
OL_STARTER_SLOTS = {"LT", "LG", "C", "RG", "RT"}

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB", "HST": "HOU",
    "JAC": "JAX", "KAN": "KC", "KCC": "KC", "LA": "LAR", "STL": "LAR",
    "SD": "LAC", "SDG": "LAC", "LVR": "LV", "OAK": "LV", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}
POSITION_ALIASES = {"T": "OT", "G": "OG"}

RETURNING_SNAP_WEIGHT = 0.60
STARTER_EQUIVALENT_WEIGHT = 0.30
MULTI_YEAR_TENURE_WEIGHT = 0.10
ESTABLISHED_RETURNER_CREDIT = 0.75
LIMITED_PRIOR_SEASON_SNAPS = 200.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical 2026 OL continuity.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_ol_continuity_v2")
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
    return re.sub(r"\.0$", "", clean_scalar(value))


def normalize_team(value: Any) -> str:
    team = clean_scalar(value).upper().replace(".", "")
    return TEAM_ALIASES.get(team, team)


def normalize_position(value: Any) -> str:
    pos = re.sub(r"[^A-Z]", "", clean_scalar(value).upper())
    return POSITION_ALIASES.get(pos, pos)


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
    df = pd.read_sql_query(f'SELECT * FROM "{escaped}"', conn)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def is_ol_row(position: str, position_group: str) -> bool:
    return normalize_position(position) in OL_POSITIONS or clean_scalar(position_group).upper() == "OL"


def build_player_history(history_raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season", "pff_team", "pff_player_id", "player_name", "pff_position",
        "offense_snaps", "games", "gsis_player_id",
    ]
    missing = [c for c in required if c not in history_raw.columns]
    if missing:
        raise RuntimeError(f"{PFF_HISTORY_TABLE} missing columns: {missing}")
    out = history_raw.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = out[out["season"].isin(HISTORICAL_SEASONS)].copy()
    out["team"] = out["pff_team"].map(normalize_team)
    out["pff_player_id"] = out["pff_player_id"].map(clean_id)
    out["gsis_player_id"] = out["gsis_player_id"].map(clean_id)
    out["player_id"] = np.where(
        out["gsis_player_id"].ne(""),
        out["gsis_player_id"],
        "PFF:" + out["pff_player_id"],
    )
    out["player_name"] = out["player_name"].map(clean_scalar)
    out["position"] = out["pff_position"].map(normalize_position)
    out["position_group"] = "OL"
    out = out[out["player_id"].ne("") & out["team"].ne("")].copy()
    out["offense_snaps"] = numeric(out, "offense_snaps", 0.0)
    out["total_snaps"] = out["offense_snaps"]
    out["games"] = numeric(out, "games", 0.0)
    out["ol_snaps"] = out["offense_snaps"]
    out = out[out["ol_snaps"].gt(0)].copy()
    aggregated = out.groupby(["season", "team", "player_id"], as_index=False).agg(
        player_name=("player_name", "first"), position=("position", "first"),
        pff_player_id=("pff_player_id", "first"), gsis_player_id=("gsis_player_id", "first"),
        offense_snaps=("offense_snaps", "max"), total_snaps=("total_snaps", "max"),
        ol_snaps=("ol_snaps", "max"), games=("games", "max"),
    )
    aggregated["team_ol_snap_rank"] = aggregated.groupby(["season", "team"])["ol_snaps"].rank(method="first", ascending=False).astype(int)
    aggregated["top_five_flag"] = aggregated["team_ol_snap_rank"].le(5).astype(int)
    aggregated["source"] = VERSION
    aggregated["build_id"] = BUILD_ID
    return aggregated.sort_values(["season", "team", "team_ol_snap_rank", "player_id"]).reset_index(drop=True)


def pair_set(ids: Iterable[str]) -> set[tuple[str, str]]:
    unique = sorted(set(clean_id(x) for x in ids if clean_id(x)))
    return {tuple(sorted(pair)) for pair in itertools.combinations(unique, 2)}


def continuity_record(
    prior: pd.DataFrame,
    current_ids: set[str],
    current_top_ids: set[str],
    season: int,
    team: str,
    comparison_type: str,
    earlier: pd.DataFrame | None = None,
) -> dict[str, Any]:
    earlier = earlier.copy() if earlier is not None else pd.DataFrame(columns=prior.columns)
    prior_total = float(prior["ol_snaps"].sum())
    prior_top = prior.sort_values("ol_snaps", ascending=False).head(5)
    prior_top_ids = set(prior_top["player_id"])
    earlier_top = earlier.sort_values("ol_snaps", ascending=False).head(5)
    earlier_top_ids = set(earlier_top["player_id"])
    returning_ids = set(prior["player_id"]) & current_ids
    returning_snap = float(prior.loc[prior["player_id"].isin(returning_ids), "ol_snaps"].sum())
    returning_snap_pct = returning_snap / prior_total if prior_total > 0 else 0.0
    returning_starters = prior_top_ids & current_top_ids
    returning_starter_pct = len(returning_starters) / 5.0
    prior_snaps_by_player = prior.groupby("player_id")["ol_snaps"].sum().to_dict()
    established_returners = {
        player_id
        for player_id in (earlier_top_ids & current_top_ids)
        if float(prior_snaps_by_player.get(player_id, 0.0)) < LIMITED_PRIOR_SEASON_SNAPS
    }
    # Never count the same player as both an ordinary returning starter and an
    # injury-returner credit.
    established_returners -= returning_starters
    returning_starter_equivalent_count = (
        len(returning_starters)
        + ESTABLISHED_RETURNER_CREDIT * len(established_returners)
    )
    returning_starter_equivalent_pct = returning_starter_equivalent_count / 5.0
    multi_year_same_team_ids = current_top_ids & (set(prior["player_id"]) | set(earlier["player_id"]))
    multi_year_same_team_pct = len(multi_year_same_team_ids) / 5.0
    prior_pairs = pair_set(prior_top_ids)
    current_pairs = pair_set(current_top_ids)
    retained_pairs = prior_pairs & current_pairs
    retained_pair_pct = len(retained_pairs) / 10.0
    score = 100.0 * (
        RETURNING_SNAP_WEIGHT * returning_snap_pct
        + STARTER_EQUIVALENT_WEIGHT * returning_starter_equivalent_pct
        + MULTI_YEAR_TENURE_WEIGHT * multi_year_same_team_pct
    )
    confidence = min(1.0, prior_total / 4500.0) * min(1.0, len(current_top_ids) / 5.0)
    return {
        "season": season, "team": team, "comparison_type": comparison_type,
        "prior_snap_season": season - 1, "prior_team_ol_snaps": prior_total,
        "prior_starter_count": len(prior_top_ids), "current_starter_count": len(current_top_ids),
        "returning_player_count": len(returning_ids), "returning_starter_count": len(returning_starters),
        "established_returner_count": len(established_returners),
        "returning_starter_equivalent_count": returning_starter_equivalent_count,
        "multi_year_same_team_count": len(multi_year_same_team_ids),
        "retained_pair_count": len(retained_pairs), "returning_snap_pct": returning_snap_pct,
        "returning_starter_pct": returning_starter_pct,
        "returning_starter_equivalent_pct": returning_starter_equivalent_pct,
        "multi_year_same_team_pct": multi_year_same_team_pct,
        "retained_pair_pct": retained_pair_pct,
        "retained_pair_score_weight": 0.0,
        "continuity_score": score, "ol_continuity_score": score,
        "continuity_confidence": confidence, "continuity_available": int(prior_total > 0 and len(current_top_ids) > 0),
        "complete_five_man_unit_flag": int(len(current_top_ids) == 5),
        "prior_starter_ids": "|".join(sorted(prior_top_ids)),
        "current_starter_ids": "|".join(sorted(current_top_ids)),
        "returning_starter_ids": "|".join(sorted(returning_starters)),
        "established_returner_ids": "|".join(sorted(established_returners)),
        "multi_year_same_team_ids": "|".join(sorted(multi_year_same_team_ids)),
        "continuity_formula": "0.60_returning_snaps+0.30_starter_equivalent+0.10_multi_year_tenure",
        "source": VERSION, "build_id": BUILD_ID,
    }


def build_historical(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in (2023, 2024, 2025):
        prior_season = season - 1
        teams = sorted(set(history.loc[history["season"].eq(season), "team"]) | set(history.loc[history["season"].eq(prior_season), "team"]))
        for team in teams:
            prior = history[(history["season"] == prior_season) & (history["team"] == team)].copy()
            current = history[(history["season"] == season) & (history["team"] == team)].copy()
            if prior.empty:
                continue
            current_ids = set(current["player_id"])
            current_top = set(current.sort_values("ol_snaps", ascending=False).head(5)["player_id"])
            earlier = history[(history["season"] == season - 2) & (history["team"] == team)].copy()
            rows.append(continuity_record(prior, current_ids, current_top, season, team, "historical", earlier))
    return pd.DataFrame(rows)


def projected_starters(depth_team: pd.DataFrame) -> pd.DataFrame:
    ol = depth_team[
        depth_team["position_group"].astype(str).str.upper().eq("OL")
        | depth_team["canonical_role"].astype(str).str.upper().isin(OL_POSITIONS)
    ].copy()
    if ol.empty:
        return ol
    ol["starter_slot_priority"] = ol["starter_slot"].isin(OL_STARTER_SLOTS).astype(int)
    ol = ol.sort_values(["starter_slot_priority", "is_projected_starter", "depth_order_score", "player_id"], ascending=[False, False, False, True])
    selected: list[int] = []
    used: set[str] = set()
    for slot in ["LT", "LG", "C", "RG", "RT"]:
        exact = ol[(ol["starter_slot"] == slot) & ~ol["player_id"].isin(used)]
        if not exact.empty:
            idx = exact.index[0]
            selected.append(idx); used.add(ol.at[idx, "player_id"])
    for idx, row in ol.iterrows():
        if len(selected) >= 5: break
        if row["player_id"] in used: continue
        selected.append(idx); used.add(row["player_id"])
    return ol.loc[selected].head(5).copy()


def build_projected(history: pd.DataFrame, depth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    teams = sorted(depth["team"].dropna().astype(str).unique())
    for team in teams:
        prior = history[(history["season"] == PRIOR_SEASON) & (history["team"] == team)].copy()
        earlier = history[(history["season"] == PRIOR_SEASON - 1) & (history["team"] == team)].copy()
        starters = projected_starters(depth[depth["team"] == team])
        current_ids = set(starters["player_id"])
        if prior.empty:
            record = {
                "season": SEASON, "team": team, "comparison_type": "projected",
                "prior_snap_season": PRIOR_SEASON, "prior_team_ol_snaps": 0.0,
                "prior_starter_count": 0, "current_starter_count": len(current_ids),
                "returning_player_count": 0, "returning_starter_count": 0,
                "established_returner_count": 0, "returning_starter_equivalent_count": 0.0,
                "multi_year_same_team_count": 0, "retained_pair_count": 0,
                "returning_snap_pct": 0.0, "returning_starter_pct": 0.0,
                "returning_starter_equivalent_pct": 0.0, "multi_year_same_team_pct": 0.0,
                "retained_pair_pct": 0.0, "retained_pair_score_weight": 0.0,
                "continuity_score": 50.0, "ol_continuity_score": 50.0,
                "continuity_confidence": 0.0, "continuity_available": 0,
                "complete_five_man_unit_flag": int(len(current_ids) == 5),
                "prior_starter_ids": "", "current_starter_ids": "|".join(sorted(current_ids)),
                "returning_starter_ids": "", "established_returner_ids": "",
                "multi_year_same_team_ids": "",
                "continuity_formula": "0.60_returning_snaps+0.30_starter_equivalent+0.10_multi_year_tenure",
                "source": VERSION, "build_id": BUILD_ID,
            }
        else:
            record = continuity_record(prior, current_ids, current_ids, SEASON, team, "projected", earlier)
        record["projected_starter_quality_score"] = float(numeric(starters, "unit_quality_grade", 0.0).mean()) if not starters.empty else 0.0
        record["projected_starter_availability_score"] = float(numeric(starters, "availability_score", 0.0).mean()) if not starters.empty else 0.0
        record["projected_starter_confidence"] = float(numeric(starters, "position_confidence_score", 0.0).mean()) if not starters.empty else 0.0
        record["quality_in_continuity_score_flag"] = 0
        record["availability_in_continuity_score_flag"] = 0
        rows.append(record)

        prior_top_ids = set(prior.sort_values("ol_snaps", ascending=False).head(5)["player_id"]) if not prior.empty else set()
        earlier_top_ids = set(earlier.sort_values("ol_snaps", ascending=False).head(5)["player_id"]) if not earlier.empty else set()
        established_ids = set(str(record.get("established_returner_ids", "")).split("|")) - {""}
        for _, player in starters.iterrows():
            audit_rows.append({
                "season": SEASON, "team": team, "player_id": player["player_id"], "player_name": player["player_name"],
                "starter_slot": player.get("starter_slot", ""), "projected_starter_flag": 1,
                "prior_top_five_flag": int(player["player_id"] in prior_top_ids),
                "two_year_prior_top_five_flag": int(player["player_id"] in earlier_top_ids),
                "established_injury_returner_flag": int(player["player_id"] in established_ids),
                "returning_same_team_flag": int(player["player_id"] in set(prior["player_id"]) if not prior.empty else False),
                "unit_quality_grade": float(player.get("unit_quality_grade", 0.0)),
                "availability_score": float(player.get("availability_score", 0.0)),
                "position_confidence_score": float(player.get("position_confidence_score", 0.0)),
                "source": VERSION, "build_id": BUILD_ID,
            })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def validate(history: pd.DataFrame, projected: pd.DataFrame, depth: pd.DataFrame) -> None:
    if set(HISTORICAL_SEASONS) - set(history["season"].astype(int).unique()):
        raise RuntimeError("OL history is missing one or more required seasons")
    if projected["team"].nunique() != 32 or len(projected) != 32:
        raise RuntimeError(f"Expected 32 projected OL continuity rows, found {len(projected)}/{projected['team'].nunique()} teams")
    if projected["team"].duplicated().any():
        raise RuntimeError("Duplicate team rows in projected continuity")
    if not projected["prior_snap_season"].eq(PRIOR_SEASON).all():
        raise RuntimeError("Projected continuity is not based on 2025")
    if not projected["quality_in_continuity_score_flag"].eq(0).all() or not projected["availability_in_continuity_score_flag"].eq(0).all():
        raise RuntimeError("Quality or availability was incorrectly folded into continuity")
    if not projected["retained_pair_score_weight"].eq(0.0).all():
        raise RuntimeError("The obsolete all-pairs continuity proxy still affects the score")
    expected_score = 100.0 * (
        RETURNING_SNAP_WEIGHT * projected["returning_snap_pct"]
        + STARTER_EQUIVALENT_WEIGHT * projected["returning_starter_equivalent_pct"]
        + MULTI_YEAR_TENURE_WEIGHT * projected["multi_year_same_team_pct"]
    )
    available = projected["continuity_available"].eq(1)
    if available.any() and not np.allclose(
        projected.loc[available, "continuity_score"],
        expected_score.loc[available],
        atol=1e-9,
    ):
        raise RuntimeError("Projected continuity does not reconcile to the approved formula")
    for col in ["continuity_score", "projected_starter_quality_score", "projected_starter_availability_score", "projected_starter_confidence"]:
        if projected[col].isna().any():
            raise RuntimeError(f"Missing values in {col}")
    if depth["team"].nunique() != 32:
        raise RuntimeError("Depth table does not contain 32 teams")


def empty_unmatched() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_table", "source_player_id", "player_name", "team", "season", "reason", "build_id"])


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    db_path = args.db_path.resolve()
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir / "build_nfl_ol_continuity.log")
    logger.info("[OL] Building canonical offensive-line continuity")
    logger.info("[OL] Build ID: %s", BUILD_ID)
    logger.info("[OL] Version: %s", VERSION)
    logger.info("[OL] Database: %s", db_path)

    try:
        with sqlite3.connect(db_path) as conn:
            raw_history = read_table(conn, PFF_HISTORY_TABLE)
            depth = read_table(conn, DEPTH_TABLE)
            depth["team"] = depth["team"].map(normalize_team)
            depth["player_id"] = depth["player_id"].map(clean_id)
            history = build_player_history(raw_history)
            historical = build_historical(history)
            projected, audit = build_projected(history, depth)
            unmatched = empty_unmatched()
            validate(history, projected, depth)

            outputs = {
                PLAYER_HISTORY_TABLE: history, HISTORICAL_TABLE: historical,
                PROJECTED_TABLE: projected, PLAYER_AUDIT_TABLE: audit,
                UNMATCHED_AUDIT_TABLE: unmatched,
            }
            for table, frame in outputs.items():
                frame.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{PROJECTED_TABLE}_team ON {PROJECTED_TABLE}(team)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{PLAYER_HISTORY_TABLE}_season_team ON {PLAYER_HISTORY_TABLE}(season, team)")
            conn.commit()

        if not args.no_csv:
            for table, frame in outputs.items():
                frame.to_csv(output_dir / f"{table}.csv", index=False, encoding="utf-8-sig")

        logger.info("[OL] Canonical OL player-season rows: %s", f"{len(history):,}")
        logger.info("[OL] Historical continuity rows: %s", f"{len(historical):,}")
        logger.info("[OL] Projected team rows: %s", f"{len(projected):,}")
        logger.info("[OL] Teams with continuity history: %s/32", int(projected["continuity_available"].sum()))
        logger.info("[OL] Complete projected five-man units: %s/32", int(projected["complete_five_man_unit_flag"].sum()))
        logger.info("[OL] Quality and availability remain separate from continuity_score")
        logger.info("[OL] Established same-team injury returners credited at %.2f starter-equivalent", ESTABLISHED_RETURNER_CREDIT)
        logger.info("[OL] All-pairs overlap retained for audit with zero score weight")
        logger.info("\n[OL] 2026 continuity summary:\n%s", projected[["team", "returning_snap_pct", "returning_starter_pct", "established_returner_count", "returning_starter_equivalent_pct", "multi_year_same_team_pct", "continuity_score", "continuity_confidence", "projected_starter_quality_score", "projected_starter_availability_score"]].sort_values("continuity_score", ascending=False).to_string(index=False))
        return 0
    except Exception as exc:
        logger.error("[OL] FAILED: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
