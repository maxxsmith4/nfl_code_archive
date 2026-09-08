#!/usr/bin/env python
"""Build canonical 2026 NFL team unit ratings.

Required SQLite input
---------------------
- nfl_projected_depth_chart_2026

Optional context input
----------------------
- nfl_ol_continuity_2026

Outputs
-------
- nfl_team_unit_ratings_2026
- nfl_position_group_ratings_2026
- nfl_team_unit_player_audit_2026
- nfl_team_unit_completeness_2026

Design rules
------------
1. The depth chart is the only player-level contract. This script does not merge
   the performance table again, eliminating prior duplicate-column collisions.
2. Player quality, confidence, depth completeness and availability remain
   separate fields.
3. OL unit quality uses the PFF blocking-talent unit_quality_grade from the
   depth chart without another shrink; continuity and availability are context
   only and are not added here.
4. Unit ratings are anchored to replacement value. No league mean/std variance
   forcing is performed. QB uses a 95 guardrail; all other units retain 80.
5. Compatibility columns are retained, but unit_rating_normalized is simply the
   replacement-anchored rating and does not imply z-score normalization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sqlite3
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SEASON = 2026
BUILD_ID = "NFL_TEAM_UNIT_RATINGS_2026_CANONICAL_V8"
VERSION = "v8_pff_ol_talent_contract_qb_ceiling_95"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

DEPTH_TABLE = "nfl_projected_depth_chart_2026"
OL_CONTINUITY_TABLE = "nfl_ol_continuity_2026"
TEAM_OUTPUT_TABLE = "nfl_team_unit_ratings_2026"
UNIT_OUTPUT_TABLE = "nfl_position_group_ratings_2026"
AUDIT_OUTPUT_TABLE = "nfl_team_unit_player_audit_2026"
COMPLETENESS_OUTPUT_TABLE = "nfl_team_unit_completeness_2026"

TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "GNB": "GB", "HST": "HOU",
    "JAC": "JAX", "KAN": "KC", "KCC": "KC", "LA": "LAR", "STL": "LAR",
    "SD": "LAC", "SDG": "LAC", "LVR": "LV", "OAK": "LV", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "WSH": "WAS", "WFT": "WAS",
}

UNIT_INDEX_MULTIPLIER = {
    "QB": 0.90, "RB": 0.75, "WR_TE": 0.75, "OL": 0.75,
    "DL_EDGE": 0.75, "LB": 0.75, "DB": 0.75, "ST": 0.60,
}

# Preserve the established 20-80 range for non-QB units. Quarterback is the
# only unit allowed above 80 because the prior universal ceiling collapsed
# distinct elite QB grades into identical ratings. A 95 ceiling remains as a
# defensive guardrail without affecting the current observed QB distribution.
UNIT_RATING_BOUNDS = {
    "QB": (20.0, 95.0),
    "RB": (20.0, 80.0),
    "WR_TE": (20.0, 80.0),
    "OL": (20.0, 80.0),
    "DL_EDGE": (20.0, 80.0),
    "LB": (20.0, 80.0),
    "DB": (20.0, 80.0),
    "ST": (20.0, 80.0),
}

OFFENSE_WEIGHTS = {"QB": 0.36, "RB": 0.10, "WR_TE": 0.27, "OL": 0.27}
DEFENSE_WEIGHTS = {"DL_EDGE": 0.42, "LB": 0.25, "DB": 0.33}
OVERALL_WEIGHTS = {"OFFENSE": 0.52, "DEFENSE": 0.44, "ST": 0.04}


@dataclass(frozen=True)
class SlotSpec:
    slot_name: str
    weight: float
    allowed_roles: tuple[str, ...]


UNIT_SLOTS: dict[str, list[SlotSpec]] = {
    "QB": [SlotSpec("QB1", 1.00, ("QB",))],
    "RB": [SlotSpec("RB1", 0.65, ("RB",)), SlotSpec("RB2", 0.35, ("RB",))],
    "WR_TE": [
        SlotSpec("WR1", 0.24, ("WR",)), SlotSpec("WR2", 0.21, ("WR",)),
        SlotSpec("WR3", 0.15, ("WR",)), SlotSpec("TE1", 0.25, ("TE",)),
        SlotSpec("TE2", 0.15, ("TE", "WR")),
    ],
    "OL": [
        SlotSpec("LT", 0.20, ("LT", "OT", "OL")),
        SlotSpec("LG", 0.20, ("LG", "OG", "OL")),
        SlotSpec("C", 0.20, ("C", "OL", "OG")),
        SlotSpec("RG", 0.20, ("RG", "OG", "OL")),
        SlotSpec("RT", 0.20, ("RT", "OT", "OL")),
    ],
    "DL_EDGE": [
        SlotSpec("EDGE1", 0.28, ("EDGE", "LB_EDGE", "DL")),
        SlotSpec("EDGE2", 0.27, ("EDGE", "LB_EDGE", "DL")),
        SlotSpec("DT1", 0.25, ("DT", "DL")), SlotSpec("DT2", 0.20, ("DT", "DL")),
    ],
    "LB": [
        SlotSpec("LB1", 0.40, ("LB",)), SlotSpec("LB2", 0.35, ("LB",)),
        SlotSpec("LB3", 0.25, ("LB",)),
    ],
    "DB": [
        SlotSpec("CB1", 0.22, ("CB", "DB")), SlotSpec("CB2", 0.20, ("CB", "DB")),
        SlotSpec("SLOT", 0.12, ("SLOT", "CB", "DB")),
        SlotSpec("FS", 0.23, ("FS", "S", "DB")),
        SlotSpec("SS", 0.23, ("SS", "S", "DB")),
    ],
    "ST": [
        SlotSpec("K", 0.45, ("K",)), SlotSpec("P", 0.35, ("P",)),
        SlotSpec("LS", 0.20, ("LS", "ST")),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical 2026 NFL team unit ratings.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_units_v6")
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


def compatible(role: str, allowed: Iterable[str]) -> bool:
    role = clean_scalar(role).upper()
    allowed = tuple(clean_scalar(x).upper() for x in allowed)
    if role in allowed:
        return True
    if role == "OT" and any(x in allowed for x in ("LT", "RT", "OT")): return True
    if role == "OG" and any(x in allowed for x in ("LG", "RG", "OG")): return True
    if role == "OL" and any(x in allowed for x in ("LT", "LG", "C", "RG", "RT", "OT", "OG", "OL")): return True
    if role == "S" and any(x in allowed for x in ("FS", "SS", "S", "DB")): return True
    if role == "DB" and any(x in allowed for x in ("CB", "SLOT", "FS", "SS", "S", "DB")): return True
    if role == "DL" and any(x in allowed for x in ("EDGE", "DT", "DL")): return True
    if role == "LB_EDGE" and any(x in allowed for x in ("EDGE", "LB_EDGE")): return True
    return False


def prepare_depth(raw: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season", "team", "player_id", "player_name", "canonical_role",
        "depth_rank", "is_projected_starter", "starter_slot", "projected_snap_share",
        "likely_unavailable", "performance_grade", "replacement_baseline",
        "unit_quality_grade", "availability_score", "position_confidence_score",
        "usable_performance_grade", "depth_order_score",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{DEPTH_TABLE} missing required columns: {missing}")
    out = raw.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out = out[out["season"].fillna(SEASON).eq(SEASON)].copy()
    out["team"] = out["team"].map(normalize_team)
    out["player_id"] = out["player_id"].map(clean_id)
    out["player_name"] = out["player_name"].map(clean_scalar)
    out["canonical_role"] = out["canonical_role"].map(lambda x: clean_scalar(x).upper())
    out["starter_slot"] = out["starter_slot"].map(lambda x: clean_scalar(x).upper())
    for col in ["depth_rank", "is_projected_starter", "projected_snap_share", "likely_unavailable", "performance_grade", "replacement_baseline", "unit_quality_grade", "availability_score", "position_confidence_score", "usable_performance_grade", "depth_order_score"]:
        out[col] = numeric(out, col, 0.0)
    out["position_confidence_score"] = out["position_confidence_score"].clip(0, 1)
    out["projected_snap_share"] = out["projected_snap_share"].clip(0, 1)
    out["availability_score"] = out["availability_score"].clip(0, 100)
    out["unit_quality_grade"] = out["unit_quality_grade"].clip(0, 100)
    out["replacement_baseline"] = out["replacement_baseline"].clip(0, 100)
    out = out[out["player_id"].ne("")].drop_duplicates("player_id", keep="last")
    if any(c.endswith("_x") or c.endswith("_y") for c in out.columns):
        raise RuntimeError("Depth contract contains merge-collision suffix columns")
    return out


def load_continuity(conn: sqlite3.Connection) -> pd.DataFrame:
    columns = [
        "team", "prior_snap_season", "continuity_score", "ol_continuity_score",
        "continuity_confidence", "continuity_available", "complete_five_man_unit_flag",
        "returning_snap_pct", "returning_starter_pct", "retained_pair_pct",
        "projected_starter_quality_score", "projected_starter_availability_score",
        "projected_starter_confidence",
    ]
    if not table_exists(conn, OL_CONTINUITY_TABLE):
        return pd.DataFrame(columns=columns)
    out = read_table(conn, OL_CONTINUITY_TABLE)
    out["team"] = out["team"].map(normalize_team)
    for col in columns[1:]:
        out[col] = numeric(out, col, 0.0)
    return out[columns].drop_duplicates("team", keep="last")


def choose_player(team_depth: pd.DataFrame, spec: SlotSpec, used: set[str]) -> pd.Series | None:
    candidates = team_depth[
        ~team_depth["player_id"].isin(used)
        & team_depth["canonical_role"].map(lambda r: compatible(r, spec.allowed_roles))
        & team_depth["likely_unavailable"].eq(0)
    ].copy()
    if candidates.empty:
        return None
    candidates["exact_slot"] = candidates["starter_slot"].eq(spec.slot_name).astype(int)
    candidates["starter_flag"] = candidates["is_projected_starter"].gt(0).astype(int)
    candidates = candidates.sort_values(
        ["exact_slot", "starter_flag", "depth_order_score", "position_confidence_score", "player_id"],
        ascending=[False, False, False, False, True],
    )
    return candidates.iloc[0]


def replacement_for_slot(team_depth: pd.DataFrame, spec: SlotSpec) -> float:
    candidates = team_depth[team_depth["canonical_role"].map(lambda r: compatible(r, spec.allowed_roles))]
    if not candidates.empty:
        values = candidates["replacement_baseline"]
        if values.notna().any():
            return float(values.median())
    unit_defaults = {"QB": 30.0, "RB": 28.0, "WR_TE": 28.0, "OL": 30.0, "DL_EDGE": 30.0, "LB": 30.0, "DB": 30.0, "ST": 20.0}
    for unit, specs in UNIT_SLOTS.items():
        if spec in specs:
            return unit_defaults[unit]
    return 30.0


def build_unit(team: str, unit_name: str, team_depth: pd.DataFrame, continuity_row: pd.Series | None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    used: set[str] = set()
    audit_rows: list[dict[str, Any]] = []
    weighted_quality = 0.0
    weighted_replacement = 0.0
    weighted_confidence = 0.0
    weighted_availability = 0.0
    weighted_snap_share = 0.0
    filled_weight = 0.0
    usable_weight = 0.0
    replacement_slots = 0
    for spec in UNIT_SLOTS[unit_name]:
        player = choose_player(team_depth, spec, used)
        if player is None:
            replacement = replacement_for_slot(team_depth, spec)
            quality = replacement
            confidence = 0.0
            availability = 0.0
            snap_share = 0.0
            usable = 0
            replacement_flag = 1
            player_id = ""
            player_name = "REPLACEMENT"
            role = ""
            depth_rank = 99
            selected_starter = 0
            replacement_slots += 1
        else:
            player_id = clean_id(player["player_id"])
            used.add(player_id)
            player_name = clean_scalar(player["player_name"])
            role = clean_scalar(player["canonical_role"])
            quality = float(player["unit_quality_grade"])
            replacement = float(player["replacement_baseline"])
            confidence = float(player["position_confidence_score"])
            availability = float(player["availability_score"])
            snap_share = float(player["projected_snap_share"])
            usable = int(player["usable_performance_grade"] > 0)
            replacement_flag = 0
            depth_rank = int(player["depth_rank"])
            selected_starter = int(player["is_projected_starter"] > 0)
            filled_weight += spec.weight
            usable_weight += spec.weight * usable
        weighted_quality += spec.weight * quality
        weighted_replacement += spec.weight * replacement
        weighted_confidence += spec.weight * confidence
        weighted_availability += spec.weight * availability
        weighted_snap_share += spec.weight * snap_share
        audit_rows.append({
            "season": SEASON, "team": team, "unit_name": unit_name, "slot_name": spec.slot_name,
            "slot_weight": spec.weight, "player_id": player_id, "player_name": player_name,
            "canonical_role": role, "depth_rank": depth_rank, "projected_starter_flag": selected_starter,
            "quality_grade": quality, "replacement_baseline": replacement,
            "value_over_replacement": quality - replacement, "position_confidence_score": confidence,
            "availability_score": availability, "projected_snap_share": snap_share,
            "usable_performance_grade": usable, "replacement_slot_flag": replacement_flag,
            "quality_source": clean_scalar(player.get("quality_source", "")) if player is not None else "replacement",
            "build_id": BUILD_ID, "unit_rating_version": VERSION,
        })
    multiplier = UNIT_INDEX_MULTIPLIER[unit_name]
    vor = weighted_quality - weighted_replacement
    rating_floor, rating_ceiling = UNIT_RATING_BOUNDS[unit_name]
    uncapped_rating = 50.0 + multiplier * vor
    rating = float(np.clip(uncapped_rating, rating_floor, rating_ceiling))
    continuity_score = float(continuity_row.get("continuity_score", 50.0)) if continuity_row is not None else 50.0
    continuity_confidence = float(continuity_row.get("continuity_confidence", 0.0)) if continuity_row is not None else 0.0
    continuity_available = int(float(continuity_row.get("continuity_available", 0.0)) > 0) if continuity_row is not None else 0
    prior_snap_season = int(float(continuity_row.get("prior_snap_season", 0.0))) if continuity_row is not None else 0
    complete_five = int(float(continuity_row.get("complete_five_man_unit_flag", 0.0)) > 0) if continuity_row is not None else 0
    unit_row = {
        "season": SEASON, "team": team, "unit_name": unit_name,
        "raw_unit_quality_grade": weighted_quality, "unit_replacement_baseline": weighted_replacement,
        "unit_value_over_replacement": vor, "unit_index_multiplier": multiplier,
        "uncapped_unit_rating": uncapped_rating, "unit_rating_floor": rating_floor,
        "unit_rating_ceiling": rating_ceiling,
        "raw_unit_rating": rating, "unit_rating_adjusted": rating, "unit_rating_normalized": rating,
        "unit_rating": rating, "normalization_method": "replacement_anchor_no_variance_forcing",
        "unit_confidence": weighted_confidence, "unit_availability_context": weighted_availability,
        "unit_projected_snap_share": weighted_snap_share, "unit_depth_completeness": filled_weight,
        "unit_usable_grade_weight": usable_weight, "replacement_slots": replacement_slots,
        "ol_continuity_score": continuity_score if unit_name == "OL" else 50.0,
        "ol_continuity_confidence": continuity_confidence if unit_name == "OL" else 0.0,
        "ol_continuity_available": continuity_available if unit_name == "OL" else 0,
        "ol_continuity_prior_snap_season": prior_snap_season if unit_name == "OL" else 0,
        "ol_complete_five_man_unit_flag": complete_five if unit_name == "OL" else 0,
        "ol_continuity_adjustment": 0.0, "continuity_applied_to_quality_flag": 0,
        "quality_adjusted_completeness": filled_weight * (0.5 + 0.5 * weighted_confidence),
        "unit_rating_version": VERSION, "build_id": BUILD_ID,
        "date_imported": dt.datetime.now().isoformat(timespec="seconds"),
    }
    completeness = {
        "season": SEASON, "team": team, "unit_name": unit_name,
        "required_slots": len(UNIT_SLOTS[unit_name]), "filled_slots": len(UNIT_SLOTS[unit_name]) - replacement_slots,
        "replacement_slots": replacement_slots, "depth_completeness": filled_weight,
        "usable_grade_weight": usable_weight, "unit_confidence": weighted_confidence,
        "unit_availability_context": weighted_availability,
        "quality_adjusted_completeness": unit_row["quality_adjusted_completeness"],
        "build_id": BUILD_ID, "unit_rating_version": VERSION,
    }
    return unit_row, audit_rows, completeness


def weighted_average(row: pd.Series, columns: dict[str, float]) -> float:
    return float(sum(float(row[col]) * weight for col, weight in columns.items()))


def build_all(depth: pd.DataFrame, continuity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    continuity_lookup = {row["team"]: row for _, row in continuity.iterrows()}
    for team in sorted(depth["team"].unique()):
        team_depth = depth[depth["team"] == team].copy()
        continuity_row = continuity_lookup.get(team)
        for unit_name in UNIT_SLOTS:
            unit_row, unit_audit, unit_complete = build_unit(team, unit_name, team_depth, continuity_row)
            unit_rows.append(unit_row)
            audit_rows.extend(unit_audit)
            completeness_rows.append(unit_complete)
    units = pd.DataFrame(unit_rows)
    audit = pd.DataFrame(audit_rows)
    completeness = pd.DataFrame(completeness_rows)

    wide = units.pivot(index=["season", "team"], columns="unit_name", values="unit_rating").reset_index()
    conf = units.pivot(index=["season", "team"], columns="unit_name", values="unit_confidence").reset_index()
    comp = units.pivot(index=["season", "team"], columns="unit_name", values="quality_adjusted_completeness").reset_index()
    wide.columns.name = None; conf.columns.name = None; comp.columns.name = None
    wide = wide.rename(columns={u: f"{u.lower()}_unit_rating" for u in UNIT_SLOTS})
    conf = conf.rename(columns={u: f"{u.lower()}_unit_confidence" for u in UNIT_SLOTS})
    comp = comp.rename(columns={u: f"{u.lower()}_quality_adjusted_completeness" for u in UNIT_SLOTS})
    teams = wide.merge(conf, on=["season", "team"], validate="one_to_one").merge(comp, on=["season", "team"], validate="one_to_one")

    offense_cols = {f"{u.lower()}_unit_rating": w for u, w in OFFENSE_WEIGHTS.items()}
    defense_cols = {f"{u.lower()}_unit_rating": w for u, w in DEFENSE_WEIGHTS.items()}
    offense_conf_cols = {f"{u.lower()}_unit_confidence": w for u, w in OFFENSE_WEIGHTS.items()}
    defense_conf_cols = {f"{u.lower()}_unit_confidence": w for u, w in DEFENSE_WEIGHTS.items()}
    teams["offense_unit_rating"] = teams.apply(lambda r: weighted_average(r, offense_cols), axis=1)
    teams["defense_unit_rating"] = teams.apply(lambda r: weighted_average(r, defense_cols), axis=1)
    teams["special_teams_unit_rating"] = teams["st_unit_rating"]
    teams["offense_unit_confidence"] = teams.apply(lambda r: weighted_average(r, offense_conf_cols), axis=1)
    teams["defense_unit_confidence"] = teams.apply(lambda r: weighted_average(r, defense_conf_cols), axis=1)
    teams["special_teams_unit_confidence"] = teams["st_unit_confidence"]
    teams["overall_team_unit_rating"] = (
        OVERALL_WEIGHTS["OFFENSE"] * teams["offense_unit_rating"]
        + OVERALL_WEIGHTS["DEFENSE"] * teams["defense_unit_rating"]
        + OVERALL_WEIGHTS["ST"] * teams["special_teams_unit_rating"]
    )
    teams["overall_team_unit_confidence"] = (
        OVERALL_WEIGHTS["OFFENSE"] * teams["offense_unit_confidence"]
        + OVERALL_WEIGHTS["DEFENSE"] * teams["defense_unit_confidence"]
        + OVERALL_WEIGHTS["ST"] * teams["special_teams_unit_confidence"]
    )
    quality_comp_cols = [c for c in teams.columns if c.endswith("_quality_adjusted_completeness")]
    teams["overall_quality_adjusted_completeness"] = teams[quality_comp_cols].mean(axis=1)
    teams["overall_team_unit_rank"] = teams["overall_team_unit_rating"].rank(method="min", ascending=False).astype(int)
    teams["normalization_method"] = "replacement_anchor_no_variance_forcing"
    teams["unit_rating_version"] = VERSION
    teams["build_id"] = BUILD_ID
    teams["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")

    if not continuity.empty:
        continuity_fields = continuity[[
            "team", "continuity_score", "continuity_confidence", "continuity_available",
            "prior_snap_season", "complete_five_man_unit_flag", "returning_snap_pct",
            "returning_starter_pct", "retained_pair_pct", "projected_starter_quality_score",
            "projected_starter_availability_score", "projected_starter_confidence",
        ]].rename(columns={
            "continuity_score": "ol_continuity_score",
            "continuity_confidence": "ol_continuity_confidence",
            "continuity_available": "ol_continuity_available",
            "prior_snap_season": "ol_continuity_prior_snap_season",
            "complete_five_man_unit_flag": "ol_complete_five_man_unit_flag",
        })
        teams = teams.merge(continuity_fields, on="team", how="left", validate="one_to_one")
    else:
        for col, default in {
            "ol_continuity_score": 50.0, "ol_continuity_confidence": 0.0,
            "ol_continuity_available": 0, "ol_continuity_prior_snap_season": 0,
            "ol_complete_five_man_unit_flag": 0, "returning_snap_pct": 0.0,
            "returning_starter_pct": 0.0, "retained_pair_pct": 0.0,
            "projected_starter_quality_score": 0.0,
            "projected_starter_availability_score": 0.0,
            "projected_starter_confidence": 0.0,
        }.items(): teams[col] = default
    teams["ol_continuity_adjustment"] = 0.0
    teams["continuity_applied_to_quality_flag"] = 0
    return teams, units, audit, completeness


def validate(teams: pd.DataFrame, units: pd.DataFrame, audit: pd.DataFrame, completeness: pd.DataFrame, depth: pd.DataFrame) -> None:
    if len(teams) != 32 or teams["team"].nunique() != 32:
        raise RuntimeError(f"Expected 32 team rows, found {len(teams)}/{teams['team'].nunique()} teams")
    expected_units = 32 * len(UNIT_SLOTS)
    if len(units) != expected_units:
        raise RuntimeError(f"Expected {expected_units} unit rows, found {len(units)}")
    if units.duplicated(["team", "unit_name"]).any():
        raise RuntimeError("Duplicate team/unit rows")
    if len(completeness) != expected_units:
        raise RuntimeError("Completeness row count does not match unit rows")
    if audit.empty:
        raise RuntimeError("Player-slot audit is empty")
    required_team = ["offense_unit_rating", "defense_unit_rating", "special_teams_unit_rating", "overall_team_unit_rating"]
    if teams[required_team].isna().any().any():
        raise RuntimeError("Null composite unit ratings")
    if not np.allclose(units["unit_rating_adjusted"], units["unit_rating_normalized"], atol=1e-12):
        raise RuntimeError("Compatibility normalized rating differs from replacement-anchored rating")
    expected_floor = units["unit_name"].map(lambda name: UNIT_RATING_BOUNDS[str(name)][0])
    expected_ceiling = units["unit_name"].map(lambda name: UNIT_RATING_BOUNDS[str(name)][1])
    if not np.allclose(units["unit_rating_floor"], expected_floor, atol=1e-12):
        raise RuntimeError("Unit rating floors do not match the configured position-specific bounds")
    if not np.allclose(units["unit_rating_ceiling"], expected_ceiling, atol=1e-12):
        raise RuntimeError("Unit rating ceilings do not match the configured position-specific bounds")
    if ((units["unit_rating"] < units["unit_rating_floor"] - 1e-12) | (units["unit_rating"] > units["unit_rating_ceiling"] + 1e-12)).any():
        raise RuntimeError("A unit rating falls outside its configured position-specific bounds")
    qb_rows = units[units["unit_name"].eq("QB")]
    if not qb_rows["unit_rating_ceiling"].eq(95.0).all():
        raise RuntimeError("QB ceiling regression: QB units must use a 95-point guardrail")
    non_qb_rows = units[~units["unit_name"].eq("QB")]
    if not non_qb_rows["unit_rating_ceiling"].eq(80.0).all():
        raise RuntimeError("Non-QB unit ceilings changed unexpectedly")
    if units.groupby("unit_name")["unit_rating_normalized"].std().round(8).eq(10.0).all():
        raise RuntimeError("Every unit still has fixed standard deviation 10; variance forcing appears active")
    if not units["continuity_applied_to_quality_flag"].eq(0).all() or not teams["continuity_applied_to_quality_flag"].eq(0).all():
        raise RuntimeError("OL continuity was applied inside unit quality")
    if not teams["ol_continuity_adjustment"].eq(0.0).all():
        raise RuntimeError("OL continuity adjustment must remain zero in unit builder")
    if any(c.endswith("_x") or c.endswith("_y") for c in teams.columns) or any(c.endswith("_x") or c.endswith("_y") for c in units.columns):
        raise RuntimeError("Merge collision suffix columns detected")
    if depth["player_id"].duplicated().any():
        raise RuntimeError("Depth input contains duplicate player IDs")
    ol = depth[depth["canonical_role"].isin({"LT", "LG", "C", "RG", "RT", "OT", "OG", "OL"})]
    if not ol.empty:
        if "ol_quality_discount_applied" in ol.columns and not numeric(ol, "ol_quality_discount_applied", 0.0).eq(0).all():
            raise RuntimeError("An obsolete OL quality discount is active in the depth contract")
        if not np.allclose(ol["unit_quality_grade"], ol["performance_grade"], atol=1e-9):
            raise RuntimeError("OL quality was shrunk again before unit aggregation")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    db_path = args.db_path.resolve()
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_dir / "build_nfl_team_unit_ratings.log")
    logger.info("[UNIT] Building canonical team unit ratings")
    logger.info("[UNIT] Build ID: %s", BUILD_ID)
    logger.info("[UNIT] Version: %s", VERSION)
    logger.info("[UNIT] Database: %s", db_path)

    try:
        with sqlite3.connect(db_path) as conn:
            depth = prepare_depth(read_table(conn, DEPTH_TABLE))
            continuity = load_continuity(conn)
            teams, units, audit, completeness = build_all(depth, continuity)
            validate(teams, units, audit, completeness, depth)
            outputs = {
                TEAM_OUTPUT_TABLE: teams, UNIT_OUTPUT_TABLE: units,
                AUDIT_OUTPUT_TABLE: audit, COMPLETENESS_OUTPUT_TABLE: completeness,
            }
            for table, frame in outputs.items():
                frame.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TEAM_OUTPUT_TABLE}_team ON {TEAM_OUTPUT_TABLE}(team)")
            conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{UNIT_OUTPUT_TABLE}_team_unit ON {UNIT_OUTPUT_TABLE}(team, unit_name)")
            conn.commit()

        if not args.no_csv:
            for table, frame in outputs.items():
                frame.to_csv(output_dir / f"{table}.csv", index=False, encoding="utf-8-sig")

        logger.info("[UNIT] Depth rows consumed: %s", f"{len(depth):,}")
        logger.info("[UNIT] Teams: %s", len(teams))
        logger.info("[UNIT] Unit rows: %s", len(units))
        logger.info("[UNIT] Player-slot audit rows: %s", len(audit))
        logger.info("[UNIT] Performance table was not re-merged; depth chart is the single player contract")
        logger.info("[UNIT] Fixed-variance normalization: DISABLED")
        logger.info("[UNIT] QB rating ceiling: 95.0")
        logger.info("[UNIT] Non-QB rating ceiling: 80.0")
        logger.info("[UNIT] OL continuity applied to quality: NO")
        stats = units.groupby("unit_name").agg(
            mean_rating=("unit_rating", "mean"), std_rating=("unit_rating", "std"),
            min_rating=("unit_rating", "min"), max_rating=("unit_rating", "max"),
            avg_confidence=("unit_confidence", "mean"), avg_completeness=("unit_depth_completeness", "mean"),
        ).reset_index()
        logger.info("\n[UNIT] Replacement-anchored unit distribution:\n%s", stats.to_string(index=False))
        qb_audit = units[units["unit_name"].eq("QB")][["team", "raw_unit_quality_grade", "unit_replacement_baseline", "uncapped_unit_rating", "unit_rating", "unit_rating_ceiling"]].sort_values("unit_rating", ascending=False)
        logger.info("\n[UNIT] QB cap audit:\n%s", qb_audit.to_string(index=False))
        logger.info("\n[UNIT] Top 10 teams:\n%s", teams.sort_values("overall_team_unit_rating", ascending=False)[["overall_team_unit_rank", "team", "offense_unit_rating", "defense_unit_rating", "special_teams_unit_rating", "overall_team_unit_rating", "overall_team_unit_confidence", "ol_continuity_score", "ol_continuity_confidence"]].head(10).to_string(index=False))
        return 0
    except Exception as exc:
        logger.error("[UNIT] FAILED: %s", exc)
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
