#!/usr/bin/env python
"""Build the canonical 2026 NFL player master from the standardized roster.

Primary output
--------------
SQLite/CSV: nfl_player_master_2026

Audit outputs
-------------
SQLite/CSV: nfl_player_master_duplicate_audit_2026
SQLite/CSV: nfl_player_master_identity_issues_2026
SQLite/CSV: nfl_player_master_build_audit_2026

Identity policy
---------------
- GSIS is the canonical working ID for nflverse player data.
- All available provider IDs are retained.
- Team is never part of permanent identity.
- Rows without GSIS remain visible for audit, but cannot silently impersonate a
  canonical player.
- Duplicate canonical IDs are deterministically resolved and fully audited.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import sqlalchemy as sql


SEASON = 2026
VERSION = "v2_1_canonical_gsis_master_explicit_audit_schema"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DEFAULT_DB_PATH = Path(
    r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"
)

RAW_ROSTER_TABLE = "nfl_rosters_2026_raw"
OUTPUT_TABLE = "nfl_player_master_2026"
DUPLICATE_AUDIT_TABLE = "nfl_player_master_duplicate_audit_2026"
IDENTITY_ISSUE_TABLE = "nfl_player_master_identity_issues_2026"
BUILD_AUDIT_TABLE = "nfl_player_master_build_audit_2026"

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

POSITION_GROUP_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB",
    "WR": "WR_TE", "TE": "WR_TE",
    "C": "OL", "G": "OL", "OG": "OL", "T": "OL", "OT": "OL",
    "LT": "OL", "RT": "OL", "LG": "OL", "RG": "OL", "OL": "OL",
    "DE": "EDGE", "EDGE": "EDGE", "ED": "EDGE",
    "OLB": "LB_EDGE", "LB_EDGE": "LB_EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL",
    "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "NB": "DB", "SLOT": "DB", "S": "DB", "FS": "DB",
    "SS": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST", "KR": "ST", "PR": "ST",
}

TEAM_ALIASES = {
    "ARZ": "ARI", "GNB": "GB", "HST": "HOU", "JAC": "JAX", "KAN": "KC",
    "KCC": "KC", "LA": "LAR", "STL": "LAR", "SD": "LAC", "SDG": "LAC",
    "LVR": "LV", "OAK": "LV", "NWE": "NE", "NOR": "NO", "SFO": "SF",
    "TAM": "TB", "WSH": "WAS", "WFT": "WAS", "BLT": "BAL", "CLV": "CLE",
}

ACTIVE_STATUS_PRIORITY = {
    "ACT": 100, "ACTIVE": 100, "A": 100,
    "RES": 80, "RESERVE": 80,
    "PUP": 70, "NFI": 70,
    "PS": 60, "PRACTICE SQUAD": 60,
    "IR": 50, "INJURED RESERVE": 50,
    "CUT": 10, "WAIVED": 10, "RELEASED": 10,
}

NULL_STRINGS = {"", "nan", "none", "null", "<na>", "nat"}

FINAL_COLUMNS = [
    "season",
    "player_id",
    "canonical_player_id",
    "gsis_id",
    "pfr_id",
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
    "clean_name_with_suffix",
    "first_last_name",
    "initial_last_key",
    "compact_name",
    "team",
    "position",
    "depth_chart_position",
    "position_group",
    "status",
    "age",
    "years_exp",
    "birth_date",
    "college",
    "height",
    "weight",
    "jersey_number",
    "rookie_year",
    "entry_year",
    "draft_club",
    "draft_number",
    "canonical_key",
    "aliases",
    "identity_quality_flag",
    "identity_resolution_method",
    "source_row_count",
    "source",
    "identity_version",
    "date_imported",
]


DUPLICATE_AUDIT_COLUMNS = [
    "duplicate_reason",
    "selected_flag",
    *ID_COLUMNS,
    "player_name",
    "first_name",
    "last_name",
    "football_name",
    "team",
    "position",
    "depth_chart_position",
    "status",
    "age",
    "years_exp",
    "birth_date",
    "college",
    "height",
    "weight",
    "jersey_number",
    "rookie_year",
    "entry_year",
    "draft_club",
    "draft_number",
    "source",
    "source_row_number",
    "status_priority",
    "identity_completeness",
    "stable_row_order",
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
    return None if text.lower() in NULL_STRINGS else text


def clean_id(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    text = text.replace("\u200b", "").replace("\ufeff", "").strip()
    if re.fullmatch(r"\d+\.0", text):
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


def position_group(value: Any) -> str:
    position = normalize_position(value)
    return POSITION_GROUP_MAP.get(position or "", "OTHER")


def ascii_text(value: Any) -> Optional[str]:
    text = clean_scalar(value)
    if text is None:
        return None
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def split_name_suffix(value: Any) -> tuple[Optional[str], Optional[str]]:
    text = ascii_text(value)
    if text is None:
        return None, None
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text).strip()
    tokens = [token for token in cleaned.split() if token]
    suffix = None
    if tokens and tokens[-1].lower() in {"jr", "sr", "ii", "iii", "iv", "v"}:
        suffix = tokens.pop(-1).upper()
    return " ".join(tokens) or None, suffix


def normalize_name(value: Any, preserve_suffix: bool = False) -> Optional[str]:
    base, suffix = split_name_suffix(value)
    if base is None:
        return None
    normalized = re.sub(r"[^A-Z0-9 ]+", " ", base.upper())
    normalized = " ".join(normalized.split())
    if preserve_suffix and suffix:
        normalized = f"{normalized} {suffix}"
    return normalized or None


def compact_name(value: Any) -> Optional[str]:
    normalized = normalize_name(value)
    return normalized.replace(" ", "") if normalized else None


def first_last_name(value: Any) -> Optional[str]:
    normalized = normalize_name(value)
    if not normalized:
        return None
    parts = normalized.split()
    return parts[0] if len(parts) == 1 else f"{parts[0]} {parts[-1]}"


def initial_last_key(value: Any) -> Optional[str]:
    normalized = normalize_name(value)
    if not normalized:
        return None
    parts = normalized.split()
    return parts[0] if len(parts) == 1 else f"{parts[0][0]}{parts[-1]}"


def first_non_null(values: Iterable[Any]) -> Any:
    for value in values:
        if clean_scalar(value) is not None:
            return value
    return None


def status_score(value: Any) -> int:
    text = clean_scalar(value)
    if text is None:
        return 0
    return ACTIVE_STATUS_PRIORITY.get(text.upper(), 40)


def read_table(engine, table_name: str) -> pd.DataFrame:
    with engine.connect() as connection:
        frame = pd.read_sql(sql.text(f'SELECT * FROM "{table_name}"'), connection)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out


def choose_canonical_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = ensure_columns(raw, [*ID_COLUMNS, "player_name", "team", "position", "status", "source_row_number"])
    for column in ID_COLUMNS:
        frame[column] = frame[column].map(clean_id)
    frame["team"] = frame["team"].map(normalize_team)
    frame["position"] = frame["position"].map(normalize_position)
    frame["player_name"] = frame["player_name"].map(clean_scalar)
    frame["status_priority"] = frame["status"].map(status_score)
    frame["identity_completeness"] = frame[[*ID_COLUMNS, "player_name", "team", "position"]].notna().sum(axis=1)
    frame["stable_row_order"] = range(len(frame))

    canonical_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []

    with_gsis = frame[frame["gsis_id"].notna()].copy()
    without_gsis = frame[frame["gsis_id"].isna()].copy()

    for gsis_id, group in with_gsis.groupby("gsis_id", sort=True):
        group = group.sort_values(
            ["status_priority", "identity_completeness", "stable_row_order"],
            ascending=[False, False, True],
        )
        chosen = group.head(1).copy()
        chosen["source_row_count"] = len(group)
        canonical_parts.append(chosen)
        if len(group) > 1:
            audit = group.copy()
            audit.insert(0, "duplicate_reason", "duplicate_gsis_id")
            audit.insert(1, "selected_flag", 0)
            audit.loc[chosen.index, "selected_flag"] = 1
            audit_parts.append(audit)

    # Unresolved rows remain separate; deduplicate only exact roster identity.
    unresolved_key = ["team", "position", "player_name", "birth_date", "pfr_id"]
    without_gsis = ensure_columns(without_gsis, unresolved_key)
    for _, group in without_gsis.groupby(unresolved_key, dropna=False, sort=False):
        group = group.sort_values(
            ["status_priority", "identity_completeness", "stable_row_order"],
            ascending=[False, False, True],
        )
        chosen = group.head(1).copy()
        chosen["source_row_count"] = len(group)
        canonical_parts.append(chosen)
        if len(group) > 1:
            audit = group.copy()
            audit.insert(0, "duplicate_reason", "duplicate_unresolved_identity")
            audit.insert(1, "selected_flag", 0)
            audit.loc[chosen.index, "selected_flag"] = 1
            audit_parts.append(audit)

    canonical = pd.concat(canonical_parts, ignore_index=True, sort=False) if canonical_parts else pd.DataFrame()
    if audit_parts:
        audit = pd.concat(audit_parts, ignore_index=True, sort=False)
        audit = ensure_columns(audit, DUPLICATE_AUDIT_COLUMNS)[DUPLICATE_AUDIT_COLUMNS]
    else:
        audit = pd.DataFrame(columns=DUPLICATE_AUDIT_COLUMNS)
    return canonical, audit


def build_aliases(row: pd.Series) -> Optional[str]:
    values: list[str] = []
    for value in [
        row.get("player_name"), row.get("football_name"), row.get("clean_name"),
        row.get("clean_name_with_suffix"), row.get("first_last_name"),
        row.get("initial_last_key"), row.get("compact_name"),
    ]:
        text = clean_scalar(value)
        if text and text not in values:
            values.append(text)
    return "|".join(values) if values else None


def build_master(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical, duplicate_audit = choose_canonical_rows(raw)
    canonical = ensure_columns(
        canonical,
        [*ID_COLUMNS, "player_name", "first_name", "last_name", "football_name", "team",
         "position", "depth_chart_position", "status", "age", "years_exp", "birth_date",
         "college", "height", "weight", "jersey_number", "rookie_year", "entry_year",
         "draft_club", "draft_number", "source", "source_row_count"],
    )

    out = pd.DataFrame(index=canonical.index)
    out["season"] = SEASON
    for column in ID_COLUMNS:
        out[column] = canonical[column].map(clean_id)
    out["player_id"] = out["gsis_id"]
    out["canonical_player_id"] = out["gsis_id"]
    out["pfr_player_id"] = out["pfr_id"]

    out["player_name"] = canonical["player_name"].map(clean_scalar)
    out["first_name"] = canonical["first_name"].map(clean_scalar)
    out["last_name"] = canonical["last_name"].map(clean_scalar)
    out["football_name"] = canonical["football_name"].map(clean_scalar)
    out["clean_name"] = out["player_name"].map(normalize_name)
    out["clean_name_with_suffix"] = out["player_name"].map(lambda value: normalize_name(value, True))
    out["first_last_name"] = out["player_name"].map(first_last_name)
    out["initial_last_key"] = out["player_name"].map(initial_last_key)
    out["compact_name"] = out["player_name"].map(compact_name)

    out["team"] = canonical["team"].map(normalize_team)
    out["position"] = canonical["position"].map(normalize_position)
    out["depth_chart_position"] = canonical["depth_chart_position"].map(normalize_position)
    out["position_group"] = out["position"].map(position_group)
    out["status"] = canonical["status"].map(clean_scalar)

    for column in ["age", "years_exp", "height", "weight", "jersey_number", "rookie_year", "entry_year", "draft_number"]:
        out[column] = pd.to_numeric(canonical[column], errors="coerce")
    out["birth_date"] = canonical["birth_date"].map(clean_scalar)
    out["college"] = canonical["college"].map(clean_scalar)
    out["draft_club"] = canonical["draft_club"].map(normalize_team)

    provider_fallback = (
        "PFR:" + out["pfr_id"].fillna("")
    ).where(out["pfr_id"].notna())
    name_dob_fallback = (
        "NAME_DOB:" + out["clean_name"].fillna("UNKNOWN") + ":" + out["birth_date"].fillna("UNKNOWN")
    )
    out["canonical_key"] = (
        "GSIS:" + out["gsis_id"].fillna("")
    ).where(out["gsis_id"].notna()).combine_first(provider_fallback).combine_first(name_dob_fallback)

    issue_flags: list[str] = []
    methods: list[str] = []
    for row in out.itertuples(index=False):
        issues: list[str] = []
        if row.player_name is None:
            issues.append("missing_name")
        if row.team is None:
            issues.append("missing_team")
        if row.position is None:
            issues.append("missing_position")
        if row.position_group == "OTHER":
            issues.append("unknown_position_group")
        if row.gsis_id is None:
            issues.append("missing_gsis_id")
        issue_flags.append("|".join(issues) if issues else "ok")
        if row.gsis_id is not None:
            methods.append("canonical_gsis")
        elif row.pfr_id is not None:
            methods.append("unresolved_pfr_only")
        else:
            methods.append("unresolved_name_identity")
    out["identity_quality_flag"] = issue_flags
    out["identity_resolution_method"] = methods
    out["source_row_count"] = pd.to_numeric(canonical["source_row_count"], errors="coerce").fillna(1).astype(int)
    out["source"] = canonical["source"].map(clean_scalar)
    out["aliases"] = out.apply(build_aliases, axis=1)
    out["identity_version"] = VERSION
    out["date_imported"] = dt.datetime.now().isoformat(timespec="seconds")

    for column in FINAL_COLUMNS:
        if column not in out.columns:
            out[column] = None
    out = out[FINAL_COLUMNS].sort_values(
        ["team", "position_group", "position", "player_name"], na_position="last"
    ).reset_index(drop=True)

    duplicate_ids = out[out["player_id"].notna()]["player_id"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = out.loc[duplicate_ids, ["player_id", "player_name", "team", "position"]]
        raise RuntimeError("Canonical master still contains duplicate GSIS IDs:\n" + examples.head(25).to_string(index=False))

    return out, duplicate_audit


def build_audit(master: pd.DataFrame, raw_rows: int) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": SEASON,
            "raw_roster_rows": raw_rows,
            "master_rows": len(master),
            "canonical_gsis_players": master["player_id"].nunique(dropna=True),
            "unresolved_players": int(master["player_id"].isna().sum()),
            "teams": master["team"].nunique(dropna=True),
            "identity_issue_rows": int(master["identity_quality_flag"].ne("ok").sum()),
            "unknown_position_rows": int(master["position_group"].eq("OTHER").sum()),
            "identity_version": VERSION,
            "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
    ])


def configure_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nfl_player_master")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(
        log_dir / f"build_nfl_player_master_{dt.datetime.now():%Y%m%d_%H%M%S}.log",
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
    parser = argparse.ArgumentParser(description="Build the canonical 2026 NFL player master.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.project_root / "outputs"
    log_dir = args.project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(log_dir)
    engine = sql.create_engine(f"sqlite:///{args.db_path}", pool_pre_ping=True)

    raw = read_table(engine, RAW_ROSTER_TABLE)
    logger.info("[PLAYER_MASTER] Raw roster rows: %s", f"{len(raw):,}")
    master, duplicate_audit = build_master(raw)
    issues = master[master["identity_quality_flag"].ne("ok")].copy()
    build_audit_df = build_audit(master, len(raw))

    if master["team"].nunique(dropna=True) < 32:
        raise RuntimeError(f"Player master has only {master['team'].nunique(dropna=True)} teams.")

    save_frame(engine, output_dir, OUTPUT_TABLE, master)
    save_frame(engine, output_dir, DUPLICATE_AUDIT_TABLE, duplicate_audit)
    save_frame(engine, output_dir, IDENTITY_ISSUE_TABLE, issues)
    save_frame(engine, output_dir, BUILD_AUDIT_TABLE, build_audit_df)

    logger.info("[PLAYER_MASTER] Master rows: %s", f"{len(master):,}")
    logger.info("[PLAYER_MASTER] Canonical GSIS players: %s", f"{master['player_id'].nunique(dropna=True):,}")
    logger.info("[PLAYER_MASTER] Unresolved players: %s", f"{master['player_id'].isna().sum():,}")
    logger.info("[PLAYER_MASTER] Duplicate audit rows: %s", f"{len(duplicate_audit):,}")
    logger.info("[PLAYER_MASTER] Identity issue rows: %s", f"{len(issues):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
