"""Build canonical 2026 quarterback ratings from 2022-2025 efficiency data.

Canonical inputs
----------------
1. nfl_player_master_2026
2. nfl_player_advanced_stats_2022_2025
3. Optional local files: inputs/rbsdm_qb_<season>.csv

Outputs
-------
1. nfl_qb_rbsdm_ratings_raw
2. nfl_qb_rbsdm_ratings_2026
3. nfl_qb_rbsdm_local_unmatched_audit

The advanced-history table is the identity authority. Local RBSDM CSVs may
supply richer metric values, but they are accepted only after they resolve to a
single canonical GSIS player-season. No name-only row is written to the final
2026 table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sqlalchemy as sql

SEASON = 2026
HIST_SEASONS = [2022, 2023, 2024, 2025]
RECENT_SEASON = 2025
BUILD_ID = "NFL_RBSDM_QB_2026_CANONICAL_V3"
RATING_VERSION = "v3_single_metric_stabilization_unshrunk_composite"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")

MASTER_TABLE = "nfl_player_master_2026"
ADVANCED_HISTORY_TABLE = "nfl_player_advanced_stats_2022_2025"
RAW_OUTPUT_TABLE = "nfl_qb_rbsdm_ratings_raw"
OUTPUT_TABLE = "nfl_qb_rbsdm_ratings_2026"
UNMATCHED_TABLE = "nfl_qb_rbsdm_local_unmatched_audit"

SEASON_WEIGHTS = {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40}

RAW_COLUMNS = [
    "season", "player_id", "qb_name", "qb_name_clean", "team", "plays",
    "epa_per_play", "success_rate", "cpoe", "completion_pct",
    "epa_cpoe_composite", "epa_score", "success_score", "cpoe_score",
    "qb_season_score_raw", "qb_season_score", "qualified_qb_volume",
    "metric_source", "rating_version", "date_imported",
]

FINAL_COLUMNS = [
    "player_id", "qb_name", "qb_name_clean", "recent_team",
    "seasons_observed", "qb_plays_3yr", "qb_plays_4yr", "qb_plays_l1",
    "qb_qualified_seasons", "qb_qualified_recent",
    "qb_multi_year_score", "qb_recent_score", "qb_avg_score",
    "qb_rating_override_raw", "qb_rating_override",
    "qb_multi_year_available", "qb_recent_available",
    "qb_multi_year_rank", "qb_recent_rank",
    "qb_multi_year_percentile", "qb_recent_percentile",
    "qb_epa_per_play_l1", "qb_success_rate_l1", "qb_cpoe_l1",
    "qb_completion_pct_l1", "qb_epa_cpoe_composite_l1",
    "rating_source", "rating_version", "date_imported",
]

UNMATCHED_COLUMNS = [
    "season", "qb_name", "qb_name_clean", "team", "plays",
    "source_file", "unmatched_reason",
]

TEAM_ALIASES = {
    "ARI": "ARI", "ARZ": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL",
    "DEN": "DEN", "DET": "DET", "GB": "GB", "GNB": "GB", "HOU": "HOU",
    "IND": "IND", "JAC": "JAX", "JAX": "JAX", "KC": "KC", "KAN": "KC",
    "KCC": "KC", "LA": "LAR", "LAR": "LAR", "STL": "LAR", "LAC": "LAC",
    "SD": "LAC", "LV": "LV", "LVR": "LV", "OAK": "LV", "MIA": "MIA",
    "MIN": "MIN", "NE": "NE", "NWE": "NE", "NO": "NO", "NOR": "NO",
    "NYG": "NYG", "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF": "SF", "SFO": "SF", "TB": "TB", "TAM": "TB", "TEN": "TEN",
    "WAS": "WAS", "WSH": "WAS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def get_engine(db_path: Path) -> sql.Engine:
    return sql.create_engine(f"sqlite:///{db_path}", pool_pre_ping=True)


def table_exists(engine: sql.Engine, table_name: str) -> bool:
    query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:name"
    with engine.connect() as conn:
        return conn.execute(sql.text(query), {"name": table_name}).fetchone() is not None


def read_table(engine: sql.Engine, table_name: str) -> pd.DataFrame:
    if not table_exists(engine, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    with engine.connect() as conn:
        frame = pd.read_sql(sql.text(f'SELECT * FROM "{table_name}"'), conn)
    frame.columns = [clean_col(c) for c in frame.columns]
    return frame


def clean_col(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("%", "pct").replace("+", "_plus_")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def clean_id(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("\u200b", "").replace("\ufeff", "")
    return None if text.lower() in {"", "nan", "none", "null", "na"} else text


def clean_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = re.sub(r"\b(JR|SR|II|III|IV|V)\b\.?", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return " ".join(text.split())


def normalize_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).upper().strip()
    return TEAM_ALIASES.get(text, text or None)


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def percentile(series: pd.Series, eligible: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(50.0, index=series.index, dtype=float)
    valid = eligible & values.notna()
    if valid.sum() >= 2:
        out.loc[valid] = values.loc[valid].rank(method="average", pct=True) * 100.0
    return out.clip(0.0, 100.0)


def load_master_qbs(engine: sql.Engine) -> pd.DataFrame:
    master = read_table(engine, MASTER_TABLE)
    for col in ["player_id", "player_name", "team", "position", "position_group"]:
        if col not in master.columns:
            master[col] = None
    master["player_id"] = master["player_id"].map(clean_id)
    master["team"] = master["team"].map(normalize_team)
    master["position_group"] = master["position_group"].astype(str).str.upper().str.strip()
    qbs = master[(master["position_group"] == "QB") & master["player_id"].notna()].copy()
    qbs = qbs[["player_id", "player_name", "team", "position", "position_group"]]
    qbs = qbs.drop_duplicates("player_id", keep="first")
    if qbs["player_id"].duplicated().any():
        raise RuntimeError("Duplicate canonical QB player_id values remain in player master.")
    return qbs


def load_advanced_qb(engine: sql.Engine) -> pd.DataFrame:
    advanced = read_table(engine, ADVANCED_HISTORY_TABLE)
    required = ["season", "player_id", "player_name", "team", "position_group"]
    for col in required:
        if col not in advanced.columns:
            advanced[col] = None

    advanced["season"] = pd.to_numeric(advanced["season"], errors="coerce")
    advanced["player_id"] = advanced["player_id"].map(clean_id)
    advanced["team"] = advanced["team"].map(normalize_team)
    advanced["position_group"] = advanced["position_group"].astype(str).str.upper().str.strip()

    plays = numeric(advanced, "qb_dropbacks")
    mask = (
        advanced["season"].isin(HIST_SEASONS)
        & advanced["player_id"].notna()
        & ((advanced["position_group"] == "QB") | (plays > 0))
    )
    advanced = advanced.loc[mask].copy()

    out = pd.DataFrame(index=advanced.index)
    out["season"] = advanced["season"].astype(int)
    out["player_id"] = advanced["player_id"]
    out["qb_name"] = advanced["player_name"].astype(str).str.strip()
    out["qb_name_clean"] = out["qb_name"].map(clean_name)
    out["team"] = advanced["team"]
    out["plays"] = plays.loc[advanced.index]
    out["epa_per_play"] = numeric(advanced, "qb_epa_per_dropback").loc[advanced.index]
    out["success_rate"] = numeric(advanced, "qb_success_rate").loc[advanced.index]
    out["cpoe"] = numeric(advanced, "qb_cpoe").loc[advanced.index]
    out["completion_pct"] = numeric(advanced, "completion_pct", np.nan).loc[advanced.index]
    out["epa_cpoe_composite"] = out["epa_per_play"] + out["cpoe"] / 100.0
    out["metric_source"] = "advanced_history"

    # One canonical row per player-season. Prefer the row with the most dropbacks.
    out = out.sort_values(["season", "player_id", "plays"], ascending=[True, True, False])
    out = out.drop_duplicates(["season", "player_id"], keep="first").reset_index(drop=True)
    return out


def standardize_local_file(path: Path, season: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [clean_col(c) for c in raw.columns]

    name_col = first_present(raw.columns, ["player", "player_name", "name", "passer", "qb", "full_name"])
    if name_col is None:
        raise RuntimeError(f"Could not identify QB name column in {path}. Columns: {list(raw.columns)}")

    team_col = first_present(raw.columns, ["team", "posteam", "recent_team", "tm"])
    plays_col = first_present(raw.columns, ["plays", "n", "dropbacks", "db", "attempts", "att"])
    epa_col = first_present(raw.columns, ["epa_per_play", "epa_play", "epa_per_dropback", "epa_db", "epa", "adj_epa_play", "adj_epa_per_play"])
    success_col = first_present(raw.columns, ["success_rate", "success", "sr"])
    cpoe_col = first_present(raw.columns, ["cpoe", "completion_pct_over_expected"])
    completion_col = first_present(raw.columns, ["comp_pct", "completion_pct", "cmp_pct", "completion_percentage"])
    composite_col = first_present(raw.columns, ["epa_plus_cpoe", "epa_cpoe", "composite", "epa_cpoe_composite"])
    id_col = first_present(raw.columns, ["player_id", "gsis_id", "gsis", "nfl_id"])

    out = pd.DataFrame(index=raw.index)
    out["season"] = season
    out["player_id"] = raw[id_col].map(clean_id) if id_col else None
    out["qb_name"] = raw[name_col].astype(str).str.strip()
    out["qb_name_clean"] = out["qb_name"].map(clean_name)
    out["team"] = raw[team_col].map(normalize_team) if team_col else None
    out["plays"] = numeric(raw, plays_col) if plays_col else 0.0
    out["epa_per_play"] = numeric(raw, epa_col, np.nan) if epa_col else np.nan
    out["success_rate"] = numeric(raw, success_col, np.nan) if success_col else np.nan
    out["cpoe"] = numeric(raw, cpoe_col, np.nan) if cpoe_col else np.nan
    out["completion_pct"] = numeric(raw, completion_col, np.nan) if completion_col else np.nan
    out["epa_cpoe_composite"] = numeric(raw, composite_col, np.nan) if composite_col else np.nan
    out["source_file"] = str(path)
    return out


def map_local_to_canonical(
    local: pd.DataFrame,
    advanced: pd.DataFrame,
    valid_canonical_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if local.empty:
        return local.copy(), pd.DataFrame(columns=UNMATCHED_COLUMNS)

    reference = advanced[["season", "player_id", "qb_name_clean", "team", "plays"]].copy()
    reference = reference.rename(columns={"plays": "advanced_plays"})

    resolved_parts: list[pd.DataFrame] = []
    unresolved = local.copy()

    # Direct canonical IDs win when they exist in the advanced identity set.
    direct_ids = set(reference["player_id"].dropna()) | set(valid_canonical_ids)
    direct_mask = unresolved["player_id"].isin(direct_ids)
    if direct_mask.any():
        resolved_parts.append(unresolved.loc[direct_mask].copy())
        unresolved = unresolved.loc[~direct_mask].copy()

    # Exact season + normalized name + team.
    if not unresolved.empty:
        exact_ref = reference.dropna(subset=["team"]).drop_duplicates(
            ["season", "qb_name_clean", "team"], keep=False
        )
        exact = unresolved.drop(columns=["player_id"], errors="ignore").merge(
            exact_ref[["season", "qb_name_clean", "team", "player_id"]],
            on=["season", "qb_name_clean", "team"], how="left", validate="many_to_one"
        )
        matched = exact["player_id"].notna()
        if matched.any():
            resolved_parts.append(exact.loc[matched].copy())
        unresolved = exact.loc[~matched].drop(columns=["player_id"], errors="ignore").copy()
        unresolved["player_id"] = None

    # Unique season + normalized name fallback.
    if not unresolved.empty:
        counts = reference.groupby(["season", "qb_name_clean"])["player_id"].nunique()
        unique_keys = counts[counts == 1].reset_index()[["season", "qb_name_clean"]]
        unique_ref = reference.merge(unique_keys, on=["season", "qb_name_clean"], how="inner")
        unique_ref = unique_ref.drop_duplicates(["season", "qb_name_clean"])
        name_match = unresolved.drop(columns=["player_id"], errors="ignore").merge(
            unique_ref[["season", "qb_name_clean", "player_id"]],
            on=["season", "qb_name_clean"], how="left", validate="many_to_one"
        )
        matched = name_match["player_id"].notna()
        if matched.any():
            resolved_parts.append(name_match.loc[matched].copy())
        unresolved = name_match.loc[~matched].copy()

    resolved = pd.concat(resolved_parts, ignore_index=True, sort=False) if resolved_parts else pd.DataFrame(columns=local.columns)
    if not resolved.empty:
        resolved["metric_source"] = "local_rbsdm"
        resolved = resolved.sort_values(["season", "player_id", "plays"], ascending=[True, True, False])
        resolved = resolved.drop_duplicates(["season", "player_id"], keep="first")

    if unresolved.empty:
        audit = pd.DataFrame(columns=UNMATCHED_COLUMNS)
    else:
        unresolved["unmatched_reason"] = "no_unique_canonical_player_season_match"
        for col in UNMATCHED_COLUMNS:
            if col not in unresolved.columns:
                unresolved[col] = None
        audit = unresolved[UNMATCHED_COLUMNS].copy()
    return resolved, audit


def overlay_local_metrics(advanced: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return advanced.copy()

    metric_cols = ["qb_name", "qb_name_clean", "team", "plays", "epa_per_play", "success_rate", "cpoe", "completion_pct", "epa_cpoe_composite"]
    local_keep = local[["season", "player_id", *metric_cols]].copy()
    local_keep = local_keep.rename(columns={col: f"local_{col}" for col in metric_cols})
    local_keep["_local_metric_row"] = 1

    combined = advanced.merge(
        local_keep,
        on=["season", "player_id"],
        how="outer",
        validate="one_to_one",
    )
    for col in metric_cols:
        local_col = f"local_{col}"
        if col not in combined.columns:
            combined[col] = np.nan
        combined[col] = combined[local_col].combine_first(combined[col])
        combined = combined.drop(columns=[local_col])

    combined["metric_source"] = np.where(
        pd.to_numeric(combined["_local_metric_row"], errors="coerce").fillna(0) > 0,
        "local_rbsdm",
        "advanced_history",
    )
    return combined.drop(columns=["_local_metric_row"])


def score_qb_seasons(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["plays", "epa_per_play", "success_rate", "cpoe", "completion_pct", "epa_cpoe_composite"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["plays"] = out["plays"].fillna(0.0).clip(lower=0.0)

    scored_parts: list[pd.DataFrame] = []
    for season, group in out.groupby("season", sort=True):
        group = group.copy()
        eligible = group["plays"] >= 30
        sample_weight = group["plays"] / (group["plays"] + 150.0)

        for metric in ["epa_per_play", "success_rate", "cpoe"]:
            values = group[metric]
            prior = values.loc[eligible & values.notna()].median()
            if pd.isna(prior):
                prior = values.median()
            if pd.isna(prior):
                prior = 0.0
            group[f"{metric}_shrunk"] = prior + sample_weight * (values.fillna(prior) - prior)

        group["epa_score"] = percentile(group["epa_per_play_shrunk"], eligible)
        group["success_score"] = percentile(group["success_rate_shrunk"], eligible)
        group["cpoe_score"] = percentile(group["cpoe_shrunk"], eligible)
        group["qualified_qb_volume"] = (group["plays"] >= 100).astype(int)
        group["qb_season_score_raw"] = (
            0.55 * group["epa_score"]
            + 0.25 * group["success_score"]
            + 0.20 * group["cpoe_score"]
        )
        # The inputs above have already received the one allowed small-sample
        # stabilization at the metric level. Do not shrink the completed
        # seasonal composite a second time; final player confidence is applied
        # once, downstream, against the QB replacement baseline.
        group["qb_season_score"] = np.where(
            group["plays"] > 0,
            group["qb_season_score_raw"],
            0.0,
        )
        scored_parts.append(group)

    scored = pd.concat(scored_parts, ignore_index=True, sort=False) if scored_parts else pd.DataFrame(columns=RAW_COLUMNS)
    scored["rating_version"] = RATING_VERSION
    scored["date_imported"] = pd.Timestamp(dt.date.today())
    for col in RAW_COLUMNS:
        if col not in scored.columns:
            scored[col] = None
    return scored[RAW_COLUMNS]


def build_current_qb(master_qbs: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    hist = scored.copy()
    hist["base_season_weight"] = hist["season"].map(SEASON_WEIGHTS).fillna(0.0)
    hist["observed_weight"] = np.where(hist["plays"] > 0, hist["base_season_weight"], 0.0)
    hist["weighted_score"] = hist["qb_season_score"] * hist["observed_weight"]

    multi = hist.groupby("player_id", dropna=False).agg(
        seasons_observed=("season", lambda s: int(pd.Series(s).nunique())),
        qb_plays_4yr=("plays", "sum"),
        qb_qualified_seasons=("qualified_qb_volume", "sum"),
        available_weight=("observed_weight", "sum"),
        weighted_score=("weighted_score", "sum"),
        qb_avg_score=("qb_season_score", lambda s: float(pd.Series(s)[pd.Series(s) > 0].mean()) if (pd.Series(s) > 0).any() else 0.0),
        rating_sources=("metric_source", lambda s: ",".join(sorted(set(str(v) for v in s if pd.notna(v))))),
    ).reset_index()

    # Preserve the fixed 2022-2025 season weights and renormalize over seasons
    # actually observed. The weighted efficiency composite is the talent
    # estimate; it is intentionally not shrunk again inside this loader.
    multi["qb_multi_year_score"] = np.where(
        multi["available_weight"] > 0,
        multi["weighted_score"] / multi["available_weight"],
        0.0,
    )
    multi["qb_multi_year_available"] = (multi["qb_plays_4yr"] >= 50).astype(int)
    multi["qb_plays_3yr"] = hist[hist["season"].isin([2023, 2024, 2025])].groupby("player_id")["plays"].sum().reindex(multi["player_id"]).fillna(0.0).to_numpy()

    recent_cols = [
        "player_id", "qb_name", "team", "plays", "epa_per_play", "success_rate", "cpoe",
        "completion_pct", "epa_cpoe_composite", "qb_season_score", "qualified_qb_volume",
    ]
    recent = hist[hist["season"] == RECENT_SEASON][recent_cols].copy()
    recent = recent.sort_values(["player_id", "plays"], ascending=[True, False]).drop_duplicates("player_id")
    recent = recent.rename(columns={
        "qb_name": "recent_qb_name", "team": "historical_recent_team", "plays": "qb_plays_l1",
        "epa_per_play": "qb_epa_per_play_l1", "success_rate": "qb_success_rate_l1",
        "cpoe": "qb_cpoe_l1", "completion_pct": "qb_completion_pct_l1",
        "epa_cpoe_composite": "qb_epa_cpoe_composite_l1", "qb_season_score": "qb_recent_score",
        "qualified_qb_volume": "qb_qualified_recent",
    })
    recent["qb_recent_available"] = (recent["qb_plays_l1"] >= 30).astype(int)

    out = master_qbs.merge(multi, on="player_id", how="left", validate="one_to_one")
    out = out.merge(recent, on="player_id", how="left", validate="one_to_one")

    out["qb_name"] = out["player_name"]
    out["qb_name_clean"] = out["qb_name"].map(clean_name)
    out["recent_team"] = out["team"]

    numeric_cols = [
        "seasons_observed", "qb_plays_3yr", "qb_plays_4yr", "qb_plays_l1",
        "qb_qualified_seasons", "qb_qualified_recent", "qb_multi_year_score",
        "qb_recent_score", "qb_avg_score", "qb_multi_year_available", "qb_recent_available",
        "qb_epa_per_play_l1", "qb_success_rate_l1", "qb_cpoe_l1",
        "qb_completion_pct_l1", "qb_epa_cpoe_composite_l1",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # Compatibility/audit field only. Production player scoring performs its
    # confidence-adjusted recent blend and single replacement shrink downstream.
    out["qb_rating_override_raw"] = np.where(
        out["qb_recent_available"] > 0,
        0.65 * out["qb_multi_year_score"] + 0.35 * out["qb_recent_score"],
        out["qb_multi_year_score"],
    )
    out["qb_rating_override"] = out["qb_rating_override_raw"].clip(0.0, 100.0)

    multi_mask = out["qb_multi_year_available"] > 0
    recent_mask = out["qb_recent_available"] > 0
    out["qb_multi_year_rank"] = np.nan
    out["qb_recent_rank"] = np.nan
    out["qb_multi_year_percentile"] = np.nan
    out["qb_recent_percentile"] = np.nan
    if multi_mask.any():
        out.loc[multi_mask, "qb_multi_year_rank"] = out.loc[multi_mask, "qb_multi_year_score"].rank(method="min", ascending=False)
        out.loc[multi_mask, "qb_multi_year_percentile"] = out.loc[multi_mask, "qb_multi_year_score"].rank(method="average", pct=True) * 100.0
    if recent_mask.any():
        out.loc[recent_mask, "qb_recent_rank"] = out.loc[recent_mask, "qb_recent_score"].rank(method="min", ascending=False)
        out.loc[recent_mask, "qb_recent_percentile"] = out.loc[recent_mask, "qb_recent_score"].rank(method="average", pct=True) * 100.0

    out["rating_source"] = out.get("rating_sources", pd.Series(index=out.index, dtype=object)).fillna("no_historical_qb_data")
    out["rating_version"] = RATING_VERSION
    out["date_imported"] = pd.Timestamp(dt.date.today())

    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    final = out[FINAL_COLUMNS].copy()
    if final["player_id"].duplicated().any():
        raise RuntimeError("Final QB table contains duplicate canonical player_id values.")
    score_cols = ["qb_multi_year_score", "qb_recent_score", "qb_rating_override"]
    if ((final[score_cols].apply(pd.to_numeric, errors="coerce") < 0) | (final[score_cols].apply(pd.to_numeric, errors="coerce") > 100)).any().any():
        raise RuntimeError("QB score outside 0-100 range.")
    return final


def save_frame(frame: pd.DataFrame, engine: sql.Engine, table_name: str, csv_path: Path | None) -> None:
    frame.to_sql(table_name, con=engine, if_exists="replace", index=False)
    if csv_path is not None:
        frame.to_csv(csv_path, index=False)


def main() -> int:
    args = parse_args()
    project_root = args.project_root
    db_path = args.db_path
    output_dir = project_root / "outputs"
    log_dir = project_root / "logs"
    input_dir = project_root / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("[RBSDM_QB] Building canonical QB ratings")
    print(f"[RBSDM_QB] Build ID: {BUILD_ID}")
    print(f"[RBSDM_QB] Historical seasons: {HIST_SEASONS}")
    print(f"[RBSDM_QB] Database: {db_path}")

    engine = get_engine(db_path)
    master_qbs = load_master_qbs(engine)
    advanced = load_advanced_qb(engine)

    local_frames: list[pd.DataFrame] = []
    for season in HIST_SEASONS:
        path = input_dir / f"rbsdm_qb_{season}.csv"
        if path.exists():
            local = standardize_local_file(path, season)
            local_frames.append(local)
            print(f"[RBSDM_QB] Local RBSDM {season}: {len(local):,} rows from {path}")
        else:
            print(f"[RBSDM_QB] Local RBSDM {season}: not found; using advanced-history metrics")

    local_all = pd.concat(local_frames, ignore_index=True, sort=False) if local_frames else pd.DataFrame()
    valid_canonical_ids = set(master_qbs["player_id"].dropna()) | set(advanced["player_id"].dropna())
    local_mapped, unmatched = map_local_to_canonical(
        local_all,
        advanced,
        valid_canonical_ids,
    )
    combined = overlay_local_metrics(advanced, local_mapped)
    scored = score_qb_seasons(combined)
    final = build_current_qb(master_qbs, scored)

    if RECENT_SEASON not in set(pd.to_numeric(scored["season"], errors="coerce").dropna().astype(int)):
        raise RuntimeError("Final QB raw table is missing 2025.")

    raw_csv = None if args.no_csv else output_dir / "nfl_qb_rbsdm_ratings_raw.csv"
    final_csv = None if args.no_csv else output_dir / "nfl_qb_rbsdm_ratings_2026.csv"
    unmatched_csv = None if args.no_csv else output_dir / "nfl_qb_rbsdm_local_unmatched_audit.csv"
    save_frame(scored, engine, RAW_OUTPUT_TABLE, raw_csv)
    save_frame(final, engine, OUTPUT_TABLE, final_csv)
    save_frame(unmatched if not unmatched.empty else pd.DataFrame(columns=UNMATCHED_COLUMNS), engine, UNMATCHED_TABLE, unmatched_csv)

    multi_available = int((pd.to_numeric(final["qb_multi_year_available"], errors="coerce").fillna(0) > 0).sum())
    recent_available = int((pd.to_numeric(final["qb_recent_available"], errors="coerce").fillna(0) > 0).sum())
    source_counts = scored["metric_source"].value_counts(dropna=False).to_dict()

    log_path = log_dir / "load_rbsdm_qb_ratings.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Run timestamp: {dt.datetime.now().isoformat()}\n")
        handle.write(f"Build ID: {BUILD_ID}\n")
        handle.write(f"Historical rows: {len(scored)}\n")
        handle.write(f"Current QB rows: {len(final)}\n")
        handle.write(f"Multi-year available: {multi_available}\n")
        handle.write(f"Recent available: {recent_available}\n")
        handle.write(f"Local unmatched: {len(unmatched)}\n")
        handle.write(f"Sources: {source_counts}\n")

    print(f"[RBSDM_QB] Historical player-seasons: {len(scored):,}")
    print(f"[RBSDM_QB] Current roster QBs: {len(final):,}")
    print(f"[RBSDM_QB] Multi-year ratings available: {multi_available:,}/{len(final):,}")
    print(f"[RBSDM_QB] 2025 ratings available: {recent_available:,}/{len(final):,}")
    print(f"[RBSDM_QB] Local RBSDM rows unmatched: {len(unmatched):,}")
    print(f"[RBSDM_QB] Metric sources: {source_counts}")
    print(f"[RBSDM_QB] Saved table: {OUTPUT_TABLE}")
    print(f"[RBSDM_QB] Log saved: {log_path}")

    top_cols = ["qb_name", "recent_team", "qb_plays_4yr", "qb_plays_l1", "qb_multi_year_score", "qb_recent_score", "qb_rating_override"]
    print("\n[RBSDM_QB] Top 25 current QB ratings:")
    print(final.sort_values("qb_rating_override", ascending=False).head(25)[top_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
