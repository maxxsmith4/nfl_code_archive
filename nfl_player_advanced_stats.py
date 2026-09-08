# load_nfl_player_advanced_stats.py

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sql


SEASON = 2026
HIST_SEASONS = [2022, 2023, 2024, 2025]
ADVANCED_BUILD_ID = "NFL_ADVANCED_STATS_2026_V3_2_FINAL"
ADVANCED_VERSION = "v3_2_final_snap_participation_identity"

PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DB_PATH = r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite"

MASTER_TABLE = "nfl_player_master_2026"
CROSSWALK_TABLE = "nfl_player_crosswalk"

OUTPUT_TABLE = "nfl_player_advanced_stats_2022_2025"
OUTPUT_TABLE_CURRENT = "nfl_player_advanced_stats_current_roster_2026"
SNAP_ID_AUDIT_TABLE = "nfl_player_advanced_stats_snap_id_audit"

SNAP_ID_AUDIT_COLUMNS = [
    "pfr_player_id",
    "player_id",
    "player_name",
    "id_player_name",
    "position",
    "id_position",
    "canonical_id_match",
    "snap_rows",
    "first_season",
    "last_season",
    "offense_snaps",
    "defense_snaps",
    "st_snaps",
    "total_snaps",
    "position_normalized",
    "position_group",
    "ol_flag",
    "mapping_status",
    "advanced_version",
    "date_imported",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


TEAM_ALIASES = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "KAN": "KC",
    "KCC": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "STL": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "LV": "LV",
    "LVR": "LV",
    "OAK": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "SFO": "SF",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}


POSITION_SEASON_WEIGHTS = {
    "QB": {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40},
    "RB": {2022: 0.02, 2023: 0.08, 2024: 0.25, 2025: 0.65},
    "WR_TE": {2022: 0.05, 2023: 0.15, 2024: 0.30, 2025: 0.50},
    "OL": {2022: 0.15, 2023: 0.20, 2024: 0.30, 2025: 0.35},
    "EDGE": {2022: 0.10, 2023: 0.15, 2024: 0.30, 2025: 0.45},
    "DL": {2022: 0.10, 2023: 0.15, 2024: 0.30, 2025: 0.45},
    "LB_EDGE": {2022: 0.10, 2023: 0.15, 2024: 0.30, 2025: 0.45},
    "LB": {2022: 0.10, 2023: 0.15, 2024: 0.30, 2025: 0.45},
    "DB": {2022: 0.05, 2023: 0.15, 2024: 0.30, 2025: 0.50},
    "ST": {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40},
    "OTHER": {2022: 0.10, 2023: 0.20, 2024: 0.30, 2025: 0.40},
}

DEFAULT_SEASON_WEIGHTS = {
    2022: 0.10,
    2023: 0.20,
    2024: 0.30,
    2025: 0.40,
}


POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB",
    "FB": "RB",
    "WR": "WR_TE",
    "TE": "WR_TE",
    "C": "OL",
    "G": "OL",
    "OG": "OL",
    "T": "OL",
    "OT": "OL",
    "LT": "OL",
    "RT": "OL",
    "LG": "OL",
    "RG": "OL",
    "OL": "OL",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "OLB": "LB_EDGE",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",
    "ILB": "LB",
    "MLB": "LB",
    "LB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "K": "ST",
    "P": "ST",
    "LS": "ST",
}


WEEKLY_COUNTING_COLUMNS = [
    "passing_attempts", "completions", "passing_yards", "passing_tds",
    "interceptions", "sacks", "carries", "rushing_yards",
    "rushing_first_downs", "rushing_epa", "targets", "receptions",
    "receiving_yards", "receiving_first_downs", "receiving_epa",
]

WEEKLY_RATE_COLUMNS = [
    "target_share", "air_yards_share", "wopr", "cpoe",
]

OFFENSIVE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


FINAL_COLS = [
    "season",
    "player_id",
    "player_name",
    "team",
    "position",
    "position_group",

    "games",
    "total_snaps",
    "offense_snaps",
    "defense_snaps",
    "st_snaps",

    "qb_dropbacks",
    "qb_epa_per_dropback",
    "qb_success_rate",
    "qb_cpoe",
    "qb_sack_rate",
    "qb_int_rate",
    "qb_yards_per_dropback",

    "rush_attempts",
    "rush_yards",
    "rush_epa",
    "rush_success_rate",
    "rush_yards_per_carry",
    "rush_epa_per_carry",
    "rush_first_down_rate",

    "targets",
    "receptions",
    "receiving_yards",
    "receiving_epa",
    "receiving_success_rate",
    "yards_per_target",
    "receiving_epa_per_target",
    "catch_rate",
    "receiving_first_down_rate",
    "target_share_avg",
    "air_yards_share_avg",
    "wopr_avg",

    "ngs_avg_separation",
    "ngs_avg_cushion",
    "ngs_avg_intended_air_yards",
    "ngs_yac_above_expectation",
    "ngs_rush_yards_over_expected",
    "ngs_rush_yards_over_expected_per_att",
    "ngs_time_to_throw",
    "ngs_completion_pct_above_expectation",

    "def_tackles",
    "def_sacks",
    "def_half_sacks",
    "def_tfl",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_forced_fumbles",
    "def_fumble_recoveries",

    "pressure_proxy",
    "coverage_playmaking_proxy",
    "run_defense_proxy",
    "defensive_playmaking_score",

    "master_matched",
    "current_team",
    "current_position",
    "current_position_group",
    "clean_name",
    "initial_last_key",
    "canonical_key",
    "identity_quality_flag",
    "identity_version",

    "seasons_observed",
    "available_season_weight",
    "games_history_sum",
    "total_snaps_history_sum",
    "games_recent_2025",
    "total_snaps_recent_2025",
    "advanced_version",
    "date_imported",
]


def get_engine():
    return sql.create_engine(f"sqlite:///{DB_PATH}", pool_pre_ping=True)


def table_exists(engine, table_name):
    query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name = :table_name
    """

    with engine.connect() as conn:
        result = conn.execute(sql.text(query), {"table_name": table_name}).fetchone()

    return result is not None


def read_table(engine, table_name):
    if not table_exists(engine, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")

    with engine.connect() as conn:
        df = pd.read_sql(sql.text(f"SELECT * FROM {table_name}"), conn)

    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def normalize_team(team):
    if pd.isna(team):
        return None

    t = str(team).upper().strip()
    return TEAM_ALIASES.get(t, t)


def normalize_position(pos):
    if pd.isna(pos):
        return None

    return str(pos).upper().strip()


def position_group(pos):
    pos = normalize_position(pos)

    if pos is None:
        return "OTHER"

    return POSITION_GROUP_MAP.get(pos, "OTHER")


def clean_player_id(series):
    out = series.astype("string")
    out = (
        out.str.strip()
        .str.replace("\u200b", "", regex=False)
        .str.replace("\ufeff", "", regex=False)
    )
    return out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def add_missing_cols(df, cols, default=0.0):
    out = df.copy()

    for col in cols:
        if col not in out.columns:
            out[col] = default

    return out


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0.0)

    return np.where(denominator > 0, numerator / denominator, 0.0)


def load_master(engine):
    master = read_table(engine, MASTER_TABLE)

    keep_cols = [
        "player_id",
        "player_name",
        "team",
        "position",
        "position_group",
        "clean_name",
        "initial_last_key",
        "canonical_key",
        "identity_quality_flag",
        "identity_version",
    ]

    for col in keep_cols:
        if col not in master.columns:
            master[col] = None

    master["player_id"] = clean_player_id(master["player_id"])
    unresolved_count = int(master["player_id"].isna().sum())
    if unresolved_count:
        print(
            f"[ADV] Excluding {unresolved_count:,} current-roster rows without canonical GSIS IDs "
            "from the advanced-stat join; they remain in the master identity audit."
        )
    master = master[master["player_id"].notna()].copy()
    master["team"] = master["team"].apply(normalize_team)
    master["position"] = master["position"].apply(normalize_position)
    master["position_group"] = master["position_group"].fillna(master["position"].apply(position_group))

    master = (
        master[keep_cols]
        .drop_duplicates(subset=["player_id"], keep="first")
        .rename(
            columns={
                "player_name": "current_player_name",
                "team": "current_team",
                "position": "current_position",
                "position_group": "current_position_group",
            }
        )
    )

    return master


def _standardize_weekly_frame(frame, requested_season=None):
    """Normalize one weekly-player source and collapse duplicate player-week rows.

    nflreadpy.load_player_stats can return a broader player participation frame
    than nfl_data_py.import_weekly_data.  The loader therefore standardizes IDs,
    filters to regular-season rows when a season-type field is present, and
    collapses repeated player-week observations before downstream game counts
    and counting statistics are calculated.
    """
    if frame is None:
        return pd.DataFrame()

    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()

    out = pd.DataFrame(frame).copy()
    if out.empty:
        return out

    out.columns = [str(column).lower().strip() for column in out.columns]

    rename_map = {
        "gsis_id": "player_id",
        "player_gsis_id": "player_id",
        "display_name": "player_name",
        "player_display_name": "player_name",
        "recent_team": "team",
        "team_abbr": "team",
        "position_group": "position",
        "completions_percentage_above_expectation": "cpoe",
    }
    out = out.rename(
        columns={
            old_name: new_name
            for old_name, new_name in rename_map.items()
            if old_name in out.columns and new_name not in out.columns
        }
    )

    if "season" not in out.columns and requested_season is not None:
        out["season"] = int(requested_season)

    for season_type_col in ["season_type", "game_type"]:
        if season_type_col in out.columns:
            regular = out[season_type_col].astype(str).str.upper().str.strip().isin(
                {"REG", "R", "REGULAR", "REGULAR SEASON", "REGULAR_SEASON"}
            )
            if regular.any():
                out = out[regular].copy()
            break

    for column in ["season", "week", "player_id", "player_name", "team", "position"]:
        if column not in out.columns:
            out[column] = None

    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out["player_id"] = clean_player_id(out["player_id"])
    out["team"] = out["team"].apply(normalize_team)
    out["position"] = out["position"].apply(normalize_position)

    for column in WEEKLY_COUNTING_COLUMNS + WEEKLY_RATE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan if column in WEEKLY_RATE_COLUMNS else 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    raw_rows = len(out)
    valid_week_key = out["week"].notna() & out["player_id"].notna()
    keyed = out[valid_week_key].copy()
    unkeyed = out[~valid_week_key].copy()

    if not keyed.empty:
        duplicate_rows = int(keyed.duplicated(["season", "week", "player_id"], keep=False).sum())
        identity_aggs = {
            "player_name": "first",
            "team": "first",
            "position": "first",
        }
        count_aggs = {column: "max" for column in WEEKLY_COUNTING_COLUMNS}
        rate_aggs = {column: "mean" for column in WEEKLY_RATE_COLUMNS}
        extra_aggs = {}
        if "games" in keyed.columns:
            extra_aggs["games"] = "max"

        keyed = (
            keyed.groupby(["season", "week", "player_id"], dropna=False)
            .agg(**{
                **{name: (name, agg) for name, agg in identity_aggs.items()},
                **{name: (name, agg) for name, agg in count_aggs.items()},
                **{name: (name, agg) for name, agg in rate_aggs.items()},
                **{name: (name, agg) for name, agg in extra_aggs.items()},
            })
            .reset_index()
        )
    else:
        duplicate_rows = 0

    out = pd.concat([keyed, unkeyed], ignore_index=True, sort=False)
    out.attrs["weekly_raw_rows"] = raw_rows
    out.attrs["weekly_duplicate_rows"] = duplicate_rows
    out.attrs["weekly_standardized_rows"] = len(out)
    return out


def _load_weekly_with_nflreadpy(season):
    errors = []

    try:
        import nflreadpy as nfl

        for function_name in [
            "load_player_stats",
            "load_weekly_data",
            "load_weekly_player_stats",
        ]:
            if not hasattr(nfl, function_name):
                continue

            loader = getattr(nfl, function_name)
            attempts = [
                lambda: loader([season]),
                lambda: loader(seasons=[season]),
                lambda: loader(season),
                lambda: loader(seasons=season),
            ]

            for attempt_number, attempt in enumerate(
                attempts,
                start=1,
            ):
                try:
                    frame = _standardize_weekly_frame(
                        attempt(),
                        requested_season=season,
                    )
                    if frame.empty:
                        continue

                    seasons_found = set(
                        pd.to_numeric(
                            frame.get("season"),
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                    )
                    if season not in seasons_found:
                        continue

                    raw_rows = int(frame.attrs.get("weekly_raw_rows", len(frame)))
                    duplicate_rows = int(frame.attrs.get("weekly_duplicate_rows", 0))
                    print(
                        "[ADV] Weekly fallback loaded "
                        f"{len(frame):,} standardized rows for {season} via "
                        f"nflreadpy.{function_name} "
                        f"(raw={raw_rows:,}, duplicate_player_week_rows={duplicate_rows:,})"
                    )
                    return frame
                except Exception as exc:
                    errors.append(
                        f"nflreadpy.{function_name} "
                        f"attempt {attempt_number}: {exc}"
                    )

    except Exception as exc:
        errors.append(f"import nflreadpy: {exc}")

    print(
        f"[ADV] Weekly nflreadpy fallback unavailable for "
        f"{season}: {' | '.join(errors)}"
    )
    return pd.DataFrame()


def load_weekly_data(seasons):
    frames = []
    missing_seasons = []

    try:
        import nfl_data_py as nfl
    except Exception as exc:
        nfl = None
        print(
            "[ADV] nfl_data_py weekly loader unavailable: "
            f"{exc}"
        )

    print("[ADV] Loading weekly data by season...")

    for season in seasons:
        season_frame = pd.DataFrame()

        if nfl is not None:
            try:
                season_frame = _standardize_weekly_frame(
                    nfl.import_weekly_data([season]),
                    requested_season=season,
                )
                if not season_frame.empty:
                    print(
                        f"[ADV] Weekly {season} rows via "
                        f"nfl_data_py: {len(season_frame):,}"
                    )
            except Exception as exc:
                print(
                    f"[ADV] Weekly nfl_data_py unavailable for "
                    f"{season}: {exc}"
                )

        if season_frame.empty:
            season_frame = _load_weekly_with_nflreadpy(
                season
            )

        if season_frame.empty:
            missing_seasons.append(int(season))
            continue

        frames.append(season_frame)

    if not frames:
        print(
            "[ADV] Weekly data unavailable for every requested "
            "season. The pipeline will rely on snaps, NGS, and PBP."
        )
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["season"] = pd.to_numeric(
        combined["season"],
        errors="coerce",
    ).astype("Int64")

    combined = combined[
        combined["season"].isin(seasons)
    ].copy()

    print(f"[ADV] Weekly rows: {len(combined):,}")
    print(
        "[ADV] Weekly seasons loaded: "
        f"{sorted(set(combined['season'].dropna().astype(int)))}"
    )

    if missing_seasons:
        print(
            "[ADV] Weekly seasons unavailable: "
            f"{missing_seasons}. These seasons will retain "
            "snap, NGS, and PBP history but may lack some "
            "weekly offensive fields."
        )

    return combined


def load_snap_counts(seasons):
    try:
        import nfl_data_py as nfl

        print("[ADV] Loading snap counts...")
        df = nfl.import_snap_counts(seasons)
        df.columns = [str(c).lower().strip() for c in df.columns]
        print(f"[ADV] Snap rows: {len(df):,}")
        return df

    except Exception as e:
        print(f"[ADV] Snap counts unavailable: {e}")
        return pd.DataFrame()



def load_crosswalk_pfr_map(engine):
    """Load the permanent one-way PFR -> canonical GSIS map.

    Multiple historical PFR IDs may legitimately resolve to one GSIS ID. The
    required uniqueness is the opposite direction: each PFR ID may map to only
    one canonical player. Ambiguous PFR IDs are rejected rather than guessed.
    """
    crosswalk = read_table(engine, CROSSWALK_TABLE)

    canonical_col = next(
        (column for column in ["canonical_player_id", "gsis_id", "player_id"] if column in crosswalk.columns),
        None,
    )
    pfr_col = next(
        (column for column in ["pfr_player_id", "pfr_id"] if column in crosswalk.columns),
        None,
    )
    if canonical_col is None or pfr_col is None:
        raise RuntimeError(
            f"{CROSSWALK_TABLE} must contain canonical GSIS and PFR columns. "
            f"Columns: {list(crosswalk.columns)}"
        )

    keep = pd.DataFrame(index=crosswalk.index)
    keep["player_id"] = clean_player_id(crosswalk[canonical_col])
    keep["pfr_player_id"] = crosswalk[pfr_col].astype("string").str.split("|")
    keep["id_player_name"] = crosswalk.get("player_name")
    keep["id_position"] = crosswalk.get("position")
    keep["match_method"] = crosswalk.get("match_method")
    keep["match_confidence"] = crosswalk.get("match_confidence")
    conflict_source = (
        crosswalk["identity_conflict_flag"]
        if "identity_conflict_flag" in crosswalk.columns
        else pd.Series(0, index=crosswalk.index)
    )
    keep["identity_conflict_flag"] = pd.to_numeric(
        conflict_source, errors="coerce"
    ).fillna(0)
    keep["crosswalk_version"] = crosswalk.get("crosswalk_version")

    keep = keep.explode("pfr_player_id")
    keep["pfr_player_id"] = clean_player_id(keep["pfr_player_id"])
    keep = keep[
        keep["player_id"].notna()
        & keep["pfr_player_id"].notna()
        & keep["identity_conflict_flag"].eq(0)
    ].copy()

    conflicts = (
        keep.groupby("pfr_player_id")["player_id"].nunique().loc[lambda values: values > 1]
    )
    if not conflicts.empty:
        examples = keep[keep["pfr_player_id"].isin(conflicts.index)].head(25)
        raise RuntimeError(
            "Permanent crosswalk contains PFR IDs mapped to multiple canonical players:\n"
            + examples.to_string(index=False)
        )

    keep = keep.sort_values(
        ["pfr_player_id", "match_confidence", "player_id"],
        ascending=[True, True, True],
    ).drop_duplicates("pfr_player_id", keep="first")

    print(f"[ADV] Permanent crosswalk PFR mappings: {len(keep):,}")
    return keep[[
        "pfr_player_id", "player_id", "id_player_name", "id_position",
        "match_method", "match_confidence", "crosswalk_version",
    ]]


def canonicalize_snap_ids(snaps, player_id_map):
    if snaps.empty:
        return snaps.copy(), pd.DataFrame(columns=SNAP_ID_AUDIT_COLUMNS)

    df = snaps.copy()
    df.columns = [
        str(c).lower().strip()
        for c in df.columns
    ]

    if (
        "pfr_player_id" not in df.columns
        and "pfr_id" in df.columns
    ):
        df = df.rename(
            columns={"pfr_id": "pfr_player_id"}
        )

    if "pfr_player_id" not in df.columns:
        raise RuntimeError(
            "Snap source does not contain pfr_player_id. "
            f"Columns: {list(df.columns)}"
        )

    if "player" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"player": "player_name"})

    for column in [
        "season",
        "week",
        "player_name",
        "position",
        "team",
        "offense_snaps",
        "defense_snaps",
        "st_snaps",
    ]:
        if column not in df.columns:
            df[column] = None

    df["pfr_player_id"] = clean_player_id(
        df["pfr_player_id"]
    )

    mapping_columns = [
        "pfr_player_id", "player_id", "id_player_name", "id_position",
        "match_method", "match_confidence", "crosswalk_version",
    ]
    mapping = player_id_map.copy()
    for column in mapping_columns:
        if column not in mapping.columns:
            mapping[column] = None
    mapping = mapping[mapping_columns].copy()

    merged = df.merge(
        mapping,
        on="pfr_player_id",
        how="left",
        validate="many_to_one",
    )

    merged["canonical_id_match"] = (
        merged["player_id"].notna()
        & merged["player_id"].astype(str).str.strip().ne("")
    ).astype(int)

    for column in [
        "offense_snaps",
        "defense_snaps",
        "st_snaps",
    ]:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(0.0)

    merged["total_snap_value"] = (
        merged["offense_snaps"]
        + merged["defense_snaps"]
        + merged["st_snaps"]
    )

    audit = (
        merged.groupby(
            [
                "pfr_player_id",
                "player_id",
                "player_name",
                "id_player_name",
                "position",
                "id_position",
                "canonical_id_match",
            ],
            dropna=False,
        )
        .agg(
            snap_rows=("pfr_player_id", "size"),
            first_season=("season", "min"),
            last_season=("season", "max"),
            offense_snaps=("offense_snaps", "sum"),
            defense_snaps=("defense_snaps", "sum"),
            st_snaps=("st_snaps", "sum"),
            total_snaps=("total_snap_value", "sum"),
        )
        .reset_index()
    )

    audit["position_normalized"] = audit[
        "position"
    ].apply(normalize_position)
    audit["position_group"] = audit[
        "position_normalized"
    ].apply(position_group)
    audit["ol_flag"] = audit[
        "position_group"
    ].eq("OL").astype(int)
    audit["mapping_status"] = np.where(
        audit["canonical_id_match"].eq(1),
        "mapped",
        "unmatched",
    )
    audit["advanced_version"] = (
        "v3_permanent_crosswalk_snap_identity"
    )
    audit["date_imported"] = pd.to_datetime(
        dt.datetime.now().strftime("%Y-%m-%d")
    )
    for column in SNAP_ID_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = None
    audit = audit[SNAP_ID_AUDIT_COLUMNS]

    total_ids = merged["pfr_player_id"].nunique()
    mapped_ids = merged.loc[
        merged["canonical_id_match"].eq(1),
        "pfr_player_id",
    ].nunique()
    total_rows = len(merged)
    mapped_rows = int(merged["canonical_id_match"].sum())

    ol = merged[
        merged["position"].apply(
            normalize_position
        ).apply(position_group).eq("OL")
    ].copy()
    ol_total_ids = ol["pfr_player_id"].nunique()
    ol_mapped_ids = ol.loc[
        ol["canonical_id_match"].eq(1),
        "pfr_player_id",
    ].nunique()

    print(
        "[ADV] Snap canonical ID matches: "
        f"{mapped_rows:,}/{total_rows:,} "
        f"({mapped_rows / total_rows:.2%})"
    )
    print(
        "[ADV] Snap PFR IDs mapped: "
        f"{mapped_ids:,}/{total_ids:,} "
        f"({mapped_ids / total_ids:.2%})"
    )
    print(
        "[ADV] OL PFR IDs mapped: "
        f"{ol_mapped_ids:,}/{ol_total_ids:,} "
        f"({ol_mapped_ids / ol_total_ids:.2%})"
    )

    canonical = merged[
        merged["canonical_id_match"].eq(1)
    ].copy()

    return canonical, audit


def load_ngs_data(seasons):
    out = []

    try:
        import nfl_data_py as nfl
    except Exception as e:
        print(f"[ADV] nfl_data_py unavailable for NGS: {e}")
        return pd.DataFrame()

    for stat_type in ["passing", "rushing", "receiving"]:
        try:
            print(f"[ADV] Loading NGS {stat_type}...")
            df = nfl.import_ngs_data(stat_type=stat_type, years=seasons)
            df.columns = [str(c).lower().strip() for c in df.columns]
            df["ngs_stat_type"] = stat_type
            print(f"[ADV] NGS {stat_type} rows: {len(df):,}")
            out.append(df)

        except TypeError:
            try:
                df = nfl.import_ngs_data(stat_type, seasons)
                df.columns = [str(c).lower().strip() for c in df.columns]
                df["ngs_stat_type"] = stat_type
                print(f"[ADV] NGS {stat_type} rows: {len(df):,}")
                out.append(df)
            except Exception as e:
                print(f"[ADV] NGS {stat_type} unavailable: {e}")

        except Exception as e:
            print(f"[ADV] NGS {stat_type} unavailable: {e}")

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True, sort=False)


def load_pbp_data(seasons):
    try:
        import nfl_data_py as nfl

        print("[ADV] Loading play-by-play data for defensive proxies...")

        columns = [
            "season",
            "week",
            "game_id",
            "posteam",
            "defteam",
            "pass",
            "rush",
            "qb_dropback",
            "pass_attempt",
            "complete_pass",
            "sack",
            "interception",
            "epa",
            "success",
            "cpoe",
            "yards_gained",
            "passer_player_id",
            "passer_player_name",
            "receiver_player_id",
            "receiver_player_name",
            "rusher_player_id",
            "rusher_player_name",
            "solo_tackle_1_player_id",
            "solo_tackle_1_player_name",
            "solo_tackle_2_player_id",
            "solo_tackle_2_player_name",
            "assist_tackle_1_player_id",
            "assist_tackle_1_player_name",
            "assist_tackle_2_player_id",
            "assist_tackle_2_player_name",
            "sack_player_id",
            "sack_player_name",
            "half_sack_1_player_id",
            "half_sack_1_player_name",
            "half_sack_2_player_id",
            "half_sack_2_player_name",
            "tackle_for_loss_1_player_id",
            "tackle_for_loss_1_player_name",
            "tackle_for_loss_2_player_id",
            "tackle_for_loss_2_player_name",
            "qb_hit_1_player_id",
            "qb_hit_1_player_name",
            "qb_hit_2_player_id",
            "qb_hit_2_player_name",
            "interception_player_id",
            "interception_player_name",
            "pass_defense_1_player_id",
            "pass_defense_1_player_name",
            "pass_defense_2_player_id",
            "pass_defense_2_player_name",
            "forced_fumble_player_1_player_id",
            "forced_fumble_player_1_player_name",
            "forced_fumble_player_2_player_id",
            "forced_fumble_player_2_player_name",
            "fumble_recovery_1_player_id",
            "fumble_recovery_1_player_name",
            "fumble_recovery_2_player_id",
            "fumble_recovery_2_player_name",
        ]

        try:
            df = nfl.import_pbp_data(seasons, columns=columns)
        except Exception as e:
            print(f"[ADV] PBP column-filtered load failed: {e}")
            print("[ADV] Retrying PBP with default columns...")
            df = nfl.import_pbp_data(seasons)

        df.columns = [str(c).lower().strip() for c in df.columns]
        print(f"[ADV] PBP rows: {len(df):,}")
        return df

    except Exception as e:
        print(f"[ADV] PBP unavailable. Continuing without defensive proxies. Error: {e}")
        return pd.DataFrame()


def build_weekly_offense(weekly):
    if weekly.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    df = weekly.copy()

    if "player_display_name" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"player_display_name": "player_name"})

    if "recent_team" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"recent_team": "team"})

    required = [
        "season",
        "player_id",
        "player_name",
        "team",
        "position",
        "passing_attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "carries",
        "rushing_yards",
        "rushing_first_downs",
        "rushing_epa",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_first_downs",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    for col in ["player_id", "player_name", "team", "position"]:
        if col not in df.columns:
            df[col] = None

    df["player_id"] = clean_player_id(df["player_id"])
    df["team"] = df["team"].apply(normalize_team)
    df["position"] = df["position"].apply(normalize_position)
    df["position_group"] = df["position"].apply(position_group)

    if "week" in df.columns and pd.to_numeric(df["week"], errors="coerce").notna().any():
        df["games"] = 1.0
    elif "games" in df.columns:
        df["games"] = pd.to_numeric(df["games"], errors="coerce").fillna(0.0)
    else:
        df["games"] = 1.0

    numeric_cols = [
        "passing_attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "carries",
        "rushing_yards",
        "rushing_first_downs",
        "rushing_epa",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_first_downs",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
    ]

    for col in numeric_cols:
        df[col] = safe_numeric(df, col)

    offense_signal = (
        df[
            [
                "passing_attempts", "completions", "passing_yards", "passing_tds",
                "interceptions", "sacks", "carries", "rushing_yards",
                "rushing_first_downs", "targets", "receptions",
                "receiving_yards", "receiving_first_downs",
            ]
        ].abs().sum(axis=1)
        > 0
    )
    recognized_offense = df["position"].isin(OFFENSIVE_POSITIONS)
    df = df[offense_signal | recognized_offense].copy()

    out = (
        df.groupby(
            ["season", "player_id", "player_name", "team", "position", "position_group"],
            dropna=False,
        )
        .agg(
            games=("games", "sum"),
            pass_att=("passing_attempts", "sum"),
            completions=("completions", "sum"),
            pass_yards=("passing_yards", "sum"),
            pass_tds=("passing_tds", "sum"),
            interceptions=("interceptions", "sum"),
            sacks_taken=("sacks", "sum"),
            rush_attempts=("carries", "sum"),
            rush_yards=("rushing_yards", "sum"),
            rush_first_downs=("rushing_first_downs", "sum"),
            rush_epa=("rushing_epa", "sum"),
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_first_downs=("receiving_first_downs", "sum"),
            receiving_epa=("receiving_epa", "sum"),
            target_share_avg=("target_share", "mean"),
            air_yards_share_avg=("air_yards_share", "mean"),
            wopr_avg=("wopr", "mean"),
        )
        .reset_index()
    )

    out["qb_dropbacks"] = out["pass_att"] + out["sacks_taken"]
    out["qb_sack_rate"] = safe_divide(out["sacks_taken"], out["qb_dropbacks"])
    out["qb_int_rate"] = safe_divide(out["interceptions"], out["pass_att"])
    out["qb_yards_per_dropback"] = safe_divide(out["pass_yards"], out["qb_dropbacks"])
    out["qb_epa_per_dropback"] = 0.0
    out["qb_success_rate"] = 0.0
    out["qb_cpoe"] = 0.0

    out["rush_yards_per_carry"] = safe_divide(out["rush_yards"], out["rush_attempts"])
    out["rush_epa_per_carry"] = safe_divide(out["rush_epa"], out["rush_attempts"])
    out["rush_first_down_rate"] = safe_divide(out["rush_first_downs"], out["rush_attempts"])
    out["rush_success_rate"] = 0.0

    out["yards_per_target"] = safe_divide(out["receiving_yards"], out["targets"])
    out["receiving_epa_per_target"] = safe_divide(out["receiving_epa"], out["targets"])
    out["catch_rate"] = safe_divide(out["receptions"], out["targets"])
    out["receiving_first_down_rate"] = safe_divide(out["receiving_first_downs"], out["targets"])
    out["receiving_success_rate"] = 0.0

    return out


def _first_non_missing(series):
    """Return the first useful scalar from a Series, else None."""
    for value in series:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "<na>"}:
            return value
    return None


def build_snap_metrics(snaps):
    """Aggregate snap participation while preserving historical identity.

    Snap counts are the broadest participation source. They provide game counts
    and identity for OL, defenders, and specialists who may never appear in the
    weekly fantasy/offensive-stat feed.
    """
    if snaps.empty:
        return pd.DataFrame(columns=[
            "season", "player_id", "snap_player_name", "snap_team",
            "snap_position", "snap_position_group", "snap_games",
            "total_snaps", "offense_snaps", "defense_snaps", "st_snaps",
        ])

    df = snaps.copy()

    if "canonical_player_id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"canonical_player_id": "player_id"})

    # Rename only when the destination is absent; otherwise coalesce below.
    if "recent_team" in df.columns:
        if "team" not in df.columns:
            df = df.rename(columns={"recent_team": "team"})
        else:
            df["team"] = df["team"].combine_first(df["recent_team"])

    if "player_display_name" in df.columns:
        if "player_name" not in df.columns:
            df = df.rename(columns={"player_display_name": "player_name"})
        else:
            df["player_name"] = df["player_name"].combine_first(
                df["player_display_name"]
            )

    for col in ["season", "player_id", "player_name", "team", "position"]:
        if col not in df.columns:
            df[col] = None

    for col in ["offense_snaps", "defense_snaps", "st_snaps"]:
        if col not in df.columns:
            df[col] = 0.0

    df["player_id"] = clean_player_id(df["player_id"])
    df["team"] = df["team"].apply(normalize_team)
    df["position"] = df["position"].apply(normalize_position)
    df["position_group"] = df["position"].apply(position_group)

    df["offense_snaps"] = safe_numeric(df, "offense_snaps")
    df["defense_snaps"] = safe_numeric(df, "defense_snaps")
    df["st_snaps"] = safe_numeric(df, "st_snaps")
    df["total_snaps"] = (
        df["offense_snaps"] + df["defense_snaps"] + df["st_snaps"]
    )

    # Count a game only when the player actually recorded a snap. This avoids
    # counting zero-filled provider rows as appearances.
    df["snap_game_played"] = df["total_snaps"].gt(0).astype(int)

    out = (
        df.groupby(["season", "player_id"], dropna=False)
        .agg(
            snap_player_name=("player_name", _first_non_missing),
            snap_team=("team", _first_non_missing),
            snap_position=("position", _first_non_missing),
            snap_position_group=("position_group", _first_non_missing),
            snap_games=("snap_game_played", "sum"),
            total_snaps=("total_snaps", "sum"),
            offense_snaps=("offense_snaps", "sum"),
            defense_snaps=("defense_snaps", "sum"),
            st_snaps=("st_snaps", "sum"),
        )
        .reset_index()
    )

    return out

def build_ngs_metrics(ngs):
    if ngs.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    df = ngs.copy()

    if "player_gsis_id" in df.columns and "player_id" not in df.columns:
        df = df.rename(columns={"player_gsis_id": "player_id"})

    if "player_display_name" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"player_display_name": "player_name"})

    if "team_abbr" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"team_abbr": "team"})

    for col in ["season", "player_id", "player_name", "team", "ngs_stat_type"]:
        if col not in df.columns:
            df[col] = None

    df["player_id"] = clean_player_id(df["player_id"])
    df["team"] = df["team"].apply(normalize_team)

    numeric_candidates = [
        "avg_separation",
        "avg_cushion",
        "avg_intended_air_yards",
        "avg_yac_above_expectation",
        "rush_yards_over_expected",
        "rush_yards_over_expected_per_att",
        "avg_time_to_throw",
        "completion_percentage_above_expectation",
    ]

    for col in numeric_candidates:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = (
        df.groupby(["season", "player_id"], dropna=False)
        .agg(
            ngs_avg_separation=("avg_separation", "mean"),
            ngs_avg_cushion=("avg_cushion", "mean"),
            ngs_avg_intended_air_yards=("avg_intended_air_yards", "mean"),
            ngs_yac_above_expectation=("avg_yac_above_expectation", "mean"),
            ngs_rush_yards_over_expected=("rush_yards_over_expected", "sum"),
            ngs_rush_yards_over_expected_per_att=("rush_yards_over_expected_per_att", "mean"),
            ngs_time_to_throw=("avg_time_to_throw", "mean"),
            ngs_completion_pct_above_expectation=("completion_percentage_above_expectation", "mean"),
        )
        .reset_index()
    )

    return out


def pbp_player_events(df, id_col, name_col, value_col):
    if df.empty or id_col not in df.columns:
        return pd.DataFrame(columns=["season", "player_id", "player_name", value_col])

    tmp = df[["season", id_col] + ([name_col] if name_col in df.columns else [])].copy()
    tmp = tmp[tmp[id_col].notna()].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["season", "player_id", "player_name", value_col])

    tmp = tmp.rename(columns={id_col: "player_id"})

    if name_col in tmp.columns:
        tmp = tmp.rename(columns={name_col: "player_name"})
    else:
        tmp["player_name"] = None

    tmp["player_id"] = clean_player_id(tmp["player_id"])
    tmp[value_col] = 1

    return (
        tmp.groupby(["season", "player_id", "player_name"], dropna=False)
        .agg(**{value_col: (value_col, "sum")})
        .reset_index()
    )


def combine_event_frames(frames):
    valid = [
        frame
        for frame in frames
        if frame is not None and not frame.empty
    ]

    if not valid:
        return pd.DataFrame(
            columns=["season", "player_id", "player_name"]
        )

    stacked = pd.concat(
        valid,
        ignore_index=True,
        sort=False,
    )

    value_cols = [
        column
        for column in stacked.columns
        if column not in [
            "season",
            "player_id",
            "player_name",
        ]
    ]

    for column in value_cols:
        stacked[column] = pd.to_numeric(
            stacked[column],
            errors="coerce",
        ).fillna(0.0)

    return (
        stacked.groupby(
            ["season", "player_id", "player_name"],
            dropna=False,
        )[value_cols]
        .sum()
        .reset_index()
    )


def build_pbp_qb_receiving_rushing(pbp):
    if pbp.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    df = pbp.copy()

    for col in ["season", "epa", "success", "cpoe", "yards_gained"]:
        if col not in df.columns:
            df[col] = 0.0

    for col in ["pass_attempt", "qb_dropback", "sack", "interception", "rush"]:
        if col not in df.columns:
            df[col] = 0.0

    for col in ["passer_player_id", "receiver_player_id", "rusher_player_id"]:
        if col not in df.columns:
            df[col] = None

    df["epa"] = safe_numeric(df, "epa")
    df["success"] = safe_numeric(df, "success")
    df["cpoe"] = safe_numeric(df, "cpoe")
    df["yards_gained"] = safe_numeric(df, "yards_gained")
    df["pass_attempt"] = safe_numeric(df, "pass_attempt")
    df["qb_dropback"] = safe_numeric(df, "qb_dropback")
    df["sack"] = safe_numeric(df, "sack")
    df["interception"] = safe_numeric(df, "interception")
    df["rush"] = safe_numeric(df, "rush")

    qb = df[df["qb_dropback"] == 1].copy()

    if not qb.empty:
        qb["player_id"] = clean_player_id(qb["passer_player_id"])
        qb = qb[qb["player_id"].notna()].copy()
        qb["dropback"] = 1

        qb_out = (
            qb.groupby(["season", "player_id"], dropna=False)
            .agg(
                pbp_qb_dropbacks=("dropback", "sum"),
                pbp_qb_epa_per_dropback=("epa", "mean"),
                pbp_qb_success_rate=("success", "mean"),
                pbp_qb_cpoe=("cpoe", "mean"),
                pbp_qb_sack_rate=("sack", "mean"),
                pbp_qb_int_rate=("interception", "mean"),
                pbp_qb_yards_per_dropback=("yards_gained", "mean"),
            )
            .reset_index()
        )
    else:
        qb_out = pd.DataFrame(columns=["season", "player_id"])

    receiving = df[df["receiver_player_id"].notna()].copy()

    if not receiving.empty:
        receiving["player_id"] = clean_player_id(receiving["receiver_player_id"])
        receiving["target_event"] = 1

        rec_out = (
            receiving.groupby(["season", "player_id"], dropna=False)
            .agg(
                pbp_targets=("target_event", "sum"),
                pbp_receiving_epa_per_target=("epa", "mean"),
                pbp_receiving_success_rate=("success", "mean"),
            )
            .reset_index()
        )
    else:
        rec_out = pd.DataFrame(columns=["season", "player_id"])

    rushing = df[(df["rush"] == 1) & (df["rusher_player_id"].notna())].copy()

    if not rushing.empty:
        rushing["player_id"] = clean_player_id(rushing["rusher_player_id"])
        rushing["rush_event"] = 1

        rush_out = (
            rushing.groupby(["season", "player_id"], dropna=False)
            .agg(
                pbp_rush_attempts=("rush_event", "sum"),
                pbp_rush_epa_per_carry=("epa", "mean"),
                pbp_rush_success_rate=("success", "mean"),
            )
            .reset_index()
        )
    else:
        rush_out = pd.DataFrame(columns=["season", "player_id"])

    out = qb_out.merge(rec_out, on=["season", "player_id"], how="outer")
    out = out.merge(rush_out, on=["season", "player_id"], how="outer")

    return out


def build_pbp_defense(pbp):
    if pbp.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    event_frames = []

    tackle_cols = [
        ("solo_tackle_1_player_id", "solo_tackle_1_player_name", "def_tackles"),
        ("solo_tackle_2_player_id", "solo_tackle_2_player_name", "def_tackles"),
        ("assist_tackle_1_player_id", "assist_tackle_1_player_name", "def_tackles"),
        ("assist_tackle_2_player_id", "assist_tackle_2_player_name", "def_tackles"),
    ]

    sack_cols = [
        ("sack_player_id", "sack_player_name", "def_sacks"),
    ]

    half_sack_cols = [
        ("half_sack_1_player_id", "half_sack_1_player_name", "def_half_sacks"),
        ("half_sack_2_player_id", "half_sack_2_player_name", "def_half_sacks"),
    ]

    tfl_cols = [
        ("tackle_for_loss_1_player_id", "tackle_for_loss_1_player_name", "def_tfl"),
        ("tackle_for_loss_2_player_id", "tackle_for_loss_2_player_name", "def_tfl"),
    ]

    qb_hit_cols = [
        ("qb_hit_1_player_id", "qb_hit_1_player_name", "def_qb_hits"),
        ("qb_hit_2_player_id", "qb_hit_2_player_name", "def_qb_hits"),
    ]

    interception_cols = [
        ("interception_player_id", "interception_player_name", "def_interceptions"),
    ]

    pass_defense_cols = [
        ("pass_defense_1_player_id", "pass_defense_1_player_name", "def_pass_defended"),
        ("pass_defense_2_player_id", "pass_defense_2_player_name", "def_pass_defended"),
    ]

    forced_fumble_cols = [
        ("forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name", "def_forced_fumbles"),
        ("forced_fumble_player_2_player_id", "forced_fumble_player_2_player_name", "def_forced_fumbles"),
    ]

    recovery_cols = [
        ("fumble_recovery_1_player_id", "fumble_recovery_1_player_name", "def_fumble_recoveries"),
        ("fumble_recovery_2_player_id", "fumble_recovery_2_player_name", "def_fumble_recoveries"),
    ]

    for id_col, name_col, value_col in (
        tackle_cols
        + sack_cols
        + half_sack_cols
        + tfl_cols
        + qb_hit_cols
        + interception_cols
        + pass_defense_cols
        + forced_fumble_cols
        + recovery_cols
    ):
        event_frames.append(pbp_player_events(pbp, id_col, name_col, value_col))

    raw = combine_event_frames(event_frames)

    if raw.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    value_cols = [
        "def_tackles",
        "def_sacks",
        "def_half_sacks",
        "def_tfl",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_forced_fumbles",
        "def_fumble_recoveries",
    ]

    for col in value_cols:
        if col not in raw.columns:
            raw[col] = 0.0

    out = (
        raw.groupby(["season", "player_id"], dropna=False)
        .agg(
            def_tackles=("def_tackles", "sum"),
            def_sacks=("def_sacks", "sum"),
            def_half_sacks=("def_half_sacks", "sum"),
            def_tfl=("def_tfl", "sum"),
            def_qb_hits=("def_qb_hits", "sum"),
            def_interceptions=("def_interceptions", "sum"),
            def_pass_defended=("def_pass_defended", "sum"),
            def_forced_fumbles=("def_forced_fumbles", "sum"),
            def_fumble_recoveries=("def_fumble_recoveries", "sum"),
        )
        .reset_index()
    )

    return out


def build_base_player_seasons(weekly_offense, snaps, defense, pbp_skill):
    frames = []

    for frame in [weekly_offense, snaps, defense, pbp_skill]:
        if frame is not None and not frame.empty and "player_id" in frame.columns:
            frames.append(frame[["season", "player_id"]].drop_duplicates())

    if not frames:
        return pd.DataFrame(columns=["season", "player_id"])

    base = pd.concat(frames, ignore_index=True).drop_duplicates()
    return base


def coalesce_player_identity(df, weekly_offense, snap_metrics, ngs_metrics=None):
    """Attach historical identity without overwriting it with 2026 identity.

    Priority is weekly offense, then snap counts, then NGS. Snap identity is
    essential for OL, defensive players, and specialists.
    """
    out = df.copy()

    weekly_identity = pd.DataFrame(columns=[
        "season", "player_id", "weekly_player_name", "weekly_team",
        "weekly_position", "weekly_position_group",
    ])
    if weekly_offense is not None and not weekly_offense.empty:
        weekly_identity = (
            weekly_offense[[
                "season", "player_id", "player_name", "team", "position",
                "position_group",
            ]]
            .drop_duplicates(subset=["season", "player_id"], keep="last")
            .rename(columns={
                "player_name": "weekly_player_name",
                "team": "weekly_team",
                "position": "weekly_position",
                "position_group": "weekly_position_group",
            })
        )
        out = out.merge(
            weekly_identity,
            on=["season", "player_id"],
            how="left",
            validate="one_to_one",
        )

    snap_identity_cols = [
        "season", "player_id", "snap_player_name", "snap_team",
        "snap_position", "snap_position_group",
    ]
    if snap_metrics is not None and not snap_metrics.empty:
        available = [c for c in snap_identity_cols if c in snap_metrics.columns]
        snap_identity = snap_metrics[available].drop_duplicates(
            ["season", "player_id"], keep="last"
        )
        out = out.merge(
            snap_identity,
            on=["season", "player_id"],
            how="left",
            validate="one_to_one",
        )

    ngs_identity = pd.DataFrame()
    if ngs_metrics is not None and not ngs_metrics.empty:
        ngs_cols = [
            c for c in [
                "season", "player_id", "ngs_player_name", "ngs_team",
                "ngs_position", "ngs_position_group",
            ] if c in ngs_metrics.columns
        ]
        if {"season", "player_id"}.issubset(ngs_cols):
            ngs_identity = ngs_metrics[ngs_cols].drop_duplicates(
                ["season", "player_id"], keep="last"
            )
            out = out.merge(
                ngs_identity,
                on=["season", "player_id"],
                how="left",
                validate="one_to_one",
            )

    def combine_candidates(columns):
        result = pd.Series(None, index=out.index, dtype="object")
        for column in columns:
            if column in out.columns:
                result = result.combine_first(out[column])
        return result

    out["player_name"] = combine_candidates([
        "weekly_player_name", "snap_player_name", "ngs_player_name"
    ])
    out["team"] = combine_candidates([
        "weekly_team", "snap_team", "ngs_team"
    ])
    out["position"] = combine_candidates([
        "weekly_position", "snap_position", "ngs_position"
    ])
    out["position_group"] = combine_candidates([
        "weekly_position_group", "snap_position_group", "ngs_position_group"
    ])

    return out

def build_advanced_stats(weekly, snaps, ngs, pbp):
    weekly_offense = build_weekly_offense(weekly)
    snap_metrics = build_snap_metrics(snaps)
    ngs_metrics = build_ngs_metrics(ngs)
    pbp_skill = build_pbp_qb_receiving_rushing(pbp)
    defense = build_pbp_defense(pbp)

    base = build_base_player_seasons(weekly_offense, snap_metrics, defense, pbp_skill)
    base = coalesce_player_identity(base, weekly_offense, snap_metrics, ngs_metrics)

    out = base.copy()

    merge_frames = [
        weekly_offense.drop(
            columns=["player_name", "team", "position", "position_group"],
            errors="ignore",
        ),
        snap_metrics.drop(
            columns=[
                "snap_player_name", "snap_team", "snap_position",
                "snap_position_group",
            ],
            errors="ignore",
        ),
        ngs_metrics,
        pbp_skill,
        defense,
    ]

    for frame in merge_frames:
        if frame is not None and not frame.empty:
            out = out.merge(frame, on=["season", "player_id"], how="left")

    numeric_cols = [c for c in FINAL_COLS if c not in [
        "season",
        "player_id",
        "player_name",
        "team",
        "position",
        "position_group",
        "master_matched",
        "current_team",
        "current_position",
        "current_position_group",
        "clean_name",
        "initial_last_key",
        "canonical_key",
        "identity_quality_flag",
        "identity_version",
        "advanced_version",
        "date_imported",
    ]]

    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # Weekly stat feeds omit many defenders, OL, and specialists. Use snap
    # participation as the authoritative game-count fallback so `games`
    # represents NFL appearances across every position group.
    if "snap_games" not in out.columns:
        out["snap_games"] = 0.0
    out["snap_games"] = pd.to_numeric(
        out["snap_games"], errors="coerce"
    ).fillna(0.0)
    out["games"] = np.maximum(
        pd.to_numeric(out["games"], errors="coerce").fillna(0.0),
        out["snap_games"],
    )

    for col in [
        "pbp_qb_dropbacks",
        "pbp_qb_epa_per_dropback",
        "pbp_qb_success_rate",
        "pbp_qb_cpoe",
        "pbp_qb_sack_rate",
        "pbp_qb_int_rate",
        "pbp_qb_yards_per_dropback",
        "pbp_targets",
        "pbp_receiving_epa_per_target",
        "pbp_receiving_success_rate",
        "pbp_rush_attempts",
        "pbp_rush_epa_per_carry",
        "pbp_rush_success_rate",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["qb_dropbacks"] = np.where(out["pbp_qb_dropbacks"] > 0, out["pbp_qb_dropbacks"], out["qb_dropbacks"])
    out["qb_epa_per_dropback"] = np.where(
        out["pbp_qb_dropbacks"] > 0,
        out["pbp_qb_epa_per_dropback"],
        out["qb_epa_per_dropback"],
    )
    out["qb_success_rate"] = np.where(
        out["pbp_qb_dropbacks"] > 0,
        out["pbp_qb_success_rate"],
        out["qb_success_rate"],
    )
    out["qb_cpoe"] = np.where(out["pbp_qb_dropbacks"] > 0, out["pbp_qb_cpoe"], out["qb_cpoe"])
    out["qb_sack_rate"] = np.where(
        out["pbp_qb_dropbacks"] > 0,
        out["pbp_qb_sack_rate"],
        out["qb_sack_rate"],
    )
    out["qb_int_rate"] = np.where(
        out["pbp_qb_dropbacks"] > 0,
        out["pbp_qb_int_rate"],
        out["qb_int_rate"],
    )
    out["qb_yards_per_dropback"] = np.where(
        out["pbp_qb_dropbacks"] > 0,
        out["pbp_qb_yards_per_dropback"],
        out["qb_yards_per_dropback"],
    )

    out["receiving_epa_per_target"] = np.where(
        out["pbp_targets"] > 0,
        out["pbp_receiving_epa_per_target"],
        out["receiving_epa_per_target"],
    )
    out["receiving_success_rate"] = np.where(
        out["pbp_targets"] > 0,
        out["pbp_receiving_success_rate"],
        out["receiving_success_rate"],
    )

    out["rush_epa_per_carry"] = np.where(
        out["pbp_rush_attempts"] > 0,
        out["pbp_rush_epa_per_carry"],
        out["rush_epa_per_carry"],
    )
    out["rush_success_rate"] = np.where(
        out["pbp_rush_attempts"] > 0,
        out["pbp_rush_success_rate"],
        out["rush_success_rate"],
    )

    out["pressure_proxy"] = out["def_sacks"] + 0.5 * out["def_half_sacks"] + 0.30 * out["def_qb_hits"]
    out["coverage_playmaking_proxy"] = (
        2.0 * out["def_interceptions"]
        + 0.75 * out["def_pass_defended"]
        + 0.50 * out["def_forced_fumbles"]
    )
    out["run_defense_proxy"] = out["def_tackles"] + 1.5 * out["def_tfl"]

    out["defensive_playmaking_score"] = (
        out["pressure_proxy"]
        + out["coverage_playmaking_proxy"]
        + 0.10 * out["run_defense_proxy"]
    )

    out["advanced_version"] = ADVANCED_VERSION
    out["date_imported"] = pd.to_datetime(dt.datetime.now().strftime("%Y-%m-%d"))

    for col in FINAL_COLS:
        if col not in out.columns:
            out[col] = None

    return out[FINAL_COLS]


def apply_master_identity(advanced, master):
    df = advanced.copy()
    df["player_id"] = clean_player_id(df["player_id"])

    # FINAL_COLS includes identity placeholders. They must be removed before
    # joining the canonical master or pandas will create current_team_x /
    # current_team_y and equivalent suffixed columns.
    placeholder_identity_cols = [
        "current_team",
        "current_position",
        "current_position_group",
        "clean_name",
        "initial_last_key",
        "canonical_key",
        "identity_quality_flag",
        "identity_version",
    ]
    df = df.drop(
        columns=placeholder_identity_cols,
        errors="ignore",
    )

    duplicate_master_ids = master["player_id"].duplicated(
        keep=False
    )
    if duplicate_master_ids.any():
        examples = master.loc[
            duplicate_master_ids,
            ["player_id", "current_player_name"],
        ].head(20)
        raise RuntimeError(
            "Master contains duplicate player_id values:\n"
            + examples.to_string(index=False)
        )

    merged = df.merge(
        master,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    required_master_columns = [
        "current_player_name",
        "current_team",
        "current_position",
        "current_position_group",
    ]
    missing_master_columns = [
        column
        for column in required_master_columns
        if column not in merged.columns
    ]
    if missing_master_columns:
        raise RuntimeError(
            "Canonical master merge is missing expected columns: "
            f"{missing_master_columns}. "
            f"Available columns: {list(merged.columns)}"
        )

    merged["master_matched"] = (
        merged["current_player_name"].notna()
    )

    # Preserve historical team and position in the historical table.  Current
    # roster identity is published separately in current_* fields and is used
    # only when the historical source omitted an identity value.
    merged["player_name"] = merged["player_name"].combine_first(
        merged["current_player_name"]
    )
    merged["team"] = merged["team"].combine_first(merged["current_team"])
    merged["position"] = merged["position"].combine_first(
        merged["current_position"]
    )
    merged["position_group"] = merged["position_group"].combine_first(
        merged["current_position_group"]
    )

    for column in FINAL_COLS:
        if column not in merged.columns:
            merged[column] = None

    return merged[FINAL_COLS]


def get_position_season_weight(position_group_value, season_value):
    position_key = str(position_group_value).upper().strip()
    try:
        season_key = int(season_value)
    except (TypeError, ValueError):
        return 0.0

    curve = POSITION_SEASON_WEIGHTS.get(
        position_key,
        DEFAULT_SEASON_WEIGHTS,
    )
    return float(curve.get(season_key, 0.0))


def build_current_roster_view(advanced, master):
    """Create the current-roster historical view with explicit sample fields.

    Existing metric columns remain position-decayed averages for compatibility.
    Separate history/recent volume fields are published so the downstream player
    model no longer needs to pretend a decayed average is a three-year total.
    """
    current = master.copy()
    hist = advanced.copy()
    hist["season"] = pd.to_numeric(hist["season"], errors="coerce")
    current_group_map = current.set_index("player_id")["current_position_group"]
    hist["weight_position_group"] = hist["player_id"].map(current_group_map).combine_first(
        hist["position_group"]
    )
    hist["season_weight"] = hist.apply(
        lambda row: get_position_season_weight(
            row.get("weight_position_group"), row.get("season")
        ),
        axis=1,
    )

    identity_columns = {
        "season", "player_id", "player_name", "team", "position",
        "position_group", "master_matched", "current_team",
        "current_position", "current_position_group", "clean_name",
        "initial_last_key", "canonical_key", "identity_quality_flag",
        "identity_version", "weight_position_group", "seasons_observed", "available_season_weight",
        "games_history_sum", "total_snaps_history_sum",
        "games_recent_2025", "total_snaps_recent_2025",
        "advanced_version", "date_imported",
    }
    numeric_cols = [column for column in FINAL_COLS if column not in identity_columns]

    numeric_frame = pd.DataFrame(
        {column: pd.to_numeric(hist.get(column), errors="coerce") for column in numeric_cols},
        index=hist.index,
    )
    hist[numeric_cols] = numeric_frame
    weighted_frame = numeric_frame.mul(hist["season_weight"], axis=0)
    weighted_frame.columns = [f"{column}_weighted" for column in numeric_cols]
    hist = pd.concat([hist, weighted_frame], axis=1)

    aggregate_spec = {
        "seasons_observed": ("season", "nunique"),
        "available_season_weight": ("season_weight", "sum"),
        "games_history_sum": ("games", "sum"),
        "total_snaps_history_sum": ("total_snaps", "sum"),
    }
    aggregate_spec.update({
        f"{column}_weighted": (f"{column}_weighted", "sum")
        for column in numeric_cols
    })

    if hist.empty:
        rolled = pd.DataFrame(columns=["player_id", *aggregate_spec])
    else:
        rolled = hist.groupby("player_id", dropna=False).agg(**aggregate_spec).reset_index()

    for column in numeric_cols:
        weighted = f"{column}_weighted"
        rolled[column] = np.where(
            pd.to_numeric(rolled.get("available_season_weight"), errors="coerce").fillna(0) > 0,
            pd.to_numeric(rolled.get(weighted), errors="coerce").fillna(0)
            / pd.to_numeric(rolled.get("available_season_weight"), errors="coerce").replace(0, np.nan),
            np.nan,
        )

    recent = hist[hist["season"].eq(2025)].copy()
    recent = (
        recent.sort_values(["player_id", "total_snaps"], ascending=[True, False])
        .drop_duplicates("player_id", keep="first")
        [["player_id", "games", "total_snaps"]]
        .rename(columns={
            "games": "games_recent_2025",
            "total_snaps": "total_snaps_recent_2025",
        })
    ) if not recent.empty else pd.DataFrame(
        columns=["player_id", "games_recent_2025", "total_snaps_recent_2025"]
    )

    keep_cols = [
        "player_id", "seasons_observed", "available_season_weight",
        "games_history_sum", "total_snaps_history_sum", *numeric_cols,
    ]
    for column in keep_cols:
        if column not in rolled.columns:
            rolled[column] = np.nan

    out = current.merge(rolled[keep_cols], on="player_id", how="left", validate="one_to_one")
    out = out.merge(recent, on="player_id", how="left", validate="one_to_one")
    out["season"] = SEASON
    out["player_name"] = out["current_player_name"]
    out["team"] = out["current_team"]
    out["position"] = out["current_position"]
    out["position_group"] = out["current_position_group"]
    out["master_matched"] = True
    out["advanced_version"] = "v3_1_verified_weekly_identity_history"
    out["date_imported"] = pd.to_datetime(dt.datetime.now().strftime("%Y-%m-%d"))

    for column in FINAL_COLS:
        if column not in out.columns:
            out[column] = None
    return out[FINAL_COLS]


def validate_current_history_rollup(advanced, current):
    """Prove that no current-player history is lost during the rollup."""
    expected_ids = set(
        advanced.loc[
            advanced["master_matched"].eq(True), "player_id"
        ].dropna().astype(str)
    )
    actual_ids = set(
        current.loc[
            pd.to_numeric(
                current["seasons_observed"], errors="coerce"
            ).fillna(0).gt(0),
            "player_id",
        ].dropna().astype(str)
    )

    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unexpected_ids:
        raise RuntimeError(
            "Current-roster history rollup failed identity reconciliation. "
            f"expected={len(expected_ids):,}, actual={len(actual_ids):,}, "
            f"missing={len(missing_ids):,}, unexpected={len(unexpected_ids):,}. "
            f"Missing examples={missing_ids[:20]}; "
            f"unexpected examples={unexpected_ids[:20]}"
        )

    # Anyone with snap history must have at least one game of participation.
    snap_history = pd.to_numeric(
        current["total_snaps_history_sum"], errors="coerce"
    ).fillna(0).gt(0)
    game_history = pd.to_numeric(
        current["games_history_sum"], errors="coerce"
    ).fillna(0).gt(0)
    bad_snap_game = current.loc[
        snap_history & ~game_history,
        ["player_id", "player_name", "position_group",
         "total_snaps_history_sum", "games_history_sum"],
    ]
    if not bad_snap_game.empty:
        raise RuntimeError(
            "Snap-backed players lost game participation in the current "
            "rollup. Examples:\n" + bad_snap_game.head(20).to_string(index=False)
        )

    return len(expected_ids)

def print_season_counts(label, frame):
    if frame is None or frame.empty or "season" not in frame.columns:
        print(f"[ADV][SEASON_CHECK] {label}: no season data")
        return

    check = frame.copy()
    check["season"] = pd.to_numeric(check["season"], errors="coerce")
    counts = (
        check.dropna(subset=["season"])
        .groupby("season")
        .size()
        .reset_index(name="rows")
        .sort_values("season")
    )
    print(f"\n[ADV][SEASON_CHECK] {label}")
    print(counts.to_string(index=False))


def validate_2025_presence(label, frame, required=True):
    if frame is None or frame.empty or "season" not in frame.columns:
        if required:
            raise RuntimeError(f"{label} has no season data; cannot validate 2025.")
        print(f"[ADV][SEASON_CHECK] Warning: {label} has no season data.")
        return

    seasons = set(
        pd.to_numeric(frame["season"], errors="coerce")
        .dropna()
        .astype(int)
    )
    if 2025 not in seasons:
        message = f"{label} does not contain 2025 rows. Seasons found: {sorted(seasons)}"
        if required:
            raise RuntimeError(message)
        print(f"[ADV][SEASON_CHECK] Warning: {message}")

def main():
    print("[ADV] Loading NFL advanced player stats")
    print(f"[ADV] Build ID: {ADVANCED_BUILD_ID}")
    print(f"[ADV] Script path: {Path(__file__).resolve()}")
    print(f"[ADV] Version: {ADVANCED_VERSION}")
    print(f"[ADV] Historical seasons: {HIST_SEASONS}")
    print(f"[ADV] DB: {DB_PATH}")

    engine = get_engine()
    master = load_master(engine)

    weekly = load_weekly_data(HIST_SEASONS)
    raw_snaps = load_snap_counts(HIST_SEASONS)
    player_id_map = load_crosswalk_pfr_map(engine)
    snaps, snap_id_audit = canonicalize_snap_ids(
        raw_snaps,
        player_id_map,
    )
    ngs = load_ngs_data(HIST_SEASONS)
    pbp = load_pbp_data(HIST_SEASONS)

    print_season_counts("weekly", weekly)
    print_season_counts("raw snaps", raw_snaps)
    print_season_counts("canonical snaps", snaps)
    print_season_counts("NGS", ngs)
    print_season_counts("PBP", pbp)

    validate_2025_presence("weekly", weekly, required=False)
    validate_2025_presence("raw snaps", raw_snaps, required=True)
    validate_2025_presence("canonical snaps", snaps, required=True)
    validate_2025_presence("NGS", ngs, required=False)
    validate_2025_presence("PBP", pbp, required=True)

    weekly_seasons = (
        set(
            pd.to_numeric(
                weekly.get("season"),
                errors="coerce",
            ).dropna().astype(int)
        )
        if not weekly.empty and "season" in weekly.columns
        else set()
    )
    if 2025 not in weekly_seasons:
        print(
            "[ADV][SEASON_CHECK] 2025 weekly player stats are "
            "not available from the configured public sources. "
            "The 2025 advanced rows will still be built from "
            "canonical snaps, NGS, and PBP."
        )

    advanced = build_advanced_stats(weekly, snaps, ngs, pbp)
    advanced = apply_master_identity(advanced, master)

    current = build_current_roster_view(advanced, master)
    reconciled_history_players = validate_current_history_rollup(
        advanced, current
    )

    output_csv = OUTPUT_DIR / "nfl_player_advanced_stats_2022_2025.csv"
    current_csv = OUTPUT_DIR / "nfl_player_advanced_stats_current_roster_2026.csv"
    unmatched_csv = OUTPUT_DIR / "nfl_player_advanced_stats_2022_2025_unmatched_master.csv"
    summary_csv = OUTPUT_DIR / "nfl_player_advanced_stats_2022_2025_position_summary.csv"
    snap_id_audit_csv = OUTPUT_DIR / "nfl_player_advanced_stats_snap_id_audit.csv"
    log_file = LOG_DIR / "load_nfl_player_advanced_stats.log"

    advanced.to_csv(output_csv, index=False)
    current.to_csv(current_csv, index=False)

    for table_name, frame in [
        (OUTPUT_TABLE, advanced),
        (OUTPUT_TABLE_CURRENT, current),
        (SNAP_ID_AUDIT_TABLE, snap_id_audit),
    ]:
        if len(frame.columns) == 0:
            raise RuntimeError(
                f"Refusing to persist {table_name}: DataFrame has zero columns. "
                "Every output, including an empty audit, must have an explicit schema."
            )

    advanced.to_sql(OUTPUT_TABLE, con=engine, if_exists="replace", index=False)
    current.to_sql(OUTPUT_TABLE_CURRENT, con=engine, if_exists="replace", index=False)
    snap_id_audit.to_sql(
        SNAP_ID_AUDIT_TABLE,
        con=engine,
        if_exists="replace",
        index=False,
    )
    snap_id_audit.to_csv(
        snap_id_audit_csv,
        index=False,
    )

    unmatched = advanced[advanced["master_matched"] != True].copy()
    unmatched.to_csv(unmatched_csv, index=False)

    summary = (
        advanced.groupby(["season", "position_group"])
        .agg(
            players=("player_id", "nunique"),
            master_matches=("master_matched", "sum"),
            avg_total_snaps=("total_snaps", "mean"),
            avg_qb_dropbacks=("qb_dropbacks", "mean"),
            avg_targets=("targets", "mean"),
            avg_rush_attempts=("rush_attempts", "mean"),
            avg_defensive_playmaking=("defensive_playmaking_score", "mean"),
        )
        .reset_index()
        .sort_values(["season", "position_group"])
    )

    summary.to_csv(summary_csv, index=False)

    master_matches = int(advanced["master_matched"].sum())
    match_rate = master_matches / len(advanced) if len(advanced) else 0.0

    current_with_history = int(
        pd.to_numeric(current["seasons_observed"], errors="coerce")
        .fillna(0)
        .gt(0)
        .sum()
    )
    current_rate = current_with_history / len(current) if len(current) else 0.0
    current_with_snap_history = int(
        pd.to_numeric(current["total_snaps_history_sum"], errors="coerce")
        .fillna(0)
        .gt(0)
        .sum()
    )
    current_with_stat_games = int(
        pd.to_numeric(current["games_history_sum"], errors="coerce")
        .fillna(0)
        .gt(0)
        .sum()
    )

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Run timestamp: {dt.datetime.now()}\n")
        f.write(f"Historical seasons: {HIST_SEASONS}\n")
        f.write(f"Weekly rows: {len(weekly)}\n")
        f.write(f"Raw snap rows: {len(raw_snaps)}\n")
        f.write(f"Canonical snap rows: {len(snaps)}\n")
        f.write(f"Snap ID audit rows: {len(snap_id_audit)}\n")
        f.write(f"NGS rows: {len(ngs)}\n")
        f.write(f"PBP rows: {len(pbp)}\n")
        f.write(f"Advanced rows: {len(advanced)}\n")
        f.write(f"Current roster rows: {len(current)}\n")
        f.write(f"Current-roster master matches: {master_matches}\n")
        f.write(f"Master match rate: {match_rate:.2%}\n")
        f.write(f"Current roster with history: {current_with_history}\n")
        f.write(f"Reconciled current history players: {reconciled_history_players}\n")
        f.write(f"Current roster with history rate: {current_rate:.2%}\n")
        f.write(f"Current roster with snap history: {current_with_snap_history}\n")
        f.write(f"Current roster with game history: {current_with_stat_games}\n")
        f.write(f"Output table: {OUTPUT_TABLE}\n")
        f.write(f"Current output table: {OUTPUT_TABLE_CURRENT}\n")
        f.write(f"Output csv: {output_csv}\n")
        f.write(f"Current csv: {current_csv}\n")
        f.write(f"Unmatched csv: {unmatched_csv}\n")
        f.write(f"Summary csv: {summary_csv}\n")
        f.write(f"Snap ID audit table: {SNAP_ID_AUDIT_TABLE}\n")
        f.write(f"Snap ID audit csv: {snap_id_audit_csv}\n")

    print(f"[ADV] Loaded {len(advanced):,} rows into {OUTPUT_TABLE}")
    print(f"[ADV] Loaded {len(current):,} rows into {OUTPUT_TABLE_CURRENT}")
    print(f"[ADV] Current-roster master matches: {master_matches:,}/{len(advanced):,} ({match_rate:.2%})")
    print(f"[ADV] Current roster with history: {current_with_history:,}/{len(current):,} ({current_rate:.2%})")
    print(f"[ADV] History rollup reconciliation: {reconciled_history_players:,}/{reconciled_history_players:,} matched players retained")
    print(f"[ADV] Current roster with snap history: {current_with_snap_history:,}/{len(current):,}")
    print(f"[ADV] Current roster with game history: {current_with_stat_games:,}/{len(current):,}")
    print(f"[ADV] CSV saved: {output_csv}")
    print(f"[ADV] Current CSV saved: {current_csv}")
    current_ol = current[
        current["current_position_group"].eq("OL")
        | current["position_group"].eq("OL")
    ].copy()
    current_ol_with_snap_history = int(
        current_ol["offense_snaps"].gt(0).sum()
    )

    print(f"[ADV] Summary saved: {summary_csv}")
    print(f"[ADV] Snap ID audit saved: {snap_id_audit_csv}")
    print(
        "[ADV] Current OL with offensive snap history: "
        f"{current_ol_with_snap_history:,}/{len(current_ol):,}"
    )
    print(f"[ADV] Log saved: {log_file}")

    print("\n[ADV] Position summary:")
    print(summary.to_string(index=False))

    history_by_position = (
        current.assign(
            has_history=pd.to_numeric(
                current["seasons_observed"], errors="coerce"
            ).fillna(0).gt(0)
        )
        .groupby("position_group", dropna=False)
        .agg(
            roster_players=("player_id", "count"),
            players_with_history=("has_history", "sum"),
        )
        .reset_index()
    )
    history_by_position["history_rate"] = np.where(
        history_by_position["roster_players"].gt(0),
        history_by_position["players_with_history"]
        / history_by_position["roster_players"],
        0.0,
    )
    print("\n[ADV] Current-roster history coverage by position:")
    print(history_by_position.to_string(index=False))

    print("\n[ADV] Top current-roster defensive playmaking scores:")
    top_cols = [
        "player_name",
        "team",
        "position",
        "position_group",
        "total_snaps",
        "def_sacks",
        "def_qb_hits",
        "def_tfl",
        "def_interceptions",
        "def_pass_defended",
        "defensive_playmaking_score",
    ]

    print(
        current[current["position_group"].isin(["EDGE", "DL", "LB", "LB_EDGE", "DB"])]
        .sort_values("defensive_playmaking_score", ascending=False)
        .head(25)[top_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
