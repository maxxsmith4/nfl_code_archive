#!/usr/bin/env python
"""Freeze the market-free learned structural consensus for the 2026 season.

The training universe is the point-in-time 2020-2025 replay, Weeks 2-17.  All
hyperparameters are selected on prior-season scoring-margin MAE; no spread,
price, ATS result, or 2026 outcome is used to fit a projection model.

The resulting bundle contains:
* a positive ridge model over eight position units;
* a positive ridge model over 26 projected-starter slots;
* a nonlinear process/personnel model;
* the frozen 2026 structural input snapshot; and
* the prospectively frozen consensus gate and OOF probability calibration.

Run once before the first 2026 prediction.  Subsequent calls reuse the frozen
bundle unless --force-rebuild is supplied.  This prevents silent in-season
coefficient or preseason-snapshot drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

import backtest_nfl_learned_structural_weights as learned
import backtest_nfl_nonlinear_matchup_consensus as nonlinear


SEASON = 2026
BUILD_ID = "NFL_LEARNED_CONSENSUS_2026_CANONICAL_V1"
VERSION = "v1_0_2020_2025_frozen_market_free_consensus"

EXPECTED_LEARNED_BUILD_ID = "NFL_LEARNED_STRUCTURAL_WEIGHTS_BACKTEST_CANONICAL_V1"
EXPECTED_LEARNED_VERSION = "v1_0_market_free_nested_season_forward"
EXPECTED_NONLINEAR_BUILD_ID = "NFL_NONLINEAR_MATCHUP_CONSENSUS_BACKTEST_CANONICAL_V1"
EXPECTED_NONLINEAR_VERSION = "v1_0_full_game_market_free_nested_consensus_audit"

TRAINING_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
MINIMUM_WEEK = 2
MAXIMUM_WEEK = 17
UNIT_VARIANT = "LEARNED_UNIT_STRUCTURAL"
SLOT_VARIANT = "LEARNED_SLOT_STRUCTURAL"

FROZEN_ENSEMBLE = "mean_unit_nonlinear"
FROZEN_EDGE_THRESHOLD = 2.0
FROZEN_MAXIMUM_PROJECTION_RANGE = 3.5
FROZEN_MINIMUM_AGREEMENT = 3
FROZEN_OOF_BETS = 252
FROZEN_OOF_WINS = 134
FROZEN_OOF_LOSSES = 111
FROZEN_OOF_PUSHES = 7

UNIT_SOURCE_TABLE = "nfl_position_group_ratings_2026"
DEPTH_SOURCE_TABLE = "nfl_projected_depth_chart_2026"
REGISTRY_TABLE = "nfl_learned_consensus_model_registry_2026"
COEFFICIENT_TABLE = "nfl_learned_consensus_linear_coefficients_2026"
UNIT_SNAPSHOT_TABLE = "nfl_learned_consensus_unit_snapshot_2026"
SLOT_SNAPSHOT_TABLE = "nfl_learned_consensus_slot_snapshot_2026"

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\maxxs\Downloads\Football Files\nfl_model")
DEFAULT_DB_PATH = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--db-path", "--database", dest="db_path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--replay-db", type=Path)
    parser.add_argument("--matchup-db", type=Path)
    parser.add_argument("--learned-weights-db", type=Path)
    parser.add_argument("--consensus-backtest-db", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    args.project_root = args.project_root.expanduser().resolve()
    args.database_root = (args.database_root or args.project_root / "backtests").expanduser().resolve()
    args.db_path = args.db_path.expanduser().resolve()
    args.replay_db = (args.replay_db or args.database_root / learned.REPLAY_DATABASE).expanduser().resolve()
    args.matchup_db = (args.matchup_db or args.database_root / nonlinear.MATCHUP_DATABASE).expanduser().resolve()
    args.learned_weights_db = (
        args.learned_weights_db
        or args.database_root / "nfl_learned_structural_weights_backtest.sqlite"
    ).expanduser().resolve()
    args.consensus_backtest_db = (
        args.consensus_backtest_db
        or args.database_root / "nfl_nonlinear_matchup_consensus_backtest.sqlite"
    ).expanduser().resolve()
    args.model_path = (
        args.model_path or args.project_root / "models" / "nfl_learned_consensus_2026.joblib"
    ).expanduser().resolve()
    args.metadata_path = (
        args.metadata_path
        or args.project_root / "models" / "nfl_learned_consensus_2026_metadata.json"
    ).expanduser().resolve()
    return args


def now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone() is not None


def read_table(connection: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    if not table_exists(connection, table_name):
        raise RuntimeError(f"Missing required table: {table_name}")
    escaped = table_name.replace('"', '""')
    return pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    canonical = frame[columns].copy().sort_values(columns[:2]).reset_index(drop=True)
    payload = canonical.to_csv(index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_research_modules() -> None:
    if learned.BUILD_ID != EXPECTED_LEARNED_BUILD_ID or learned.VERSION != EXPECTED_LEARNED_VERSION:
        raise RuntimeError("The learned-weight research script is not the validated canonical version.")
    if nonlinear.BUILD_ID != EXPECTED_NONLINEAR_BUILD_ID or nonlinear.VERSION != EXPECTED_NONLINEAR_VERSION:
        raise RuntimeError("The nonlinear-consensus research script is not the validated canonical version.")


def validate_paths(args: argparse.Namespace) -> None:
    sources = [
        args.db_path,
        args.replay_db,
        args.matchup_db,
        args.learned_weights_db,
        args.consensus_backtest_db,
        *(args.database_root / f"{season}.sqlite" for season in TRAINING_SEASONS),
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))


def validate_frozen_candidate(args: argparse.Namespace) -> dict[str, Any]:
    with sqlite3.connect(f"{args.consensus_backtest_db.as_uri()}?mode=ro", uri=True) as connection:
        frozen = read_table(connection, "nfl_consensus_frozen_research_candidate")
    row = frozen.loc[frozen["scope"].astype(str).eq("OOF_ALL")]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one OOF_ALL frozen consensus row.")
    row = row.iloc[0]
    expected = {
        "ensemble": FROZEN_ENSEMBLE,
        "edge_threshold": FROZEN_EDGE_THRESHOLD,
        "maximum_projection_range": FROZEN_MAXIMUM_PROJECTION_RANGE,
        "minimum_agreement": FROZEN_MINIMUM_AGREEMENT,
        "bets": FROZEN_OOF_BETS,
        "wins": FROZEN_OOF_WINS,
        "losses": FROZEN_OOF_LOSSES,
        "pushes": FROZEN_OOF_PUSHES,
        "build_id": EXPECTED_NONLINEAR_BUILD_ID,
        "version": EXPECTED_NONLINEAR_VERSION,
    }
    for column, value in expected.items():
        actual = row[column]
        if isinstance(value, float):
            valid = np.isclose(float(actual), value, atol=1e-12)
        elif isinstance(value, int):
            valid = int(actual) == value
        else:
            valid = str(actual) == value
        if not valid:
            raise RuntimeError(
                f"Frozen consensus verification failed for {column}: expected={value!r}, found={actual!r}."
            )
    return {key: expected[key] for key in expected}


def load_oof_probability_calibration(args: argparse.Namespace) -> dict[str, Any]:
    with sqlite3.connect(f"{args.consensus_backtest_db.as_uri()}?mode=ro", uri=True) as connection:
        oof = read_table(connection, "nfl_consensus_oof_predictions")
    required = {"season", "actual_home_margin", "mean_unit_nonlinear_projection"}
    missing = required - set(oof.columns)
    if missing:
        raise RuntimeError(f"Consensus OOF table is missing columns: {sorted(missing)}")
    oof = oof[oof["season"].isin((2022, 2023, 2024, 2025))].copy()
    residual = pd.to_numeric(oof["actual_home_margin"], errors="coerce") - pd.to_numeric(
        oof["mean_unit_nonlinear_projection"], errors="coerce"
    )
    residual = residual.dropna()
    if len(residual) < 900 or float(residual.std(ddof=0)) <= 0:
        raise RuntimeError("OOF residual calibration is incomplete.")
    return {
        "method": "normal_oof_2022_2025_mean_unit_nonlinear_residual",
        "rows": int(len(residual)),
        "mean": float(residual.mean()),
        "sigma": float(residual.std(ddof=0)),
    }


def load_training_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, tuple[str, ...]]:
    shared = SimpleNamespace(
        database_root=args.database_root,
        replay_db=args.replay_db,
        circa_db=args.database_root / learned.CIRCA_DATABASE,
        target_seasons=TRAINING_SEASONS,
        minimum_week=MINIMUM_WEEK,
        maximum_week=MAXIMUM_WEEK,
    )
    frame = learned.load_games(shared)
    frame = learned.attach_form(shared, frame)
    frame = learned.attach_units(shared, frame)
    frame, slot_names = learned.attach_slots(shared, frame)
    learned.validate_modeling_inputs(frame, slot_names)
    if set(frame["season"].astype(int)) != set(TRAINING_SEASONS):
        raise RuntimeError("The training frame does not contain exactly 2020-2025.")
    return frame, slot_names


def fit_linear_models(
    frame: pd.DataFrame, slot_names: tuple[str, ...]
) -> tuple[dict[str, Any], pd.DataFrame]:
    models: dict[str, Any] = {}
    coefficient_rows: list[dict[str, Any]] = []
    variants = {variant.name: variant for variant in learned.VARIANTS}
    for variant_name in (UNIT_VARIANT, SLOT_VARIANT):
        variant = variants[variant_name]
        parameters = learned.choose_parameters(frame, variant, slot_names, SEASON)
        features = learned.make_features(
            frame,
            variant,
            slot_names,
            parameters["transition_k"],
            parameters["maximum_current_season_weight"],
        )
        model = learned.fit_model(
            features,
            frame["actual_home_margin"],
            parameters["ridge_alpha"],
            parameters["target_margin_cap"],
        )
        models[variant_name] = {
            "model": model,
            "parameters": parameters,
            "feature_names": list(features.columns),
        }
        raw = model.named_steps["ridge"].coef_ / model.named_steps["scale"].scale_
        standardized = model.named_steps["ridge"].coef_
        for feature, raw_value, standardized_value in zip(features.columns, raw, standardized):
            coefficient_rows.append(
                {
                    "season": SEASON,
                    "model_variant": variant_name,
                    "feature": feature,
                    "feature_group": learned.feature_group(feature),
                    "raw_coefficient": float(raw_value),
                    "standardized_coefficient": float(standardized_value),
                }
            )
        print(f"[FREEZE] {variant_name}: {json.dumps(parameters, sort_keys=True)}", flush=True)
    return models, pd.DataFrame(coefficient_rows)


def linear_coefficients_from_bundle(bundle: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_name in (UNIT_VARIANT, SLOT_VARIANT):
        details = bundle["linear_models"][variant_name]
        model = details["model"]
        raw = model.named_steps["ridge"].coef_ / model.named_steps["scale"].scale_
        standardized = model.named_steps["ridge"].coef_
        for feature, raw_value, standardized_value in zip(
            details["feature_names"], raw, standardized
        ):
            rows.append(
                {
                    "season": SEASON,
                    "model_variant": variant_name,
                    "feature": feature,
                    "feature_group": learned.feature_group(feature),
                    "raw_coefficient": float(raw_value),
                    "standardized_coefficient": float(standardized_value),
                }
            )
    return pd.DataFrame(rows)


def fit_nonlinear_model(args: argparse.Namespace) -> dict[str, Any]:
    shared = SimpleNamespace(
        database_root=args.database_root,
        replay_db=args.replay_db,
        circa_db=args.database_root / learned.CIRCA_DATABASE,
        matchup_db=args.matchup_db,
        target_seasons=TRAINING_SEASONS,
        minimum_week=MINIMUM_WEEK,
        maximum_week=MAXIMUM_WEEK,
    )
    structural = nonlinear.load_structural_inputs(shared)
    feature_names = nonlinear.nonlinear_feature_names()
    matrices = {
        alpha: nonlinear.load_matchup_matrix(shared, structural, alpha)
        for alpha in nonlinear.RATING_ALPHA_GRID
    }
    parameters, inner_mae = nonlinear.choose_nonlinear_parameters(
        matrices, feature_names, SEASON
    )
    frame = matrices[parameters.rating_alpha]
    model = nonlinear.build_nonlinear_model(parameters)
    model.fit(
        frame.loc[:, feature_names],
        frame["actual_home_margin"].clip(
            -parameters.target_margin_cap, parameters.target_margin_cap
        ),
    )
    values = {
        "rating_alpha": float(parameters.rating_alpha),
        "l2_regularization": float(parameters.l2_regularization),
        "max_leaf_nodes": int(parameters.max_leaf_nodes),
        "target_margin_cap": float(parameters.target_margin_cap),
        "inner_validation_mae": float(inner_mae),
    }
    print(f"[FREEZE] NONLINEAR: {json.dumps(values, sort_keys=True)}", flush=True)
    return {
        "model": model,
        "parameters": values,
        "feature_names": list(feature_names),
    }


def freeze_live_structural_inputs(
    connection: sqlite3.Connection, slot_names: tuple[str, ...], frozen_at: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    form = read_table(connection, "nfl_2026_form_ratings")
    if {"through_week", "games_played"} - set(form.columns):
        raise RuntimeError("The 2026 form table is missing freeze-control columns.")
    through = pd.to_numeric(form["through_week"], errors="coerce")
    games = pd.to_numeric(form["games_played"], errors="coerce")
    if (
        len(form) != 32
        or through.isna().any()
        or games.isna().any()
        or through.ne(0).any()
        or games.ne(0).any()
    ):
        raise RuntimeError(
            "The first 2026 structural freeze must occur before Week 1 with a "
            "32-team, through_week=0 form table. Existing frozen bundles may be "
            "reused in season; structural inputs may not be refrozen in season."
        )
    units = read_table(connection, UNIT_SOURCE_TABLE)
    required_units = {"season", "team", "unit_name", "unit_rating"}
    if required_units - set(units.columns):
        raise RuntimeError(f"{UNIT_SOURCE_TABLE} is missing required columns.")
    units = units[list(required_units)].copy()
    units["team"] = units["team"].astype(str).str.upper().str.strip()
    units["unit_name"] = units["unit_name"].astype(str).str.lower().str.strip()
    units["unit_rating"] = pd.to_numeric(units["unit_rating"], errors="coerce")
    units = units[units["season"].eq(SEASON)].copy()
    if units.duplicated(["team", "unit_name"]).any():
        raise RuntimeError("Current unit source contains duplicate team/unit rows.")
    expected_units = set(learned.UNIT_NAMES)
    if set(units["unit_name"]) != expected_units or units["team"].nunique() != 32:
        raise RuntimeError("Current unit source is not a complete 32-team x 8-unit matrix.")
    unit_wide = units.pivot(index="team", columns="unit_name", values="unit_rating").reset_index()
    for name in learned.UNIT_NAMES:
        unit_wide[name] = unit_wide[name] - unit_wide[name].mean()
    unit_wide["season"] = SEASON
    unit_wide["frozen_at"] = frozen_at

    depth = read_table(connection, DEPTH_SOURCE_TABLE)
    required_depth = {
        "season",
        "team",
        "starter_slot",
        "is_projected_starter",
        "performance_grade",
        "replacement_baseline",
    }
    if required_depth - set(depth.columns):
        raise RuntimeError(f"{DEPTH_SOURCE_TABLE} is missing required columns.")
    depth = depth[list(required_depth)].copy()
    depth = depth[
        depth["season"].eq(SEASON)
        & pd.to_numeric(depth["is_projected_starter"], errors="coerce").fillna(0).eq(1)
        & depth["starter_slot"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    depth["team"] = depth["team"].astype(str).str.upper().str.strip()
    depth["starter_slot"] = depth["starter_slot"].astype(str).str.strip()
    if depth.duplicated(["team", "starter_slot"]).any():
        raise RuntimeError("Current depth source contains duplicate projected starter slots.")
    if set(depth["starter_slot"]) - set(slot_names):
        raise RuntimeError("Current depth source contains an out-of-contract starter slot.")
    depth["slot_value"] = pd.to_numeric(depth["performance_grade"], errors="coerce") - pd.to_numeric(
        depth["replacement_baseline"], errors="coerce"
    )
    slot_wide = depth.pivot(index="team", columns="starter_slot", values="slot_value").reset_index()
    if slot_wide["team"].nunique() != 32:
        raise RuntimeError("Current slot source does not contain 32 teams.")
    for slot in slot_names:
        if slot not in slot_wide:
            slot_wide[slot] = np.nan
        slot_wide[slot] = slot_wide[slot].fillna(slot_wide[slot].median()).fillna(0.0)
        slot_wide[slot] = slot_wide[slot] - slot_wide[slot].mean()
    slot_wide = slot_wide[["team", *slot_names]]
    slot_wide["season"] = SEASON
    slot_wide["frozen_at"] = frozen_at

    unit_hash = stable_frame_hash(unit_wide, ["season", "team", *learned.UNIT_NAMES])
    slot_hash = stable_frame_hash(slot_wide, ["season", "team", *slot_names])
    snapshot_hash = hashlib.sha256(f"{unit_hash}|{slot_hash}".encode("utf-8")).hexdigest()
    return unit_wide, slot_wide, snapshot_hash


def write_registry(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    coefficients: pd.DataFrame,
    unit_snapshot: pd.DataFrame,
    slot_snapshot: pd.DataFrame,
    model_sha256: str,
) -> None:
    created_at = bundle["created_at"]
    registry = pd.DataFrame(
        [
            {
                "season": SEASON,
                "build_id": BUILD_ID,
                "version": VERSION,
                "deployment_status": "PROSPECTIVE_2026_LIVE_TEST",
                "training_seasons": ",".join(map(str, TRAINING_SEASONS)),
                "training_week_range": f"{MINIMUM_WEEK}-{MAXIMUM_WEEK}",
                "training_games": bundle["training_games"],
                "model_path": str(args.model_path),
                "metadata_path": str(args.metadata_path),
                "model_sha256": model_sha256,
                "structural_snapshot_hash": bundle["structural_snapshot_hash"],
                "unit_parameters_json": json.dumps(bundle["linear_models"][UNIT_VARIANT]["parameters"], sort_keys=True),
                "slot_parameters_json": json.dumps(bundle["linear_models"][SLOT_VARIANT]["parameters"], sort_keys=True),
                "nonlinear_parameters_json": json.dumps(bundle["nonlinear_model"]["parameters"], sort_keys=True),
                "ensemble": FROZEN_ENSEMBLE,
                "minimum_edge_points": FROZEN_EDGE_THRESHOLD,
                "maximum_projection_range_points": FROZEN_MAXIMUM_PROJECTION_RANGE,
                "minimum_agreement": FROZEN_MINIMUM_AGREEMENT,
                "week1_eligible": 0,
                "week18_eligible_by_default": 0,
                "projection_uses_market_inputs": 0,
                "gate_uses_market_after_projection_freeze": 1,
                "coefficients_frozen_during_2026": 1,
                "structural_inputs_frozen_during_2026": 1,
                "created_at": created_at,
            }
        ]
    )
    for frame in (coefficients, unit_snapshot, slot_snapshot):
        frame["build_id"] = BUILD_ID
        frame["version"] = VERSION
        frame["structural_snapshot_hash"] = bundle["structural_snapshot_hash"]
        frame["date_imported"] = created_at
    with sqlite3.connect(args.db_path) as connection:
        registry.to_sql(REGISTRY_TABLE, connection, if_exists="replace", index=False)
        coefficients.to_sql(COEFFICIENT_TABLE, connection, if_exists="replace", index=False)
        unit_snapshot.to_sql(UNIT_SNAPSHOT_TABLE, connection, if_exists="replace", index=False)
        slot_snapshot.to_sql(SLOT_SNAPSHOT_TABLE, connection, if_exists="replace", index=False)
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if check != "ok":
        raise RuntimeError(f"Production database integrity check failed: {check}")


def reusable_bundle(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.force_rebuild or not args.model_path.exists() or not args.metadata_path.exists():
        return None
    bundle = joblib.load(args.model_path)
    required = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "season": SEASON,
        "training_seasons": TRAINING_SEASONS,
    }
    for key, expected in required.items():
        actual = bundle.get(key)
        if key == "training_seasons":
            actual = tuple(actual or ())
        if actual != expected:
            raise RuntimeError(
                f"Existing model bundle has incompatible {key}: expected={expected!r}, found={actual!r}. "
                "Use --force-rebuild only after resolving the lineage change."
            )
    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    actual_sha256 = sha256_file(args.model_path)
    if metadata.get("model_sha256") != actual_sha256:
        raise RuntimeError(
            "Existing model file does not match its metadata SHA-256; refusing to reuse it."
        )
    return bundle


def main() -> int:
    started = datetime.now(timezone.utc)
    args = parse_args()
    print("=" * 112)
    print("[FREEZE] NFL LEARNED CONSENSUS — 2026 PRODUCTION FREEZE")
    print("=" * 112)
    print(f"[FREEZE] Build ID: {BUILD_ID}")
    print(f"[FREEZE] Version: {VERSION}")
    print("[FREEZE] Projection model uses market inputs: NO")
    print("[FREEZE] Hyperparameters use ATS results: NO")
    validate_research_modules()
    validate_paths(args)

    existing = reusable_bundle(args)
    if existing is not None:
        model_sha256 = sha256_file(args.model_path)
        unit_snapshot = existing["unit_snapshot"].copy()
        slot_snapshot = existing["slot_snapshot"].copy()
        coefficients = linear_coefficients_from_bundle(existing)
        unit_hash = stable_frame_hash(
            unit_snapshot, ["season", "team", *learned.UNIT_NAMES]
        )
        slot_hash = stable_frame_hash(
            slot_snapshot, ["season", "team", *tuple(existing["slot_names"])]
        )
        reconstructed_hash = hashlib.sha256(
            f"{unit_hash}|{slot_hash}".encode("utf-8")
        ).hexdigest()
        if reconstructed_hash != existing["structural_snapshot_hash"]:
            raise RuntimeError("Frozen unit snapshot does not match the reusable bundle.")
        write_registry(args, existing, coefficients, unit_snapshot, slot_snapshot, model_sha256)
        print(f"[FREEZE] Reused frozen model: {args.model_path}")
        print(f"[FREEZE] SHA-256: {model_sha256}")
        return 0

    frozen_candidate = validate_frozen_candidate(args)
    probability = load_oof_probability_calibration(args)
    training_frame, slot_names = load_training_frame(args)
    linear_models, coefficients = fit_linear_models(training_frame, slot_names)
    nonlinear_model = fit_nonlinear_model(args)
    created_at = now_string()
    with sqlite3.connect(args.db_path) as connection:
        unit_snapshot, slot_snapshot, snapshot_hash = freeze_live_structural_inputs(
            connection, slot_names, created_at
        )

    bundle: dict[str, Any] = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "season": SEASON,
        "created_at": created_at,
        "deployment_status": "PROSPECTIVE_2026_LIVE_TEST",
        "training_seasons": TRAINING_SEASONS,
        "training_week_range": (MINIMUM_WEEK, MAXIMUM_WEEK),
        "training_games": int(len(training_frame)),
        "market_features_in_projection": 0,
        "hyperparameters_selected_on_ats": 0,
        "structural_inputs_frozen_during_2026": 1,
        "structural_snapshot_hash": snapshot_hash,
        "unit_snapshot": unit_snapshot,
        "slot_snapshot": slot_snapshot,
        "slot_names": slot_names,
        "linear_models": linear_models,
        "nonlinear_model": nonlinear_model,
        "probability_calibration": probability,
        "gate": {
            "ensemble": FROZEN_ENSEMBLE,
            "edge_threshold": FROZEN_EDGE_THRESHOLD,
            "maximum_projection_range": FROZEN_MAXIMUM_PROJECTION_RANGE,
            "minimum_agreement": FROZEN_MINIMUM_AGREEMENT,
            "week1_eligible": 0,
            "week18_eligible_by_default": 0,
        },
        "frozen_candidate_verification": frozen_candidate,
        "research_build_ids": {
            "learned": EXPECTED_LEARNED_BUILD_ID,
            "nonlinear": EXPECTED_NONLINEAR_BUILD_ID,
        },
    }
    atomic_joblib_dump(bundle, args.model_path)
    model_sha256 = sha256_file(args.model_path)
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"linear_models", "nonlinear_model", "unit_snapshot", "slot_snapshot"}
    }
    metadata["linear_parameters"] = {
        name: details["parameters"] for name, details in linear_models.items()
    }
    metadata["nonlinear_parameters"] = nonlinear_model["parameters"]
    metadata["model_sha256"] = model_sha256
    atomic_json_dump(metadata, args.metadata_path)
    write_registry(
        args,
        bundle,
        coefficients,
        unit_snapshot,
        slot_snapshot,
        model_sha256,
    )

    print(f"[FREEZE] Training games: {len(training_frame):,}")
    print(f"[FREEZE] Structural snapshot: {snapshot_hash}")
    print(f"[FREEZE] Model: {args.model_path}")
    print(f"[FREEZE] Metadata: {args.metadata_path}")
    print(f"[FREEZE] SHA-256: {model_sha256}")
    print(f"[FREEZE] Completed in {(datetime.now(timezone.utc) - started).total_seconds():.2f} seconds")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[FREEZE] Cancelled by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"[FREEZE] FAILED: {exc}", file=sys.stderr)
        raise
