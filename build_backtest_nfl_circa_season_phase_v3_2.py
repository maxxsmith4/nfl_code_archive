#!/usr/bin/env python
"""Rebuild the fixed Circa V3.2 season-phase ceiling after QB repair.

This stage deliberately preserves the existing production architecture:

* Weeks 1-9 use the locked V1 model exactly.
* Weeks 10-18 use a Ridge model trained on strict historical Circa rows.
* The selected late alpha remains 1000 and the blend weight remains 1.0.

It does not reuse the old V3.2 selection-gate statistics because those were
calculated while the QB EPA/CPOE source fields were dead.  The repaired V1
bundle and repaired Circa backtest database must both pass explicit QB
coverage and variance gates.  The output is cryptographically tied to the V1
bundle that produced it, so mixed old/new model generations cannot load.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BUILD_ID = "NFL_CIRCA_CONTEST_SEASON_PHASE_V3_2_QB_REBUILT"
VERSION = "v3_2_1_fixed_architecture_qb_integrity_lineage"

DEFAULT_PROJECT_ROOT = Path(
    r"C:\Users\maxxs\Downloads\Football Files\nfl_model"
)
DATABASE_RELATIVE_PATH = Path("backtests/nfl_circa_contest_lines.sqlite")
V1_MODEL_RELATIVE_PATH = Path("models/nfl_circa_contest_model_v1.joblib")
OUTPUT_MODEL_RELATIVE_PATH = Path(
    "models/nfl_circa_contest_season_phase_v3_2.joblib"
)
OUTPUT_METADATA_RELATIVE_PATH = Path(
    "models/nfl_circa_contest_season_phase_v3_2_metadata.json"
)

BACKTEST_TABLE = "nfl_circa_backtest_matrix"
RUN_AUDIT_TABLE = "nfl_circa_run_audit"
LATE_ALPHAS = (500.0, 750.0, 1000.0, 1500.0, 2000.0)
SELECTED_LATE_ALPHA = 1000.0
SELECTED_BLEND_WEIGHT = 1.0
LATE_FIRST_WEEK = 10
LATE_LAST_WEEK = 18
MINIMUM_QB_COVERAGE = 0.99
MINIMUM_QB_STANDARD_DEVIATION = 1e-6


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--v1-model-path", type=Path)
    parser.add_argument("--output-model-path", type=Path)
    parser.add_argument("--output-metadata-path", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.database_path = (
        args.database_path.expanduser().resolve()
        if args.database_path
        else args.project_root / DATABASE_RELATIVE_PATH
    )
    args.v1_model_path = (
        args.v1_model_path.expanduser().resolve()
        if args.v1_model_path
        else args.project_root / V1_MODEL_RELATIVE_PATH
    )
    args.output_model_path = (
        args.output_model_path.expanduser().resolve()
        if args.output_model_path
        else args.project_root / OUTPUT_MODEL_RELATIVE_PATH
    )
    args.output_metadata_path = (
        args.output_metadata_path.expanduser().resolve()
        if args.output_metadata_path
        else args.project_root / OUTPUT_METADATA_RELATIVE_PATH
    )
    return args


def now_string() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not exists:
        raise RuntimeError(f"Required table is missing: {table}")
    return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def qb_integrity_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "season",
        "week",
        "home_qb_recent_epa",
        "away_qb_recent_epa",
        "home_qb_recent_cpoe",
        "away_qb_recent_cpoe",
        "qb_recent_epa_advantage",
        "qb_recent_cpoe_advantage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Season-phase QB integrity fields are missing: {missing}")

    scope = frame[
        numeric(frame, "season").between(2020, 2025)
        & numeric(frame, "week").between(2, 17)
    ].copy()
    if scope.empty:
        raise RuntimeError("Season-phase QB integrity audit has no eligible rows.")

    side_epa = pd.concat(
        [numeric(scope, "home_qb_recent_epa"), numeric(scope, "away_qb_recent_epa")],
        ignore_index=True,
    )
    side_cpoe = pd.concat(
        [
            numeric(scope, "home_qb_recent_cpoe"),
            numeric(scope, "away_qb_recent_cpoe"),
        ],
        ignore_index=True,
    )
    epa_advantage = numeric(scope, "qb_recent_epa_advantage")
    cpoe_advantage = numeric(scope, "qb_recent_cpoe_advantage")
    metrics = {
        "qb_feature_integrity_passed": 0,
        "qb_side_epa_coverage": float(side_epa.notna().mean()),
        "qb_side_cpoe_coverage": float(side_cpoe.notna().mean()),
        "qb_epa_advantage_std": float(epa_advantage.std(ddof=0)),
        "qb_cpoe_advantage_std": float(cpoe_advantage.std(ddof=0)),
        "qb_epa_advantage_nonzero_rows": int(
            epa_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
        "qb_cpoe_advantage_nonzero_rows": int(
            cpoe_advantage.fillna(0).abs().gt(1e-12).sum()
        ),
    }
    passed = (
        metrics["qb_side_epa_coverage"] >= MINIMUM_QB_COVERAGE
        and metrics["qb_side_cpoe_coverage"] >= MINIMUM_QB_COVERAGE
        and metrics["qb_epa_advantage_std"] > MINIMUM_QB_STANDARD_DEVIATION
        and metrics["qb_cpoe_advantage_std"] > MINIMUM_QB_STANDARD_DEVIATION
        and metrics["qb_epa_advantage_nonzero_rows"] > 0
        and metrics["qb_cpoe_advantage_nonzero_rows"] > 0
    )
    metrics["qb_feature_integrity_passed"] = int(passed)
    if not passed:
        raise RuntimeError(
            "Season-phase QB feature-integrity gate failed; rebuild blocked: "
            + json.dumps(metrics, sort_keys=True)
        )
    return metrics


def assert_saved_qb_integrity(bundle: dict[str, Any], label: str) -> None:
    required = {
        "qb_feature_integrity_passed",
        "qb_side_epa_coverage",
        "qb_side_cpoe_coverage",
        "qb_epa_advantage_std",
        "qb_cpoe_advantage_std",
        "qb_epa_advantage_nonzero_rows",
        "qb_cpoe_advantage_nonzero_rows",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(
            f"{label} predates the QB repair; missing metadata: {missing}"
        )
    metrics = {
        key: bundle[key]
        for key in required
    }
    valid = (
        int(metrics["qb_feature_integrity_passed"]) == 1
        and float(metrics["qb_side_epa_coverage"]) >= MINIMUM_QB_COVERAGE
        and float(metrics["qb_side_cpoe_coverage"]) >= MINIMUM_QB_COVERAGE
        and float(metrics["qb_epa_advantage_std"]) > MINIMUM_QB_STANDARD_DEVIATION
        and float(metrics["qb_cpoe_advantage_std"]) > MINIMUM_QB_STANDARD_DEVIATION
        and int(metrics["qb_epa_advantage_nonzero_rows"]) > 0
        and int(metrics["qb_cpoe_advantage_nonzero_rows"]) > 0
    )
    if not valid:
        raise RuntimeError(
            f"{label} failed saved QB integrity: "
            + json.dumps(metrics, sort_keys=True)
        )


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def matrix_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(["season", "week", "game_id"])[columns]
    payload = ordered.to_csv(index=False, float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_bundle(
    database_path: Path,
    v1_model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not database_path.exists():
        raise FileNotFoundError(database_path)
    if not v1_model_path.exists():
        raise FileNotFoundError(v1_model_path)

    with sqlite3.connect(database_path) as connection:
        frame = read_table(connection, BACKTEST_TABLE)
        run_audit = read_table(connection, RUN_AUDIT_TABLE)
    if len(run_audit) != 1:
        raise RuntimeError(
            f"Expected one Circa run-audit row; found {len(run_audit)}."
        )
    source_audit = run_audit.iloc[0].to_dict()
    if int(source_audit.get("circa_qb_feature_integrity_passed", 0)) != 1:
        raise RuntimeError(
            "Circa database was not produced by the repaired QB pipeline."
        )

    v1_bundle = joblib.load(v1_model_path)
    if not isinstance(v1_bundle, dict):
        raise RuntimeError("V1 model bundle is not a dictionary.")
    assert_saved_qb_integrity(v1_bundle, "Circa V1 model bundle")
    if abs(float(v1_bundle.get("qb_epa_standardized_coefficient", 0.0))) <= 1e-12:
        raise RuntimeError(
            "Circa V1 bundle has no nonzero repaired QB EPA coefficient."
        )

    required_v1 = {
        "model",
        "build_id",
        "version",
        "created_at",
        "feature_order",
        "feature_set",
        "rating_alpha",
        "model_alpha",
        "selection_policy",
    }
    missing = sorted(required_v1 - set(v1_bundle))
    if missing:
        raise RuntimeError(f"V1 model bundle is missing keys: {missing}")
    feature_order = [str(value) for value in v1_bundle["feature_order"]]
    missing_features = sorted(set(feature_order) - set(frame.columns))
    if missing_features:
        raise RuntimeError(
            f"Circa backtest matrix is missing V1 features: {missing_features}"
        )
    if "actual_circa_residual" not in frame.columns:
        raise RuntimeError("Circa backtest matrix lacks actual_circa_residual.")

    qb_metrics = qb_integrity_metrics(frame)
    for key, value in qb_metrics.items():
        if key in v1_bundle and not math.isclose(
            float(v1_bundle[key]), float(value), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"V1 model/database QB integrity mismatch for {key}: "
                f"model={v1_bundle[key]!r}, database={value!r}"
            )

    strict = frame[
        numeric(frame, "week").between(LATE_FIRST_WEEK, LATE_LAST_WEEK)
        & numeric(frame, "reference_fallback_used").fillna(0).eq(0)
        & numeric(frame, "actual_circa_residual").notna()
    ].copy()
    if strict.empty:
        raise RuntimeError("No strict Weeks 10-18 Circa rows are available.")
    if strict[feature_order].notna().sum().eq(0).any():
        dead = strict[feature_order].notna().sum().loc[lambda value: value.eq(0)]
        raise RuntimeError(
            "Late-model training features are entirely missing: "
            + ", ".join(dead.index)
        )

    late_models: dict[float, Pipeline] = {}
    for alpha in LATE_ALPHAS:
        model = build_pipeline(alpha)
        model.fit(strict[feature_order], strict["actual_circa_residual"])
        late_models[float(alpha)] = model

    selected_model = late_models[SELECTED_LATE_ALPHA]
    qb_coefficient = float(
        selected_model.named_steps["ridge"].coef_[
            feature_order.index("qb_recent_epa_advantage")
        ]
    )
    if abs(qb_coefficient) <= 1e-12:
        raise RuntimeError(
            "Selected late model has a zero QB EPA coefficient after repair."
        )

    lineage = {
        "source_v1_build_id": str(v1_bundle["build_id"]),
        "source_v1_version": str(v1_bundle["version"]),
        "source_v1_created_at": str(v1_bundle["created_at"]),
        "source_v1_sha256": sha256_file(v1_model_path),
    }
    fingerprint_columns = [
        "season",
        "week",
        "game_id",
        *feature_order,
        "actual_circa_residual",
    ]
    bundle: dict[str, Any] = {
        "build_id": BUILD_ID,
        "version": VERSION,
        "created_at": now_string(),
        "source_database": str(database_path),
        "source_database_sha256": sha256_file(database_path),
        "source_circa_build_id": str(source_audit.get("build_id", "")),
        "source_circa_version": str(source_audit.get("version", "")),
        **lineage,
        "feature_order": feature_order,
        "feature_set": str(v1_bundle["feature_set"]),
        "rating_alpha": float(v1_bundle["rating_alpha"]),
        "v1_model_alpha": float(v1_bundle["model_alpha"]),
        "v1_selection_policy": str(v1_bundle["selection_policy"]),
        "v1_model": v1_bundle["model"],
        "late_alphas": tuple(float(value) for value in LATE_ALPHAS),
        "late_models": late_models,
        "early_week_policy": "EXACT_V1_WEEKS_1_9",
        "late_week_policy": "FIXED_ALPHA_1000_WEEKS_10_18",
        "ceiling_source": {
            "late_alpha": SELECTED_LATE_ALPHA,
            "blend_weight": SELECTED_BLEND_WEIGHT,
            "direction_policy": "BLENDED_SIGN",
        },
        "selection_provenance": (
            "PRESERVED_PREEXISTING_V3_2_ARCHITECTURE_NOT_RESELECTED"
        ),
        "stale_pre_repair_gate_statistics_reused": False,
        "implementation_ready": bool(v1_bundle.get("implementation_ready", False)),
        "shadow_only": False,
        "late_training_rows": int(len(strict)),
        "late_training_seasons": sorted(
            int(value) for value in numeric(strict, "season").dropna().unique()
        ),
        "late_training_matrix_sha256": matrix_fingerprint(
            strict, fingerprint_columns
        ),
        "selected_late_qb_epa_standardized_coefficient": qb_coefficient,
        **qb_metrics,
    }
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"v1_model", "late_models"}
    }
    metadata["late_model_alphas_saved"] = list(LATE_ALPHAS)
    return bundle, metadata


def atomic_save_joblib(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)


def atomic_save_json(metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_self_test() -> int:
    frame = pd.DataFrame(
        {
            "season": [2020, 2021, 2022, 2023, 2024, 2025],
            "week": [2, 3, 4, 5, 6, 7],
            "home_qb_recent_epa": [0.1, 0.2, 0.3, 0.1, -0.1, 0.0],
            "away_qb_recent_epa": [0.0, -0.1, 0.1, -0.2, 0.2, 0.1],
            "home_qb_recent_cpoe": [1, 2, 3, 4, 5, 6],
            "away_qb_recent_cpoe": [0, -1, 1, 2, 1, 0],
            "qb_recent_epa_advantage": [0.1, 0.3, 0.2, 0.3, -0.3, -0.1],
            "qb_recent_cpoe_advantage": [1, 3, 2, 2, 4, 6],
        }
    )
    if qb_integrity_metrics(frame)["qb_feature_integrity_passed"] != 1:
        raise AssertionError("Populated QB fixture should pass.")
    dead = frame.copy()
    dead["qb_recent_epa_advantage"] = 0.0
    try:
        qb_integrity_metrics(dead)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Dead QB fixture should fail.")
    print("[CIRCA_PHASE_V3_2] Self-test passed.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    print("[CIRCA_PHASE_V3_2] Rebuilding fixed season-phase ceiling")
    print(f"[CIRCA_PHASE_V3_2] Build ID: {BUILD_ID}")
    print(f"[CIRCA_PHASE_V3_2] Version: {VERSION}")
    bundle, metadata = build_bundle(args.database_path, args.v1_model_path)
    atomic_save_joblib(bundle, args.output_model_path)
    atomic_save_json(metadata, args.output_metadata_path)

    reloaded = joblib.load(args.output_model_path)
    assert_saved_qb_integrity(reloaded, "Reloaded V3.2 ceiling bundle")
    if str(reloaded.get("source_v1_created_at", "")) != str(
        bundle["source_v1_created_at"]
    ):
        raise RuntimeError("V3.2 bundle reload lineage validation failed.")
    print(
        "[CIRCA_PHASE_V3_2] QB integrity: "
        f"EPA coverage={bundle['qb_side_epa_coverage']:.2%} | "
        f"CPOE coverage={bundle['qb_side_cpoe_coverage']:.2%} | "
        f"EPA std={bundle['qb_epa_advantage_std']:.6f} | "
        f"CPOE std={bundle['qb_cpoe_advantage_std']:.6f}"
    )
    print(
        "[CIRCA_PHASE_V3_2] Selected late QB EPA coefficient: "
        f"{bundle['selected_late_qb_epa_standardized_coefficient']:.8f}"
    )
    print(f"[CIRCA_PHASE_V3_2] Model bundle: {args.output_model_path}")
    print(f"[CIRCA_PHASE_V3_2] Metadata: {args.output_metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[CIRCA_PHASE_V3_2] Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[CIRCA_PHASE_V3_2] FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
