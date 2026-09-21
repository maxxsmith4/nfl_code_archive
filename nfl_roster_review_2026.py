"""Score current all-position roster inputs with the existing frozen estimators.

This is one combined roster/QB scenario. It never reads a manual QB file,
modifies a fitted estimator, or changes baseline projection/staking fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_nfl_team_unit_ratings as units_builder
import nfl_live_depth_2026 as live_depth

VERSION = "v1_1_source_outage_baseline_recovery"
BUILD_ID = "NFL_ROSTER_REVIEW_2026_CANONICAL_V1"
SEASON = 2026
CSV_COLUMNS = (
    "away_expected_qb", "home_expected_qb", "roster_review_flag", "roster_review_reason",
    "roster_adjustment_status", "roster_adjustment_home_points", "roster_adjusted_home_margin",
    "roster_adjusted_spread", "roster_source_as_of_utc",
)


def read_table(connection: Any, name: str) -> pd.DataFrame:
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is None:
        raise ValueError(f"Missing current-roster table {name}; run the updated weekly runner")
    frame = pd.read_sql_query(f'SELECT * FROM "{name}"', connection)
    frame.columns = [str(column).lower() for column in frame]
    return frame


def frame_hash(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    payload = frame[columns].sort_values(columns, na_position="last").to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def check_refresh_status(connection: Any) -> None:
    """Never reuse an older snapshot after a failed current refresh attempt."""
    name = live_depth.REFRESH_STATUS_TABLE
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is None:
        return  # Earlier snapshots and fixtures predate the attempt marker.
    marker = read_table(connection, name)
    if len(marker) != 1 or not {"status", "attempted_at_utc", "error"}.issubset(marker):
        raise ValueError("Invalid current-roster refresh status marker")
    record = marker.iloc[0]
    status = live_depth.text(record.status).upper()
    attempted = live_depth.utc(record.attempted_at_utc).isoformat()
    preview = " ".join(live_depth.text(record.error).split())
    if len(preview) > 400:
        preview = preview[:397] + "..."
    message = f"{status}: latest roster refresh at {attempted}: {preview}"
    if status == "SOURCE_UNAVAILABLE":
        message += "; details: outputs/nfl_live_roster_fetch_diagnostics_2026.json"
        raise live_depth.SourceUnavailable(message)
    if status != "SUCCESS":
        raise ValueError(message)


def load_current_inputs(connection: Any, bundle: dict, learned: Any, cutoff: pd.Timestamp) -> dict:
    check_refresh_status(connection)
    depth = read_table(connection, "nfl_projected_depth_chart_2026")
    units = read_table(connection, "nfl_position_group_ratings_2026")
    source = read_table(connection, live_depth.SOURCE_TABLE)
    availability = read_table(connection, live_depth.AVAILABILITY_TABLE)
    teams = set(bundle["unit_snapshot"]["team"].astype(str))
    required = {"team", "player_id", "player_name", "starter_slot", "is_projected_starter",
                "performance_grade", "replacement_baseline", "likely_unavailable", "date_imported",
                "live_fetched_at_utc", "live_injury_status", "live_uncertain", "live_depth_rank"}
    if required - set(depth):
        raise ValueError(f"Depth chart lacks automatic availability fields: {sorted(required - set(depth))}")
    if set(source.team) != teams or source.team.duplicated().any():
        raise ValueError("Current source audit must contain exactly the 32 model teams")
    if depth.player_id.duplicated().any() or set(depth.team) != teams:
        raise ValueError("Current depth has duplicate players or an incomplete team set")
    if not pd.to_numeric(depth.season, errors="coerce").eq(SEASON).all():
        raise ValueError("Current depth contains the wrong season")
    if not pd.to_numeric(units.season, errors="coerce").eq(SEASON).all():
        raise ValueError("Current units contain the wrong season")
    units["unit_name"] = units.unit_name.astype(str).str.lower()
    if units.duplicated(["team", "unit_name"]).any():
        raise ValueError("Current units have duplicate team/unit rows")
    if set(zip(units.team, units.unit_name)) != {(team, unit) for team in teams for unit in learned.UNIT_NAMES}:
        raise ValueError("Current unit table is not a complete 32 x 8 matrix")
    raw_units = units.pivot(index="team", columns="unit_name", values="unit_rating").astype(float)
    if not np.isfinite(raw_units.to_numpy()).all():
        raise ValueError("Non-finite current unit ratings")
    # Verify that the database units actually consumed this depth snapshot.
    prepared = units_builder.prepare_depth(depth)
    _, rebuilt, contributors, _ = units_builder.build_all(prepared, pd.DataFrame())
    rebuilt["unit_name"] = rebuilt.unit_name.str.lower()
    checked = rebuilt.pivot(index="team", columns="unit_name", values="unit_rating")
    if not np.allclose(raw_units.loc[sorted(teams), list(learned.UNIT_NAMES)],
                       checked.loc[sorted(teams), list(learned.UNIT_NAMES)], atol=1e-8, rtol=0):
        raise ValueError("Current unit ratings do not reconcile to the refreshed depth chart")
    selected = depth[pd.to_numeric(depth.is_projected_starter, errors="coerce").eq(1)].copy()
    if pd.to_numeric(selected.likely_unavailable, errors="coerce").ne(0).any():
        raise ValueError("An unavailable player is still assigned a starter slot")
    selected["starter_slot"] = selected.starter_slot.astype(str)
    if selected.duplicated(["team", "starter_slot"]).any():
        raise ValueError("Duplicate current starter slots")
    selected["slot_value"] = pd.to_numeric(selected.performance_grade, errors="raise") - pd.to_numeric(selected.replacement_baseline, errors="raise")
    raw_slots = selected.pivot(index="team", columns="starter_slot", values="slot_value").reindex(
        index=sorted(teams), columns=list(bundle["slot_names"]))
    missing_slots = {team: [column for column in raw_slots if pd.isna(raw_slots.loc[team, column])] for team in teams}
    # Same numeric preparation as the original model freezer; missing slots are
    # explicitly flagged below instead of being mistaken for confirmed starters.
    raw_slots = raw_slots.fillna(raw_slots.median()).fillna(0.0)
    unit_centers, slot_centers = raw_units.mean(), raw_slots.mean()
    centered_units = raw_units - unit_centers
    centered_slots = raw_slots - slot_centers
    details = {}
    frozen_units = bundle["unit_snapshot"].set_index("team")
    frozen_slots = bundle["slot_snapshot"].set_index("team")
    for team in sorted(teams):
        team_source = source[source.team.eq(team)].iloc[0]
        current = selected[selected.team.eq(team)]
        team_depth = depth[depth.team.eq(team)]
        team_units = units[units.team.eq(team)]
        warnings, errors = [], []
        identity_issues = live_depth.text(team_source.get("identity_issues"))
        if identity_issues:
            errors.append("SOURCE_IDENTITY_CONFLICT: " + identity_issues)
        stamps = []
        for column in ("fetched_at_utc", "roster_source_timestamp", "depth_source_timestamp"):
            stamp = live_depth.utc(team_source[column])
            stamps.append(stamp)
            if stamp > cutoff:
                errors.append(f"SOURCE_AFTER_PREDICTION_CUTOFF:{column}")
            elif cutoff - stamp > pd.Timedelta(hours=24):
                errors.append(f"SOURCE_OLDER_THAN_24_HOURS:{column}")
        if team_source.source_state != "LIVE":
            warnings.append(f"SOURCE_{team_source.source_state}")
        consumed = team_depth.live_fetched_at_utc.map(live_depth.text)
        consumed = consumed[consumed.ne("")].map(live_depth.utc)
        if consumed.empty or not consumed.eq(live_depth.utc(team_source.fetched_at_utc)).all():
            errors.append("DEPTH_DID_NOT_CONSUME_CURRENT_SOURCE_SNAPSHOT")
        depth_stamps = team_depth.date_imported.map(live_depth.utc)
        unit_stamps = team_units.date_imported.map(live_depth.utc)
        if unit_stamps.min() < depth_stamps.max():
            errors.append("UNITS_OLDER_THAN_DEPTH")
        if depth_stamps.max() > cutoff or unit_stamps.max() > cutoff:
            errors.append("BUILD_AFTER_PREDICTION_CUTOFF")
        selected_ids = set(contributors.loc[contributors.team.eq(team), "player_id"].astype(str)) - {""}
        used = team_depth[team_depth.player_id.astype(str).isin(selected_ids)]
        uncertain = used[pd.to_numeric(used.live_uncertain, errors="coerce").fillna(0).gt(0)]
        uncertain_names = [f"{row.player_name} ({row.live_injury_status})" for row in uncertain.itertuples()]
        if uncertain_names:
            warnings.append("UNCERTAIN_AVAILABILITY: " + ", ".join(uncertain_names))
        unknown = used[pd.to_numeric(used.usable_performance_grade, errors="coerce").fillna(0).ne(1)]
        if not unknown.empty:
            warnings.append("REPLACEMENT_LEVEL_GRADE: " + ", ".join(unknown.player_name.astype(str)))
        if missing_slots[team]:
            errors.append("MISSING_CURRENT_STARTER_SLOTS: " + ",".join(missing_slots[team]))
        qbs = current[current.starter_slot.eq("QB1")]
        if len(qbs) != 1:
            errors.append("UPCOMING_QB_NOT_UNIQUELY_IDENTIFIED")
        elif float(qbs.iloc[0].live_depth_rank) > 1:
            warnings.append(f"BACKUP_QB_EXPECTED: {qbs.iloc[0].player_name} (published depth rank {int(qbs.iloc[0].live_depth_rank)})")
        changed_units = [unit for unit in learned.UNIT_NAMES if not np.isclose(
            centered_units.loc[team, unit], float(frozen_units.loc[team, unit]), atol=1e-9, rtol=0)]
        changed_slots = [slot for slot in bundle["slot_names"] if not np.isclose(
            centered_slots.loc[team, slot], float(frozen_slots.loc[team, slot]), atol=1e-9, rtol=0)]
        if changed_units or changed_slots:
            warnings.insert(0, "CURRENT_ROSTER_INPUTS_DIFFER_FROM_FROZEN_SNAPSHOT")
        absent = availability[availability.team.eq(team) & pd.to_numeric(availability.live_unavailable, errors="coerce").eq(1)]
        details[team] = {
            "team": team, "can_adjust": not errors,
            "expected_qb": str(qbs.iloc[0].player_name) if len(qbs) == 1 else "",
            "expected_qb_player_id": str(qbs.iloc[0].player_id) if len(qbs) == 1 else "",
            "changed_units": changed_units, "changed_slots": changed_slots,
            "review_reasons": [*errors, *warnings], "source_state": str(team_source.source_state),
            "source_as_of_utc": min(stamps).isoformat(), "source_sha256": str(team_source.source_sha256),
            "roster_url": str(team_source.roster_url), "depth_url": str(team_source.depth_url),
            "current_starters": {row.starter_slot: {"player_id": str(row.player_id), "player_name": str(row.player_name)} for row in current.itertuples()},
            "unavailable_players": [f"{row.player_name} ({row.live_injury_status})" for row in absent.itertuples()],
            "uncertain_contributors": uncertain_names,
            "unit_inputs": {unit: {"frozen": float(frozen_units.loc[team, unit]),
                                    "current": float(centered_units.loc[team, unit]),
                                    "raw_current": float(raw_units.loc[team, unit]),
                                    "current_league_center": float(unit_centers[unit])} for unit in learned.UNIT_NAMES},
            "slot_inputs": {slot: {"frozen": float(frozen_slots.loc[team, slot]),
                                    "current": float(centered_slots.loc[team, slot]),
                                    "raw_current": float(raw_slots.loc[team, slot]),
                                    "current_league_center": float(slot_centers[slot])} for slot in bundle["slot_names"]},
        }
    return {"units": centered_units, "slots": centered_slots, "details": details,
            "depth_sha256": frame_hash(depth), "units_sha256": frame_hash(units),
            "source_sha256": frame_hash(source)}


def append_scenarios(structural: pd.DataFrame, nonlinear_features: pd.DataFrame,
                     output: pd.DataFrame, bundle: dict, inputs: dict,
                     learned: Any, frozen: Any) -> pd.DataFrame:
    original = output.copy(deep=True)
    structural_live = structural.copy(deep=True)
    nonlinear_live = nonlinear_features.copy(deep=True)
    eligible, records = [], []
    incomplete_center = any(not detail["can_adjust"] for detail in inputs["details"].values())
    columns = [f"{side}_unit_{unit}" for side in ("home", "away") for unit in learned.UNIT_NAMES]
    columns += [f"{side}_slot_{slot}" for side in ("home", "away") for slot in bundle["slot_names"]]
    for position, game in enumerate(structural.to_dict("records")):
        home, away = (inputs["details"][str(game[f"{side}_team"])] for side in ("home", "away"))
        for side in ("home", "away"):
            team = str(game[f"{side}_team"])
            for unit in learned.UNIT_NAMES:
                structural_live.iloc[position, structural_live.columns.get_loc(f"{side}_unit_{unit}")] = inputs["units"].loc[team, unit]
            for slot in bundle["slot_names"]:
                structural_live.iloc[position, structural_live.columns.get_loc(f"{side}_slot_{slot}")] = inputs["slots"].loc[team, slot]
        for unit in learned.UNIT_NAMES:
            nonlinear_live.iloc[position, nonlinear_live.columns.get_loc(f"unit_{unit}")] = (
                inputs["units"].loc[str(game["home_team"]), unit] - inputs["units"].loc[str(game["away_team"]), unit])
        valid = home["can_adjust"] and away["can_adjust"]
        # A common league center cancels when both teams have equal season
        # weights. After a bye, uncertain inputs elsewhere could influence the
        # center without cancelling, so withhold that scenario too.
        center_error = incomplete_center and float(game["home_games_played"]) != float(game["away_games_played"])
        valid = valid and not center_error
        changed = not np.allclose(structural_live.iloc[position][columns].astype(float),
                                  structural.iloc[position][columns].astype(float), atol=1e-9, rtol=0)
        record = {
            "home_expected_qb": home["expected_qb"], "away_expected_qb": away["expected_qb"],
            "roster_review_flag": int(bool(home["review_reasons"] or away["review_reasons"] or center_error)),
            "roster_review_reason": " | ".join([*(f"{row['team']}: {'; '.join(row['review_reasons'])}" for row in (away, home) if row["review_reasons"]),
                                                 *(["UNRESOLVED_LEAGUE_INPUTS_WITH_UNEQUAL_GAMES_PLAYED"] if center_error else [])]),
            "roster_adjustment_status": "CURRENT_DEPTH_SCENARIO" if valid and changed else "NO_CHANGE" if valid else "REVIEW_ONLY",
            "roster_review_details_json": json.dumps({"home": home, "away": away}),
            "roster_review_version": VERSION, "roster_scenario_used_for_staking": 0,
            "roster_scenario_validation_scope": "SUPPLEMENTAL_CURRENT_ROSTER_INPUTS_NOT_HISTORICALLY_VALIDATED",
            "roster_source_as_of_utc": min(home["source_as_of_utc"], away["source_as_of_utc"]),
            **{f"roster_{name}": inputs[name] for name in ("depth_sha256", "units_sha256", "source_sha256")},
        }
        for component in ("unit", "slot", "nonlinear"):
            record[f"roster_adjusted_{component}_projection"] = float(output.iloc[position][f"{component}_projection"]) if valid else np.nan
        if valid and changed:
            eligible.append(position)
        records.append(record)
    extra = pd.DataFrame(records, index=output.index)
    if eligible:
        try:
            variants = {variant.name: variant for variant in learned.VARIANTS}
            for name, component in ((frozen.UNIT_VARIANT, "unit"), (frozen.SLOT_VARIANT, "slot")):
                detail = bundle["linear_models"][name]
                features = learned.make_features(structural_live.iloc[eligible], variants[name], tuple(bundle["slot_names"]),
                                                  detail["parameters"]["transition_k"], detail["parameters"]["maximum_current_season_weight"])
                if set(features.columns) != set(detail["feature_names"]):
                    raise ValueError(f"Current-roster {component} feature contract mismatch")
                values = detail["model"].predict(features[list(detail["feature_names"])])
                extra.iloc[eligible, extra.columns.get_loc(f"roster_adjusted_{component}_projection")] = values
            detail = bundle["nonlinear_model"]
            extra.iloc[eligible, extra.columns.get_loc("roster_adjusted_nonlinear_projection")] = detail["model"].predict(
                nonlinear_live.iloc[eligible][list(detail["feature_names"])])
            if not np.isfinite(extra.iloc[eligible][[f"roster_adjusted_{c}_projection" for c in ("unit", "slot", "nonlinear")]].to_numpy(float)).all():
                raise ValueError("Non-finite current-roster prediction")
        except Exception as exc:
            for position in eligible:
                extra.iloc[position, extra.columns.get_loc("roster_adjustment_status")] = "SCENARIO_ERROR"
                extra.iloc[position, extra.columns.get_loc("roster_review_flag")] = 1
                extra.iloc[position, extra.columns.get_loc("roster_review_reason")] += f" | SCENARIO_ERROR: {type(exc).__name__}: {exc}"
                for component in ("unit", "slot", "nonlinear"):
                    extra.iloc[position, extra.columns.get_loc(f"roster_adjusted_{component}_projection")] = np.nan
    extra["roster_adjusted_home_margin"] = (extra.roster_adjusted_unit_projection + extra.roster_adjusted_nonlinear_projection) / 2.0
    unchanged = extra.roster_adjustment_status.eq("NO_CHANGE")
    extra.loc[unchanged, "roster_adjusted_home_margin"] = output.loc[unchanged, "consensus_projection"]
    extra["roster_adjustment_home_points"] = extra.roster_adjusted_home_margin - output.consensus_projection
    extra["roster_adjusted_projection_range"] = (
        extra[[f"roster_adjusted_{c}_projection" for c in ("unit", "slot", "nonlinear")]].max(axis=1)
        - extra[[f"roster_adjusted_{c}_projection" for c in ("unit", "slot", "nonlinear")]].min(axis=1))
    if set(extra) & set(output):
        raise ValueError("Roster scenario attempted to overwrite baseline columns")
    result = pd.concat([output, extra], axis=1)
    pd.testing.assert_frame_equal(result[original.columns], original, check_exact=True)
    return result


def review_error(output: pd.DataFrame, message: str, status: str = "REVIEW_ERROR") -> pd.DataFrame:
    result = output.copy(deep=True)
    for column in ("home_expected_qb", "away_expected_qb", "roster_source_as_of_utc"):
        result[column] = ""
    result["roster_review_flag"] = 1
    result["roster_review_reason"] = message
    result["roster_adjustment_status"] = status
    result["roster_review_details_json"] = "{}"
    result["roster_review_version"] = VERSION
    result["roster_scenario_used_for_staking"] = 0
    result["roster_scenario_validation_scope"] = "SUPPLEMENTAL_CURRENT_ROSTER_INPUTS_NOT_HISTORICALLY_VALIDATED"
    for column in ("roster_adjusted_home_margin", "roster_adjustment_home_points", "roster_adjusted_unit_projection",
                   "roster_adjusted_slot_projection", "roster_adjusted_nonlinear_projection", "roster_adjusted_projection_range"):
        result[column] = np.nan
    for column in ("roster_depth_sha256", "roster_units_sha256", "roster_source_sha256"):
        result[column] = ""
    return result


def apply_review(connection: Any, structural: pd.DataFrame, nonlinear_features: pd.DataFrame,
                 output: pd.DataFrame, bundle: dict, args: Any, learned: Any, frozen: Any) -> pd.DataFrame:
    try:
        inputs = load_current_inputs(connection, bundle, learned, live_depth.prediction_cutoff(getattr(args, "as_of_date", None)))
        result = append_scenarios(structural, nonlinear_features, output, bundle, inputs, learned, frozen)
    except live_depth.SourceUnavailable as exc:
        message = f"ROSTER_SOURCE_UNAVAILABLE: {exc}"
        print(f"[ROSTER_REVIEW] {message}")
        result = review_error(output, message, status="SOURCE_UNAVAILABLE")
    except Exception as exc:
        message = f"ROSTER_REVIEW_ERROR: {type(exc).__name__}: {exc}"
        print(f"[ROSTER_REVIEW] {message}")
        result = review_error(output, message)
    pd.testing.assert_frame_equal(result[output.columns], output, check_exact=True)
    print(f"[ROSTER_REVIEW] version={VERSION} | current_scenarios={int(result.roster_adjustment_status.eq('CURRENT_DEPTH_SCENARIO').sum())}"
          f" | review_flags={int(result.roster_review_flag.sum())} | staking=FROZEN_BASELINE | manual_QB_file=NOT_USED")
    return result


def append_execution_columns(output: pd.DataFrame, frame: pd.DataFrame, spread_label: Any) -> pd.DataFrame:
    if "roster_review_flag" not in frame:
        return output
    result = output.copy(deep=True)
    for column in CSV_COLUMNS:
        if column == "roster_adjusted_spread":
            result[column] = [spread_label(home, away, margin) if pd.notna(margin) else "" for home, away, margin in
                              zip(frame.home_team, frame.away_team, frame.roster_adjusted_home_margin)]
        elif column == "roster_review_flag":
            result[column] = np.where(frame[column].eq(1), "YES", "NO")
        elif column in ("roster_adjustment_home_points", "roster_adjusted_home_margin"):
            result[column] = pd.to_numeric(frame[column], errors="coerce").round(3)
        else:
            result[column] = frame[column].fillna("").astype(str)
    pd.testing.assert_frame_equal(result[output.columns], output, check_exact=True)
    return result


def write_audit(frame: pd.DataFrame, current_csv: Path | None, snapshot_csv: Path | None) -> None:
    if current_csv is None and snapshot_csv is None:
        return
    rows = []
    for game in frame.to_dict("records"):
        details = json.loads(game["roster_review_details_json"])
        common = {key: game.get(key) for key in (
            "run_id", "game_id", "week", "home_team", "away_team", "projected_home_margin",
            "roster_adjusted_home_margin", "roster_adjustment_home_points", "roster_adjustment_status",
            "roster_adjusted_unit_projection", "roster_adjusted_slot_projection", "roster_adjusted_nonlinear_projection",
            "roster_review_reason", "roster_review_version", "roster_source_as_of_utc",
            "roster_scenario_used_for_staking", "roster_scenario_validation_scope", "learned_bundle_sha256",
            "roster_depth_sha256", "roster_units_sha256", "roster_source_sha256")}
        for side in ("home", "away"):
            row = {**common, "side": side, **details.get(side, {})}
            rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    audit = pd.DataFrame(rows)
    for path in (current_csv, snapshot_csv):
        if path is not None:
            name = path.name.replace("nfl_weekly_power_spread_predictions_2026", "nfl_roster_review_audit_2026")
            if name == path.name:
                name = "roster_review_" + path.name
            audit.to_csv(path.with_name(name), index=False, encoding="utf-8-sig")
    print("[ROSTER_REVIEW] Current starters, source times and model-input changes saved to nfl_roster_review_audit_2026.csv")
