# NFL pipeline guide

## 1. What the models do

There are two distinct prediction systems. The regular weekly model estimates a fair spread independently of the market. The Circa system works with official contest lines and a market-residual model to produce two five-pick cards. Their backtest results and assumptions should be tracked separately.

### Weekly learned consensus

`build_nfl_learned_consensus_2026.py` packages three models: positive Ridge models using eight position units and 26 starter slots, plus a nonlinear process/personnel model. Training uses reconstructed 2020–2025 games, Weeks 2–17. Hyperparameters are selected using prior-season scoring-margin error, not ATS results.

`predict_nfl_weekly_learned_consensus_2026.py` calculates all three projections before attaching prices. The fair margin is the mean of the unit and nonlinear projections; the slot model supplies another agreement check. The frozen gate requires at least a 2-point edge, no more than 3.5 points of range across projections, and all three models on the same side. Week 1 stakes require the explicit `--allow-week1-stakes` override and remain labeled outside the validated scope. Week 18 is outside the default validating scope as well.

The bundle contains frozen 2026 unit and starter snapshots. A structural refresh does **not** automatically replace those snapshots. Current-season process and personnel information can still change through the weekly feature builders. Do not assume a new projected starter is instantly reflected in every frozen structural feature, and do not use `--force-rebuild` as a routine injury update.

### Circa two-entry portfolio

The board loader obtains official Circa PDFs, renders/OCRs them, matches teams to the schedule, and checks complete weeks and opposite spread signs. Live sportsbook lines are not interchangeable with the contest board.

The two policies in the supplied predictor are `TB_V1_P37_D050` and `CEILING_LATE_HOME_FAVORITE_7P5_VETO`. The ceiling bundle uses V1 for Weeks 1–9 and its fixed late-season Ridge architecture for Weeks 10–18. Its lineage is tied to the matching V1 model.

The structural overlay is used for confidence/manual-review context, not to change the model cards (`STRUCTURAL_USED_TO_CHANGE_MODEL_CARD = 0`). However, its source checks still expect the legacy structural projection while the weekly runner now writes learned-consensus projections. This is an unresolved integration issue in the supplied files; see [Known gaps](KNOWN_GAPS.md).

## 2. Structural refresh: where the ratings come from

`run_nfl_structural_refresh.py` runs the following stages sequentially and stops on failure.

| Order | File | Logic |
|---|---|---|
| 1 | `load_nfl_rosters.py` | Load current roster records and preserve player/provider identity. |
| 2 | `build_nfl_player_master.py` | Create a canonical GSIS player population; resolve and audit duplicates. |
| 3 | `build_nfl_player_crosswalk.py` | Reconcile provider/name identifiers to that population, with ambiguity audits. |
| 4 | `nfl_player_advanced_stats.py` | Build 2022–2025 player history and map it to current rosters; include play-by-play, weekly statistics, snaps and Next Gen features where available. |
| 5 | `load_rbsdm_qb_ratings.py` | Build dedicated QB efficiency ratings with sample stabilization and historical season weights; accept optional local QB CSVs. |
| 6 | `load_nfl_defensive_front_metrics.py` | Build front-seven measures using canonical history and snap denominators. |
| 7 | `load_nfl_coverage_metrics.py` | Build coverage measures using canonical history and snap denominators. |
| 8 | `build_nfl_defensive_metrics_summary.py` | Combine available front and coverage components for current defenders. |
| 9 | `build_nfl_player_performance.py` | Combine canonical inputs into position-relative talent grades; apply a single confidence shrink; use PFF blocking grades as OL talent. |
| 10 | `build_nfl_projected_depth_chart.py` | Assign starters using roster eligibility and authoritative QB/position-specific OL evidence. |
| 11 | `build_nfl_ol_continuity.py` | Measure retained OL teammates/combinations; missing history is neutral with zero confidence. |
| 12 | `build_nfl_team_unit_ratings.py` | Aggregate projected players into units and offense/defense/special teams. |
| 13 | `build_nfl_team_strength.py` | Combine units; apply OL continuity once; avoid adding QB strength twice. |
| 14 | `build_nfl_power_ratings.py` | Convert the structural strength into neutral-field point ratings and calibration audits. |

Historical performance follows the player's identity onto the current team. Team is not a permanent identity key. Talent, sample confidence, availability, and starter assignment are separate concepts in the revised stack.

The base unit builder has explicit offense weights (QB 36%, RB 10%, WR/TE 27%, OL 27%), defense weights (DL/EDGE 42%, LB 25%, DB 33%), and overall weights (offense 52%, defense 44%, special teams 4%). These are upstream base-model choices; they are not the learned consensus model's fitted coefficients.

## 3. Weekly production order

`run_nfl_weekly_2026_final.py` runs depth → OL continuity → unit ratings → team strength → power → form → frozen learned model verification/build → live market → learned prediction. `--form-and-predict-only` starts with the weekly form/model stages. `--predict-only` uses existing upstream inputs; it should not be treated as a fresh-data run.

The form builder uses completed games before the target week. Its legacy live rating blends current structural power with opponent-adjusted season/recent form. The documented base formula uses 70% process and 30% results, then 70% season and 30% recent form. The current-season contribution is `games / (games + 5)`, capped at 75%. The learned predictor has its own fitted combination; these base form formulas do not fully describe the final learned model.

The market loader uses the Odds API after independent upstream ratings are built. The predictor attaches the market after freezing its fair projection, then writes edges, eligibility, and the recorded flat/Kelly fields. Use the execution output actually generated by the chosen run; the presence of a stake column is not proof that a row passed its execution gate.

### Usual commands on the existing computer

These document the supplied interfaces, not a claim of successful end-to-end validation of this archive. Resolve the known runner mismatches and restore required databases first.

```powershell
# Full weekly stack; change the week number as needed.
& "C:\Users\maxxs\anaconda3\python.exe" -u `
  "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" `
  --week 2

# Form/prediction refresh when structural inputs are already current.
& "C:\Users\maxxs\anaconda3\python.exe" -u `
  "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" `
  --form-and-predict-only --week 2

# Infrequent complete structural refresh.
& "C:\Users\maxxs\anaconda3\python.exe" -u `
  "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_structural_refresh.py"
```

For Circa, the intended entry point is `run_nfl_circa_weekly_2026.py --week 2`. Its certificate currently has stale version expectations, so the archived collection must be aligned before treating a Circa run as certified. Never bypass that certificate just to get a success message.

## 4. Human review

- Maintain the authoritative QB-starter table and check OL positional assignments after injuries, transactions, and depth-chart changes. A successful data download cannot verify future availability by itself.
- Check injuries announced after the most recent completed game. Prior-game participation is not a forward-looking injury report.
- Verify weather, unusual travel, coaching/context changes, and other late information separately. The presence of weather/rest fields elsewhere does not establish their use by the final learned projection.
- Keep any discretionary override explicit, dated, and separate from the untouched model output. Do not assume the legacy predictor's manual-adjustment table affects the learned model.
- Reconcile the official Circa board, both five-pick entries, deadline, and manual-review flags before submitting. These scripts generate outputs; they are not proof of a completed contest submission or placed wager.

## 5. Historical/research order

1. `prepare_nfl_historical_reconstruction.py`: isolated season rosters, schedule, player master and Week 1 QB seed.
2. `backfill_nfl_historical_advanced_stats.py`: legal rolling four-year player windows.
3. `build_nfl_historical_qb_defense.py`: reuse canonical QB/defense builders with isolated database/table settings.
4. `build_nfl_historical_player_performance_depth.py`: prior-only player/OL history and frozen Week 1 QB/OL personnel.
5. **Missing wrapper:** build isolated OL continuity, units, team strength and power, and mark structural readiness.
6. `replay_nfl_historical_weekly_spreads.py`: replay legacy structural/form projections by week and combine the outputs into `nfl_rebuilt_historical_replay.sqlite`.
7. `audit_nfl_rebuilt_historical_replay.py`: independently recalculate results and audit leakage/duplicates without modifying source databases.

The historical Week 1 personnel seed uses observed Week 1 usage, so graded replay begins at Week 2. The selected replay defaults to 2023–2025 and Weeks 2–17 even though it supports earlier reconstruction. A full 2020–2025 research run requires explicitly consistent target seasons and earlier prior-only inputs across stages; do not infer coverage from the filename or header.

The matchup-residual builder separately creates its research matrix/database and source cache. Official Circa line reconstruction produces the contest database and V1 bundle; the V3.2 builder creates the matching late-season bundle. Learned structural-weight research runs before nonlinear-consensus research. The freeze builder consumes those results, the historical replay, matchup data, and current preseason structural inputs.

Do not run research jobs against the live database as an improvised substitute for the missing historical wrapper. Preserve original frozen models before any intentional rebuild.
