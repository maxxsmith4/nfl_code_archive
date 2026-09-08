# Known gaps and compatibility issues

This is an evidence-based backup of the files supplied in this conversation. It is not yet a verified installation that can rebuild everything from an empty machine.

## Historical Stage 5 is absent

The current replay explicitly consumes `nfl_power_ratings_target`, `nfl_power_rating_calibration_season_audit_target`, and a context row with `structural_inputs_ready_flag = 1` in every isolated season database. The supplied historical wrappers cover Stages 1–4. No supplied script was identified as the Stage 5 wrapper that produces those isolated structural outputs. The equivalent production components are included, but simply running them would target live 2026 tables rather than reconstruct historical seasons correctly.

To locate the missing wrapper on Windows:

```powershell
Get-ChildItem "C:\Users\maxxs\Downloads\Football Files\nfl_model" -Recurse -Filter *.py |
    Select-String -SimpleMatch "structural_inputs_ready_flag" |
    Select-Object -ExpandProperty Path -Unique
```

Existing validated season databases can preserve that state for recovery. Rebuilding them from scratch still requires the correct wrapper and its inputs. No guessed replacement is included.

## Circa and weekly output expectations differ

| Consumer | Supplied expectation | Supplied current producer |
|---|---|---|
| `run_nfl_circa_weekly_2026.py` | Form V2 | Form V3 |
| `run_nfl_circa_weekly_2026.py` | Structural weekly V3 | Main weekly runner writes learned-consensus V1 |
| `predict_nfl_circa_top5_2026.py` | Legacy structural V4 / `STRUCTURAL_FORM_HFA` | Main weekly runner writes `LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS` to the same table |
| `run_nfl_weekly_form.py` | Learned predictor version `v1_0_frozen_learned_weights_point_in_time_consensus` | Supplied predictor version `v1_3_readable_execution_csv_scope_fix` |

These exact contracts can reject otherwise populated tables. Both predictor families remain included because historical replay and shared helper imports require the legacy file. Running the legacy predictor after the learned one also overwrites the current output table; that is not a documented production fix.

The next integration change should align the Circa confidence overlay and certificate with the approved weekly source and verify both cards and audit lineage. This archive does not silently change the accepted model/overlay source or suppress a guard.

## Frozen-model reuse still requires research databases

`build_nfl_learned_consensus_2026.py` calls `validate_paths()` before `reusable_bundle()`. Consequently, its ordinary reuse path requires the live database, isolated 2020–2025 databases, replay database, matchup database, learned-weight database, and nonlinear-consensus backtest database even when the joblib and metadata exist. Back those up. The archive has not reordered this validation.

The learned predictor also needs historical play-by-play/snap cache data for current matchup construction. The model bundle alone is insufficient.

## Data, environment, and operations

- Live and historical SQLite files, source caches, PFF inputs, manual QB/OL records, schedules stored in SQLite, official-board caches/corrections, and private configuration were not uploaded as a full recovery set.
- Requirements are a proposed compatibility specification, not a captured working environment. The prior artifact review identified scikit-learn 1.3.0; export the actual working environment as described in the recovery guide.
- Several scripts hard-code Windows paths. The old `2026_nfl_files` root is normalized, but this is still a Windows-oriented archive.
- The weekly runner can prefer an older alternate depth-chart filename if it is left in the installation. Only the canonical supplied depth script is in this package; inspect that preference when restoring over an old directory.
- The structural runner's lock cleanup should be reviewed before parallel/scheduled launches. Keep runs sequential; an overlapping failed launch can interfere with lock ownership.
- No full end-to-end database or API run was performed for this archive. Compilation and file checks do not validate business logic or live data coverage.
