# File-by-file reference

Every supplied current Python stage is listed below. Read the purpose first; expand the technical reference only when investigating a specific table, argument or function. Constants and source comments describe implemented intent, not proof of a successful live run. [Known gaps](KNOWN_GAPS.md) takes precedence over stale source headers.

## audit_nfl_rebuilt_historical_replay.py

Read-only independent recalculation of historical replay results, leakage/duplicate checks, legacy comparison, calibration and flat/Kelly summaries.

[Open source](../audit_nfl_rebuilt_historical_replay.py)

<details>
<summary>Source design notes</summary>

```text
Final read-only audit of the rebuilt 2022-2025 NFL weekly spread replay.

This script independently recomputes the primary backtest statistics directly
from nfl_rebuilt_weekly_replay_predictions, compares the locked 2024-2025
holdout with the legacy benchmark, checks leakage/duplication contracts, reviews
probability calibration, and simulates the recorded flat and fractional-Kelly
stakes.

It does not modify any SQLite database.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_REBUILT_HISTORICAL_REPLAY_FINAL_AUDIT_V1` |
| `VERSION` | `v1_independent_recalculation_legacy_comparison_calibration_staking` |
| `DEFAULT_REBUILT_DB_NAME` | `nfl_rebuilt_historical_replay.sqlite` |
| `PREDICTIONS_TABLE` | `nfl_rebuilt_weekly_replay_predictions` |
| `SAVED_THRESHOLD_TABLE` | `nfl_rebuilt_weekly_replay_threshold_summary` |
| `SAVED_SEASON_TABLE` | `nfl_rebuilt_weekly_replay_season_summary` |
| `LEGACY_THRESHOLD_TABLE` | `nfl_weekly_power_spread_backtest_threshold_summary` |
| `LEGACY_SEASON_TABLE` | `nfl_weekly_power_spread_backtest_season_summary` |

**Supported named options:** `--project-root`, `--rebuilt-db`, `--legacy-db`, `--primary-threshold`, `--maximum-disagreement`, `--starting-bankroll`, `--flat-stake`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 64 |  |
| `table_exists` | 93 |  |
| `read_table` | 100 |  |
| `to_numeric` | 109 |  |
| `wilson_lower` | 115 |  |
| `max_drawdown` | 126 |  |
| `american_profit` | 134 |  |
| `eligible_sample` | 144 |  |
| `summarize_sample` | 166 |  |
| `build_threshold_summary` | 204 |  |
| `build_season_summary` | 218 |  |
| `build_edge_bucket_summary` | 238 |  |
| `build_probability_bins` | 259 |  |
| `simulate_staking` | 286 |  |
| `validate_predictions` | 341 |  |
| `load_legacy_summary` | 383 |  |
| `compare_primary` | 398 |  |
| `main` | 431 |  |

</details>

## backfill_nfl_historical_advanced_stats.py

Historical Stage 2: import the canonical advanced-stats builder, create shared history, then restrict each target season to its legal preceding four-year window.

[Open source](../backfill_nfl_historical_advanced_stats.py)

<details>
<summary>Source design notes</summary>

```text
Backfill rolling historical advanced-player inputs for the canonical NFL model.

This Stage 2 historical-reconstruction script imports the approved production
`nfl_player_advanced_stats.py`, builds the shared 2018-2024 raw player-season
history once, and writes only the legal four-year window to each isolated
season database:

    2022 <- 2018-2021
    2023 <- 2019-2022
    2024 <- 2020-2023
    2025 <- 2021-2024

The production database is never written.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_HISTORICAL_ADVANCED_STATS_CANONICAL_V1` |
| `VERSION` | `v1_reuse_v3_2_final_rolling_four_year_windows` |
| `EXPECTED_BUILD_ID` | `NFL_ADVANCED_STATS_2026_V3_2_FINAL` |
| `EXPECTED_VERSION` | `v3_2_final_snap_participation_identity` |
| `CONTEXT_TABLE` | `nfl_historical_reconstruction_context` |
| `MASTER_TABLE` | `nfl_player_master_target` |
| `CROSSWALK_TABLE` | `nfl_player_crosswalk` |
| `CACHE_DB_NAME` | `nfl_historical_advanced_cache.sqlite` |
| `CACHE_RAW_TABLE` | `nfl_player_advanced_stats_raw_union` |
| `CACHE_SNAP_TABLE` | `nfl_player_advanced_stats_snap_id_audit_union` |
| `CACHE_AUDIT_TABLE` | `nfl_historical_advanced_cache_audit` |
| `HISTORY_TABLE` | `nfl_player_advanced_stats_history` |
| `CURRENT_TABLE` | `nfl_player_advanced_stats_current_roster_target` |
| `SNAP_TABLE` | `nfl_player_advanced_stats_snap_id_audit` |
| `READINESS_TABLE` | `nfl_historical_advanced_stats_readiness_audit` |

**Supported named options:** `--project-root`, `--canonical-script`, `--target-seasons`, `--rebuild-cache`, `--allow-build-id-mismatch`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_seasons` | 70 |  |
| `parse_args` | 77 |  |
| `now` | 95 |  |
| `table_exists` | 99 |  |
| `read_table` | 106 |  |
| `add_column` | 114 |  |
| `frame_hash` | 120 |  |
| `import_canonical` | 128 |  |
| `configure` | 154 |  |
| `load_mapping` | 179 |  |
| `copy_crosswalk_to_cache` | 189 |  |
| `build_cache` | 199 |  |
| `load_cache` | 252 |  |
| `validate_context` | 263 |  |
| `update_context` | 278 |  |
| `build_target` | 311 |  |
| `main` | 387 |  |

</details>

## backtest_nfl_learned_structural_weights.py

Learn positive unit/slot weights, form/process/result mix and transition parameters using nested season-forward scoring-margin validation. Attach Circa afterward for ATS grading.

[Open source](../backtest_nfl_learned_structural_weights.py)

<details>
<summary>Source design notes</summary>

```text
Leakage-controlled NFL structural-weight research backtest.

This script learns, rather than hard-codes:

* position-unit or projected-starter-slot weights;
* prior-season versus current-season contribution weights;
* season/recent and process/result form weights;
* home-field advantage;
* the current-season transition curve g / (g + k), subject to a learned cap;
* ridge shrinkage and the training-margin cap.

The independent projection is trained only on actual scoring margin. Circa is
attached after each frozen out-of-fold projection and is used only for ATS
grading. The first honest test season defaults to 2022, leaving 2020 for the
first training window and 2021 for the first inner validation window.

Source databases are read-only. Results are written to a separate SQLite file.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_LEARNED_STRUCTURAL_WEIGHTS_BACKTEST_CANONICAL_V1` |
| `VERSION` | `v1_0_market_free_nested_season_forward` |
| `REPLAY_PREDICTION_TABLE` | `nfl_rebuilt_weekly_replay_predictions` |
| `REPLAY_FORM_TABLE` | `nfl_rebuilt_weekly_form_history` |
| `UNIT_TABLE` | `nfl_position_group_ratings_target` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_target` |
| `CIRCA_TABLE` | `nfl_circa_game_lines` |

**Supported named options:** `--database-root`, `--replay-db`, `--circa-db`, `--output-db`, `--target-seasons`, `--test-start-season`, `--minimum-week`, `--maximum-week`, `--thresholds`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `Variant` | 77 |  |
| `parse_int_csv` | 91 |  |
| `parse_float_csv` | 98 |  |
| `build_parser` | 105 |  |
| `resolve_paths` | 120 |  |
| `read_sql` | 155 |  |
| `table_columns` | 163 |  |
| `require_columns` | 176 |  |
| `load_games` | 182 |  |
| `attach_form` | 215 |  |
| `attach_units` | 257 |  |
| `attach_slots` | 291 |  |
| `attach_circa` | 341 |  |
| `validate_modeling_inputs` | 362 |  |
| `season_weights` | 379 |  |
| `make_features` | 387 |  |
| `fit_model` | 428 |  |
| `choose_parameters` | 442 |  |
| `feature_group` | 479 |  |
| `fit_out_of_fold` | 489 |  |
| `add_baseline_predictions` | 579 |  |
| `wilson_interval` | 589 |  |
| `grade_threshold` | 600 |  |
| `summarize_thresholds` | 642 |  |
| `summarize_seasons` | 662 |  |
| `add_metadata` | 687 |  |
| `write_results` | 695 |  |
| `main` | 726 |  |

</details>

## backtest_nfl_nonlinear_matchup_consensus.py

Train nonlinear matchup/personnel projections and compare consensuses with learned structural out-of-fold results. Fit projections market-free; evaluate wagering gates separately using prior out-of-fold outcomes.

[Open source](../backtest_nfl_nonlinear_matchup_consensus.py)

**Local imports:** `backtest_nfl_learned_structural_weights`.

<details>
<summary>Source design notes</summary>

```text
Market-free nonlinear NFL matchup and consensus research backtest.

The nonlinear projection is trained on every reconstructed game, including
games without a valid Circa contest line. Circa is attached only after the
out-of-fold projection is frozen. The wagering-gate audit is separate: it may
use prior out-of-fold ATS results, but never the season it is grading.

This script requires the output of backtest_nfl_learned_structural_weights.py.
It does not modify any source or production database.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_NONLINEAR_MATCHUP_CONSENSUS_BACKTEST_CANONICAL_V1` |
| `VERSION` | `v1_0_full_game_market_free_nested_consensus_audit` |
| `MATCHUP_TABLE` | `nfl_matchup_game_matrix` |
| `LEARNED_PREDICTION_TABLE` | `nfl_learned_structural_oof_predictions` |

**Supported named options:** `--database-root`, `--replay-db`, `--matchup-db`, `--learned-weights-db`, `--output-db`, `--target-seasons`, `--test-start-season`, `--gate-test-start-season`, `--minimum-week`, `--maximum-week`, `--thresholds`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `NonlinearParameters` | 103 |  |
| `Gate` | 111 |  |
| `parse_int_csv` | 118 |  |
| `parse_float_csv` | 125 |  |
| `build_parser` | 132 |  |
| `resolve_paths` | 149 |  |
| `read_sql` | 192 |  |
| `table_columns` | 200 |  |
| `require_columns` | 213 |  |
| `load_structural_inputs` | 219 |  |
| `nonlinear_feature_names` | 247 |  |
| `load_matchup_matrix` | 263 |  |
| `build_nonlinear_model` | 316 |  |
| `parameter_grid` | 335 |  |
| `choose_nonlinear_parameters` | 345 |  |
| `fit_nonlinear_oof` | 384 |  |
| `load_learned_oof` | 451 |  |
| `build_consensus_matrix` | 509 |  |
| `wilson_interval` | 558 |  |
| `gate_mask` | 576 |  |
| `grade_gate` | 589 |  |
| `enumerate_gates` | 628 |  |
| `summarize_posthoc_grid` | 638 |  |
| `bayesian_gate_score` | 659 |  |
| `select_walk_forward_gates` | 671 |  |
| `nonlinear_threshold_summary` | 803 |  |
| `add_metadata` | 828 |  |
| `write_results` | 836 |  |
| `main` | 854 |  |

</details>

## build_backtest_nfl_circa_contest_lines.py

Discover and cache official historical Circa PDFs, OCR/reconcile complete boards, and refit/backtest the residual architecture on strict contest lines. Saves V1 model, line tables and integrity audits.

[Open source](../build_backtest_nfl_circa_contest_lines.py)

<details>
<summary>Source design notes</summary>

```text
Discover, download, OCR, audit, and backtest official Circa Million weekly
contest point-spread boards for the 2020-2025 NFL seasons.

Why this stage exists
---------------------
The previous NFL reconstruction used one nflverse spread per completed game.
That did not establish whether the model could beat the static Circa Million
contest number posted around Thursday morning. This stage targets that exact
market.

Workflow
--------
    plan      Inspect dependencies and expected season/week coverage.
    discover  Find and permanently cache official Circa PDF boards.
    parse     OCR the cached boards, reconcile every line to the NFL schedule,
              and refuse silent partial weeks.
    backtest  Refit the locked canonical matchup-residual architecture against
              the Circa number, enforce QB EPA/CPOE integrity, and evaluate
              both threshold bets and mandatory top-five weekly selections.
    all       Run discover, parse, and backtest.

No paid odds API is used.

Official-source policy
----------------------
Only URLs on www.circasports.com under /wp-content/uploads/ are accepted.
Every downloaded PDF is retained permanently. When multiple official versions
exist, all are parsed and the earliest valid "updated" timestamp is selected.

OCR policy
----------
Circa's historical boards are generally image-only PDFs. The script renders
page 1 with PyMuPDF and runs local Tesseract OCR. It then uses the known weekly
NFL schedule to match team names, locate each team's adjacent spread, enforce
opposite-line symmetry, and reject incomplete or ambiguous weeks.

Required local dependencies
---------------------------
    pip install pymupdf requests joblib scikit-learn pandas numpy

Tesseract must also be installed. On Anaconda Windows:
    conda install -c conda-forge tesseract

Inputs
------
- backtests/nfl_weekly_matchup_residual.sqlite
- nflreadpy/nfl_data_py schedules, or --schedule-path CSV
- optional --manifest-path CSV for manually supplied official PDF URLs
- optional --manual-lines-path CSV for audited line corrections

Outputs
-------
- backtests/nfl_circa_contest_lines.sqlite
- backtests/nfl_circa_contest_pdfs/
- backtests/nfl_circa_contest_ocr/
- outputs/historical_weekly_replay/circa_contest_lines/
- models/nfl_circa_contest_model_v1.joblib
- models/nfl_circa_contest_model_v1_metadata.json
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_CIRCA_CONTEST_LINES_CANONICAL_V1` |
| `VERSION` | `v1_5_qb_identity_integrity_guard` |
| `MATCHUP_MATRIX_TABLE` | `nfl_matchup_game_matrix` |
| `MATCHUP_AUDIT_TABLE` | `nfl_matchup_run_audit` |
| `MATCHUP_QB_AUDIT_TABLE` | `nfl_matchup_qb_feature_integrity_audit` |
| `MODEL_FILENAME` | `nfl_circa_contest_model_v1.joblib` |
| `MANIFEST_TABLE` | `nfl_circa_pdf_manifest` |
| `OCR_ATTEMPT_TABLE` | `nfl_circa_ocr_attempts` |
| `TEAM_LINE_TABLE` | `nfl_circa_team_lines` |
| `GAME_LINE_TABLE` | `nfl_circa_game_lines` |
| `WEEK_AUDIT_TABLE` | `nfl_circa_week_audit` |
| `BACKTEST_MATRIX_TABLE` | `nfl_circa_backtest_matrix` |
| `OOF_PREDICTION_TABLE` | `nfl_circa_oof_predictions` |
| `THRESHOLD_VALIDATION_TABLE` | `nfl_circa_threshold_validation` |
| `TOP5_VALIDATION_TABLE` | `nfl_circa_top5_validation` |
| `BENCHMARK_PREDICTION_TABLE` | `nfl_circa_benchmark_predictions` |
| `BENCHMARK_SUMMARY_TABLE` | `nfl_circa_benchmark_summary` |
| `COEFFICIENT_TABLE` | `nfl_circa_model_coefficients` |
| `RUN_AUDIT_TABLE` | `nfl_circa_run_audit` |

**Supported named options:** `--mode`, `--project-root`, `--schedule-path`, `--manifest-path`, `--manual-lines-path`, `--seasons`, `--tesseract-path`, `--ocr-dpi`, `--ocr-psms`, `--thresholds`, `--fallback-ats-price`, `--request-timeout`, `--request-pause`, `--probe-missing`, `--rebuild-downloads`, `--rebuild-ocr`, `--allow-incomplete`, `--backfill-missing-2025-with-reference`, `--no-csv`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_int_list` | 267 |  |
| `parse_float_list` | 274 |  |
| `parse_args` | 281 |  |
| `now_string` | 333 |  |
| `normalize_team` | 337 |  |
| `normalize_token` | 342 |  |
| `first_existing` | 346 |  |
| `table_exists` | 354 |  |
| `read_table` | 361 |  |
| `numeric` | 370 |  |
| `frame_from_any` | 376 |  |
| `sha256_file` | 384 |  |
| `official_circa_pdf_url` | 395 |  |
| `load_schedule_from_package` | 412 |  |
| `standardize_schedule` | 445 |  |
| `load_schedule` | 493 |  |
| `expected_week_table` | 502 |  |
| `infer_season_week_from_url` | 520 |  |
| `discover_via_media_api` | 532 |  |
| `discover_via_sitemap` | 575 |  |
| `candidate_months_for_week` | 615 |  |
| `candidate_filenames` | 628 |  |
| `probe_candidate_urls` | 654 |  |
| `load_manual_manifest` | 706 |  |
| `safe_pdf_filename` | 732 |  |
| `download_manifest` | 738 |  |
| `discover_and_download` | 779 |  |
| `find_tesseract` | 847 |  |
| `render_pdf_first_page` | 870 |  |
| `run_tesseract_tsv` | 885 |  |
| `alias_similarity` | 923 |  |
| `team_word_candidates` | 931 |  |
| `assign_unique_team_words` | 961 |  |
| `looks_like_odds_token` | 977 |  |
| `parse_odds_token` | 986 |  |
| `nearest_odds_word` | 1029 |  |
| `extract_updated_timestamp` | 1060 |  |
| `parse_ocr_attempt` | 1087 |  |
| `ocr_pdf_versions` | 1203 |  |
| `reconcile_ocr_sources` | 1313 | Reconcile valid game lines across OCR modes for one source PDF. |
| `load_manual_lines` | 1565 |  |
| `finalize_lines` | 1586 | Reconcile OCR modes at game level and merge audited lines to schedule. |
| `apply_2025_reference_backfill` | 1781 | Fill only missing 2025 rows with nflverse final spreads when enabled. |
| `load_matchup_inputs` | 1918 |  |
| `locked_configuration` | 1947 |  |
| `build_backtest_matrix` | 1956 |  |
| `assert_circa_qb_feature_integrity` | 2006 |  |
| `build_pipeline` | 2076 |  |
| `rolling_oof_predictions` | 2086 |  |
| `grade_selected_rows` | 2109 |  |
| `select_top5_each_week` | 2151 |  |
| `max_drawdown` | 2163 |  |
| `strategy_summary` | 2170 |  |
| `validation_outputs` | 2210 |  |
| `top5_validation_passed` | 2246 |  |
| `benchmark_outputs` | 2269 |  |
| `benchmark_passed` | 2307 |  |
| `save_model_bundle` | 2331 |  |
| `save_database_tables` | 2434 |  |
| `load_existing_tables` | 2446 |  |
| `save_csv_outputs` | 2463 |  |
| `run_self_test` | 2480 |  |
| `main` | 2572 |  |
| `_module_available` | 2900 |  |
| `_tesseract_available` | 2908 |  |

</details>

## build_backtest_nfl_circa_season_phase_v3_2.py

Rebuild the fixed ceiling architecture after QB repair: locked V1 early, fixed Ridge late. Require QB feature integrity and tie the bundle to the exact source V1 hash.

[Open source](../build_backtest_nfl_circa_season_phase_v3_2.py)

<details>
<summary>Source design notes</summary>

```text
Rebuild the fixed Circa V3.2 season-phase ceiling after QB repair.

This stage deliberately preserves the existing production architecture:

* Weeks 1-9 use the locked V1 model exactly.
* Weeks 10-18 use a Ridge model trained on strict historical Circa rows.
* The selected late alpha remains 1000 and the blend weight remains 1.0.

It does not reuse the old V3.2 selection-gate statistics because those were
calculated while the QB EPA/CPOE source fields were dead.  The repaired V1
bundle and repaired Circa backtest database must both pass explicit QB
coverage and variance gates.  The output is cryptographically tied to the V1
bundle that produced it, so mixed old/new model generations cannot load.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_CIRCA_CONTEST_SEASON_PHASE_V3_2_QB_REBUILT` |
| `VERSION` | `v3_2_1_fixed_architecture_qb_integrity_lineage` |
| `BACKTEST_TABLE` | `nfl_circa_backtest_matrix` |
| `RUN_AUDIT_TABLE` | `nfl_circa_run_audit` |

**Supported named options:** `--project-root`, `--database-path`, `--v1-model-path`, `--output-model-path`, `--output-metadata-path`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 65 |  |
| `now_string` | 98 |  |
| `sha256_file` | 102 |  |
| `read_table` | 110 |  |
| `numeric` | 120 |  |
| `qb_integrity_metrics` | 126 |  |
| `assert_saved_qb_integrity` | 191 |  |
| `build_pipeline` | 226 |  |
| `matrix_fingerprint` | 236 |  |
| `build_bundle` | 242 |  |
| `atomic_save_joblib` | 399 |  |
| `atomic_save_json` | 406 |  |
| `run_self_test` | 413 |  |
| `main` | 440 |  |

</details>

## build_backtest_nfl_weekly_matchup_residual.py

Build prior-only matchup and personnel history and a market-anchored residual model. Audit QB ID/feature coverage and variance, forward validation, coefficient stability and deployment gates. Also supplies cached data/features used by other models.

[Open source](../build_backtest_nfl_weekly_matchup_residual.py)

<details>
<summary>Source design notes</summary>

```text
Build prior-only weekly NFL matchup and personnel ratings, validate a compact
market-residual model across four rolling seasons, and produce a guarded
implementation bundle.

Architecture
------------
    actual_home_margin = market_home_margin + predicted_market_residual

The market remains the anchor. The model is permitted to adjust it only when
prior-only matchup and personnel features demonstrate stable residual value.

Historical information policy
-----------------------------
For prediction week W:
- matchup ratings use current-season games through W-1 plus a low-weight
  prior-season sample;
- personnel ratings use snap participation only through W-1;
- quarterback identity and recent quarterback performance use only games
  completed through W-1;
- Week 1 is never graded.

QB identity policy
------------------
Snap-count QB names are resolved to nflverse GSIS passer identifiers through
an audited first-initial/surname crosswalk.  Ambiguous mappings are fatal, and
recent QB performance follows the player across team changes.  Model fitting
is blocked unless QB identity coverage, EPA/CPOE coverage, nonzero counts, and
feature variance all clear explicit integrity thresholds.

Evaluation policy
-----------------
- 2018-2019: initial development.
- 2020-2023: rolling validation, one season at a time.
- 2024-2025: fixed benchmark safety check only; never used to select the
  feature set, model alpha, rating alpha, or betting threshold.
- 2026: true forward evaluation.

A model is marked implementation-ready only when:
1. residual correlation is positive in at least three of four validation years;
2. average validation residual correlation is positive;
3. coefficient signs are stable across rolling folds;
4. the selected threshold has sufficient bets in every validation year;
5. at least three validation years are profitable and no year breaches the
   fixed ROI floor;
6. the locked 2024-2025 benchmark clears a fixed safety floor without any
   retuning.

Output database
---------------
    backtests/nfl_weekly_matchup_residual.sqlite

Deployment bundle
-----------------
    models/nfl_matchup_residual_model_v2.joblib
    models/nfl_matchup_residual_model_v2_metadata.json

No production or previously built historical database is modified.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V3` |
| `VERSION` | `v3_qb_gsis_identity_crosswalk_dead_feature_guard` |
| `DEFAULT_MINIMUM_PROFITABLE_VALIDATION_SEASONS` | `3` |
| `MODEL_FILENAME` | `nfl_matchup_residual_model_v2.joblib` |
| `TEAM_GAME_TABLE` | `nfl_matchup_team_game_features` |
| `TEAM_WEEK_TABLE` | `nfl_matchup_team_week_ratings` |
| `PERSONNEL_WEEK_TABLE` | `nfl_matchup_weekly_personnel_features` |
| `GAME_MATRIX_TABLE` | `nfl_matchup_game_matrix` |
| `DIRECTION_AUDIT_TABLE` | `nfl_matchup_feature_direction_audit` |
| `CANDIDATE_VALIDATION_TABLE` | `nfl_matchup_candidate_validation` |
| `COEFFICIENT_STABILITY_TABLE` | `nfl_matchup_coefficient_stability` |
| `THRESHOLD_VALIDATION_TABLE` | `nfl_matchup_threshold_validation` |
| `BENCHMARK_PREDICTION_TABLE` | `nfl_matchup_benchmark_predictions` |
| `BENCHMARK_SUMMARY_TABLE` | `nfl_matchup_benchmark_summary` |
| `MODEL_COEFFICIENT_TABLE` | `nfl_matchup_model_coefficients` |
| `MODEL_COMPARISON_TABLE` | `nfl_matchup_model_comparison` |
| `QB_IDENTITY_AUDIT_TABLE` | `nfl_matchup_qb_identity_crosswalk_audit` |
| `QB_FEATURE_AUDIT_TABLE` | `nfl_matchup_qb_feature_integrity_audit` |
| `RUN_AUDIT_TABLE` | `nfl_matchup_run_audit` |

**Supported named options:** `--project-root`, `--pbp-path`, `--schedule-path`, `--snap-counts-path`, `--rebuild-cache`, `--rating-alphas`, `--model-alphas`, `--thresholds`, `--ats-price`, `--prior-season-weight`, `--current-decay`, `--prior-decay`, `--minimum-total-validation-bets`, `--minimum-season-validation-bets`, `--minimum-profitable-validation-seasons`, `--validation-roi-floor`, `--benchmark-combined-roi-floor`, `--benchmark-season-roi-floor`, `--benchmark-correlation-floor`, `--minimum-sign-stability`, `--minimum-benchmark-bets`, `--minimum-rolling-train-rows`, `--minimum-rolling-validation-rows`, `--preflight-only`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_float_list` | 319 |  |
| `parse_args` | 334 |  |
| `now_string` | 461 |  |
| `normalize_team` | 465 |  |
| `normalize_position` | 470 |  |
| `position_group` | 475 |  |
| `normalize_name` | 479 |  |
| `qb_name_match_key` | 485 | Return a stable first-initial plus surname QB identity key. |
| `first_existing` | 504 |  |
| `numeric` | 515 |  |
| `frame_from_any` | 525 |  |
| `table_exists` | 533 |  |
| `read_table` | 543 |  |
| `safe_correlation` | 552 |  |
| `load_package_pbp` | 571 |  |
| `load_package_schedule` | 620 |  |
| `load_package_snaps` | 661 |  |
| `standardize_pbp` | 704 |  |
| `standardize_schedule` | 794 |  |
| `standardize_snaps` | 852 |  |
| `load_or_build_cache` | 928 |  |
| `safe_mean` | 1006 |  |
| `build_team_game_features` | 1011 |  |
| `observation_weights` | 1151 |  |
| `fit_metric_model` | 1175 |  |
| `weighted_special_teams_ratings` | 1232 |  |
| `build_team_week_ratings` | 1264 |  |
| `build_qb_identity_crosswalk` | 1348 |  |
| `share_dictionary` | 1429 |  |
| `weighted_overlap` | 1450 |  |
| `average_share_dictionary` | 1462 |  |
| `missing_core_share` | 1475 |  |
| `build_snapshots` | 1493 |  |
| `weighted_qb_performance` | 1542 |  |
| `build_weekly_personnel_features` | 1591 |  |
| `side_ratings` | 1720 |  |
| `side_personnel` | 1734 |  |
| `expected_metric` | 1748 |  |
| `build_game_matrix` | 1769 |  |
| `build_qb_feature_integrity_audit` | 1867 | Profile and enforce QB identity/performance integrity before fitting. |
| `build_direction_audit` | 1997 |  |
| `build_pipeline` | 2047 |  |
| `prediction_quality` | 2057 |  |
| `extract_base_coefficients` | 2074 |  |
| `coefficient_stability` | 2088 |  |
| `rolling_validation` | 2130 |  |
| `evaluate_candidates` | 2194 |  |
| `win_profit` | 2395 |  |
| `grade_predictions` | 2399 |  |
| `maximum_drawdown` | 2448 |  |
| `betting_summary` | 2455 |  |
| `validate_thresholds` | 2480 |  |
| `build_benchmark` | 2561 |  |
| `benchmark_safety_check` | 2639 |  |
| `build_comparison` | 2683 |  |
| `save_deployment_bundle` | 2733 |  |
| `main` | 2848 |  |

</details>

## build_nfl_2026_form_rating.py

Use completed games before the prediction week to create opponent-adjusted process/result and season/recent form. Blend with current structural power and preserve the immutable preseason snapshot for audit.

[Open source](../build_nfl_2026_form_rating.py)

<details>
<summary>Source design notes</summary>

```text
Build an opponent-adjusted 2026 NFL form rating and blend it incrementally into
the current roster/depth-chart structural power rating.

Purpose
-------
This script creates the in-season layer that sits on top of
nfl_power_ratings_2026.

The immutable preseason snapshot remains an audit comparator. The current
market-independent structural rating is the active prior, so authoritative
roster, quarterback, depth-chart, and offensive-line refreshes reach the live
weekly model. Completed 2026 games enter gradually using:

    current_season_weight = games_played / (games_played + 5)

The weight is capped at 75 percent. Therefore:
    0 games  ->  0.0% 2026 form
    1 game   -> 16.7% 2026 form
    2 games  -> 28.6% 2026 form
    4 games  -> 44.4% 2026 form
    8 games  -> 61.5% 2026 form
    15+ games -> 75.0% maximum

The 2026 form rating is built from two independent signals:

1. Process rating
   Predictive play-by-play efficiency:
       EPA/play
       success rate
       passing EPA
       rushing EPA
       early-down EPA
       explosive-play rate
       turnover rate
       sack rate
       CPOE
       special-teams EPA

   Historical 2018-2025 play-by-play is used to estimate how these game-level
   process metrics translate to points.

2. Result rating
   Capped actual scoring margin.

Both signals are opponent-adjusted through a ridge SRS system:

    observed_home_margin
    = home_team_rating - away_team_rating + home_field

The final form rating is:

    season_form = 70% process + 30% results
    recent_form = 70% process + 30% results

    form_rating = 70% season_form + 30% recent_form

The live internal rating is:

    live_power_rating
    = structural_prior * (1 - current_season_weight)
    + form_rating * current_season_weight

Before any 2026 regular-season games are completed, this script safely writes
a 32-team output where live_power_rating equals the current structural rating
exactly. Preseason drift remains explicit in separate audit columns.

Inputs
------
SQLite:
    nfl_power_ratings_2026

External data:
    nflreadpy.load_schedules
    nfl_data_py.import_schedules fallback
    nfl_data_py.import_pbp_data
    nflreadpy play-by-play fallback

Outputs
-------
SQLite and CSV:
    nfl_2026_form_ratings
    nfl_2026_form_game_audit
    nfl_2026_form_team_game_audit
    nfl_2026_form_process_model

Primary point-spread field
--------------------------
Use:

    nfl_2026_form_ratings.power_rating_points

This field is a neutral-field points-above-average rating. Once games begin it
is the incrementally blended live internal rating. Before Week 1 it equals the
current structural power_rating_points value.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_2026_FORM_RATING_CANONICAL_V3` |
| `VERSION` | `v3_current_structural_prior_asof_opponent_adjusted_form` |
| `FEATURE_VERSION` | `v2_0_pbp_process_features` |
| `PROCESS_MODEL_VERSION` | `v2_0_2018_2025_process_to_points` |
| `EXPECTED_STRUCTURAL_POWER_BUILD_ID` | `NFL_POWER_RATINGS_2026_CANONICAL_V2` |
| `PRESEASON_POWER_TABLE` | `nfl_power_ratings_2026` |
| `PRESEASON_SNAPSHOT_TABLE` | `nfl_2026_preseason_power_snapshot` |
| `OUTPUT_TABLE` | `nfl_2026_form_ratings` |
| `GAME_AUDIT_TABLE` | `nfl_2026_form_game_audit` |
| `TEAM_GAME_AUDIT_TABLE` | `nfl_2026_form_team_game_audit` |
| `PROCESS_MODEL_TABLE` | `nfl_2026_form_process_model` |
| `HISTORICAL_FEATURE_CACHE_TABLE` | `nfl_form_historical_team_game_features_2018_2025` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--as-of-date`, `--through-week`, `--refresh-preseason-snapshot`, `--rebuild-process-cache`, `--prepare-process-model`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 269 |  |
| `configure_runtime` | 331 |  |
| `configure_logging` | 344 |  |
| `get_engine` | 376 |  |
| `table_exists` | 383 |  |
| `table_columns` | 398 |  |
| `read_table` | 414 |  |
| `drop_table_if_exists` | 433 |  |
| `parse_as_of_date` | 441 |  |
| `stable_rating_hash` | 452 |  |
| `stable_current_structural_hash` | 464 |  |
| `empty_process_model` | 480 |  |
| `normalize_team` | 503 |  |
| `first_existing` | 520 |  |
| `numeric_series` | 531 |  |
| `ensure_column` | 549 |  |
| `frame_from_any` | 558 |  |
| `safe_mean` | 568 |  |
| `safe_divide` | 580 |  |
| `center_series` | 600 |  |
| `clip_and_recenter` | 612 |  |
| `empty_game_audit` | 630 |  |
| `empty_team_game_audit` | 655 |  |
| `prepare_power_frame` | 705 |  |
| `load_current_structural_power` | 758 |  |
| `load_or_create_preseason_snapshot` | 762 |  |
| `load_schedules_external` | 876 |  |
| `load_schedules_database` | 934 |  |
| `load_schedules` | 987 |  |
| `standardize_schedules` | 1011 |  |
| `filter_schedule_as_of` | 1188 |  |
| `filter_pbp_to_completed_games` | 1209 |  |
| `load_pbp_for_season` | 1229 |  |
| `standardize_pbp` | 1312 |  |
| `aggregate_offense` | 1418 |  |
| `aggregate_special_teams` | 1611 |  |
| `build_schedule_team_rows` | 1682 |  |
| `build_team_game_features` | 1750 |  |
| `historical_cache_is_valid` | 1894 |  |
| `load_or_build_historical_features` | 1930 |  |
| `model_cache_is_valid` | 2024 |  |
| `weighted_ridge_fit` | 2078 |  |
| `fit_process_model` | 2121 |  |
| `load_or_fit_process_model` | 2326 |  |
| `apply_process_model` | 2376 |  |
| `solve_srs` | 2501 |  |
| `build_game_audit` | 2619 |  |
| `build_no_games_output` | 2717 |  |
| `build_live_form_output` | 2789 |  |
| `finalize_output_columns` | 2936 |  |
| `validate_output` | 3008 |  |
| `save_outputs` | 3136 |  |
| `print_report` | 3194 |  |
| `main` | 3288 |  |

</details>

## build_nfl_defensive_metrics_summary.py

Join canonical front and coverage outputs to defenders and blend the components actually available.

[Open source](../build_nfl_defensive_metrics_summary.py)

<details>
<summary>Source design notes</summary>

```text
Combine canonical front and coverage metrics for current 2026 defenders.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_DEFENSIVE_SUMMARY_2026_CANONICAL_V3` |
| `METRICS_VERSION` | `v3_available_component_front_coverage_blend` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `FRONT_TABLE` | `nfl_defensive_front_metrics_current_roster_2026` |
| `COVERAGE_TABLE` | `nfl_coverage_metrics_current_roster_2026` |
| `OUTPUT_TABLE` | `nfl_defensive_player_metrics_current_roster_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 45 |  |
| `get_engine` | 53 |  |
| `table_exists` | 57 |  |
| `read_table` | 63 |  |
| `clean_id` | 72 |  |
| `load_master` | 79 |  |
| `prepare_front` | 95 |  |
| `prepare_coverage` | 120 |  |
| `build_summary` | 144 |  |
| `main` | 211 |  |

</details>

## build_nfl_historical_player_performance_depth.py

Historical Stage 4: reuse canonical player/performance and depth logic with prior-only PFF inputs and frozen Week 1 OL/QB evidence. Validate 32 QB1s and 160 OL starters with full reconciliation.

[Open source](../build_nfl_historical_player_performance_depth.py)

<details>
<summary>Source design notes</summary>

```text
Build historical player-performance grades and projected depth charts.

This is Stage 4 of the true canonical NFL historical reconstruction.

For each isolated target-season database, this wrapper:

1. verifies the approved canonical source files:
       build_nfl_player_performance.py
       build_nfl_projected_depth_chart.py
2. creates temporary runtime copies;
3. changes only season, legal history window, recent season, and table names;
4. executes the canonical player-performance V4 model with prior-only PFF OL data;
5. reconstructs the Week 1 OL unit from observed Week 1 snaps, the frozen
   roster, and prior-only PFF position evidence;
6. executes the canonical projected-depth V5 model with frozen Week 1 OL/QB starters;
7. treats non-point-in-time roster statuses as non-vetoing only for those
   frozen authoritative starters and audits every such override;
8. validates complete player reconciliation, 32 QB1s, and 160 OL starters;
9. writes historical readiness and audit tables;
10. leaves production scripts and the production database unchanged.

Historical information policy
-----------------------------
- Target-season personnel comes from the frozen Week 1 roster snapshot.
- Player performance uses only the legal four-year window ending in S-1.
- The recent-season component is S-1, never target season S.
- QB1 comes from the Week 1 prior established by
  prepare_nfl_historical_reconstruction.py.
- OL personnel comes from Week 1 offensive participation. Position assignment
  combines the Week 1 snap position, frozen roster position, and prior-only PFF
  position; no target-season performance grade is used to choose starters.
- Historical roster status is retained for audit but is not an as-of-Week-1
  injury feed. It therefore cannot veto a frozen authoritative QB/OL starter.
- Graded game replay begins in Week 2.

Required prior stages
---------------------
1. prepare_nfl_historical_reconstruction.py
2. backfill_nfl_historical_advanced_stats.py
3. build_nfl_historical_qb_defense.py
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_HISTORICAL_PLAYER_PERFORMANCE_DEPTH_CANONICAL_V3` |
| `VERSION` | `v3_3_week1_snap_ol_bootstrap_prior_only` |
| `CONTEXT_TABLE` | `nfl_historical_reconstruction_context` |
| `MASTER_TABLE` | `nfl_player_master_target` |
| `ADVANCED_CURRENT_TABLE` | `nfl_player_advanced_stats_current_roster_target` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_history` |
| `QB_RATING_TABLE` | `nfl_qb_rbsdm_ratings_target` |
| `DEFENSE_TABLE` | `nfl_defensive_player_metrics_current_roster_target` |
| `HISTORICAL_QB_PRIOR_TABLE` | `nfl_projected_qb_starters_target` |
| `DEPTH_QB_INPUT_TABLE` | `nfl_projected_qb_starters_depth_input` |
| `DEPTH_OL_INPUT_TABLE` | `nfl_projected_ol_starters_depth_input` |
| `MATCHUP_SNAP_TABLE` | `snap_source` |
| `PERFORMANCE_TABLE` | `nfl_player_performance_inputs_target` |
| `SEASONAL_PERFORMANCE_TABLE` | `nfl_player_performance_seasonal_history` |
| `PERFORMANCE_POSITION_SUMMARY_TABLE` | `nfl_player_performance_position_summary_target` |
| `PERFORMANCE_COMPONENT_AUDIT_TABLE` | `nfl_player_performance_component_audit_target` |
| `PERFORMANCE_UNRESOLVED_TABLE` | `nfl_player_performance_unresolved_master_audit_target` |
| `PFF_OL_HISTORY_TABLE` | `nfl_ol_pff_player_season_history_target` |
| `PFF_OL_IDENTITY_AUDIT_TABLE` | `nfl_ol_pff_identity_audit_target` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_target` |
| `DEPTH_AUDIT_TABLE` | `nfl_projected_depth_chart_audit_target` |
| `READINESS_TABLE` | `nfl_historical_player_performance_depth_readiness_audit` |
| `STEP_AUDIT_TABLE` | `nfl_historical_player_performance_depth_step_audit` |
| `EXPECTED_PERFORMANCE_BUILD_ID` | `NFL_PLAYER_PERFORMANCE_2026_CANONICAL_V4` |
| `EXPECTED_PERFORMANCE_VERSION` | `v4_1_pff_row_identity_collision_safe` |
| `EXPECTED_DEPTH_BUILD_ID` | `NFL_PROJECTED_DEPTH_CHART_2026_CANONICAL_V5` |
| `EXPECTED_DEPTH_VERSION` | `v5_pff_ol_talent_no_second_shrink_authoritative_starters` |

**Supported named options:** `--project-root`, `--database-root`, `--output-root`, `--source-cache-db`, `--target-seasons`, `--python`, `--keep-patched-scripts`, `--allow-source-mismatch`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `RuntimeScript` | 159 |  |
| `parse_seasons` | 166 |  |
| `parse_args` | 183 |  |
| `now_string` | 259 |  |
| `table_exists` | 263 |  |
| `read_table` | 278 |  |
| `add_column` | 298 |  |
| `stable_hash` | 318 |  |
| `replace_assignment` | 340 |  |
| `replace_mapping_assignment` | 361 |  |
| `replace_exact_once` | 381 |  |
| `clean_pff_column` | 388 |  |
| `normalize_pff_position` | 393 |  |
| `pff_history_seasons` | 404 |  |
| `resolve_pff_input_paths` | 416 |  |
| `derive_pff_replacement_anchors` | 452 |  |
| `copy_pff_inputs` | 492 |  |
| `verify_source` | 499 |  |
| `patch_performance_script` | 525 |  |
| `patch_depth_script` | 657 |  |
| `prepare_qb_depth_input` | 747 |  |
| `prepare_ol_depth_input` | 827 | Build the frozen Week 1 OL input without a pre-existing depth table. |
| `insert_step_audit` | 1123 |  |
| `run_child` | 1136 |  |
| `validate_numeric_range` | 1231 |  |
| `normalize_names` | 1258 |  |
| `normalize_historical_ol_names` | 1279 |  |
| `ol_role_family` | 1283 |  |
| `ol_role_evidence` | 1296 |  |
| `validate_target` | 1311 |  |
| `write_mirror_tables` | 1678 |  |
| `update_context` | 1738 |  |
| `export_outputs` | 1815 |  |
| `main` | 1855 |  |

</details>

## build_nfl_historical_qb_defense.py

Historical Stage 3: run verified production QB and defense sources against isolated season/table settings, preserving prior-only windows and readiness audits.

[Open source](../build_nfl_historical_qb_defense.py)

<details>
<summary>Source design notes</summary>

```text
Build historical QB and defensive metrics for isolated NFL reconstruction DBs.

Stage 3 of the canonical historical reconstruction.

The wrapper does not copy or simplify production formulas. For each target
season it verifies the approved canonical source Build IDs, creates temporary
season-patched copies, executes those copies against backtests/<season>.sqlite,
and validates the resulting QB, defensive-front, coverage, and defensive-summary
tables. Production scripts and the production database are never modified.

Required previous stages
------------------------
1. prepare_nfl_historical_reconstruction.py
2. backfill_nfl_historical_advanced_stats.py

Canonical source files
----------------------
- load_rbsdm_qb_ratings.py
- load_nfl_defensive_front_metrics.py
- load_nfl_coverage_metrics.py
- build_nfl_defensive_metrics_summary.py

Primary outputs in each isolated DB
-----------------------------------
- nfl_qb_rbsdm_ratings_raw_history
- nfl_qb_rbsdm_ratings_target
- nfl_qb_rbsdm_local_unmatched_audit_history
- nfl_defensive_front_metrics_history
- nfl_defensive_front_metrics_current_roster_target
- nfl_coverage_metrics_history
- nfl_coverage_metrics_current_roster_target
- nfl_defensive_player_metrics_current_roster_target
- nfl_projected_qb_starter_ratings_target
- nfl_historical_qb_defense_readiness_audit
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_HISTORICAL_QB_DEFENSE_CANONICAL_V2` |
| `VERSION` | `v2_qb_v3_isolated_database_root_prior_only` |
| `CONTEXT_TABLE` | `nfl_historical_reconstruction_context` |
| `MASTER_TABLE` | `nfl_player_master_target` |
| `STARTER_QB_TABLE` | `nfl_projected_qb_starters_target` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_history` |
| `QB_RAW_TABLE` | `nfl_qb_rbsdm_ratings_raw_history` |
| `QB_TABLE` | `nfl_qb_rbsdm_ratings_target` |
| `QB_UNMATCHED_TABLE` | `nfl_qb_rbsdm_local_unmatched_audit_history` |
| `FRONT_HISTORY_TABLE` | `nfl_defensive_front_metrics_history` |
| `FRONT_CURRENT_TABLE` | `nfl_defensive_front_metrics_current_roster_target` |
| `COVERAGE_HISTORY_TABLE` | `nfl_coverage_metrics_history` |
| `COVERAGE_CURRENT_TABLE` | `nfl_coverage_metrics_current_roster_target` |
| `DEFENSE_SUMMARY_TABLE` | `nfl_defensive_player_metrics_current_roster_target` |
| `STARTER_RATING_TABLE` | `nfl_projected_qb_starter_ratings_target` |
| `READINESS_TABLE` | `nfl_historical_qb_defense_readiness_audit` |
| `STEP_AUDIT_TABLE` | `nfl_historical_qb_defense_step_audit` |

**Supported named options:** `--project-root`, `--database-root`, `--output-root`, `--target-seasons`, `--python`, `--keep-patched-scripts`, `--allow-source-mismatch`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `PatchedScript` | 116 |  |
| `parse_seasons` | 123 |  |
| `parse_args` | 130 |  |
| `now_string` | 166 |  |
| `table_exists` | 170 |  |
| `read_table` | 177 |  |
| `add_column` | 186 |  |
| `stable_hash` | 202 |  |
| `replace_assignment` | 214 |  |
| `verified_source` | 224 |  |
| `patch_source` | 240 |  |
| `prepare_runtime` | 317 |  |
| `run_child` | 355 |  |
| `group_count` | 426 |  |
| `validate_range` | 431 |  |
| `historical_seasons` | 441 |  |
| `validate_outputs` | 453 |  |
| `write_mirrors` | 598 |  |
| `update_context` | 621 |  |
| `export_csvs` | 673 |  |
| `main` | 701 |  |

</details>

## build_nfl_learned_consensus_2026.py

Fit or reuse the frozen three-model 2026 bundle, its structural snapshots, consensus gate and out-of-fold probability calibration. Reuse validates metadata/hash but presently requires research paths first.

[Open source](../build_nfl_learned_consensus_2026.py)

**Local imports:** `backtest_nfl_learned_structural_weights`, `backtest_nfl_nonlinear_matchup_consensus`.

<details>
<summary>Source design notes</summary>

```text
Freeze the market-free learned structural consensus for the 2026 season.

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
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_LEARNED_CONSENSUS_2026_CANONICAL_V1` |
| `VERSION` | `v1_0_2020_2025_frozen_market_free_consensus` |
| `EXPECTED_LEARNED_BUILD_ID` | `NFL_LEARNED_STRUCTURAL_WEIGHTS_BACKTEST_CANONICAL_V1` |
| `EXPECTED_LEARNED_VERSION` | `v1_0_market_free_nested_season_forward` |
| `EXPECTED_NONLINEAR_BUILD_ID` | `NFL_NONLINEAR_MATCHUP_CONSENSUS_BACKTEST_CANONICAL_V1` |
| `EXPECTED_NONLINEAR_VERSION` | `v1_0_full_game_market_free_nested_consensus_audit` |
| `UNIT_SOURCE_TABLE` | `nfl_position_group_ratings_2026` |
| `DEPTH_SOURCE_TABLE` | `nfl_projected_depth_chart_2026` |
| `REGISTRY_TABLE` | `nfl_learned_consensus_model_registry_2026` |
| `COEFFICIENT_TABLE` | `nfl_learned_consensus_linear_coefficients_2026` |
| `UNIT_SNAPSHOT_TABLE` | `nfl_learned_consensus_unit_snapshot_2026` |
| `SLOT_SNAPSHOT_TABLE` | `nfl_learned_consensus_slot_snapshot_2026` |

**Supported named options:** `--project-root`, `--database-root`, `--db-path`, `--database`, `--replay-db`, `--matchup-db`, `--learned-weights-db`, `--consensus-backtest-db`, `--model-path`, `--metadata-path`, `--force-rebuild`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 77 |  |
| `now_string` | 115 |  |
| `table_exists` | 119 |  |
| `read_table` | 125 |  |
| `sha256_file` | 132 |  |
| `stable_frame_hash` | 140 |  |
| `atomic_joblib_dump` | 146 |  |
| `atomic_json_dump` | 159 |  |
| `validate_research_modules` | 172 |  |
| `validate_paths` | 179 |  |
| `validate_frozen_candidate` | 193 |  |
| `load_oof_probability_calibration` | 227 |  |
| `load_training_frame` | 249 |  |
| `fit_linear_models` | 268 |  |
| `linear_coefficients_from_bundle` | 312 |  |
| `fit_nonlinear_model` | 335 |  |
| `freeze_live_structural_inputs` | 377 |  |
| `write_registry` | 461 |  |
| `reusable_bundle` | 516 |  |
| `main` | 544 |  |

</details>

## build_nfl_ol_continuity.py

Measure retained starters and combinations from canonical PFF history and the projected OL. Publish confidence and availability separately; do not rescore OL talent.

[Open source](../build_nfl_ol_continuity.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical historical and projected NFL offensive-line continuity.

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
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_OL_CONTINUITY_2026_CANONICAL_V3` |
| `VERSION` | `v3_injury_returner_credit_bounded_secondary_context` |
| `PFF_HISTORY_TABLE` | `nfl_ol_pff_player_season_2022_2025` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_2026` |
| `PLAYER_HISTORY_TABLE` | `nfl_ol_player_snap_history` |
| `HISTORICAL_TABLE` | `nfl_ol_continuity_historical` |
| `PROJECTED_TABLE` | `nfl_ol_continuity_2026` |
| `PLAYER_AUDIT_TABLE` | `nfl_ol_continuity_player_audit` |
| `UNMATCHED_AUDIT_TABLE` | `nfl_ol_continuity_unmatched_audit` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 82 |  |
| `configure_logging` | 90 |  |
| `clean_scalar` | 105 |  |
| `clean_id` | 117 |  |
| `normalize_team` | 121 |  |
| `normalize_position` | 126 |  |
| `numeric` | 131 |  |
| `table_exists` | 137 |  |
| `read_table` | 141 |  |
| `is_ol_row` | 150 |  |
| `build_player_history` | 154 |  |
| `pair_set` | 195 |  |
| `continuity_record` | 200 |  |
| `build_historical` | 273 |  |
| `projected_starters` | 290 |  |
| `build_projected` | 313 |  |
| `validate` | 370 |  |
| `empty_unmatched` | 402 |  |
| `main` | 406 |  |

</details>

## build_nfl_player_crosswalk.py

Map provider identifiers and name evidence to canonical players. Uses vetted identity sources and snap records, retains ambiguity/reconciliation audits, and prevents unverified name collisions from silently merging players.

[Open source](../build_nfl_player_crosswalk.py)

<details>
<summary>Source design notes</summary>

```text
Build a permanent NFL player identity crosswalk.

Outputs SQLite tables:
    nfl_player_crosswalk
    nfl_player_crosswalk_unmatched
    nfl_player_crosswalk_duplicate_audit
    nfl_player_crosswalk_conflict_audit

Outputs matching CSV files and a timestamped log.

The script is intentionally schema-tolerant:
- It inspects available SQLite columns before selecting data.
- It supports nflreadpy first and nfl_data_py as a fallback.
- It searches SQLite for snap-count tables containing pfr_player_id.
- It refuses ambiguous many-to-many identity mappings.
- It treats GSIS from nfl_player_master_2026 as the canonical working ID.
- It includes 2025 historical rosters and snap identities.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `CURRENT_ROSTER_TABLE` | `nfl_rosters_2026_raw` |
| `PLAYER_MASTER_TABLE` | `nfl_player_master_2026` |
| `OUTPUT_TABLE` | `nfl_player_crosswalk` |
| `UNMATCHED_TABLE` | `nfl_player_crosswalk_unmatched` |
| `DUPLICATE_AUDIT_TABLE` | `nfl_player_crosswalk_duplicate_audit` |
| `CONFLICT_AUDIT_TABLE` | `nfl_player_crosswalk_conflict_audit` |
| `CROSSWALK_VERSION` | `2026.2.1` |

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `configure_logging` | 217 |  |
| `collapse_duplicate_columns` | 251 | Coalesce duplicate column labels without discarding non-null values. |
| `apply_aliases_without_collisions` | 276 | Apply aliases by coalescing source values into existing targets. |
| `clean_scalar` | 306 |  |
| `normalize_id` | 324 |  |
| `normalize_gsis_id` | 331 |  |
| `normalize_team` | 338 |  |
| `normalize_position` | 346 |  |
| `position_group` | 355 |  |
| `ascii_text` | 362 |  |
| `split_name_suffix` | 367 |  |
| `normalize_name` | 379 |  |
| `compact_name` | 389 |  |
| `initial_last_key` | 396 |  |
| `normalize_date` | 413 |  |
| `numeric_or_none` | 423 |  |
| `integer_or_none` | 432 |  |
| `first_non_null` | 439 |  |
| `most_common_non_null` | 446 |  |
| `joined_unique` | 455 |  |
| `dataframe_from_any` | 471 |  |
| `coalesce_column` | 479 |  |
| `ensure_columns` | 488 |  |
| `safe_json` | 496 |  |
| `sqlite_table_exists` | 504 |  |
| `sqlite_tables` | 512 |  |
| `sqlite_columns` | 519 |  |
| `read_sqlite_table` | 525 |  |
| `discover_snap_tables` | 530 |  |
| `load_historical_rosters` | 560 |  |
| `load_snap_records` | 606 | Load raw snap-count identities from authoritative sources only. |
| `standardize_source` | 702 |  |
| `standardize_snap_records` | 843 |  |
| `CandidateResult` | 914 |  |
| `duplicate_audit` | 923 |  |
| `build_conflicted_provider_values` | 972 |  |
| `make_provider_maps` | 1041 |  |
| `build_gsis_profiles` | 1064 |  |
| `profile_indexes` | 1103 |  |
| `score_name_candidate` | 1119 |  |
| `resolve_snap_pfr_id` | 1174 |  |
| `identity_quality` | 1312 |  |
| `aggregate_canonical_rows` | 1329 |  |
| `build_snap_summary` | 1466 |  |
| `append_unmatched_rows_to_crosswalk` | 1498 |  |
| `construct_outputs` | 1562 |  |
| `validate_outputs` | 1948 |  |
| `add_sqlite_indexes` | 1987 |  |
| `persist_outputs` | 2000 |  |
| `explode_pfr_crosswalk` | 2035 |  |
| `print_report` | 2048 |  |
| `main` | 2154 |  |

</details>

## build_nfl_player_master.py

Resolve the standardized roster into one canonical GSIS-based player population and write duplicate/identity/build audits. A team change does not create a new player.

[Open source](../build_nfl_player_master.py)

<details>
<summary>Source design notes</summary>

```text
Build the canonical 2026 NFL player master from the standardized roster.

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
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `VERSION` | `v2_1_canonical_gsis_master_explicit_audit_schema` |
| `RAW_ROSTER_TABLE` | `nfl_rosters_2026_raw` |
| `OUTPUT_TABLE` | `nfl_player_master_2026` |
| `DUPLICATE_AUDIT_TABLE` | `nfl_player_master_duplicate_audit_2026` |
| `IDENTITY_ISSUE_TABLE` | `nfl_player_master_identity_issues_2026` |
| `BUILD_AUDIT_TABLE` | `nfl_player_master_build_audit_2026` |

**Supported named options:** `--project-root`, `--db-path`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `clean_scalar` | 186 |  |
| `clean_id` | 198 |  |
| `normalize_team` | 208 |  |
| `normalize_position` | 216 |  |
| `position_group` | 223 |  |
| `ascii_text` | 228 |  |
| `split_name_suffix` | 235 |  |
| `normalize_name` | 247 |  |
| `compact_name` | 258 |  |
| `first_last_name` | 263 |  |
| `initial_last_key` | 271 |  |
| `first_non_null` | 279 |  |
| `status_score` | 286 |  |
| `read_table` | 293 |  |
| `ensure_columns` | 300 |  |
| `choose_canonical_rows` | 308 |  |
| `build_aliases` | 367 |  |
| `build_master` | 380 |  |
| `build_audit` | 474 |  |
| `configure_logger` | 491 |  |
| `save_frame` | 509 |  |
| `parse_args` | 519 |  |
| `main` | 526 |  |

</details>

## build_nfl_player_performance.py

Join all sources by canonical ID; convert supported position metrics to talent grades; use dedicated QB ratings and actual PFF OL grades; apply one replacement-level confidence shrink. Publish reconciliation and PFF identity audits.

[Open source](../build_nfl_player_performance.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical 2026 NFL player-performance inputs.

Required SQLite inputs
----------------------
1. nfl_player_master_2026
2. nfl_player_advanced_stats_current_roster_2026
3. nfl_player_advanced_stats_2022_2025
4. nfl_qb_rbsdm_ratings_2026
5. nfl_defensive_player_metrics_current_roster_2026

Required local OL input
-----------------------
- PFF offense-blocking exports for 2022, 2023, 2024 and 2025. The default
  filenames are offense_blocking (3).csv, offense_blocking (2).csv,
  offense_blocking (1).csv and offense_blocking.csv, respectively.

Primary outputs
---------------
1. nfl_player_performance_inputs_2026
2. nfl_player_performance_seasonal_2022_2025
3. nfl_ol_pff_player_season_2022_2025

Audit outputs
-------------
1. nfl_player_performance_position_summary_2026
2. nfl_player_performance_component_audit_2026
3. nfl_player_performance_unresolved_master_audit_2026
4. nfl_ol_pff_identity_audit_2026

Design principles
-----------------
- Join every upstream source by canonical GSIS player_id only.
- Preserve 2022-2025 as a true multi-year prior and score 2025 separately.
- Use the dedicated QB efficiency model as the authoritative QB talent source.
- Compute percentiles only among players with relevant samples.
- Renormalize component weights when an optional metric is unavailable.
- Make actual PFF blocking performance the authoritative OL talent signal.
- Treat OL snaps as sample confidence and availability, never as talent.
- Carry qualified pre-injury OL performance forward when a recent season is
  missing; an injury absence does not turn an established player into a
  replacement-level blocker.
- Apply one confidence shrink toward a data-derived position replacement
  baseline.
- Keep QB durability and age as metadata; do not mix them into QB talent.
- Do not globally re-standardize final player grades.
- Keep positional scarcity as metadata; team/unit weighting owns positional value.
- Retain rookies and low-history players at conservative replacement grades.

This file is the canonical full-file replacement for
build_nfl_player_performance.py.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_PLAYER_PERFORMANCE_2026_CANONICAL_V4` |
| `PERFORMANCE_VERSION` | `v4_1_pff_row_identity_collision_safe` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `ADVANCED_CURRENT_TABLE` | `nfl_player_advanced_stats_current_roster_2026` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_2022_2025` |
| `QB_TABLE` | `nfl_qb_rbsdm_ratings_2026` |
| `DEFENSE_TABLE` | `nfl_defensive_player_metrics_current_roster_2026` |
| `OUTPUT_TABLE` | `nfl_player_performance_inputs_2026` |
| `SEASONAL_OUTPUT_TABLE` | `nfl_player_performance_seasonal_2022_2025` |
| `POSITION_SUMMARY_TABLE` | `nfl_player_performance_position_summary_2026` |
| `COMPONENT_AUDIT_TABLE` | `nfl_player_performance_component_audit_2026` |
| `UNRESOLVED_AUDIT_TABLE` | `nfl_player_performance_unresolved_master_audit_2026` |
| `PFF_OL_HISTORY_TABLE` | `nfl_ol_pff_player_season_2022_2025` |
| `PFF_OL_IDENTITY_AUDIT_TABLE` | `nfl_ol_pff_identity_audit_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 390 |  |
| `configure_logging` | 405 |  |
| `get_engine` | 422 |  |
| `table_exists` | 426 |  |
| `read_table` | 432 |  |
| `save_frame` | 441 |  |
| `clean_column` | 458 |  |
| `clean_scalar` | 463 |  |
| `clean_id_series` | 470 |  |
| `normalize_team` | 474 |  |
| `normalize_position_group` | 482 |  |
| `ensure_columns` | 492 |  |
| `numeric_series` | 500 |  |
| `clip_0_100` | 506 |  |
| `reliability` | 510 |  |
| `age_curve_score` | 519 |  |
| `weighted_component_score` | 537 |  |
| `percentile_metric` | 559 |  |
| `validate_unique_ids` | 584 |  |
| `prepare_master` | 601 |  |
| `prepare_advanced` | 630 |  |
| `prepare_qb` | 653 |  |
| `prepare_defense` | 671 |  |
| `normalize_person_name` | 694 |  |
| `initial_last_name_key` | 706 |  |
| `normalize_ol_position` | 717 |  |
| `ol_replacement_baseline` | 728 |  |
| `resolve_pff_paths` | 733 |  |
| `load_pff_ol_history` | 777 |  |
| `master_alias_keys` | 853 |  |
| `match_pff_ol_to_master` | 861 |  |
| `position_volume` | 1201 |  |
| `build_history_context` | 1225 |  |
| `score_position_frame` | 1304 | Return one advanced-only position score without dedicated-source reuse. |
| `merge_sources` | 1457 |  |
| `build_recent_scores` | 1518 |  |
| `add_final_scores` | 1583 |  |
| `build_position_summary` | 1905 |  |
| `build_component_audit` | 1926 |  |
| `validate_outputs` | 1948 |  |
| `main` | 2100 |  |

</details>

## build_nfl_power_ratings.py

Convert structural strength to a neutral-field point scale; produce power and historical calibration audits. These are upstream/legacy ratings, not the full final learned-consensus equation.

[Open source](../build_nfl_power_ratings.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical 2026 NFL neutral-field power ratings.

Purpose
-------
Convert the validated replacement-anchored team-strength index into neutral-
field point ratings without forcing the 2026 distribution to match historical
market-rating variance.

Core principles
---------------
1. `nfl_team_strength_2026.point_ready_index` is the native structural point
   scale and remains the default point rating.
2. Historical closing spreads are used for:
   - home-field diagnostics;
   - market-rating dispersion diagnostics;
   - reproducible audit tables.
3. Historical market standard deviation is NOT used to rescale 2026 ratings.
4. An empirical slope may be learned only from an explicit paired calibration
   table containing historical structural differences and market-neutral target
   differences. If that table is absent or insufficient, the mapping is exactly
   identity: 1 structural point = 1 neutral-field point.
5. Home field, current injuries, rest, travel, weather, and matchup adjustments
   remain outside this preseason neutral-field rating.

Inputs
------
SQLite:
    nfl_team_strength_2026

Optional SQLite paired calibration table:
    nfl_power_rating_calibration_observations

The optional table must provide either:
    structural_diff
    market_neutral_diff

or recognizable aliases documented in `prepare_paired_calibration`.

External historical schedule source:
    nflreadpy.load_schedules
    nfl_data_py.import_schedules fallback

Outputs
-------
SQLite and CSV:
    nfl_power_ratings_2026
    nfl_power_rating_calibration_season_audit
    nfl_power_rating_calibration_team_audit
    nfl_power_rating_point_mapping_audit

Interpretation
--------------
A `power_rating_points` value of +5.0 means the team is rated five points
better than an average NFL team on a neutral field.

Neutral-field matchup:
    team_a_power_rating - team_b_power_rating

Home-field matchup:
    home_power - away_power + current_home_field_adjustment
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_POWER_RATINGS_2026_CANONICAL_V2` |
| `VERSION` | `v2_native_point_scale_optional_paired_calibration` |
| `TEAM_STRENGTH_TABLE` | `nfl_team_strength_2026` |
| `DEFAULT_PAIRED_CALIBRATION_TABLE` | `nfl_power_rating_calibration_observations` |
| `OUTPUT_TABLE` | `nfl_power_ratings_2026` |
| `SEASON_AUDIT_TABLE` | `nfl_power_rating_calibration_season_audit` |
| `TEAM_AUDIT_TABLE` | `nfl_power_rating_calibration_team_audit` |
| `MAPPING_AUDIT_TABLE` | `nfl_power_rating_point_mapping_audit` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--paired-calibration-table`, `--native-point-slope`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 226 |  |
| `configure_logging` | 271 |  |
| `get_engine` | 299 |  |
| `table_exists` | 307 |  |
| `read_table` | 322 |  |
| `read_optional_table` | 340 |  |
| `normalize_team` | 346 |  |
| `numeric_column` | 361 |  |
| `first_existing` | 371 |  |
| `frame_from_any` | 382 |  |
| `clip_and_recenter` | 390 |  |
| `load_historical_schedules` | 420 |  |
| `standardize_schedules` | 463 |  |
| `infer_spread_sign` | 579 |  |
| `fit_market_ratings_for_season` | 625 |  |
| `build_historical_market_diagnostics` | 733 |  |
| `prepare_paired_calibration` | 799 |  |
| `select_point_slope` | 870 |  |
| `prepare_team_strength` | 1004 |  |
| `build_power_ratings` | 1103 |  |
| `validate_outputs` | 1254 |  |
| `save_outputs` | 1346 |  |
| `add_indexes` | 1378 |  |
| `print_report` | 1398 |  |
| `main` | 1498 |  |

</details>

## build_nfl_projected_depth_chart.py

Project starters from canonical talent, eligibility and authoritative QB/OL evidence. Preserve the QB authority table and manual OL rows. Talent does not invent OL position assignments.

[Open source](../build_nfl_projected_depth_chart.py)

<details>
<summary>Source design notes</summary>

```text
Build the canonical 2026 NFL projected depth chart.

Required SQLite inputs
----------------------
- nfl_player_performance_inputs_2026
- nfl_player_master_2026

Persistent authoritative inputs
--------------------------------
- nfl_projected_qb_starters_2026
- nfl_projected_ol_starters_2026 (refreshed from ESPN; manual rows preserved)

Outputs
-------
- nfl_projected_depth_chart_2026
- nfl_projected_depth_chart_audit_2026

Design rules
------------
1. The validated player-performance table is the canonical player population.
2. Performance and roster/master fields are joined once by canonical player_id.
3. Player quality, availability and depth-order evidence remain separate.
4. OL performance consumes the PFF blocking-talent grade from the canonical
   player table without a second shrink or availability discount.
5. LT/LG/C/RG/RT assignments come from a position-specific current depth chart;
   player quality and historical durability never invent starter assignments.
6. Reserve/injured/developmental roster statuses cannot be projected starters.
7. The persistent QB starter table remains authoritative and is never silently
   overwritten after it has been created.
8. No empirical mean/std normalization is performed in this file.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_PROJECTED_DEPTH_CHART_2026_CANONICAL_V5` |
| `VERSION` | `v5_pff_ol_talent_no_second_shrink_authoritative_starters` |
| `PERFORMANCE_TABLE` | `nfl_player_performance_inputs_2026` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `QB_STARTER_TABLE` | `nfl_projected_qb_starters_2026` |
| `OL_STARTER_TABLE` | `nfl_projected_ol_starters_2026` |
| `OUTPUT_TABLE` | `nfl_projected_depth_chart_2026` |
| `AUDIT_TABLE` | `nfl_projected_depth_chart_audit_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-refresh-ol-starters`, `--espn-timeout`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 166 |  |
| `configure_logging` | 180 |  |
| `clean_scalar` | 195 |  |
| `clean_id` | 207 |  |
| `normalize_team` | 212 |  |
| `normalize_name` | 217 |  |
| `player_match_key` | 223 |  |
| `normalize_position` | 229 |  |
| `canonical_role` | 234 |  |
| `numeric` | 254 |  |
| `table_exists` | 260 |  |
| `read_table` | 264 |  |
| `load_inputs` | 273 |  |
| `ensure_qb_starter_table` | 285 |  |
| `load_qb_starters` | 307 |  |
| `ensure_ol_starter_table` | 333 |  |
| `http_json` | 353 |  |
| `fetch_espn_team_ol_starters` | 374 |  |
| `fetch_espn_ol_starters` | 432 |  |
| `validate_ol_starters` | 451 |  |
| `load_cached_ol_starters` | 494 |  |
| `load_ol_starters` | 505 |  |
| `likely_unavailable` | 586 |  |
| `prepare_depth` | 594 |  |
| `role_rank_key` | 729 |  |
| `compatible` | 737 |  |
| `assign_starters_and_ranks` | 750 |  |
| `finalize` | 803 |  |
| `validate` | 844 |  |
| `main` | 890 |  |

</details>

## build_nfl_team_strength.py

Combine replacement-anchored units, apply a small confidence/availability-weighted OL continuity contribution once, and expose component/QB/completeness audits. Avoid a second QB adjustment.

[Open source](../build_nfl_team_strength.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical 2026 NFL preseason team-strength ratings.

Required SQLite inputs
----------------------
- nfl_team_unit_ratings_2026
- nfl_position_group_ratings_2026
- nfl_team_unit_completeness_2026
- nfl_projected_depth_chart_2026

Outputs
-------
- nfl_team_strength_2026
- nfl_team_strength_component_audit_2026
- nfl_team_strength_qb_audit_2026
- nfl_team_strength_completeness_audit_2026

Design rules
------------
1. Preserve replacement-anchored offense, defense, and special-teams scales.
2. Do not standardize individual units or force any component variance.
3. Do not add a separate QB adjustment. QB is already embedded in offense.
4. Apply OL continuity exactly once, only through the OL share of offense.
5. Weight the OL continuity context by its explicit confidence and availability,
   with a deliberately small cap so individual blocking talent remains primary.
6. Do not re-merge player performance. The depth chart is the QB audit contract.
7. Publish a centered raw index and an overall z-score for downstream historical
   point calibration; neither is itself claimed to be spread points.
8. Keep injuries, home field, rest, travel, weather, and weekly form outside the
   permanent preseason base-strength layer.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_TEAM_STRENGTH_2026_CANONICAL_V3` |
| `VERSION` | `v3_pff_ol_talent_primary_bounded_continuity_no_double_count` |
| `TEAM_UNIT_TABLE` | `nfl_team_unit_ratings_2026` |
| `POSITION_UNIT_TABLE` | `nfl_position_group_ratings_2026` |
| `COMPLETENESS_TABLE` | `nfl_team_unit_completeness_2026` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_2026` |
| `OUTPUT_TABLE` | `nfl_team_strength_2026` |
| `COMPONENT_AUDIT_TABLE` | `nfl_team_strength_component_audit_2026` |
| `QB_AUDIT_TABLE` | `nfl_team_strength_qb_audit_2026` |
| `COMPLETENESS_AUDIT_TABLE` | `nfl_team_strength_completeness_audit_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 107 |  |
| `configure_logging` | 119 |  |
| `clean_scalar` | 146 |  |
| `clean_id` | 167 |  |
| `normalize_team` | 171 |  |
| `numeric` | 176 |  |
| `table_exists` | 189 |  |
| `read_table` | 205 |  |
| `reject_collision_columns` | 223 |  |
| `safe_zscore` | 238 |  |
| `prepare_team_units` | 256 |  |
| `prepare_position_units` | 387 |  |
| `prepare_completeness` | 474 |  |
| `prepare_depth` | 547 |  |
| `select_authoritative_qbs` | 624 |  |
| `build_team_strength` | 716 |  |
| `build_completeness_audit` | 1000 |  |
| `validate_outputs` | 1066 |  |
| `save_outputs` | 1223 |  |
| `add_indexes` | 1254 |  |
| `print_report` | 1282 |  |
| `main` | 1416 |  |

</details>

## build_nfl_team_unit_ratings.py

Aggregate projected starter/depth grades into position units and team offense, defense and special teams with explicit completeness/confidence contracts.

[Open source](../build_nfl_team_unit_ratings.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical 2026 NFL team unit ratings.

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
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_TEAM_UNIT_RATINGS_2026_CANONICAL_V8` |
| `VERSION` | `v8_pff_ol_talent_contract_qb_ceiling_95` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_2026` |
| `OL_CONTINUITY_TABLE` | `nfl_ol_continuity_2026` |
| `TEAM_OUTPUT_TABLE` | `nfl_team_unit_ratings_2026` |
| `UNIT_OUTPUT_TABLE` | `nfl_position_group_ratings_2026` |
| `AUDIT_OUTPUT_TABLE` | `nfl_team_unit_player_audit_2026` |
| `COMPLETENESS_OUTPUT_TABLE` | `nfl_team_unit_completeness_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `SlotSpec` | 97 |  |
| `parse_args` | 140 |  |
| `configure_logging` | 148 |  |
| `clean_scalar` | 163 |  |
| `clean_id` | 175 |  |
| `normalize_team` | 179 |  |
| `numeric` | 184 |  |
| `table_exists` | 190 |  |
| `read_table` | 194 |  |
| `compatible` | 203 |  |
| `prepare_depth` | 218 |  |
| `load_continuity` | 250 |  |
| `choose_player` | 267 |  |
| `replacement_for_slot` | 284 |  |
| `build_unit` | 297 |  |
| `weighted_average` | 399 |  |
| `build_all` | 403 |  |
| `validate` | 486 |  |
| `main` | 535 |  |

</details>

## load_nfl_coverage_metrics.py

Create current-roster coverage metrics from canonical history with snap-based denominators and confidence shrinkage.

[Open source](../load_nfl_coverage_metrics.py)

<details>
<summary>Source design notes</summary>

```text
Build 2022-2025 coverage metrics and a 2026 current-roster view.

The script reads canonical player-season statistics from
nfl_player_advanced_stats_2022_2025. It does not reload play-by-play and does
not perform name-based player matching.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_COVERAGE_2026_CANONICAL_V3` |
| `METRICS_VERSION` | `v3_canonical_2022_2025_snap_denominator_shrunk` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_2022_2025` |
| `OUTPUT_TABLE` | `nfl_coverage_metrics_2022_2025` |
| `CURRENT_OUTPUT_TABLE` | `nfl_coverage_metrics_current_roster_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 58 |  |
| `get_engine` | 66 |  |
| `table_exists` | 70 |  |
| `read_table` | 76 |  |
| `clean_id` | 85 |  |
| `numeric` | 92 |  |
| `percentile` | 98 |  |
| `empirical_bayes_rate` | 107 |  |
| `load_master` | 115 |  |
| `load_history` | 126 |  |
| `score_historical` | 144 |  |
| `weighted_rollup` | 223 |  |
| `build_current` | 253 |  |
| `save_frame` | 284 |  |
| `main` | 290 |  |

</details>

## load_nfl_defensive_front_metrics.py

Create current-roster front metrics from historical defensive statistics using consistent snap denominators and confidence shrinkage.

[Open source](../load_nfl_defensive_front_metrics.py)

<details>
<summary>Source design notes</summary>

```text
Build 2022-2025 defensive-front metrics and a 2026 current-roster view.

This script consumes the validated canonical advanced-history table. It does
not reload play-by-play and it does not perform name-based identity matching.
All joins use canonical GSIS player_id values.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_DEFENSIVE_FRONT_2026_CANONICAL_V3` |
| `METRICS_VERSION` | `v3_canonical_2022_2025_snap_denominator_shrunk` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_2022_2025` |
| `OUTPUT_TABLE` | `nfl_defensive_front_metrics_2022_2025` |
| `CURRENT_OUTPUT_TABLE` | `nfl_defensive_front_metrics_current_roster_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 61 |  |
| `get_engine` | 69 |  |
| `table_exists` | 73 |  |
| `read_table` | 79 |  |
| `clean_id` | 88 |  |
| `numeric` | 95 |  |
| `safe_divide` | 101 |  |
| `percentile` | 107 |  |
| `empirical_bayes_rate` | 116 |  |
| `load_master` | 124 |  |
| `load_history` | 136 |  |
| `score_historical` | 154 |  |
| `weighted_rollup` | 252 |  |
| `build_current` | 282 |  |
| `save_frame` | 313 |  |
| `main` | 319 |  |

</details>

## load_nfl_live_market_odds.py

Fetch current NFL spread markets through the Odds API, reconcile against the schedule and write live market output/audits for execution comparisons. Requires private credentials and the database schedule.

[Open source](../load_nfl_live_market_odds.py)

<details>
<summary>Source design notes</summary>

```text
Load one sharp reference spread per 2026 NFL matchup.

The loader uses The Odds API current NFL spreads endpoint and selects the first
available, non-stale bookmaker in an explicit priority order. It writes one
row per scheduled matchup for the requested week. It does not label any proxy
book as Circa; official Circa contest boards remain a separate manual input.

API-key setup (PowerShell session):
    $env:ODDS_API_KEY = "YOUR_KEY"

Typical run:
    python load_nfl_live_market_odds.py --week 1
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_LIVE_MARKET_ODDS_2026_CANONICAL_V1` |
| `VERSION` | `v1_odds_api_sharp_priority_audited_single_line_per_game` |
| `CURRENT_TABLE` | `nfl_live_market_odds_2026` |
| `HISTORY_TABLE` | `nfl_live_market_odds_history_2026` |
| `AUDIT_TABLE` | `nfl_live_market_odds_run_audit_2026` |
| `AUDIT_HISTORY_TABLE` | `nfl_live_market_odds_run_audit_history_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--week`, `--api-key-env`, `--bookmaker-priority`, `--maximum-line-age-minutes`, `--require-primary-bookmaker`, `--allow-partial-week`, `--schedule-table`, `--output-path`, `--response-json-path`, `--timeout-seconds`, `--no-raw-json`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 152 |  |
| `normalize_team` | 219 |  |
| `first_existing` | 224 |  |
| `parse_priority` | 235 |  |
| `table_exists` | 253 |  |
| `load_schedule` | 260 |  |
| `utc_now` | 336 |  |
| `iso_utc` | 340 |  |
| `parse_api_time` | 344 |  |
| `fetch_api_response` | 354 |  |
| `load_response_json` | 395 |  |
| `bookmaker_spread_offer` | 406 |  |
| `select_week_lines` | 470 |  |
| `append_with_schema_evolution` | 574 |  |
| `save_database` | 605 |  |
| `save_files` | 636 |  |
| `build_audit` | 671 |  |
| `run_self_test` | 717 |  |
| `main` | 808 |  |

</details>

## load_nfl_rosters.py

Import roster data, standardize team/identity fields and preserve provider IDs. Separates raw roster capture from canonical player identity.

[Open source](../load_nfl_rosters.py)

<details>
<summary>Source design notes</summary>

```text
Load and standardize the current 2026 NFL roster while preserving identities.

Primary output
--------------
SQLite/CSV: nfl_rosters_2026_raw

Audit outputs
-------------
SQLite/CSV: nfl_rosters_2026_load_audit
SQLite/CSV: nfl_rosters_2026_duplicate_id_audit
SQLite/CSV: nfl_rosters_2026_identity_issues

Design rules
------------
1. GSIS is the canonical working identifier used by downstream nflverse data.
2. Every available provider ID is preserved; no provider ID is renamed away.
3. Text placeholders such as "nan" and "None" are converted to missing values.
4. Duplicate/conflicting IDs are audited rather than silently discarded.
5. The standardized raw roster remains a source table; canonical row selection
   belongs in build_nfl_player_master.py.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `VERSION` | `v2_0_identity_preserving_roster` |
| `OUTPUT_TABLE` | `nfl_rosters_2026_raw` |
| `LOAD_AUDIT_TABLE` | `nfl_rosters_2026_load_audit` |
| `DUPLICATE_AUDIT_TABLE` | `nfl_rosters_2026_duplicate_id_audit` |
| `IDENTITY_ISSUE_TABLE` | `nfl_rosters_2026_identity_issues` |

**Supported named options:** `--season`, `--project-root`, `--db-path`, `--input-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `clean_scalar` | 156 |  |
| `clean_id` | 170 |  |
| `normalize_team` | 180 |  |
| `normalize_position` | 188 |  |
| `clean_column_name` | 195 |  |
| `coalesce_columns` | 201 |  |
| `load_rosters` | 210 |  |
| `standardize_rosters` | 237 |  |
| `build_duplicate_audit` | 330 |  |
| `build_load_audit` | 347 |  |
| `configure_logger` | 366 |  |
| `save_frame` | 384 |  |
| `parse_args` | 394 |  |
| `main` | 403 |  |

</details>

## load_rbsdm_qb_ratings.py

Build the dedicated quarterback efficiency prior using available EPA/CPOE history and optional local QB files. Stabilize individual metrics with sample confidence; preserve the unshrunk composite contract expected downstream.

[Open source](../load_rbsdm_qb_ratings.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical 2026 quarterback ratings from 2022-2025 efficiency data.

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
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_RBSDM_QB_2026_CANONICAL_V3` |
| `RATING_VERSION` | `v3_single_metric_stabilization_unshrunk_composite` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `ADVANCED_HISTORY_TABLE` | `nfl_player_advanced_stats_2022_2025` |
| `RAW_OUTPUT_TABLE` | `nfl_qb_rbsdm_ratings_raw` |
| `OUTPUT_TABLE` | `nfl_qb_rbsdm_ratings_2026` |
| `UNMATCHED_TABLE` | `nfl_qb_rbsdm_local_unmatched_audit` |

**Supported named options:** `--project-root`, `--db-path`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 92 |  |
| `get_engine` | 100 |  |
| `table_exists` | 104 |  |
| `read_table` | 110 |  |
| `clean_col` | 119 |  |
| `clean_id` | 125 |  |
| `clean_name` | 132 |  |
| `normalize_team` | 142 |  |
| `numeric` | 149 |  |
| `first_present` | 155 |  |
| `percentile` | 160 |  |
| `load_master_qbs` | 169 |  |
| `load_advanced_qb` | 185 |  |
| `standardize_local_file` | 225 |  |
| `map_local_to_canonical` | 258 |  |
| `overlay_local_metrics` | 326 |  |
| `score_qb_seasons` | 356 |  |
| `build_current_qb` | 406 |  |
| `save_frame` | 505 |  |
| `main` | 511 |  |

</details>

## nfl_player_advanced_stats.py

Build prior player statistics from weekly data, play-by-play, Next Gen and participation sources. Apply position-specific season weights and reconcile snap IDs, then publish historical and current-roster views.

[Open source](../nfl_player_advanced_stats.py)

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `ADVANCED_BUILD_ID` | `NFL_ADVANCED_STATS_2026_V3_2_FINAL` |
| `ADVANCED_VERSION` | `v3_2_final_snap_participation_identity` |
| `DB_PATH` | `C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite` |
| `MASTER_TABLE` | `nfl_player_master_2026` |
| `CROSSWALK_TABLE` | `nfl_player_crosswalk` |
| `OUTPUT_TABLE` | `nfl_player_advanced_stats_2022_2025` |
| `OUTPUT_TABLE_CURRENT` | `nfl_player_advanced_stats_current_roster_2026` |
| `SNAP_ID_AUDIT_TABLE` | `nfl_player_advanced_stats_snap_id_audit` |

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `get_engine` | 267 |  |
| `table_exists` | 271 |  |
| `read_table` | 285 |  |
| `safe_numeric` | 296 |  |
| `normalize_team` | 303 |  |
| `normalize_position` | 311 |  |
| `position_group` | 318 |  |
| `clean_player_id` | 327 |  |
| `add_missing_cols` | 337 |  |
| `safe_divide` | 347 |  |
| `load_master` | 354 |  |
| `_standardize_weekly_frame` | 402 | Normalize one weekly-player source and collapse duplicate player-week rows. |
| `_load_weekly_with_nflreadpy` | 506 |  |
| `load_weekly_data` | 576 |  |
| `load_snap_counts` | 661 |  |
| `load_crosswalk_pfr_map` | 677 | Load the permanent one-way PFR -> canonical GSIS map. |
| `canonicalize_snap_ids` | 747 |  |
| `load_ngs_data` | 921 |  |
| `load_pbp_data` | 958 |  |
| `build_weekly_offense` | 1041 |  |
| `_first_non_missing` | 1188 | Return the first useful scalar from a Series, else None. |
| `build_snap_metrics` | 1204 | Aggregate snap participation while preserving historical identity. |
| `build_ngs_metrics` | 1280 |  |
| `pbp_player_events` | 1336 |  |
| `combine_event_frames` | 1363 |  |
| `build_pbp_qb_receiving_rushing` | 1407 |  |
| `build_pbp_defense` | 1500 |  |
| `build_base_player_seasons` | 1604 |  |
| `coalesce_player_identity` | 1618 | Attach historical identity without overwriting it with 2026 identity. |
| `build_advanced_stats` | 1708 |  |
| `apply_master_identity` | 1872 |  |
| `get_position_season_weight` | 1957 |  |
| `build_current_roster_view` | 1971 | Create the current-roster historical view with explicit sample fields. |
| `validate_current_history_rollup` | 2076 | Prove that no current-player history is lost during the rollup. |
| `print_season_counts` | 2123 |  |
| `validate_2025_presence` | 2141 |  |
| `main` | 2159 |  |

</details>

## predict_nfl_circa_top5_2026.py

Validate frozen Circa bundles, construct or reuse point-in-time features, score official contest lines and produce both fixed-policy cards plus confidence/manual-review outputs. Its structural overlay contract is legacy.

[Open source](../predict_nfl_circa_top5_2026.py)

<details>
<summary>Source design notes</summary>

```text
Generate the production 2026 Circa Million two-entry portfolio.

Entry 1 uses the locked V1 residual model:
    models/nfl_circa_contest_model_v1.joblib

Its selection policy is the promoted adjusted-V1 rule:
    - raw residual ranks 1-4 are locked;
    - raw ranks 5-8 may compete for the fifth slot only when within 0.50
      residual points of raw rank 5;
    - the tiebreak uses prior-only conditional final-margin mass at keys 3/7.

Entry 2 uses the promoted Ceiling rule:
    Weeks 1-9: exact raw V1 predictions and ranks
    Weeks 10-18: full late-season BLENDED_SIGN ceiling residual model, with
                 selected home favorites of 7.5 points or more excluded before
                 residual ranking and replaced by the next eligible residual.

Both cards use the same point-in-time matchup matrix and official Circa board.
The predicted residual models remain unchanged; only the promoted, fully
audited contest-selection policies are applied.

Required line input
-------------------
CSV, XLSX, or Parquet with:
    week, away_team, home_team, circa_home_margin

Alternatively provide home_spread, where:
    home_spread = -3.5 -> circa_home_margin = +3.5

Default model bundles
---------------------
Locked V1:
    models/nfl_circa_contest_model_v1.joblib

Ceiling production model:
    models/nfl_circa_contest_season_phase_v3_1.joblib

If the V3.1 bundle is unavailable, the runner automatically checks the V3.2
bundle and extracts its unchanged A1000/W1/BLENDED_SIGN ceiling source.

Outputs
-------
Existing SQLite tables remain stable for downstream compatibility:
    nfl_circa_top5_predictions_2026
    nfl_circa_top5_prediction_history_2026

Production portfolio tables:
    nfl_circa_ceiling_predictions_2026
    nfl_circa_ceiling_prediction_history_2026
    nfl_circa_v1_vs_ceiling_comparison_2026
    nfl_circa_v1_vs_ceiling_comparison_history_2026
    nfl_circa_final_portfolio_predictions_2026
    nfl_circa_final_portfolio_prediction_history_2026
    nfl_circa_selection_confidence_2026
    nfl_circa_selection_confidence_history_2026
    nfl_circa_manual_submission_draft_2026

CSV snapshots:
    outputs/circa_2026/nfl_circa_top5_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_ceiling_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_v1_vs_ceiling_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_final_portfolio_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_confidence_week_<W>_<timestamp>.csv
    outputs/circa_2026/nfl_circa_manual_submission_week_<W>_<timestamp>.csv

The confidence score is an advisory selection-robustness score, not a cover
probability. The immutable model card remains separate from the editable
manual-submission draft. The market-independent STRUCTURAL_FORM_HFA fair line
is also joined as a confidence-only overlay. A contradiction of at least 1.5
points against within-card residual-strength ranks 4-5 raises manual-review
priority; it never changes either production card or invents a numeric score
penalty.

No sportsbook stake is recommended by this script.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_CIRCA_TOP5_2026_PRODUCTION_CONFIDENCE_BOARD` |
| `VERSION` | `v8_3_structural_lineage_live_feature_freshness` |
| `DEFAULT_MODEL_FILENAME` | `nfl_circa_contest_model_v1.joblib` |
| `MARGIN_HISTORY_TABLE` | `nfl_circa_backtest_matrix` |
| `OUTPUT_TABLE` | `nfl_circa_top5_predictions_2026` |
| `OUTPUT_HISTORY_TABLE` | `nfl_circa_top5_prediction_history_2026` |
| `RUN_AUDIT_TABLE` | `nfl_circa_top5_run_audit_2026` |
| `RUN_AUDIT_HISTORY_TABLE` | `nfl_circa_top5_run_audit_history_2026` |
| `CEILING_OUTPUT_TABLE` | `nfl_circa_ceiling_predictions_2026` |
| `CEILING_OUTPUT_HISTORY_TABLE` | `nfl_circa_ceiling_prediction_history_2026` |
| `COMPARISON_OUTPUT_TABLE` | `nfl_circa_v1_vs_ceiling_comparison_2026` |
| `COMPARISON_OUTPUT_HISTORY_TABLE` | `nfl_circa_v1_vs_ceiling_comparison_history_2026` |
| `PORTFOLIO_OUTPUT_TABLE` | `nfl_circa_final_portfolio_predictions_2026` |
| `PORTFOLIO_OUTPUT_HISTORY_TABLE` | `nfl_circa_final_portfolio_prediction_history_2026` |
| `CONFIDENCE_OUTPUT_TABLE` | `nfl_circa_selection_confidence_2026` |
| `CONFIDENCE_OUTPUT_HISTORY_TABLE` | `nfl_circa_selection_confidence_history_2026` |
| `MANUAL_SUBMISSION_DRAFT_TABLE` | `nfl_circa_manual_submission_draft_2026` |
| `STRUCTURAL_LIVE_TABLE` | `nfl_weekly_power_spread_predictions_2026` |
| `STRUCTURAL_RUN_AUDIT_TABLE` | `nfl_weekly_power_spread_run_audit_2026` |
| `EXPECTED_STRUCTURAL_BUILD_ID` | `NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4` |
| `EXPECTED_STRUCTURAL_VERSION` | `v4_structural_form_hfa_fail_closed_lineage` |
| `EXPECTED_FORM_BUILD_ID` | `NFL_2026_FORM_RATING_CANONICAL_V3` |
| `EXPECTED_FORM_VERSION` | `v3_current_structural_prior_asof_opponent_adjusted_form` |
| `EXPECTED_STRUCTURAL_POWER_BUILD_ID` | `NFL_POWER_RATINGS_2026_CANONICAL_V2` |
| `EXPECTED_LIVE_MATCHUP_BUILD_ID` | `NFL_WEEKLY_MATCHUP_MARKET_RESIDUAL_CANONICAL_V3` |
| `EXPECTED_LIVE_MATCHUP_VERSION` | `v3_qb_gsis_identity_crosswalk_dead_feature_guard` |
| `EXPECTED_V1_MODEL_BUILD_ID` | `NFL_CIRCA_CONTEST_LINES_CANONICAL_V1` |
| `EXPECTED_V1_MODEL_VERSION` | `v1_5_qb_identity_integrity_guard` |
| `EXPECTED_CEILING_MODEL_BUILD_ID` | `NFL_CIRCA_CONTEST_SEASON_PHASE_V3_2_QB_REBUILT` |
| `EXPECTED_CEILING_MODEL_VERSION` | `v3_2_1_fixed_architecture_qb_integrity_lineage` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--margin-history-db-path`, `--margin-history-table`, `--structural-table`, `--week`, `--as-of-date`, `--circa-lines-path`, `--features-path`, `--feature-table`, `--schedule-path`, `--model-path`, `--ceiling-model-path`, `--matchup-builder-path`, `--approve-contest-forward-test`, `--rebuild-live-features`, `--reuse-live-features`, `--skip-auto-feature-build`, `--no-csv`, `--preflight-only`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 225 |  |
| `now_string` | 387 |  |
| `sha256_file` | 391 |  |
| `normalize_team` | 399 |  |
| `first_existing` | 404 |  |
| `table_exists` | 415 |  |
| `read_table` | 426 |  |
| `numeric` | 441 |  |
| `require_columns` | 451 |  |
| `prepare_margin_history` | 461 |  |
| `load_margin_history` | 496 |  |
| `one_required_text` | 521 |  |
| `prepare_structural_confidence_projection` | 539 |  |
| `load_structural_confidence_projection` | 805 |  |
| `margin_probability_mass` | 832 |  |
| `conditional_margin_probability_mass` | 853 |  |
| `crossed_integer_margins` | 897 |  |
| `add_primary_key_features` | 911 |  |
| `read_frame` | 964 |  |
| `find_default_lines_path` | 980 |  |
| `prepare_lines` | 1002 |  |
| `assert_model_bundle_qb_integrity` | 1086 |  |
| `load_bundle` | 1131 |  |
| `late_model_from_mapping` | 1182 |  |
| `load_ceiling_bundle` | 1218 |  |
| `load_schedule_package` | 1333 |  |
| `augment_schedule_for_target_week` | 1372 |  |
| `patched_builder_source` | 1468 |  |
| `build_live_feature_database` | 1499 |  |
| `load_validated_live_feature_matrix` | 1572 |  |
| `load_feature_matrix` | 1673 |  |
| `merge_features_and_lines` | 1703 |  |
| `assert_live_qb_feature_integrity` | 1787 |  |
| `add_selection_fields` | 1882 |  |
| `apply_adjusted_v1_policy` | 1933 | Lock raw ranks 1-4 and use prior-only 3/7 mass for the fifth slot. |
| `apply_late_home_favorite_veto` | 2016 | Exclude late selected home favorites >= 7.5 before Ceiling ranking. |
| `predict_raw_v1_card` | 2089 |  |
| `predict_adjusted_v1_card` | 2108 |  |
| `predict_ceiling_card` | 2118 |  |
| `assert_early_ceiling_identity` | 2169 |  |
| `build_card_comparison` | 2196 |  |
| `build_final_portfolio` | 2331 |  |
| `load_historical_rank_analogs` | 2404 | Load descriptive production-rank results without using them in score. |
| `confidence_tier` | 2492 |  |
| `boundary_component` | 2502 |  |
| `agreement_component` | 2507 |  |
| `build_confidence_board` | 2525 | Rank picks by robustness plus a non-substitutive structural review flag. |
| `add_confidence_to_portfolio` | 2985 |  |
| `build_manual_submission_draft` | 3052 |  |
| `sqlite_type_for_series` | 3123 |  |
| `append_with_schema_evolution` | 3131 |  |
| `save_outputs` | 3169 |  |
| `run_self_test` | 3510 |  |
| `main` | 3934 |  |

</details>

## predict_nfl_weekly_learned_consensus_2026.py

Generate independent unit, starter-slot and nonlinear projections; average unit/nonlinear for the fair margin; apply agreement/range/edge/scope gates after market attachment; export readable betting and audit outputs.

[Open source](../predict_nfl_weekly_learned_consensus_2026.py)

**Local imports:** `backtest_nfl_learned_structural_weights`, `backtest_nfl_nonlinear_matchup_consensus`, `build_backtest_nfl_weekly_matchup_residual`, `build_nfl_learned_consensus_2026`, `predict_nfl_weekly_power_spreads_2026`.

<details>
<summary>Source design notes</summary>

```text
Produce the canonical 2026 learned structural-consensus weekly projection.

All three projections are completed before a market is attached:

1. learned eight-unit positive ridge;
2. learned 26-slot positive ridge; and
3. nonlinear unit/form/process/personnel model.

The fair margin is the mean of the unit and nonlinear projections.  The frozen
prospective gate requires at least a two-point edge, no more than 3.5 points of
range across the three projections, and all three models selecting the same
side.  Week 1 is outside the validating backtest scope and is stake-ineligible
by default.  The explicit --allow-week1-stakes override permits otherwise
qualifying Week 1 bets while preserving an unvalidated-scope audit label.

This script replaces the legacy structural/HFA branch.  It deliberately keeps
the established schedule, market, price, history-table, and CSV contracts so
the separate Circa engine is not modified.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_WEEKLY_LEARNED_CONSENSUS_2026_CANONICAL_V1` |
| `VERSION` | `v1_3_readable_execution_csv_scope_fix` |

**Supported named options:** `--project-root`, `--database-root`, `--db-path`, `--database`, `--model-path`, `--source-cache-db`, `--week`, `--prediction-week`, `--as-of-date`, `--schedule-path`, `--schedule-factors-path`, `--market-path`, `--market-line-preference`, `--current-pbp-path`, `--current-snaps-path`, `--bankroll`, `--flat-stake`, `--minimum-spread-difference`, `--maximum-market-disagreement`, `--quarter-kelly-multiplier`, `--max-kelly-bet-fraction`, `--default-spread-price`, `--allow-week1-stakes`, `--include-week18`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `build_parser` | 67 |  |
| `parse_args` | 114 |  |
| `now_string` | 163 |  |
| `table_exists` | 167 |  |
| `read_table` | 173 |  |
| `one_value` | 182 |  |
| `validate_form_contract` | 193 | Validate only the live inputs used by the replacement model. |
| `validate_bundle` | 253 |  |
| `load_historical_cache` | 290 |  |
| `load_current_raw` | 313 |  |
| `build_process_ratings` | 360 |  |
| `build_personnel` | 404 |  |
| `build_live_matchup_features` | 507 |  |
| `prepare_projection_frame` | 607 |  |
| `ensure_feature_contract` | 642 |  |
| `feature_hash` | 655 |  |
| `generate_independent_projections` | 660 |  |
| `projection_hash` | 732 |  |
| `normal_cdf` | 745 |  |
| `apply_consensus` | 749 |  |
| `validate_output` | 965 |  |
| `print_report` | 1010 |  |
| `build_execution_csv` | 1059 | Return the concise, execution-facing weekly CSV. |
| `write_execution_csv` | 1210 |  |
| `main` | 1223 |  |

</details>

## predict_nfl_weekly_power_spreads_2026.py

Legacy structural/form/HFA spread engine and shared utility module. Adds bounded schedule/rest/manual context, then market comparisons. Retained for historical replay and imports; do not confuse it with the active learned predictor.

[Open source](../predict_nfl_weekly_power_spreads_2026.py)

<details>
<summary>Source design notes</summary>

```text
Build canonical weekly 2026 NFL market-independent spread projections.

Independent projection
----------------------
    projected_home_margin
    =
    home live neutral-field form rating
    - away live neutral-field form rating
    + venue-adjusted historical home-field advantage
    + bounded rest adjustment
    + bounded travel/schedule adjustment
    + explicit manual availability adjustment

Market information is attached only after the independent projection is frozen.
It is never used to create the projected spread.

The manual availability layer is intentionally explicit. It supports temporary
QB/injury information without silently rebuilding or double-counting the
structural ratings. The authoritative structural QB remains embedded in the
live form rating.

Required input
--------------
SQLite:
    nfl_2026_form_ratings

Preferred supporting inputs
---------------------------
SQLite:
    nfl_power_ratings_2026
    nfl_power_rating_calibration_season_audit
    nfl_weekly_manual_adjustments_2026

Schedule sources, in priority order:
    --schedule-path
    recognized SQLite schedule tables
    recognized project CSV/XLSX files
    nflreadpy.load_schedules([2026])

Optional schedule-factor source:
    --schedule-factors-path
    SQLite table nfl_schedule_factors_2026
    outputs/nfl_schedule_factors_2026.csv
    project-root/nfl_schedule_factors_2026.csv

Optional market source:
    --market-path

Outputs
-------
SQLite and CSV:
    nfl_weekly_power_spread_predictions_2026
    nfl_weekly_power_spread_prediction_history_2026
    nfl_weekly_power_spread_run_audit_2026
    nfl_weekly_power_spread_run_audit_history_2026

Important interpretation
------------------------
The probability layer uses historical NFL game-noise dispersion from the market
calibration audit when available. It is a transparent provisional probability
mapping, not a claim that the new structural/form model has already been
historically calibrated. The forthcoming leakage-controlled backtest should
replace this provisional mapping with model-specific residual calibration.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4` |
| `VERSION` | `v4_structural_form_hfa_fail_closed_lineage` |
| `FORM_RATING_TABLE` | `nfl_2026_form_ratings` |
| `POWER_RATING_TABLE` | `nfl_power_ratings_2026` |
| `POWER_CALIBRATION_TABLE` | `nfl_power_rating_calibration_season_audit` |
| `MANUAL_ADJUSTMENT_TABLE` | `nfl_weekly_manual_adjustments_2026` |
| `SCHEDULE_FACTOR_TABLE` | `nfl_schedule_factors_2026` |
| `OUTPUT_TABLE` | `nfl_weekly_power_spread_predictions_2026` |
| `OUTPUT_HISTORY_TABLE` | `nfl_weekly_power_spread_prediction_history_2026` |
| `RUN_AUDIT_TABLE` | `nfl_weekly_power_spread_run_audit_2026` |
| `RUN_AUDIT_HISTORY_TABLE` | `nfl_weekly_power_spread_run_audit_history_2026` |
| `EXPECTED_FORM_BUILD_ID` | `NFL_2026_FORM_RATING_CANONICAL_V3` |
| `EXPECTED_FORM_VERSION` | `v3_current_structural_prior_asof_opponent_adjusted_form` |
| `EXPECTED_STRUCTURAL_POWER_BUILD_ID` | `NFL_POWER_RATINGS_2026_CANONICAL_V2` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--schedule-path`, `--schedule-factors-path`, `--market-path`, `--week`, `--as-of-date`, `--home-field-points`, `--market-line-preference`, `--bankroll`, `--flat-stake`, `--minimum-spread-difference`, `--maximum-market-disagreement`, `--quarter-kelly-multiplier`, `--max-kelly-bet-fraction`, `--default-spread-price`, `--include-week18`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 359 |  |
| `configure_logging` | 488 |  |
| `close_logger` | 515 |  |
| `now_string` | 528 |  |
| `normalize_team` | 532 |  |
| `first_existing` | 546 |  |
| `bool_series` | 561 |  |
| `normal_cdf` | 578 |  |
| `implied_probability_from_american` | 582 |  |
| `net_profit_per_unit` | 588 |  |
| `sanitize_american_price` | 594 |  |
| `spread_label` | 607 |  |
| `selected_market_line_label` | 622 |  |
| `independent_projection_hash` | 642 |  |
| `connect_database` | 672 |  |
| `table_exists` | 678 |  |
| `read_table` | 693 |  |
| `load_tabular_file` | 711 |  |
| `append_with_schema_evolution` | 724 |  |
| `load_live_ratings` | 781 |  |
| `stable_current_power_hash` | 857 |  |
| `one_text` | 871 |  |
| `prediction_date` | 885 |  |
| `validate_live_rating_contract` | 897 |  |
| `load_home_field` | 1075 |  |
| `load_probability_noise` | 1110 |  |
| `market_margin_from_column` | 1167 |  |
| `standardize_schedule` | 1178 |  |
| `load_external_schedule` | 1400 |  |
| `schedule_file_candidates` | 1428 |  |
| `load_primary_schedule` | 1450 |  |
| `standardize_factors` | 1526 |  |
| `load_schedule_factors` | 1613 |  |
| `merge_market_source` | 1652 |  |
| `attach_factors` | 1725 |  |
| `select_prediction_week` | 1827 |  |
| `choose_market_line` | 1857 |  |
| `ensure_manual_adjustment_table` | 1891 |  |
| `load_manual_adjustments` | 1911 |  |
| `compute_rest_adjustment` | 2002 |  |
| `compute_travel_adjustment` | 2035 |  |
| `build_predictions` | 2087 |  |
| `validate_predictions` | 2565 |  |
| `save_outputs` | 2683 |  |
| `print_report` | 2741 |  |
| `main` | 2856 |  |

</details>

## prepare_nfl_historical_reconstruction.py

Historical Stage 1: construct isolated season roster/master/schedule contexts and seed authoritative QB1 from Week 1 participation; grade only from Week 2.

[Open source](../prepare_nfl_historical_reconstruction.py)

<details>
<summary>Source design notes</summary>

```text
Prepare isolated point-in-time personnel contexts for historical NFL reconstruction.

Stage 1 only: this script creates the season-specific roster, player-master,
QB-starter, schedule, context, and readiness tables required before the rebuilt
structural model can be replayed historically.

Default cutoff policy
---------------------
- Target seasons: 2022-2025.
- Personnel snapshot: weekly roster for Week 1.
- QB1: highest Week 1 offensive QB usage, observed from completed Week 1 PBP.
- First graded week: Week 2, so the QB seed is prior information.
- Allowed player history for target season S: S-4 through S-1.
- One isolated SQLite database per season under backtests/<season>.sqlite.
- The production/source database is opened read-only and never modified.

Optional manual QB override CSV columns:
    season,team,player_id,player_name,reason
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_HISTORICAL_RECONSTRUCTION_CONTEXT_CANONICAL_V1` |
| `VERSION` | `v1_weekly_roster_week1_qb_prior_isolated_databases` |
| `CONTEXT_TABLE` | `nfl_historical_reconstruction_context` |
| `ROSTER_TABLE` | `nfl_weekly_roster_snapshot_target` |
| `MASTER_TABLE` | `nfl_player_master_target` |
| `QB_TABLE` | `nfl_projected_qb_starters_target` |
| `SCHEDULE_TABLE` | `nfl_schedule_target` |
| `READINESS_TABLE` | `nfl_historical_personnel_readiness_audit` |
| `ADVANCED_ALIAS_TABLE` | `nfl_player_advanced_stats_history_available` |
| `CROSSWALK_TABLE` | `nfl_player_crosswalk` |

**Supported named options:** `--project-root`, `--source-db`, `--target-seasons`, `--snapshot-week`, `--first-graded-week`, `--manual-qb-overrides`, `--allow-season-roster-fallback`, `--overwrite`, `--no-csv`, `--synthetic-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `now` | 85 |  |
| `parse_seasons` | 89 |  |
| `parse_args` | 96 |  |
| `frame` | 120 |  |
| `clean` | 128 |  |
| `clean_id` | 142 |  |
| `team` | 147 |  |
| `position` | 155 |  |
| `pos_group` | 163 |  |
| `clean_name` | 167 |  |
| `initial_last` | 172 |  |
| `first` | 177 |  |
| `table_exists` | 185 |  |
| `read_table` | 191 |  |
| `hash_frame` | 200 |  |
| `try_loader` | 205 |  |
| `load_weekly_rosters` | 225 |  |
| `load_season_rosters` | 238 |  |
| `load_schedules` | 251 |  |
| `load_pbp` | 264 |  |
| `standardize_rosters` | 277 |  |
| `standardize_schedule` | 323 |  |
| `standardize_pbp` | 366 |  |
| `build_master` | 393 |  |
| `infer_qbs` | 413 |  |
| `load_overrides` | 429 |  |
| `apply_overrides` | 441 |  |
| `copy_source` | 462 |  |
| `synthetic` | 478 |  |
| `build_season` | 492 |  |
| `main` | 542 |  |

</details>

## replay_nfl_historical_weekly_spreads.py

Replay the canonical legacy form/spread pair on structurally ready isolated season databases, using only prior-week form; attach market and grade after freezing predictions. Combine rows into the historical replay DB. Selected V4.1 supports a dynamic early-season calibration-history floor.

[Open source](../replay_nfl_historical_weekly_spreads.py)

<details>
<summary>Source design notes</summary>

```text
Replay the rebuilt canonical NFL model week by week for 2020-2025.

This is the decisive historical validation stage for the rewritten model.
It consumes each isolated season database created by Stages 1-5 and reuses the
approved canonical form and spread-prediction scripts without changing their
model formulas.

For target season S and prediction week W:
- the immutable preseason prior is the rebuilt S structural power table;
- form uses only completed S games through W-1;
- the process model is calibrated only on 2018 through S-1;
- the independent spread is frozen before any market field is attached;
- the historical market line is attached afterward;
- grading occurs only after prediction output has been saved;
- Week 1 is excluded because historical QB1 was established from Week 1 usage;
- Week 18 is projected but excluded from default betting summaries.

Primary outputs
---------------
Inside every <database-root>/<season>.sqlite:
- nfl_rebuilt_weekly_replay_predictions
- nfl_rebuilt_weekly_form_history
- nfl_rebuilt_weekly_replay_run_audit
- nfl_rebuilt_weekly_replay_threshold_summary
- nfl_rebuilt_weekly_replay_season_summary

Combined output database:
- <database-root>/nfl_rebuilt_historical_replay.sqlite

The production database and production scripts are never modified.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_REBUILT_HISTORICAL_WEEKLY_REPLAY_CANONICAL_V4` |
| `VERSION` | `v4_1_dynamic_form_process_history_floor` |
| `EXPECTED_FORM_BUILD_ID` | `NFL_2026_FORM_RATING_CANONICAL_V3` |
| `EXPECTED_FORM_VERSION` | `v3_current_structural_prior_asof_opponent_adjusted_form` |
| `EXPECTED_PREDICT_BUILD_ID` | `NFL_WEEKLY_POWER_SPREADS_2026_CANONICAL_V4` |
| `EXPECTED_PREDICT_VERSION` | `v4_structural_form_hfa_fail_closed_lineage` |
| `CONTEXT_TABLE` | `nfl_historical_reconstruction_context` |
| `SCHEDULE_TABLE` | `nfl_schedule_target` |
| `POWER_TABLE` | `nfl_power_ratings_target` |
| `POWER_CALIBRATION_TABLE` | `nfl_power_rating_calibration_season_audit_target` |
| `FORM_TABLE` | `nfl_form_ratings_replay_current` |
| `FORM_SNAPSHOT_TABLE` | `nfl_preseason_power_snapshot_replay` |
| `FORM_GAME_AUDIT_TABLE` | `nfl_form_game_audit_replay_current` |
| `FORM_TEAM_GAME_AUDIT_TABLE` | `nfl_form_team_game_audit_replay_current` |
| `FORM_PROCESS_MODEL_TABLE` | `nfl_form_process_model_replay` |
| `FORM_FEATURE_CACHE_TABLE` | `nfl_form_historical_feature_cache_replay` |
| `PREDICTION_TABLE` | `nfl_weekly_spread_predictions_replay_current` |
| `PREDICTION_HISTORY_TABLE` | `nfl_weekly_spread_predictions_replay_history` |
| `PREDICTION_RUN_AUDIT_TABLE` | `nfl_weekly_spread_run_audit_replay_current` |
| `PREDICTION_RUN_HISTORY_TABLE` | `nfl_weekly_spread_run_audit_replay_history` |
| `MANUAL_ADJUSTMENT_TABLE` | `nfl_weekly_manual_adjustments_replay` |
| `SCHEDULE_FACTOR_TABLE` | `nfl_schedule_factors_replay` |
| `REPLAY_PREDICTIONS_TABLE` | `nfl_rebuilt_weekly_replay_predictions` |
| `REPLAY_FORM_HISTORY_TABLE` | `nfl_rebuilt_weekly_form_history` |
| `REPLAY_RUN_AUDIT_TABLE` | `nfl_rebuilt_weekly_replay_run_audit` |
| `REPLAY_THRESHOLD_TABLE` | `nfl_rebuilt_weekly_replay_threshold_summary` |
| `REPLAY_SEASON_TABLE` | `nfl_rebuilt_weekly_replay_season_summary` |
| `COMBINED_DB_NAME` | `nfl_rebuilt_historical_replay.sqlite` |

**Supported named options:** `--project-root`, `--database-root`, `--output-root`, `--target-seasons`, `--first-week`, `--last-week`, `--thresholds`, `--maximum-market-disagreement`, `--flat-stake`, `--bankroll`, `--python`, `--preflight-only`, `--keep-runtime`, `--no-csv`, `--allow-source-mismatch`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `RuntimeScript` | 154 |  |
| `parse_seasons` | 161 |  |
| `parse_thresholds` | 168 |  |
| `parse_args` | 175 |  |
| `now_string` | 224 |  |
| `normalize_team` | 228 |  |
| `table_exists` | 233 |  |
| `read_table` | 240 |  |
| `drop_tables` | 249 |  |
| `append_schema_evolving` | 256 |  |
| `assignment_nodes` | 274 |  |
| `patch_assignments` | 285 |  |
| `verify_source` | 300 |  |
| `form_process_minimum_games` | 312 |  |
| `literal_assignment` | 324 |  |
| `validate_form_patch` | 334 |  |
| `patch_form` | 360 |  |
| `patch_predictor` | 394 |  |
| `compile_script` | 422 |  |
| `prepare_runtime_scripts` | 433 |  |
| `preflight_all` | 453 |  |
| `standardize_schedule` | 466 |  |
| `haversine_miles` | 509 |  |
| `build_schedule_factors` | 517 |  |
| `prepare_prediction_schedule` | 577 |  |
| `run_child` | 585 |  |
| `prediction_date_for_week` | 602 |  |
| `grade_predictions` | 609 |  |
| `wilson_lower` | 632 |  |
| `max_drawdown` | 643 |  |
| `summarize_thresholds` | 651 |  |
| `summarize_seasons` | 689 |  |
| `validate_week` | 715 |  |
| `build_one_season` | 733 |  |
| `save_combined` | 870 |  |
| `main` | 914 |  |

</details>

## run_nfl_circa_qb_repair.py

Deliberate rebuild chain for matchup/QB features, Circa V1 backtest and repaired ceiling model, with verification. Not the routine weekly job.

[Open source](../run_nfl_circa_qb_repair.py)

<details>
<summary>Source design notes</summary>

```text
Run the canonical QB repair, one Circa backtest, and model rebuild chain.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_CIRCA_QB_REPAIR_CANONICAL_RUNNER_V1` |
| `VERSION` | `v1_2_verification_only_windows_self_test_close` |

**Supported named options:** `--project-root`, `--python`, `--rebuild-source-cache`, `--resume-from-circa`, `--verification-only`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 32 |  |
| `TeeLog` | 61 |  |
| `run_command` | 74 |  |
| `read_one_row` | 94 |  |
| `assert_positive_qb_metadata` | 112 |  |
| `final_verification` | 137 |  |
| `main` | 180 |  |

</details>

## run_nfl_circa_weekly_2026.py

Official Circa PDF ingestion, prediction and independent weekly integrity certification. Reconciles schedule/board and audits source capture, cutoffs, QB/OL and model lineage. Stale producer expectations require alignment.

[Open source](../run_nfl_circa_weekly_2026.py)

<details>
<summary>Source design notes</summary>

```text
Pull the official 2026 Circa board, run picks, and certify data integrity.

The runner reuses the audited official-PDF discovery and OCR implementation in
``build_backtest_nfl_circa_contest_lines.py``.  It never substitutes a
sportsbook line, never accepts a non-Circa host, and does not launch the
production predictor unless every scheduled game for the requested week has a
valid reconciled contest spread.

After prediction, a fail-closed weekly certificate independently verifies the
2026 source-game capture, prior-week cutoff, form, structural projection,
authoritative QB/OL inputs, frozen model lineage, and both five-pick entries.
The certificate is saved as JSON and appended to two audit tables.  ``--audit-only``
rechecks an already-generated week without fetching a PDF or rerunning models.

Normal use requires no week number: the current regular-season week is derived
from ``identifier.sqlite::nfl_schedule_2026``.  ``--week`` remains available
for an explicit replay or preflight.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_CIRCA_WEEKLY_2026_CANONICAL_V1` |
| `VERSION` | `v1_1_fail_closed_weekly_data_integrity_certificate` |
| `SCHEDULE_TABLE` | `nfl_schedule_2026` |
| `EXPECTED_HISTORICAL_VERSION` | `v1_5_qb_identity_integrity_guard` |
| `FORM_TABLE` | `nfl_2026_form_ratings` |
| `POWER_TABLE` | `nfl_power_ratings_2026` |
| `DEPTH_TABLE` | `nfl_projected_depth_chart_2026` |
| `STRUCTURAL_TABLE` | `nfl_weekly_power_spread_predictions_2026` |
| `PREDICTOR_AUDIT_TABLE` | `nfl_circa_top5_run_audit_2026` |
| `PREDICTOR_AUDIT_HISTORY_TABLE` | `nfl_circa_top5_run_audit_history_2026` |
| `PORTFOLIO_TABLE` | `nfl_circa_final_portfolio_predictions_2026` |
| `PORTFOLIO_HISTORY_TABLE` | `nfl_circa_final_portfolio_prediction_history_2026` |
| `TEAM_GAME_TABLE` | `nfl_matchup_team_game_features` |
| `PERSONNEL_TABLE` | `nfl_matchup_weekly_personnel_features` |
| `INTEGRITY_RUN_TABLE` | `nfl_circa_weekly_integrity_runs_2026` |
| `INTEGRITY_DETAIL_TABLE` | `nfl_circa_weekly_integrity_details_2026` |
| `EXPECTED_V1_MODEL_VERSION` | `v1_5_qb_identity_integrity_guard` |
| `EXPECTED_CEILING_MODEL_VERSION` | `v3_2_1_fixed_architecture_qb_integrity_lineage` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--week`, `--historical-builder-path`, `--predictor-path`, `--tesseract-path`, `--wait-minutes`, `--poll-seconds`, `--fetch-only`, `--rebuild-ocr`, `--rebuild-live-features`, `--audit-only`, `--preflight-only`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `BoardNotPublished` | 103 | The official target-week Circa PDF is not available yet. |
| `parse_args` | 107 |  |
| `now_string` | 182 |  |
| `normalize_team` | 186 |  |
| `first_existing` | 191 |  |
| `sha256_file` | 196 |  |
| `json_value` | 207 | Convert pandas/numpy/path values into stable JSON-safe values. |
| `stable_frame_sha256` | 231 |  |
| `read_sqlite_table` | 242 |  |
| `table_exists` | 257 |  |
| `resolve_script` | 264 |  |
| `source_marker` | 286 |  |
| `load_historical_builder` | 297 |  |
| `verify_predictor` | 325 |  |
| `standardize_database_schedule` | 336 |  |
| `load_schedule` | 424 |  |
| `detect_current_week` | 447 |  |
| `historical_args` | 475 |  |
| `revision_number` | 499 |  |
| `select_latest_manifest` | 504 |  |
| `validate_complete_board` | 541 |  |
| `fetch_board_once` | 591 |  |
| `write_json_atomic` | 617 |  |
| `save_live_outputs` | 630 |  |
| `run_predictor` | 684 |  |
| `add_integrity_check` | 732 |  |
| `frame_for_season_week` | 759 |  |
| `latest_predictor_run` | 771 |  |
| `portfolio_for_run` | 816 |  |
| `parse_feature_source` | 850 |  |
| `file_hash_if_present` | 862 |  |
| `load_saved_board` | 869 |  |
| `expected_team_games` | 892 |  |
| `table_or_empty` | 905 |  |
| `audit_model_files` | 937 |  |
| `build_integrity_certificate` | 999 |  |
| `sqlite_type_for_series` | 1855 |  |
| `append_with_schema_evolution` | 1863 |  |
| `persist_integrity_certificate` | 1887 |  |
| `certify_week` | 1940 |  |
| `preflight` | 1996 |  |
| `run_self_test` | 2034 |  |
| `main` | 2430 |  |

</details>

## run_nfl_structural_refresh.py

Infrequent 14-stage rebuild of roster identity, historical player inputs and structural team ratings. Uses the current Windows root and live database; run sequentially.

[Open source](../run_nfl_structural_refresh.py)

<details>
<summary>Source design notes</summary>

```text
Run the complete canonical 2026 NFL structural-rating refresh.

CADENCE
-------
Run this script infrequently, not as the normal weekly job.

Appropriate triggers:
- material roster transactions;
- authoritative starting-quarterback changes;
- meaningful projected depth-chart changes;
- an intentional refresh of historical player inputs;
- a methodology/code change to any structural model stage;
- a full preseason rebuild.

Do NOT run merely because another NFL week finished. The weekly form runner
handles completed-game information separately and preserves the immutable
preseason snapshot.

CANONICAL SEQUENCE
------------------
1. load_nfl_rosters.py
2. build_nfl_player_master.py
3. build_nfl_player_crosswalk.py
4. nfl_player_advanced_stats.py
5. load_rbsdm_qb_ratings.py
6. load_nfl_defensive_front_metrics.py
7. load_nfl_coverage_metrics.py
8. build_nfl_defensive_metrics_summary.py
9. build_nfl_player_performance.py
10. build_nfl_projected_depth_chart.py
11. build_nfl_ol_continuity.py
12. build_nfl_team_unit_ratings.py
13. build_nfl_team_strength.py
14. build_nfl_power_ratings.py

The live player-performance experiment is intentionally excluded.

After this runner succeeds, run:
    run_nfl_weekly_form.py

That updates the live form table and records any drift between the immutable
preseason prior and the newly rebuilt current structural rating.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `RUNNER_VERSION` | `canonical_structural_refresh_v1` |
| `RUN_AUDIT_TABLE` | `nfl_structural_refresh_runner_audit_2026` |
| `STEP_AUDIT_TABLE` | `nfl_structural_refresh_step_audit_2026` |

**Supported named options:** `--start-at`, `--stop-after`, `--dry-run`, `--python`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `ExpectedTable` | 80 |  |
| `PipelineStep` | 88 |  |
| `TeeLogger` | 273 |  |
| `parse_args` | 301 |  |
| `selected_steps` | 331 |  |
| `acquire_lock` | 355 |  |
| `release_lock` | 381 |  |
| `connect_database` | 388 |  |
| `table_exists` | 397 |  |
| `validate_expected_table` | 413 |  |
| `insert_audit` | 464 |  |
| `run_step` | 500 |  |
| `validate_environment` | 583 |  |
| `validate_final_power` | 612 |  |
| `main` | 642 |  |

</details>

## run_nfl_weekly_2026_final.py

Main weekly orchestrator: refresh depth/units/power, build form, verify the frozen model, load market prices and produce learned-consensus predictions. Stops on child failure and records run/step audits.

[Open source](../run_nfl_weekly_2026_final.py)

<details>
<summary>Source design notes</summary>

```text
Run the canonical 2026 NFL weekly production pipeline.

Official weekly model
---------------------
LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS is the independent weekly spread model.

Canonical sequence
------------------
1. build_nfl_projected_depth_chart.py
   or build_nfl_projected_depth_chart_v2_3_authoritative_qbs.py
2. build_nfl_ol_continuity.py
3. build_nfl_team_unit_ratings.py
4. build_nfl_team_strength.py
5. build_nfl_power_ratings.py
6. build_nfl_2026_form_rating.py
7. build_nfl_learned_consensus_2026.py
8. load_nfl_live_market_odds.py
9. predict_nfl_weekly_learned_consensus_2026.py

The market loader runs only after the independent structural/form inputs have
been built. Its output is attached by the predictor after the fair spread is
frozen. Supply --market-path to use an existing market file, or
--skip-market-load for an intentional fair-spread-only run.

The Circa contest residual model is not run here. After Circa posts its Thursday
contest board, run predict_nfl_circa_top5_2026.py separately.

The runner:
- uses the same Python interpreter that launched it;
- resolves only canonical filenames, with the authoritative-QB depth-chart
  filename preferred when present;
- streams every child process to the console and a timestamped master log;
- stops immediately after the first failure;
- prevents overlapping runs with a lock file;
- passes only command-line options actually supported by each child script;
- validates expected SQLite outputs after each successful step;
- records run-level and step-level audits in the production SQLite database.

Typical commands
----------------
Full refresh and projection (the week is required):
    python run_nfl_weekly_2026_final.py --week 1

After the structural prior has already been refreshed and only form/projection
need to run:
    python run_nfl_weekly_2026_final.py --form-and-predict-only --week 2

Prediction only:
    python run_nfl_weekly_2026_final.py --predict-only --week 2
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_WEEKLY_2026_FINAL_RUNNER_CANONICAL_V5` |
| `VERSION` | `v6_1_explicit_unvalidated_week1_stake_override` |
| `RUN_AUDIT_TABLE` | `nfl_weekly_final_runner_audit_2026` |
| `RUN_AUDIT_HISTORY_TABLE` | `nfl_weekly_final_runner_audit_history_2026` |
| `STEP_AUDIT_TABLE` | `nfl_weekly_final_runner_step_audit_2026` |
| `STEP_AUDIT_HISTORY_TABLE` | `nfl_weekly_final_runner_step_audit_history_2026` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--week`, `--prediction-week`, `--as-of-date`, `--through-week`, `--schedule-path`, `--schedule-factors-path`, `--market-path`, `--current-pbp-path`, `--current-snaps-path`, `--skip-market-load`, `--odds-api-key-env`, `--market-bookmaker-priority`, `--market-max-age-minutes`, `--market-require-primary-bookmaker`, `--market-allow-partial-week`, `--market-line-preference`, `--bankroll`, `--flat-stake`, `--minimum-spread-difference`, `--maximum-market-disagreement`, `--quarter-kelly-multiplier`, `--max-kelly-bet-fraction`, `--default-spread-price`, `--home-field-points`, `--allow-week1-stakes`, `--include-week18`, `--no-csv`, `--prepare-process-model`, `--rebuild-process-cache`, `--form-and-predict-only`, `--predict-only`, `--start-at`, `--stop-after`, `--dry-run`, `--force-unlock`, `--self-test`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `TableExpectation` | 91 |  |
| `PipelineStep` | 101 |  |
| `TeeLogger` | 232 |  |
| `parse_args` | 256 |  |
| `table_exists` | 394 |  |
| `table_columns` | 405 |  |
| `first_existing` | 416 |  |
| `resolve_script` | 427 |  |
| `supported_help_options` | 437 |  |
| `add_if_supported` | 461 | Append a supported CLI option without emitting empty value options. |
| `default_market_path` | 485 | Return the deterministic live-market CSV used by the predictor. |
| `effective_market_path` | 500 | Resolve explicit, automatic, or intentionally absent market input. |
| `build_child_command` | 509 |  |
| `selected_steps` | 700 |  |
| `stream_process` | 743 |  |
| `validate_expectation` | 765 |  |
| `write_audits` | 846 |  |
| `acquire_lock` | 885 |  |
| `run_self_test` | 907 |  |
| `main` | 996 |  |

</details>

## run_nfl_weekly_form.py

Alternate wrapper around form, frozen bundle and learned prediction with postflight lineage checks. Its expected predictor version differs from the supplied predictor; see Known gaps.

[Open source](../run_nfl_weekly_form.py)

<details>
<summary>Source design notes</summary>

```text
Build and verify the canonical learned-consensus weekly projection.

This is the orchestration entry point between the point-in-time NFL inputs and
the Circa top-five workflow. It rebuilds form, verifies the frozen 2026 learned
model, generates market-independent fair lines, and checks exact row lineage.
```

</details>

<details>
<summary>Tables, version markers, command options and functions</summary>

| Constant | Value |
|---|---|
| `BUILD_ID` | `NFL_WEEKLY_FORM_RUNNER_CANONICAL_V2` |
| `VERSION` | `v2_form_v3_frozen_learned_consensus_fail_closed_postflight` |
| `FORM_TABLE` | `nfl_2026_form_ratings` |
| `PREDICTION_TABLE` | `nfl_weekly_power_spread_predictions_2026` |
| `RUN_AUDIT_TABLE` | `nfl_weekly_power_spread_run_audit_2026` |
| `EXPECTED_FORM_BUILD_ID` | `NFL_2026_FORM_RATING_CANONICAL_V3` |
| `EXPECTED_FORM_VERSION` | `v3_current_structural_prior_asof_opponent_adjusted_form` |
| `EXPECTED_SPREAD_BUILD_ID` | `NFL_WEEKLY_LEARNED_CONSENSUS_2026_CANONICAL_V1` |
| `EXPECTED_SPREAD_VERSION` | `v1_0_frozen_learned_weights_point_in_time_consensus` |
| `EXPECTED_BUNDLE_BUILD_ID` | `NFL_LEARNED_CONSENSUS_2026_CANONICAL_V1` |
| `EXPECTED_BUNDLE_VERSION` | `v1_0_2020_2025_frozen_market_free_consensus` |

**Supported named options:** `--project-root`, `--db-path`, `--database`, `--week`, `--as-of-date`, `--python`, `--verification-only`, `--no-csv`.

Function/class index (line numbers refer to this archived source). Undocumented helper names are navigation pointers, not inferred behavior.

| Name | Line | Source description |
|---|---:|---|
| `parse_args` | 54 |  |
| `now_string` | 102 |  |
| `run_command` | 106 |  |
| `read_table` | 135 |  |
| `unique_text` | 150 |  |
| `unique_integer` | 165 |  |
| `exact_date` | 176 |  |
| `postflight` | 193 |  |
| `main` | 323 |  |

</details>

## Other files

| File | Purpose |
|---|---|
| `models/nfl_learned_consensus_2026.joblib` + metadata | Frozen learned models, structural snapshots, gate and probability calibration. |
| `models/nfl_circa_contest_model_v1.joblib` + metadata | Circa V1 fitted model and validation/lineage metadata. |
| `models/nfl_circa_contest_season_phase_v3_2.joblib` + metadata | Fixed early/late Circa architecture, linked to its source V1. |
| `models/nfl_matchup_residual_model_v2.joblib` + metadata | Market-residual research/deployment bundle; do not confuse with market-free learned model. |
| `nfl_schedule_2026.csv` | Uploaded schedule snapshot, UTF-8; SQLite import may be required. |
| `requirements.txt`, `environment.yml` | Proposed dependency setup; replace with an actual working environment export when available. |
| `verify_archive.py` | Read-only source/data/model checksum and Python syntax check. |
