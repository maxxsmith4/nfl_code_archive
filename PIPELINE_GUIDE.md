# NFL pipeline guide

Updated September 22, 2026 for automatic roster/availability refresh, the supplemental roster projection, source recovery, Circa certificate corrections, and distinct OL starter selection. Paths and commands refer to the existing Windows installation.

## 1. What the models do

The regular weekly model estimates a fair spread independently of the market. The Circa system uses official contest lines and a market-residual model to produce two five-pick cards. Track their backtest results separately from each other and from the new supplemental roster scenario.

### Weekly learned consensus: the preserved baseline

`build_nfl_learned_consensus_2026.py` packages positive Ridge models using eight position units and 26 starter slots, plus a nonlinear process/personnel model. Training uses reconstructed 2020–2025 games, Weeks 2–17. Hyperparameters are selected using prior-season scoring-margin error, not ATS results.

`predict_nfl_weekly_learned_consensus_2026.py` calculates all three projections before attaching prices. The baseline fair margin is the mean of the unit and nonlinear projections; the slot model supplies another agreement check. The frozen gate requires at least a 2-point edge, no more than 3.5 points of range across projections, and all three models on the same side. Week 1 stakes require `--allow-week1-stakes` and remain labeled outside the validated scope. Week 18 is also outside the default validation scope.

The bundle contains frozen 2026 unit and starter snapshots. A structural refresh does **not** replace those snapshots. Weekly completed-game process/personnel and form inputs can still change the baseline normally. Preserving the fitted model means preserving its formulas, coefficients, calibration, gates, and staking rules; it does not mean the baseline number must stay constant as new weekly data arrives. Do not force a model rebuild for an injury update.

### Supplemental current-roster projection

The predictor now adds one combined scenario for current starters and availability across positions. It rebuilds current unit/starter inputs, prepares them consistently with the frozen model, and scores copies of the feature frames with the same fitted unit, slot, and nonlinear estimators. It retains weekly form/process information. The adjusted margin remains the mean of the unit and nonlinear scenario projections.

This is not a fixed point deduction for an injured player. It compares the current-roster scenario with the baseline. It can affect more games than the earlier QB-only review because all supported positions are refreshed.

**The scenario has not been historically backtested.** The baseline `final_model_spread`, probabilities, agreement, betting eligibility, and stakes retain their original meaning. No adjusted cover probability or automatic adjusted stake is produced. The original baseline columns are preserved; the supplemental fields are described below.

A multiweek absence is reevaluated from current inputs on each run. The code does not accumulate last week's adjustment or deduct another fixed penalty every week. Completed-game form may already reflect games without the player, however, so this design does not prove that all overlap between personnel and form effects has been removed. Keep the adjusted line as a separate research/review indicator. A comparable historical test would require point-in-time availability/depth evidence, or a separately disclosed reconstruction with weaker assumptions.

### Circa two-entry portfolio

The board loader obtains official Circa PDFs, renders/OCRs them, matches teams to the schedule, and checks complete weeks and opposite spread signs. Live sportsbook lines are not interchangeable with the contest board.

The policies are `TB_V1_P37_D050` and `CEILING_LATE_HOME_FAVORITE_7P5_VETO`. The ceiling bundle uses V1 for Weeks 1–9 and its fixed late-season Ridge architecture for Weeks 10–18. Its lineage is tied to the matching V1 model.

The current predictor accepts the baseline `LEARNED_STRUCTURAL_NONLINEAR_CONSENSUS` projection for confidence ordering and manual-review context. The older documentation's legacy-source mismatch is resolved in this version. `STRUCTURAL_USED_TO_CHANGE_MODEL_CARD = 0`: this context does not change the selected model cards. The supplemental `roster_adjusted_*` line is not automatically substituted into Circa; review it separately when assessing flagged games.

## 2. What to run each week

1. Run the full weekly pipeline for the target week once required prior-week completed-game inputs are available. This refreshes current availability and writes both baseline and supplemental projections.
2. Inspect expected QBs, roster review flags, source times, and any missing adjusted lines. A successful download can still reflect a provider that has not yet published a reported injury.
3. Run the Circa weekly pipeline after the official board is available and the learned forecast is current. Review the cards and the integrity certificate.
4. Rerun the appropriate pipeline when information changes. `--audit-only` verifies saved Circa outputs; it does not refresh data or generate new picks.

A full structural refresh is a deeper player-history/talent rebuild. Run it periodically or when underlying ratings need updating, then run the weekly prediction. A separate full structural rebuild is not required solely to fetch an injury: the full weekly runner already refreshes availability.

### Weekly stage order and shorter modes

The full weekly runner executes these 11 stages:

| Order | Stage key | Script |
|---|---|---|
| 1 | `rosters` | `load_nfl_rosters.py` |
| 2 | `player_master` | `build_nfl_player_master.py` |
| 3 | `depth_chart` | `build_nfl_projected_depth_chart.py` |
| 4 | `ol_continuity` | `build_nfl_ol_continuity.py` |
| 5 | `team_units` | `build_nfl_team_unit_ratings.py` |
| 6 | `team_strength` | `build_nfl_team_strength.py` |
| 7 | `power_ratings` | `build_nfl_power_ratings.py` |
| 8 | `form_ratings` | `build_nfl_2026_form_rating.py` |
| 9 | `learned_model` | `build_nfl_learned_consensus_2026.py` |
| 10 | `live_market` | `load_nfl_live_market_odds.py` |
| 11 | `weekly_predictions` | `predict_nfl_weekly_learned_consensus_2026.py` |

The learned-model stage verifies/reuses the existing valid bundle. Missing model/metadata assets follow the original required build path; incompatible lineage or a hash mismatch in existing assets stops validation rather than silently rebuilding. Weekly refresh is not an instruction to retrain the model.

| Mode | Behavior |
|---|---|
| Full run | All stages above, subject to explicit market options and the limited source-outage recovery described below. |
| `--form-and-predict-only` | Refreshes rosters, player master, depth, and units; then form, model verification, market, and prediction. Skips OL continuity, team strength, and power rebuilds. |
| `--predict-only` | Refreshes rosters, player master, depth, and units; then model verification, market, and prediction. Reuses existing form, which must match both the target-week cutoff and prediction as-of date. |
| `--start-at STAGE` | Resumes at that stage within the selected mode. Earlier refreshes can be skipped, so this is not proof that those inputs are current. |
| `--market-path PATH` | Uses an existing market file and skips the live market loader. |
| `--skip-market-load` | Skips fetching live odds and passes no explicit market file. Existing schedule prices may remain; this flag alone does not guarantee blank prices or disable stakes. |

For prediction Week N, the required form cutoff is Week N−1. Week 3 therefore requires form through Week 2. Form must also have the same normalized as-of date as the prediction; use a full or form-and-predict run when that date needs refreshing. The prior-week requirement is not relaxed by allowing partial market coverage.

The form builder uses completed games before the target week. Its legacy live rating blends current structural power with opponent-adjusted season/recent form: 70% process and 30% results, then 70% season and 30% recent form. Its current-season contribution is `games / (games + 5)`, capped at 75%. After one game that base-form contribution is about 16.7%; this is **not** a claim that the final learned-consensus line is 16.7% Week 1 data. The learned predictor uses its own fitted combination and process/personnel inputs.

The market loader runs after independent upstream inputs are built. The predictor attaches prices after calculating the fair projection, then writes edges, eligibility, and flat/Kelly fields. When an explicit market file is used, it is authoritative for that week's execution prices: omitted games do not regain older prices from the schedule. The forecast still retains scheduled games. A stake column alone does not establish that a row passed its execution gate.

### Copyable PowerShell commands

Each command below is one line. Change `3` to the target prediction week.

Full weekly run:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3
```

Allow partial sportsbook market coverage:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3 --market-allow-partial-week
```

The weekly runner does **not** accept `--allow-partial-week`. Its supported flag is `--market-allow-partial-week`; this affects the market loader only, not results, snaps, form cutoffs, or Circa integrity requirements.

Availability, units, form, and prediction when existing structural power is suitable:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3 --form-and-predict-only
```

Availability, units, and prediction using existing form through Week 2 with the matching prediction as-of date:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3 --predict-only
```

Full structural refresh, followed by a weekly run to produce new lines:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_structural_refresh.py"
```

Resume a structural refresh whose stages through player performance already completed successfully:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_structural_refresh.py" --start-at depth_chart
```

Run the official Circa workflow:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_circa_weekly_2026.py" --week 3
```

Recheck an existing Circa run without regenerating its board/features/cards:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_circa_weekly_2026.py" --week 3 --audit-only
```

Normal Circa runs rebuild live matchup features by default. `--rebuild-live-features` is not an extra weekly requirement; explicit reuse modes should only be used when their saved inputs are appropriate.

## 3. How availability reaches the supplemental line

1. `load_nfl_rosters.py` refreshes current rosters, then the player-master builder reconciles canonical identities.
2. `nfl_live_depth_2026.py`, called by the depth builder, fetches ESPN ordered depth and roster feeds for all 32 teams, including published injury designations. If roster retrieval fails but fresh depth is available, the refreshed player master can supply roster membership/reserve status with explicit provenance.
3. Unavailable players, including out/IR/suspended/practice-squad players, are excluded. Questionable and doubtful players remain assumed available and receive uncertainty flags; the code does not invent probabilities of playing.
4. The builder selects the next available listed QB and uses source depth order within supported position/slot definitions. Current depth priority drives the rebuilt unit inputs. Existing talent grades, unit weights, and replacement-level logic remain in place; identified players without historical grades use replacement level and are flagged.
5. The review helper checks source age, identities, timestamps, and whether units actually consumed the refreshed depth chart before producing the scenario.

Provider IDs are preferred. A name match must be unique and must not conflict with a known provider ID. Unresolved selected QB/OL identities stop the refresh. Unresolved other starters can withhold that team's adjusted line and flag the game. An automatic feed is still limited by what the provider publishes; a news announcement does not guarantee an immediate structured-feed update.

The old `nfl_qb_review_2026.py`, `config/nfl_qb_weekly_status_2026.csv`, and frozen QB-reference workflow are superseded. They may remain on disk, but the current predictor does not read them. No manual weekly QB file or player-ID entry is required for this workflow.

### Distinct OL starters

The September 21 correction selects five distinct OL players together from each position's published candidates. It first preserves as many available rank-one starters as possible, then minimizes combined position-specific depth ranks. Equally preferred assignments, insufficient candidates, and unresolved selected identities fail rather than silently choosing a lineup. The duplicate-player validator remains active.

Resolved source cross-listings are recorded in `ol_assignment_note` in the source audit, and selected rows retain their actual position-specific ranks. These assignments are inferences from published depth/availability, not official lineup announcements. No team or player is hardcoded.

### Reading the execution CSV

The usual output remains `outputs/nfl_weekly_power_spread_predictions_2026.csv`. Its original 26 baseline columns retain their meaning. The old QB-only supplemental columns are replaced with these nine combined-roster fields:

| Field | Meaning |
|---|---|
| `away_expected_qb`, `home_expected_qb` | QBs selected from current available depth order. Interpret them alongside the source/status flags; source-outage output withholds current-QB claims. |
| `roster_adjusted_home_margin` | Supplemental projected home margin; positive favors home, negative favors away. |
| `roster_adjusted_spread` | The same projection displayed as a team and spread. A home margin of +4 means the home team is favored by 4. |
| `roster_adjustment_home_points` | Adjusted home margin minus baseline home margin; positive moves toward the home team. |
| `roster_adjustment_status` | `CURRENT_DEPTH_SCENARIO`, `NO_CHANGE`, or a review/error status. |
| `roster_review_flag` | `YES` when inputs changed, availability is uncertain, or data needs review. |
| `roster_review_reason` | Reasons such as changed personnel, a backup QB, uncertain availability, or source/identity problems. |
| `roster_source_as_of_utc` | Oldest relevant source/fetch timestamp for the matchup; not a guarantee of the provider's publication time. |

`REVIEW_ONLY`, `REVIEW_ERROR`, `SCENARIO_ERROR`, and `SOURCE_UNAVAILABLE` have blank adjusted lines. Blank means the scenario is unavailable, not a zero-point adjustment. The baseline forecast can still exist in those rows.

### Outputs to inspect

Paths are relative to the project root. SQLite also retains the applicable source, review, and run audits.

| File | Purpose |
|---|---|
| `outputs/nfl_weekly_power_spread_predictions_2026.csv` | Baseline execution output plus supplemental fields. |
| `outputs/nfl_roster_review_audit_2026.csv` | Current starters, unavailable/uncertain players, feature changes, scenario components, and source hashes. |
| `outputs/nfl_projected_depth_chart_2026.csv` | Rebuilt depth/slot assignments and source metadata. |
| `outputs/nfl_live_player_availability_2026.csv` | Source player identities and availability decisions. |
| `outputs/nfl_live_roster_source_audit_2026.csv` | Team source timestamps, provenance, cache/identity issues, and OL assignment notes. |
| `outputs/nfl_rosters_2026_refresh_status.csv` | Roster fetch/reuse result and original source age. |
| `outputs/nfl_live_roster_fetch_diagnostics_2026.json` | Endpoint-specific depth/roster retrieval diagnostics. |
| `outputs/nfl_source_diagnostics_2026.json` | Network, dependencies, and local inputs checked by the diagnostic command. |
| `outputs/circa_2026/` | Timestamped cards, comparisons, final portfolio, confidence board, and manual submission draft. |
| `outputs/audits/nfl_circa_weekly_integrity_2026_week_3.json` | Week 3 Circa certificate; the week number follows the run. |

## 4. Source failures and recovery

The roster loader uses `nflreadpy.load_rosters` with the supported `nfl_data_py.import_seasonal_rosters` fallback. The invalid `import_rosters` call was corrected. On identified transport/dependency failures, an existing roster may be reused only after validating season, all-team coverage, identities/schema, and an original import age of at most 24 hours. Reuse records `REUSED_RECENT_CACHE` and does not turn old rows into a new fetch.

Availability inputs also have a 24-hour freshness limit and cannot be later than the prediction cutoff. Eligible cached inputs are labeled. Rebuilding a player master cannot refresh the underlying roster timestamp.

If the depth stage records a genuine availability-source outage, the weekly runner can use a specific recovery path: validate existing baseline requirements, preserve structural tables, skip the current structural rebuilds, and label the run `BASELINE_ONLY_SOURCE_UNAVAILABLE`. Adjusted lines and current-QB claims are withheld, including when an older successful snapshot exists. This is a limited recovery path, not permission to ignore schema, identity, model, missing completed-game data, or other pipeline failures. The structural runner still stops on stage failure.

For source errors, first run the diagnostic without executing model stages:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3 --diagnose-sources
```

Review `outputs/nfl_source_diagnostics_2026.json`; if depth retrieval was attempted, also inspect `outputs/nfl_live_roster_fetch_diagnostics_2026.json`. The diagnostic does not certify a full pipeline run. A script cannot repair Windows DNS or guarantee that a provider will grant access.

The weekly runner supports `--roster-input-csv`, `--current-pbp-path`, `--current-snaps-path`, and `--schedule-path` for available local inputs. These are optional source-file overrides using the expected schemas, not manual starter-status files. PBP and schedule overrides also reach the form builder. A historical cache does not substitute for required current-season games, and a valid roster cache does not supply missing odds or injury reports.

The form builder checks for its existing valid process model before requesting historical build inputs. This source-recovery change does not alter its fitted coefficients, calculations, or cutoff rules.

## 5. Circa integrity certificate

Require the actual run's certificate to report `PASS` before treating its saved cards as having passed the required data checks. A `5/5` overlap between entries is a selection comparison, not evidence that the inputs passed validation. Do not bypass a failed certificate.

The corrected runner is `v1_6_prior_result_reconciliation_single_game_ol_certificate`, paired with the current predictor `v8_5_timezone_safe_freshness`. The reported Week 2 audit passed **33/33** required checks, captured **16/16** Week 1 games, and reported **100% QB identity**. This confirms that run; each later run must pass its own checks.

Two certificate checks were corrected without changing model cards or formulas:

- `prior_schedule_results_complete`: missing prior-game scores can be reconciled in an audit-only copy from the exact feature matrix recorded for the predictor run. Season, week, oriented teams, game IDs, and selected rating alpha must agree. Conflicting/ambiguous results and target-week placeholders cannot supply valid prior results. The schedule table is not overwritten.
- `ol_live_signal`: after exactly one prior game per team, the existing feature builder initializes continuity/stability to 1 and missing-core share to 0. The certificate accepts this initialization only with matching personnel/feature metadata and source evidence, including at least five distinct OL player keys with positive finite snaps for every team. This is not a blanket Week 2 exception; later histories retain the variation requirement.

`--audit-only` reads the existing board/predictor run and writes a new certificate plus records in `nfl_circa_weekly_integrity_runs_2026` and `nfl_circa_weekly_integrity_details_2026`. It does not fetch a new board, refresh injuries, rebuild features, or rerun predictions. Refresh availability and the learned forecast with the weekly pipeline, then run the normal Circa workflow for its updated board, features, and cards.

The certificate and manual submission draft do not submit a contest entry or place a wager. Check late injury news, weather/context, the official board, deadline, and both entries separately. Keep discretionary changes explicit and dated, separate from the stored model cards.

## 6. Full structural refresh

`run_nfl_structural_refresh.py` runs these 14 stages sequentially and stops on failure:

| Order | File | Logic |
|---|---|---|
| 1 | `load_nfl_rosters.py` | Current roster records and provider identities. |
| 2 | `build_nfl_player_master.py` | Canonical GSIS player population and duplicate audit. |
| 3 | `build_nfl_player_crosswalk.py` | Provider/name reconciliation and ambiguity audits. |
| 4 | `nfl_player_advanced_stats.py` | 2022–2025 player history mapped to current rosters; PBP, weekly stats, snaps, and Next Gen features where available. |
| 5 | `load_rbsdm_qb_ratings.py` | QB efficiency, sample stabilization, and historical season weights; optional local source CSVs. |
| 6 | `load_nfl_defensive_front_metrics.py` | Front-seven measures with canonical history and snap denominators. |
| 7 | `load_nfl_coverage_metrics.py` | Coverage measures with canonical history and snap denominators. |
| 8 | `build_nfl_defensive_metrics_summary.py` | Available front/coverage components for current defenders. |
| 9 | `build_nfl_player_performance.py` | Position-relative talent grades, a single confidence shrink, and PFF blocking grades as OL talent. |
| 10 | `build_nfl_projected_depth_chart.py` | Automatic source depth/availability across positions and distinct OL starter assignment. |
| 11 | `build_nfl_ol_continuity.py` | Retained OL teammates/combinations; missing history is neutral with zero confidence. |
| 12 | `build_nfl_team_unit_ratings.py` | Aggregate projected players into units using current depth priority. |
| 13 | `build_nfl_team_strength.py` | Combine units, apply OL continuity once, and avoid adding QB strength twice. |
| 14 | `build_nfl_power_ratings.py` | Neutral-field point ratings and calibration audits. |

History follows a player's canonical identity onto the current team. Talent, confidence, availability, and starter assignment remain separate concepts.

The base unit builder weights offense as QB 36%, RB 10%, WR/TE 27%, OL 27%; defense as DL/EDGE 42%, LB 25%, DB 33%; and overall strength as offense 52%, defense 44%, special teams 4%. These upstream choices are not the learned consensus model's fitted coefficients. A structural refresh rebuilds inputs without replacing the frozen learned bundle or its preseason snapshots.

## 7. Recent GitHub file updates

Add or replace these **10 canonical Python files at the repository root**. This is the recent change set, not the full dependency inventory. Preserve the other scripts, model bundles, databases, and required source inputs.

| File | Recent change |
|---|---|
| `load_nfl_rosters.py` | Correct seasonal-roster fallback, validated recent-cache recovery, and refresh audit. |
| `build_nfl_projected_depth_chart.py` | Automatic current availability/depth integration; latest version `v5_4_unique_ol_starter_assignment`. |
| `build_nfl_team_unit_ratings.py` | Consume current source depth priority with existing rating/replacement calculations. |
| `build_nfl_2026_form_rating.py` | Reuse the valid process model before historical downloads; local current-input support. |
| `predict_nfl_weekly_learned_consensus_2026.py` | Append the combined roster scenario, expected QBs, review flags, and audits; respect explicit market-file coverage without reviving older schedule prices. |
| `run_nfl_weekly_2026_final.py` | Automatic roster/master refresh, source recovery, diagnostics, and source overrides; `v6_5_roster_dns_recovery_and_local_source_inputs`. |
| `nfl_live_depth_2026.py` | New shared source/availability helper; latest version `v1_3_unique_ol_starter_assignment`. |
| `nfl_roster_review_2026.py` | New supplemental scenario and review/audit helper. |
| `diagnose_nfl_sources_2026.py` | New read-only source/dependency/local-input diagnostic. |
| `run_nfl_circa_weekly_2026.py` | Prior-result reconciliation and evidence-based one-game OL certificate correction; version `v1_6_prior_result_reconciliation_single_game_ol_certificate`. |

Use the latest two OL files, not their earlier September 17 copies. The structural runner itself and the existing Circa predictor did not require replacement for these specific fixes. The learned predictor deliberately retains its baseline version contract; its supplemental output marker is `EXECUTION_CSV_VERSION = v1_7_automatic_roster_review`.

Replace the root `README.md` and `docs/PIPELINE_GUIDE.md` with this documentation update. Do not add numbered download copies as separate production scripts. The other archive documents/manifests were not updated here and may retain superseded compatibility notes or checksums. Reconcile those inventories separately; a manifest mismatch after an intentional update is not an instruction to restore the older production file.

## 8. Historical research and recovery limits

The historical/research order remains:

1. `prepare_nfl_historical_reconstruction.py`: isolated season rosters, schedule, player master, and Week 1 QB seed.
2. `backfill_nfl_historical_advanced_stats.py`: legal rolling four-year player windows.
3. `build_nfl_historical_qb_defense.py`: canonical QB/defense builders using isolated databases/tables.
4. `build_nfl_historical_player_performance_depth.py`: prior-only player/OL history and frozen Week 1 QB/OL personnel.
5. **Missing wrapper:** isolated OL continuity, units, team strength, power, and structural-readiness marking.
6. `replay_nfl_historical_weekly_spreads.py`: legacy structural/form replay and combined `nfl_rebuilt_historical_replay.sqlite` output.
7. `audit_nfl_rebuilt_historical_replay.py`: independent results/leakage/duplicate audit without modifying source databases.

The historical Week 1 seed uses observed Week 1 usage, so graded replay begins at Week 2. The selected replay defaults to 2023–2025 and Weeks 2–17. A full 2020–2025 reconstruction requires explicitly consistent target seasons and prior-only inputs across stages; coverage cannot be inferred from a filename.

The matchup-residual builder separately creates its research matrix/database and cache. Official Circa line reconstruction produces the contest database and V1 bundle; the V3.2 builder creates the matching late-season bundle. Learned structural-weight research precedes nonlinear-consensus research. The freeze builder consumes those results, historical replay, matchup data, and current preseason structural inputs.

These repairs did not retrain the frozen models or retroactively validate the availability scenario. Do not run research against the live database as an improvised substitute for the missing wrapper. Preserve the original fitted bundles before an intentional rebuild.

GitHub code/model backups do not contain the entire operating state. Keep separate production SQLite and local-input backups. The [recovery guide](DISASTER_RECOVERY.md) and [known gaps](KNOWN_GAPS.md) retain useful archive limitations, although their original weekly/Circa mismatch notes must be read alongside the current behavior documented here. Full recovery from a clean computer remains unverified.
