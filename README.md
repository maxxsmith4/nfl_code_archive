# Union Avenue Analytics — NFL Code Archive

An organized backup of the 2026 NFL pipeline: weekly learned-consensus spreads, two Circa contest entries, structural ratings, and historical research.

**Documentation updated September 22, 2026.** The current workflow refreshes player availability automatically and adds a separate roster-adjusted projection. The fitted learned-consensus models, baseline prediction rules, betting gates, and staking rules remain unchanged. The updated Circa certificate passed all 33 required checks in the reported Week 2 audit.

**Recovery status:** this remains a source/model backup, not a verified complete machine restore. Production databases, local inputs, and the historical Stage 5 wrapper remain outside this archive. Keep separate backups and consult the [recovery guide](docs/DISASTER_RECOVERY.md).

## Which script do I use?

| Task | Entry point | When |
|---|---|---|
| Refresh availability and produce weekly learned-consensus spreads | `run_nfl_weekly_2026_final.py` | Each prediction week and again when availability or prices change |
| Official Circa board, two contest cards, and integrity certificate | `run_nfl_circa_weekly_2026.py` | After the weekly learned forecast is current and the official board is available |
| Full player-history and structural rebuild | `run_nfl_structural_refresh.py` | Periodic deeper refresh or material changes to underlying player ratings; it does not produce the final weekly prediction |
| Alternate form wrapper | `run_nfl_weekly_form.py` | Reference only; its older version check has not been updated by these changes |
| Historical structural replay | `replay_nfl_historical_weekly_spreads.py` | Research only, using isolated season databases |
| QB-repair/Circa model rebuild | `run_nfl_circa_qb_repair.py` | Deliberate research, not the normal weekly workflow |

The normal weekly run now includes rosters, player identities, depth charts, availability, and current unit ratings. You do **not** need a separate full structural refresh merely to capture a newly published injury. A structural refresh rebuilds the broader player-rating stack; run the weekly pipeline afterward to generate updated lines.

## Normal weekly commands

Copy each command as one PowerShell line. Change `3` to the prediction week.

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_weekly_2026_final.py" --week 3
```

Then, when the official Circa board is available:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -u "C:\Users\maxxs\Downloads\Football Files\nfl_model\run_nfl_circa_weekly_2026.py" --week 3
```

For partial sportsbook coverage, the weekly runner accepts `--market-allow-partial-week`. It does not accept `--allow-partial-week`, and partial market coverage does not waive completed-game data requirements. See the [pipeline guide](docs/PIPELINE_GUIDE.md) for refresh modes, structural commands, diagnostics, outputs, and audit-only verification.

## Availability and the adjusted line

The automatic depth helper combines published depth order with injury/roster status across positions. It excludes unavailable players, flags uncertain availability, and resolves five distinct offensive-line starters. The September 22 correction also reconciles conflicting backup depth entries when a unique ESPN roster and fresh exact-ID canonical master agree on the player's team; ambiguous starter conflicts still stop the refresh. No manual QB-status CSV is required; the previous `config/nfl_qb_weekly_status_2026.csv` is no longer read by this workflow.

`outputs/nfl_weekly_power_spread_predictions_2026.csv` retains the baseline line and adds `roster_adjusted_home_margin`, `roster_adjusted_spread`, expected QBs, source timestamps, and review flags. The supplemental scenario scores current unit/starter inputs with the existing fitted models. It does not apply a fixed injury-point penalty or retrain the models.

**The adjusted line is a review indicator and has not been historically backtested.** It does not replace the baseline line, probabilities, betting decisions, or stakes. Circa continues to use the baseline learned-consensus line for confidence/review context; the supplemental line does not automatically change either contest card. Inspect blank adjusted lines and review flags before interpreting the output as a current-availability forecast.

## Repository contents and maintenance

Canonical Python filenames belong at the repository root. Keep the four trained model files and their four companion metadata files in `models/`, the UTF-8 `nfl_schedule_2026.csv`, and the documentation in `docs/`.

The [pipeline guide](docs/PIPELINE_GUIDE.md) lists the **10 Python files added or updated in the recent fixes**, including the OL and September 22 roster-conflict corrections. Replace canonical files rather than committing numbered download copies. These changes require no replacement model bundle or routine retraining.

The [file reference](docs/FILE_REFERENCE.md), [known gaps](docs/KNOWN_GAPS.md), [archive changes](docs/ARCHIVE_CHANGES.md), and [source inventory](docs/SOURCE_MANIFEST.json) describe the original archive and may still contain older version expectations. This README and the updated pipeline guide document the current weekly/Circa behavior; the other files were not revised in this documentation update. `python verify_archive.py` remains an archive check, not a pipeline certification. Its original manifest may flag intentionally replaced files until that inventory is updated.

The operating paths are:

```text
Project: C:\Users\maxxs\Downloads\Football Files\nfl_model
Python:  C:\Users\maxxs\anaconda3\python.exe
DB:      C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite
```

Keep GitHub source/model backups and separate database/input backups. A successful weekly certificate verifies that saved run's required data checks; it does not establish a complete cold restore or validate the new supplemental projection's historical performance.
