# Union Avenue Analytics — NFL Code Archive

An organized backup of the supplied 2026 NFL pipeline: weekly learned-consensus spreads, two Circa contest entries, structural ratings, and historical research. Start here when running the model, looking up its logic, or restoring a computer.

**Archive status: source and model backup; full recovery is not yet verified.** All 14 scripts named by the structural-refresh runner are included. Historical Stage 5, databases, and local source data remain external. The supplied weekly/Circa runners also have version mismatches. Read [Known gaps](docs/KNOWN_GAPS.md) before running this collection as a replacement installation.

## What is included

- 36 Python scripts under their canonical filenames, at the repository root.
- Four trained model files and four companion metadata files in `models/`.
- Your 272-game `nfl_schedule_2026.csv`, converted to UTF-8.
- Plain-language [pipeline guide](docs/PIPELINE_GUIDE.md), [file reference](docs/FILE_REFERENCE.md), [recovery guide](docs/DISASTER_RECOVERY.md), and [source inventory](docs/SOURCE_MANIFEST.json).
- A proposed Python environment and a read-only archive integrity checker.

## Which script do I use?

| Task | Entry point | When |
|---|---|---|
| Weekly learned-consensus spreads and wagering output | `run_nfl_weekly_2026_final.py` | Each prediction week; refresh again when inputs or prices change |
| Official Circa board and two contest entries | `run_nfl_circa_weekly_2026.py` | After the official weekly board is available; version alignment remains unresolved |
| Full roster/player/structural refresh | `run_nfl_structural_refresh.py` | Material roster/QB/OL changes or deliberate preseason/input refresh |
| Alternate form/learned-consensus wrapper | `run_nfl_weekly_form.py` | Included for reference; its version check is stale |
| Historical structural replay | `replay_nfl_historical_weekly_spreads.py` | Research only, after isolated season databases pass structural readiness |
| QB-repair and Circa model rebuild | `run_nfl_circa_qb_repair.py` | Deliberate research/rebuild; not the normal weekly job |

## First steps

1. Upload the **contents** of this extracted folder to your `nfl_code_archive` repository, preserving `models/` and `docs/`.
2. Read [Known gaps](docs/KNOWN_GAPS.md) and [Disaster recovery](docs/DISASTER_RECOVERY.md).
3. Keep a separate backup of your SQLite databases and local inputs. GitHub code alone cannot restore the full current operating state.
4. Check the downloaded archive with `python verify_archive.py`. This does not run the pipeline or load models.

The current machine layout is:

```text
Project: C:\Users\maxxs\Downloads\Football Files\nfl_model
Python:  C:\Users\maxxs\anaconda3\python.exe
DB:      C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite
```

## Version policy

Keep one canonical filename per stage. Replace that file when a change is approved; Git history preserves earlier versions. The newer replay was selected by its internal build/version and compatible dependencies, not its download suffix. See [Archive changes](docs/ARCHIVE_CHANGES.md).

This package does not change model formulas, gates, coefficients, or staking. Only obsolete project-root strings and the schedule encoding were normalized. Known compatibility issues are documented rather than hidden by weakening validation.
