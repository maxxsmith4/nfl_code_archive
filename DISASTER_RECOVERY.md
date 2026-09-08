# Recovering the NFL pipeline

## What to back up now

| Item | Why it matters | Included here? |
|---|---|---|
| Canonical Python source and model metadata | Logic and version contracts | Yes, supplied set |
| Four joblib models | Frozen fitted parameters and embedded snapshots | Yes |
| `identifier.sqlite` | Current tables, authoritative/manual starters, run state and history; shared with other sports | No |
| Entire `backtests/` data directory | Isolated seasons, replay, matchup, Circa, learned/consensus research DBs and caches | No |
| `inputs/`, local PFF exports, local corrections | Data not guaranteed reproducible from an API | No |
| Official Circa PDFs, OCR cache and manual reconciliations | Reproduce exact contest lines and corrections | No |
| Outputs, logs, execution ledgers, scheduled-task settings | Operational history and proof of what ran/was placed | No |
| Working Python environment export | Exact installed versions | No; proposed requirements supplied |
| Odds API credential | Market access | No; restore privately |

Keep private/raw datasets and secrets in a separate backed-up location. An empty ignored directory is not a backup. Because `identifier.sqlite` is shared, restoring an old NFL copy can also roll back other sports; restore a copy first and coordinate any replacement.

## Capture a consistent SQLite backup

Use SQLite's backup API rather than copying only the main file while a WAL database may be active. Change the destination to a backed-up drive/folder:

```powershell
@'
from pathlib import Path
from datetime import datetime
import sqlite3
source = Path(r"C:\Users\maxxs\DataGripProjects\NFL\identifier.sqlite")
folder = Path(r"C:\Users\maxxs\Documents\Model_Backups")
folder.mkdir(parents=True, exist_ok=True)
dest = folder / ("identifier_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".sqlite")
with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
    with sqlite3.connect(dest) as dst:
        src.backup(dst)
        result = dst.execute("PRAGMA integrity_check").fetchall()
        if result != [("ok",)]:
            raise RuntimeError(result)
print(dest)
'@ | & "C:\Users\maxxs\anaconda3\python.exe"
```

Use the same method for each research database, or copy the complete directory only when all writers are stopped and database connections are closed. Keep at least one backup off the computer.

Capture the current environment from the working computer:

```powershell
& "C:\Users\maxxs\anaconda3\python.exe" -m pip freeze > `
  "C:\Users\maxxs\Downloads\Football Files\nfl_model\requirements-working.txt"
conda env export > "C:\Users\maxxs\Downloads\Football Files\nfl_model\environment-working.yml"
```

Review those exports for private package URLs/local paths before committing. The conda export must be made from the environment actually used by the runner.

## Restore in order

1. Extract the repository to `C:\Users\maxxs\Downloads\Football Files\nfl_model`. Keep scripts at the root and model files in `models/`.
2. Restore a consistent `identifier.sqlite` and the complete research/cache data tree to their original locations. Preserve authoritative QB/OL tables and manual changes.
3. Restore the actual working Python environment if captured. Otherwise use `environment.yml` as a starting point in an isolated environment and validate imports/model loading. Do not upgrade scikit-learn just because a newer version exists.
4. Install/restore Tesseract for historical/weekly Circa PDF OCR and make it discoverable or supply the scripts' Tesseract path option.
5. Restore licensed PFF exports and optional `inputs/rbsdm_qb_<season>.csv` files to the locations expected by the scripts. Production PFF history needs 2022–2025; historical replay requires the relevant earlier prior-only seasons as well.
6. Restore the Odds API key outside Git. Restore Task Scheduler settings only after manual validation.
7. Run `python verify_archive.py` from the extracted repository. It checks source hashes and Python syntax without API calls or database writes.
8. Check each restored database with `PRAGMA integrity_check` and inspect expected tables, last successful runs, dates, and week cutoffs.
9. Resolve the explicit issues in [Known gaps](KNOWN_GAPS.md). Do not remove version checks to force a run through.
10. Validate the chosen week's schedule, 32-team ratings, authoritative starter coverage, prior-week feature cutoff, model lineage, predictions and outputs. Compare to saved outputs from a known successful run before resuming execution.

If restoring to another username/drive, update all hard-coded paths consistently. The structural runner has no project-root/DB override. Other scripts expose different options; consult each file's CLI options in the file reference.

## Schedule restoration

The included CSV is a reference snapshot; it does not automatically populate SQLite. If the schedule table is absent, import the CSV through DataGrip as `nfl_schedule_2026`, preserving column names and sensible numeric/date types. Check 272 rows, Weeks 1–18 and unique matchups. Compare against any more recent operational schedule before replacing existing data.

## GitHub updates

Upload the extracted folder's contents, not the ZIP as a single repository file. Preserve `models/` and `docs/`. Replace scripts at the same canonical paths and use a descriptive commit message. Check that all four model/metadata pairs and the expected files appear after upload.

For future code changes, record the reason, affected stage and validation result. When replacing a model, retain its matching metadata and hash; do not mix generations. Regenerate the integrity inventory after intentional changes—the shipped inventory describes this archive snapshot only.
