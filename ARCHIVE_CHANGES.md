# Archive changes and validation

Archive identifier: `UAA_NFL_CODE_ARCHIVE_20260908_A`.

## Selected sources

- `replay_nfl_historical_weekly_spreads(5).py` becomes `replay_nfl_historical_weekly_spreads.py`: build `NFL_REBUILT_HISTORICAL_WEEKLY_REPLAY_CANONICAL_V4`, version `v4_1_dynamic_form_process_history_floor`. Its required form V3 and legacy spread V4 match the supplied files.
- The `(6)` replay is excluded. It is V1, expects form/spread V2 and references the obsolete root. A larger download suffix did not indicate a newer implementation.
- The latest uploaded `audit_nfl_rebuilt_historical_replay(3).py` is retained; only one canonical audit file is included.
- `nfl_player_advanced_stats(7).py` has the exact `NFL_ADVANCED_STATS_2026_V3_2_FINAL` / `v3_2_final_snap_participation_identity` contract expected by the historical wrapper.
- All other selected filenames are normalized by removing download counters/timestamps. Source-to-archive names and SHA-256 hashes are recorded in `SOURCE_MANIFEST.json`.

## Content changes

Only three supplied Python files had project-root text changed from `2026_nfl_files` to `nfl_model`: historical context preparation, advanced-stats backfill and historical replay audit. The schedule was transcoded from Windows-1252 to UTF-8. Python text was written as UTF-8.

No formulas, model contracts, selections, stakes or frozen model bytes were changed. The additional `verify_archive.py` is a read-only utility created for this package. Documentation and environment starting files are new.

## Validation performed

- All 36 supplied canonical Python files and the new verifier compile without executing their code.
- All 45 archived source/model/metadata/schedule files match the archive hashes in the source manifest.
- The UTF-8 schedule parses as 272 games across 18 weeks with 272 unique week/home/away matchup keys.
- The learned model hash matches its metadata's `model_sha256`.
- The ceiling metadata's `source_v1_sha256` matches the included V1 model.
- All four metadata JSON files parse. Hashes for every joblib are retained in the source manifest; not every metadata file contains a self-hash, so the two checks above are distinguished from archive checksum verification.
- The selected replay's source-version expectations and advanced-stats historical import contract were compared to the supplied producers.

## Not validated

This archive was not run end to end against the user's live or reconstructed databases, external data providers, local PFF data, or Windows OCR installation. No model was retrained and no ROI was recalculated. The proposed environment was not installed or certified as equivalent to the user's Anaconda environment. Version mismatches and missing historical Stage 5 prevent a claim of full recovery readiness.
