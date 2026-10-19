#!/usr/bin/env python
"""Read-only Circa live-QB audit. Never fills, changes or reassigns QB features.

The builder-wide research scope retains its existing >=99% audit requirement.
Every side of every target-week game must have an evidenced GSIS identity and
finite prior-game QB features (except the pre-existing Week 1 convention).
This certifies the historical-snapshot feature QB, NOT an upcoming-game starter.
"""
from __future__ import annotations
from contextlib import closing
import argparse
import datetime as dt
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

IMPLEMENTATION_VERSION = 'v1_1_windows_sqlite_lifecycle_20261009'
HISTORICAL_MINIMUM = 0.99
SEASON = 2026
PERSONNEL_TABLE = 'nfl_matchup_weekly_personnel_features'
TEAM_GAME_TABLE = 'nfl_matchup_team_game_features'
AUDIT_TABLE = 'nfl_matchup_qb_feature_integrity_audit'
MATRIX_TABLE = 'nfl_matchup_game_matrix'
RUN_TABLE = 'nfl_matchup_run_audit'


def text(value: Any) -> str:
    return '' if pd.isna(value) else str(value).strip()


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def name_key(value: Any) -> str:
    value = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', text(value).lower())
    pieces = re.findall(r'[a-z0-9]+', value)
    return '' if not pieces else pieces[0] if len(pieces) == 1 else pieces[0][0] + pieces[-1]


def close(left: Any, right: Any) -> bool:
    a, b = number(left), number(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def required(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame))
    if missing:
        raise RuntimeError(f'{label} missing columns: {missing}')


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, (pd.Timestamp, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def read_table(connection: sqlite3.Connection, name: str) -> pd.DataFrame:
    if not connection.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?', (name,)).fetchone():
        raise RuntimeError(f'Missing audit evidence table: {name}')
    return pd.read_sql_query(f'SELECT * FROM "{name}"', connection)


def evaluate(subset: pd.DataFrame, run: pd.DataFrame, audit: pd.DataFrame,
             personnel: pd.DataFrame, team_games: pd.DataFrame, week: int) -> dict:
    if not isinstance(week, int) or not 1 <= week <= 18:
        raise RuntimeError('Prediction week must be an integer from 1 through 18.')
    required(run, {'qb_identity_match_rate', 'qb_feature_integrity_passed'}, 'Run audit')
    required(audit, {'scope','seasons','personnel_rows','qb_identity_matched_rows',
                    'qb_identity_match_rate','qb_recent_epa_coverage','qb_recent_cpoe_coverage',
                    'qb_epa_advantage_std','qb_cpoe_advantage_std','qb_epa_advantage_nonzero_rows',
                    'qb_cpoe_advantage_nonzero_rows','feature_integrity_passed'}, 'Builder QB audit')
    required(personnel, {'season','prediction_week','team','current_qb_name','current_qb_pbp_key',
                         'qb_identity_matched','last_completed_game_week','qb_recent_epa',
                         'qb_recent_cpoe','qb_recent_games'}, 'Personnel')
    required(team_games, {'season','week','offense_team','primary_qb_name','primary_qb_key'}, 'PBP team-game evidence')
    required(subset, {'season','week','home_team','away_team','rating_alpha'}, 'Target-week matrix')
    if len(run) != 1 or audit.empty or subset.empty:
        raise RuntimeError('Need one run audit, nonempty builder audit and nonempty target-week games.')
    if audit['scope'].duplicated().any():
        raise RuntimeError('Duplicate builder audit scopes.')
    expected_scopes = {'ALL_2018_2025','VALIDATION_2020_2023','BENCHMARK_2024_2025'}
    if set(audit['scope'].astype(str)) != expected_scopes:
        raise RuntimeError('Unrecognized builder audit scopes; no scope is silently skipped.')
    errors: list[str] = []
    scopes = []
    ps = pd.to_numeric(personnel.season, errors='coerce')
    pw = pd.to_numeric(personnel.prediction_week, errors='coerce')
    named = personnel.current_qb_name.map(text).ne('')
    matched = pd.to_numeric(personnel.qb_identity_matched, errors='coerce').eq(1)
    # Keep the builder's exact historical denominator and all original rows.
    for row in audit.to_dict('records'):
        try:
            seasons = [int(v) for v in str(row['seasons']).split('|')]
        except ValueError as exc:
            raise RuntimeError('Invalid seasons in builder audit.') from exc
        scope_rows = personnel[ps.isin(seasons) & pw.between(2,17) & named]
        n = len(scope_rows)
        successes = int(pd.to_numeric(scope_rows.qb_identity_matched, errors='coerce').eq(1).sum())
        rate = successes / n if n else float('nan')
        item = {'scope':row['scope'],'actual_seasons':seasons,'rows':n,
                'matched':successes,'unmatched':n-successes,'rate':rate,
                'minimum_retained':HISTORICAL_MINIMUM}
        scopes.append(item)
        if not close(row['personnel_rows'], n) or not close(row['qb_identity_matched_rows'], successes) or not close(row['qb_identity_match_rate'],rate):
            errors.append(f"{row['scope']}: stored audit counts/rate disagree with personnel rows")
        if not math.isfinite(rate) or rate < HISTORICAL_MINIMUM:
            errors.append(f"{row['scope']}: coverage {rate!r} below retained 99% builder minimum")
        if number(row['feature_integrity_passed']) != 1:
            errors.append(f"{row['scope']}: original feature_integrity_passed is not 1")
        for metric in ('qb_recent_epa','qb_recent_cpoe'):
            coverage = float(pd.to_numeric(scope_rows[metric], errors='coerce').notna().mean()) if n else float('nan')
            if not close(row[f'{metric}_coverage'], coverage) or not math.isfinite(coverage) or coverage < HISTORICAL_MINIMUM:
                errors.append(f"{row['scope']}: inconsistent/insufficient {metric} coverage")
        for metric in ('qb_epa','qb_cpoe'):
            if not math.isfinite(number(row[f'{metric}_advantage_std'])) or number(row[f'{metric}_advantage_std']) <= 1e-6 or number(row[f'{metric}_advantage_nonzero_rows']) <= 0:
                errors.append(f"{row['scope']}: original {metric} signal gate failed")
    all_scope = next(s for s in scopes if s['scope']=='ALL_2018_2025')
    if not close(run.iloc[0]['qb_identity_match_rate'], all_scope['rate']):
        errors.append('Run-audit match rate disagrees with the full builder scope')
    if number(run.iloc[0]['qb_feature_integrity_passed']) != 1:
        errors.append('Run-audit original feature_integrity_passed is not 1')
    if not pd.to_numeric(subset.season,errors='coerce').eq(SEASON).all() or not pd.to_numeric(subset.week,errors='coerce').eq(week).all():
        errors.append('Target matrix has rows outside requested season/week')
    if subset.duplicated(['home_team','away_team']).any():
        errors.append('Duplicate target-week matchups')
    if pd.to_numeric(subset.rating_alpha,errors='coerce').nunique()!=1:
        errors.append('Target matrix mixes rating alpha values')

    # This is diagnostic evidence only. No absent identity is inferred or repaired.
    missing_personnel = personnel[~matched].copy()
    missing_columns = ['season','prediction_week','team','last_completed_game_week',
                       'current_qb_name','current_qb_pbp_key','qb_identity_matched']
    all_unmatched = missing_personnel[missing_columns].to_dict('records')
    live_rows: list[dict] = []
    if week > 1:
        fields = {'current_qb_name','current_qb_pbp_key','qb_identity_matched',
                  'last_completed_game_week','qb_recent_epa','qb_recent_cpoe','qb_recent_games'}
        required(subset, {f'{side}_{f}' for side in ('home','away') for f in fields}
                 | {'qb_recent_epa_advantage','qb_recent_cpoe_advantage'}, 'Target QB features')
        evidence = team_games.copy()
        evidence['_name_key'] = evidence.primary_qb_name.map(name_key)
        es = pd.to_numeric(evidence.season, errors='coerce')
        ew = pd.to_numeric(evidence.week, errors='coerce')
        for game in subset.to_dict('records'):
            game_label = f"{text(game['away_team'])}@{text(game['home_team'])}"
            for side in ('home','away'):
                team = text(game[f'{side}_team'])
                name = text(game[f'{side}_current_qb_name'])
                pid = text(game[f'{side}_current_qb_pbp_key'])
                snapshot = number(game[f'{side}_last_completed_game_week'])
                issues=[]
                if number(game[f'{side}_qb_identity_matched']) != 1: issues.append('IDENTITY_FLAG_NOT_1')
                if not re.fullmatch(r'00-\d+',pid): issues.append('MISSING_OR_INVALID_GSIS_ID')
                if not name: issues.append('MISSING_QB_NAME')
                if not math.isfinite(snapshot) or not snapshot.is_integer() or not 1 <= snapshot < week:
                    issues.append('INVALID_OR_NONPRIOR_SNAPSHOT_WEEK')
                for field in ('qb_recent_epa','qb_recent_cpoe'):
                    if not math.isfinite(number(game[f'{side}_{field}'])): issues.append('NONFINITE_'+field.upper())
                if not math.isfinite(number(game[f'{side}_qb_recent_games'])) or number(game[f'{side}_qb_recent_games']) <= 0:
                    issues.append('NO_PRIOR_QB_PERFORMANCE_GAMES')
                p = personnel[ps.eq(SEASON) & pw.eq(week) & personnel.team.map(text).eq(team)]
                if len(p)!=1:
                    issues.append(f'PERSONNEL_ROWS_{len(p)}_EXPECTED_1')
                else:
                    pr=p.iloc[0]
                    for field in fields:
                        equal = close(game[f'{side}_{field}'],pr[field]) if field not in ('current_qb_name','current_qb_pbp_key') else text(game[f'{side}_{field}'])==text(pr[field])
                        if not equal: issues.append('MATRIX_PERSONNEL_MISMATCH_'+field.upper())
                proof = evidence[es.eq(SEASON) & ew.lt(week) & ew.le(snapshot)
                                 & evidence.offense_team.map(text).eq(team) & evidence._name_key.eq(name_key(name))]
                exact = proof[pd.to_numeric(proof.week,errors='coerce').eq(snapshot)]
                candidates = exact if not exact.empty else proof
                ids = sorted(set(candidates.primary_qb_key.map(text)) - {''})
                if ids != [pid]: issues.append('GSIS_NOT_UNIQUELY_CORROBORATED_BY_PRIOR_TEAM_PBP')
                live_rows.append({'game':game_label,'side':side,'team':team,'feature_qb':name,
                                  'gsis_id':pid,'snapshot_week':snapshot,'evidence_ids':ids,
                                  'identity_flag':game[f'{side}_qb_identity_matched'],
                                  'issues':issues,'passed':not issues})
            for metric in ('epa','cpoe'):
                h=number(game[f'home_qb_recent_{metric}']); a=number(game[f'away_qb_recent_{metric}'])
                if not close(game[f'qb_recent_{metric}_advantage'],h-a):
                    errors.append(f'{game_label}: QB {metric.upper()} advantage differs from side features')
        teams=[r['team'] for r in live_rows]
        if len(teams)!=len(set(teams)): errors.append('A team appears more than once in the target-week matrix')
        ids=[r['gsis_id'] for r in live_rows if r['gsis_id']]
        if len(ids)!=len(set(ids)): errors.append('One QB GSIS ID appears on multiple target-week teams')
        for item in live_rows:
            if item['issues']:
                errors.append(f"{item['game']} {item['team']} {item['feature_qb'] or '<blank>'}: " + '; '.join(item['issues']))
    else:
        # Week 1 has no same-season prior-game QB features. Do not invent them.
        for side in ('home','away'):
            for f in ('qb_identity_matched','current_qb_pbp_key','qb_recent_epa','qb_recent_cpoe'):
                col=f'{side}_{f}'
                if col in subset and subset[col].map(text).ne('').any():
                    errors.append('Week 1 unexpectedly contains current-season QB-history features')
    return jsonable({'implementation':IMPLEMENTATION_VERSION,'week':week,'season':SEASON,
                     'passed':not errors,'errors':errors,'builder_scopes':scopes,
                     'builder_wide_identity_rate':all_scope['rate'],
                     'live_games':len(subset),'live_sides':len(live_rows),
                     'live_sides_passed':sum(r['passed'] for r in live_rows),
                     'live_identity_rate':sum(number(r['identity_flag'])==1 for r in live_rows)/len(live_rows) if live_rows else None,
                     'live_scope':'PRIOR_GAME_FEATURE_IDENTITIES_NOT_UPCOMING_STARTER_CERTIFICATION',
                     'week1_no_same_season_history':week==1,
                     'live_qb_rows':live_rows,'all_unmatched_personnel_rows':all_unmatched,
                     'audit_rows_modified':False,'feature_values_modified':False})


def write_report(report: dict, directory: Path, week: int) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path=directory/f'nfl_circa_qb_audit_2026_week_{week:02d}.json'
    text_path=json_path.with_suffix('.txt')
    lines=[f'CIRCA QB SCOPE AUDIT | Week {week}',f'Implementation: {IMPLEMENTATION_VERSION}',
           f"Passed: {report.get('passed',False)}",f"Database: {report.get('database','')}",
           f"Builder-wide identity rate: {report.get('builder_wide_identity_rate')}",
           f"Target-week identity rate: {report.get('live_identity_rate')}",
           f"Target-week fully verified QB sides: {report.get('live_sides_passed',0)}/{report.get('live_sides',0)}",
           'The target-week check covers prior-game feature QBs, not current starter availability.',
           '', 'ERRORS:', *report.get('errors',[]),'','TARGET-WEEK QB ROWS:']
    for row in report.get('live_qb_rows',[]):
        lines.append(f"{row['game']} | {row['team']} | {row['feature_qb']} | GSIS={row['gsis_id']} | snapshot_week={row['snapshot_week']} | {'PASS' if row['passed'] else ';'.join(row['issues'])}")
    lines+=['','ALL UNMATCHED PERSONNEL ROWS (including historical):']
    for row in report.get('all_unmatched_personnel_rows',[]):
        lines.append(json.dumps(row,sort_keys=True))
    payload=json.dumps(jsonable(report),indent=2,allow_nan=False)+'\n'
    for path,content in ((json_path,payload),(text_path,'\n'.join(lines)+'\n')):
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(content,encoding='utf-8');temp.replace(path)
    return json_path,text_path


def audit_database(database: Path, week: int, rating_alpha: float, output_dir: Path | None=None,
                   subset: pd.DataFrame | None=None) -> dict:
    database=database.resolve()
    report={'passed':False,'errors':[],'database':str(database),'week':week,'implementation':IMPLEMENTATION_VERSION,
            'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        if not database.is_file(): raise RuntimeError(f'Live database not found: {database}')
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as c, c:
            c.execute('PRAGMA query_only=ON')
            run=read_table(c,RUN_TABLE);audits=read_table(c,AUDIT_TABLE)
            p=read_table(c,PERSONNEL_TABLE);t=read_table(c,TEAM_GAME_TABLE)
            if subset is None:
                m=read_table(c,MATRIX_TABLE)
                required(m,{'season','week','rating_alpha'},'Matrix')
                subset=m[pd.to_numeric(m.season,errors='coerce').eq(SEASON)&pd.to_numeric(m.week,errors='coerce').eq(week)&pd.to_numeric(m.rating_alpha,errors='coerce').eq(rating_alpha)].copy()
        report.update(evaluate(subset,run,audits,p,t,week))
    except Exception as exc:
        report['passed']=False;report['errors']=[f'{type(exc).__name__}: {exc}']
    directory=output_dir or database.parent.parent/'outputs'
    jp,tp=write_report(report,directory,week)
    report['json_report']=str(jp);report['text_report']=str(tp)
    print(f"[CIRCA_QB_AUDIT] builder-wide={report.get('builder_wide_identity_rate')} | target-week={report.get('live_identity_rate')} | verified={report.get('live_sides_passed',0)}/{report.get('live_sides',0)} | passed={report['passed']}",flush=True)
    print(f'[CIRCA_QB_AUDIT] Report: {tp}',flush=True)
    if not report['passed']:
        raise RuntimeError('QB scope audit blocked:\n'+'\n'.join(report['errors'])+f'\nFull evidence: {tp}')
    return report


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--database',type=Path,default=Path(r'C:\Users\maxxs\Downloads\Football Files\nfl_model\backtests\nfl_weekly_matchup_residual_live_2026.sqlite'))
    ap.add_argument('--week',type=int,required=True)
    ap.add_argument('--rating-alpha',type=float,default=10.0)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    audit_database(args.database,args.week,args.rating_alpha,args.output_dir)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('[CIRCA_QB_AUDIT] FAILED: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
