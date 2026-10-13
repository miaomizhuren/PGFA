#!/usr/bin/env python3
"""Summarize completed final-supplement and referenced historical SQLite runs."""
from __future__ import annotations
import csv, sqlite3, statistics
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=ROOT/'outputs/pgfa_final_supplement/experiment_manifest.csv'; OUT=ROOT/'outputs/pgfa_final_supplement/summary_available_results.csv'
def one(record):
    cfg=yaml.safe_load((ROOT/record['config_path']).read_text())
    db=ROOT/cfg['experiment']['output_dir']/cfg['tracking']['database_name']
    if not db.exists(): return None
    con=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    q='select round_index,clean_accuracy,attack_success_rate from round_metrics where round_index between 3201 and 3700 order by round_index'
    data=con.execute(q).fetchall(); end=con.execute('select attack_success_rate from round_metrics where round_index=3200').fetchone()
    asr=[r[2] for r in data if r[2] is not None]; clean=[r[1] for r in data if r[1] is not None]
    if not asr: return None
    at={r[0]:r[2] for r in data}
    return dict(record,db_path=str(db.relative_to(ROOT)),CleanACC_mean_post=sum(clean)/len(clean),ASR_attack_end=end[0] if end else '',ASR_post_100=at.get(3300,''),ASR_post_200=at.get(3400,''),ASR_post_300=at.get(3500,''),ASR_post_400=at.get(3600,''),ASR_post_500=at.get(3700,''),PA_AUC_1_500=sum(asr)/len(asr),N80=sum(x>=80 for x in asr),N90=sum(x>=90 for x in asr),Last50_ASR_mean=sum(asr[-50:])/len(asr[-50:]))
def main():
    rows=[x for r in csv.DictReader(MANIFEST.open()) if (x:=one(r))]
    cols=list(rows[0]) if rows else ['experiment_id']
    with OUT.open('w',newline='') as h: w=csv.DictWriter(h,cols);w.writeheader();w.writerows(rows)
    print(f'wrote {OUT} ({len(rows)} completed runs; missing runs omitted)')
if __name__=='__main__': main()
