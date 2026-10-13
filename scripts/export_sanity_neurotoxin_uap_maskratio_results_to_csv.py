"""Export completed Neurotoxin-UAP mask-ratio sanity experiments."""
from pathlib import Path
import sqlite3
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/sanity_check_neurotoxin_uap'
SAVE=ROOT/'outputs/saved_results/sanity_neurotoxin_uap_maskratio_20260909'
NAMES=['neuro_sanity_mr005_s60_e80_r100','neuro_sanity_mr010_s60_e80_r100','neuro_sanity_mr020_s60_e80_r100','neuro_sanity_mr030_s60_e80_r100','neuro_sanity_mr050_s60_e80_r100']
KEY=[59,60,61,70,80,81,90,100]


def at(frame, round_index, column):
    """Return the last value for a key round, or None when it is unavailable."""
    matches = frame.loc[frame["round"] == round_index, column]
    return None if matches.empty else matches.iloc[-1]


def difference(left, right):
    """Keep derived fields empty if either required key round is missing."""
    return None if left is None or right is None else left - right


def level(a80,a100):
    if a80 is None or a100 is None:
        return 'missing_key_rounds'
    if a80>=90 and a100>=50:return 'strong_and_persistent'
    if a80>=90:return 'injected_but_not_persistent'
    if a80>=60:return 'usable'
    if a80>=30:return 'weak_but_injected'
    return 'failed'
def main():
    SAVE.mkdir(parents=True,exist_ok=True); rows=[]; missing_by_run={}
    for name in NAMES:
        cfg=yaml.safe_load((CONFIG/f'{name}.yaml').read_text())
        db=ROOT/cfg['experiment']['output_dir'].replace('./','')/cfg['tracking']['database_name']
        if not db.is_file(): raise SystemExit(f'Missing completed run: {db}')
        with sqlite3.connect(db) as c: f=pd.read_sql_query('SELECT * FROM round_metrics ORDER BY round_index',c)
        f=f.rename(columns={'round_index':'round','clean_accuracy':'clean_acc','attack_success_rate':'asr'}); f['source_file']=str(db)
        # SQLite/pandas may represent the same round column as integer, float,
        # or text. Normalize it before all comparisons and explicit column access.
        f["round"] = pd.to_numeric(f["round"], errors="coerce").astype("Int64")
        f = f.dropna(subset=["round"])
        f["round"] = f["round"].astype(int)
        missing = sorted(set(KEY) - set(f["round"]))
        missing_by_run[name] = missing
        # Emit every requested key row; unavailable rounds remain blank instead
        # of aborting the export.
        key_frame = (
            f.drop_duplicates(subset=["round"], keep="last")
            .set_index("round")
            .reindex(KEY)
            .rename_axis("round")
            .reset_index()
        )
        key_frame["source_file"] = str(db)
        key_frame.to_csv(SAVE/f'FINAL_key_rounds_{name}.csv',index=False,encoding='utf-8-sig')
        a80,a81,a100=at(f,80,'asr'),at(f,81,'asr'),at(f,100,'asr')
        rows.append({'experiment':name,'mask_ratio':cfg['attack']['neurotoxin']['mask_ratio'],
        'effective_keep_ratio':1-cfg['attack']['neurotoxin']['mask_ratio'],'selected_direction':'high_gradient masked; low_gradient kept','attack_start':60,'attack_end':80,'total_rounds':100,
        'CleanACC@59':at(f,59,'clean_acc'),**{f'ASR@{r}':at(f,r,'asr') for r in (60,61,70,80,81,90,100)},'ASR_gain_60_to_80':difference(a80,at(f,60,'asr')),'ASR_drop_80_to_81':difference(a80,a81),'ASR_retention_100_vs_80':difference(a100,a80),'Neurotoxin_success_level':level(a80,a100),'source_file':str(db)})
    out=SAVE/'FINAL_compare_5groups.csv';pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
    missing_lines = [f'- {name}: ' + (', '.join(map(str, missing)) if missing else 'none') for name, missing in missing_by_run.items()]
    (SAVE/'README_where_to_open.txt').write_text(
        f'Open comparison table:\n{out}\n\nMissing key rounds (blank in CSV when absent):\n'
        + '\n'.join(missing_lines) + '\n', encoding='utf-8')
    print(f'Exported: {out}')
    print('Missing key rounds: ' + '; '.join(
        f'{name}=' + (','.join(map(str, missing)) if missing else 'none')
        for name, missing in missing_by_run.items()))
if __name__=='__main__': main()
