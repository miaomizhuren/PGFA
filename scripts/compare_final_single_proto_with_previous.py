"""Compare the final single-prototype rerun with the prior OK-equivalent key rounds."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
save = ROOT / 'outputs/saved_results/final_pbu_ok_single_proto_20260908'
old = pd.read_csv(ROOT / 'outputs/saved_results/ok_equiv_reproduce_20260908/FINAL_key_rounds.csv')
new = pd.read_csv(save / 'FINAL_key_rounds.csv')
old = old[['round','clean_acc','asr']].rename(columns={'clean_acc':'old_clean_acc','asr':'old_asr'})
new = new[['round','clean_acc','asr']].rename(columns={'clean_acc':'new_clean_acc','asr':'new_asr'})
out = old.merge(new, on='round', how='outer').sort_values('round')
out['asr_diff'] = out['new_asr'] - out['old_asr']
out['clean_acc_diff'] = out['new_clean_acc'] - out['old_clean_acc']
path = save / 'COMPARE_with_previous_ok_equiv.csv'
out.to_csv(path, index=False, encoding='utf-8-sig')
print(out.to_string(index=False)); print(f'Saved {path}')
