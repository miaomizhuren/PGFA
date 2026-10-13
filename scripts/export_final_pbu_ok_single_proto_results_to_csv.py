"""Export final single-prototype metrics to fixed CSV files after training completes."""
from pathlib import Path
import re
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'outputs/final_best/final_pbu_ok_single_proto_s60_e110_seed2026'
LOGS = ROOT / 'experiments/logs/final_best'
SAVE = ROOT / 'outputs/saved_results/final_pbu_ok_single_proto_20260908'
KEY = [110, 111, 120, 130, 150, 180, 200, 240, 250]

def norm(frame):
    aliases = {'round_index':'round', 'round_id':'round', 'clean_accuracy':'clean_acc',
               'cleanacc':'clean_acc', 'attack_success_rate':'asr', 'attack_success':'asr'}
    frame = frame.copy()
    frame.columns = [aliases.get(re.sub(r'[^a-z0-9]+', '_', str(c).lower()).strip('_'),
                                 re.sub(r'[^a-z0-9]+', '_', str(c).lower()).strip('_')) for c in frame.columns]
    return frame

def quality(frame):
    return 10 * ('round' in frame) + 10 * ('asr' in frame) + 8 * ('clean_acc' in frame)

SAVE.mkdir(parents=True, exist_ok=True)
candidates = []
for base in (RESULT, LOGS):
    if not base.exists(): continue
    for path in [*base.rglob('*.db'), *base.rglob('*.sqlite'), *base.rglob('*.sqlite3')]:
        try:
            with sqlite3.connect(path) as conn:
                for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                    frame = norm(pd.read_sql_query(f'SELECT * FROM "{table}"', conn))
                    if quality(frame) >= 20: candidates.append((quality(frame), 'sqlite', path, table, frame))
        except Exception as exc: print(f'[WARN] {path}: {exc}')
    for path in base.rglob('*.csv'):
        try:
            frame = norm(pd.read_csv(path))
            if quality(frame) >= 20: candidates.append((quality(frame), 'csv', path, '', frame))
        except Exception as exc: print(f'[WARN] {path}: {exc}')
if not candidates: raise SystemExit('No completed result DB/CSV with round and ASR found.')
_, kind, source, table, frame = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
frame['round'] = pd.to_numeric(frame['round'], errors='coerce'); frame = frame.dropna(subset=['round'])
frame['round'] = frame['round'].astype(int)
for col in ('clean_acc','asr','psnr','ssim','linf'):
    if col in frame: frame[col] = pd.to_numeric(frame[col], errors='coerce')
preferred = ['round','clean_acc','asr','selected_clients','selected_malicious_clients','num_selected_malicious','attack_active','average_train_loss','test_loss','num_attack_samples','num_successful_attacks','psnr','ssim','linf','recorded_at']
frame = frame[[c for c in preferred if c in frame] + [c for c in frame if c not in preferred]].sort_values('round')
all_path, key_path = SAVE/'FINAL_all_round_metrics.csv', SAVE/'FINAL_key_rounds.csv'
frame.to_csv(all_path, index=False, encoding='utf-8-sig')
frame[frame['round'].isin(KEY)].to_csv(key_path, index=False, encoding='utf-8-sig')
(SAVE/'README_where_to_open.txt').write_text('用户只需要打开：\n' + str(key_path) + '\n\n完整结果：\n' + str(all_path) + f'\n\n来源：{kind} {source}' + (f' table={table}' if table else '') + '\n', encoding='utf-8')
print(f'Exported {key_path}\nExported {all_path}\nSource: {kind} {source} {table}')
