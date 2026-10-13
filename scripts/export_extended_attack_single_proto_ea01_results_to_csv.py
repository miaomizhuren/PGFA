"""Export ea01 metrics and compare them with the original OK w01 baseline."""
from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'outputs/extended_attack_single_proto/ea01_s60_e150_r350'
SAVE = ROOT / 'outputs/saved_results/extended_attack_single_proto_ea01_20260909'
KEY = [59, 60, 100, 110, 150, 151, 160, 170, 190, 220, 240, 280, 300, 350]


def normalize(frame):
    aliases = {'round_index': 'round', 'round_id': 'round', 'clean_accuracy': 'clean_acc',
               'cleanacc': 'clean_acc', 'attack_success_rate': 'asr', 'attack_success': 'asr'}
    frame = frame.copy()
    names = [re.sub(r'[^a-z0-9]+', '_', str(column).lower()).strip('_') for column in frame.columns]
    frame.columns = [aliases.get(name, name) for name in names]
    return frame


def quality(frame):
    return 10 * ('round' in frame) + 10 * ('asr' in frame) + 8 * ('clean_acc' in frame)


def prepare(frame):
    frame = normalize(frame)
    frame['round'] = pd.to_numeric(frame['round'], errors='coerce')
    frame = frame.dropna(subset=['round']).copy()
    frame['round'] = frame['round'].astype(int)
    for field in ('clean_acc', 'asr', 'psnr', 'ssim', 'linf'):
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors='coerce')
    return frame.sort_values('round').drop_duplicates('round', keep='last')


def load_ea01():
    candidates = []
    if RESULT.exists():
        for path in [*RESULT.rglob('*.db'), *RESULT.rglob('*.sqlite'), *RESULT.rglob('*.sqlite3')]:
            try:
                with sqlite3.connect(path) as connection:
                    for (table,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                        frame = normalize(pd.read_sql_query(f'SELECT * FROM "{table}"', connection))
                        if quality(frame) >= 20:
                            candidates.append((quality(frame), 'sqlite', path, table, frame))
            except Exception as exc:
                print(f'[WARN] {path}: {exc}')
        for path in RESULT.rglob('*.csv'):
            try:
                frame = normalize(pd.read_csv(path))
                if quality(frame) >= 20:
                    candidates.append((quality(frame), 'csv', path, '', frame))
            except Exception as exc:
                print(f'[WARN] {path}: {exc}')
    if not candidates:
        raise SystemExit(f'No completed ea01 round/ASR metrics found under {RESULT}. Train ea01 first.')
    _, kind, source, table, frame = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return prepare(frame), kind, source, table


def at(frame, round_index, field):
    values = frame.loc[frame['round'] == round_index, field] if field in frame else pd.Series(dtype=float)
    return values.iloc[-1] if not values.empty else float('nan')


def post_auc(frame, start, end):
    points = frame[(frame['round'] >= start) & (frame['round'] <= end)][['round', 'asr']].dropna().sort_values('round')
    if len(points) < 2:
        return float('nan')
    area = sum((a + b) * (x2 - x1) / 2 for x1, a, x2, b in zip(points['round'], points['asr'], points['round'].iloc[1:], points['asr'].iloc[1:]))
    return area / (end - start)


def row(name, frame, start, end, total, source):
    final = frame[frame['round'] == total].tail(1)
    return {
        'experiment': name, 'attack_start': start, 'attack_end': end, 'total_rounds': total,
        'attack_length': end - start + 1, 'post_attack_length': total - end,
        'CleanACC@attack_end': at(frame, end, 'clean_acc'), 'CleanACC@final': at(frame, total, 'clean_acc'),
        'ASR@attack_end': at(frame, end, 'asr'), 'ASR@post1': at(frame, end + 1, 'asr'),
        'ASR@post10': at(frame, end + 10, 'asr'), 'ASR@post20': at(frame, end + 20, 'asr'),
        'ASR@post40': at(frame, end + 40, 'asr'), 'ASR@post70': at(frame, end + 70, 'asr'),
        'ASR@post90': at(frame, end + 90, 'asr'), 'ASR@post130': at(frame, end + 130, 'asr'),
        'ASR@final': at(frame, total, 'asr'), 'PA-AUC@post': post_auc(frame, end + 1, total),
        'PSNR': final['psnr'].iloc[0] if not final.empty and 'psnr' in final else float('nan'),
        'SSIM': final['ssim'].iloc[0] if not final.empty and 'ssim' in final else float('nan'),
        'Linf': final['linf'].iloc[0] if not final.empty and 'linf' in final else float('nan'),
        'source_file': str(source),
    }


SAVE.mkdir(parents=True, exist_ok=True)
ea01, kind, source, table = load_ea01()
preferred = ['round', 'clean_acc', 'asr', 'selected_clients', 'selected_malicious_clients', 'num_selected_malicious', 'attack_active', 'protocol_warning', 'average_train_loss', 'test_loss', 'num_attack_samples', 'num_successful_attacks', 'psnr', 'ssim', 'linf', 'recorded_at']
ea01 = ea01[[field for field in preferred if field in ea01] + [field for field in ea01 if field not in preferred]]
ea01.to_csv(SAVE / 'FINAL_all_round_metrics.csv', index=False, encoding='utf-8-sig')
ea01[ea01['round'].isin(KEY)].to_csv(SAVE / 'FINAL_key_rounds.csv', index=False, encoding='utf-8-sig')

rows = [row('ea01_s60_e150_r350', ea01, 60, 150, 350, source)]
for baseline in (ROOT / 'outputs/saved_results/final_pbu_ok_single_proto_20260908/FINAL_key_rounds.csv', ROOT / 'outputs/saved_results/ok_equiv_reproduce_20260908/FINAL_key_rounds.csv'):
    if baseline.exists():
        rows.append(row('original_ok_w01_s60_e110_r250', prepare(pd.read_csv(baseline)), 60, 110, 250, baseline))
        break
compare = SAVE / 'COMPARE_with_original_ok_w01.csv'
pd.DataFrame(rows).to_csv(compare, index=False, encoding='utf-8-sig')
(SAVE / 'README_where_to_open.txt').write_text(
    '用户最终需要打开：\n' + str(SAVE / 'FINAL_key_rounds.csv') + '\n' + str(compare) +
    f'\n\nea01 来源：{kind} {source}' + (f' table={table}' if table else '') + '\n', encoding='utf-8')
print(f'Exported: {SAVE / "FINAL_key_rounds.csv"}\nExported: {SAVE / "FINAL_all_round_metrics.csv"}\nExported: {compare}')
