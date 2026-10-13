"""Export the five completed window-search runs into comparable CSV files.

Run this after w02--w05 have completed; w01 is read as an existing baseline.
"""
from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / 'outputs/saved_results/window_search_final_single_proto_20260908'
SPECS = {
    'w01_s60_e110_r250': (60, 110, 250),
    'w02_s70_e120_r270': (70, 120, 270),
    'w03_s80_e130_r280': (80, 130, 280),
    'w04_s100_e150_r300': (100, 150, 300),
    'w05_s120_e170_r320': (120, 170, 320),
}


def normalize(frame):
    aliases = {
        'round_index': 'round', 'round_id': 'round',
        'clean_accuracy': 'clean_acc', 'cleanacc': 'clean_acc',
        'attack_success_rate': 'asr', 'attack_success': 'asr',
    }
    frame = frame.copy()
    normalized = [re.sub(r'[^a-z0-9]+', '_', str(c).lower()).strip('_') for c in frame.columns]
    frame.columns = [aliases.get(c, c) for c in normalized]
    return frame


def score(frame):
    return 10 * ('round' in frame) + 10 * ('asr' in frame) + 8 * ('clean_acc' in frame)


def load_metrics(run_name):
    if run_name == 'w01_s60_e110_r250':
        for path in (
            ROOT / 'outputs/saved_results/final_pbu_ok_single_proto_20260908/FINAL_key_rounds.csv',
            ROOT / 'outputs/saved_results/ok_equiv_reproduce_20260908/FINAL_key_rounds.csv',
        ):
            if path.exists():
                frame = normalize(pd.read_csv(path))
                if score(frame) >= 20:
                    frame['round'] = pd.to_numeric(frame['round'], errors='coerce')
                    frame = frame.dropna(subset=['round']).copy()
                    frame['round'] = frame['round'].astype(int)
                    for field in ('clean_acc', 'asr', 'psnr', 'ssim', 'linf'):
                        if field in frame:
                            frame[field] = pd.to_numeric(frame[field], errors='coerce')
                    return frame.sort_values('round').drop_duplicates('round', keep='last'), 'baseline_csv', path, ''
        raise SystemExit('w01 baseline missing: expected final_pbu_ok_single_proto or ok_equiv_reproduce key CSV.')
    bases = [
        ROOT / 'outputs/window_search_final_single_proto' / run_name,
        ROOT / 'experiments/logs/window_search_final_single_proto',
    ]
    candidates = []
    for base in bases:
        if not base.exists():
            continue
        for path in [*base.rglob('*.db'), *base.rglob('*.sqlite'), *base.rglob('*.sqlite3')]:
            try:
                with sqlite3.connect(path) as conn:
                    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                        frame = normalize(pd.read_sql_query(f'SELECT * FROM "{table}"', conn))
                        if score(frame) >= 20:
                            candidates.append((score(frame), 'sqlite', path, table, frame))
            except Exception as exc:
                print(f'[WARN] {path}: {exc}')
        for path in base.rglob('*.csv'):
            try:
                frame = normalize(pd.read_csv(path))
                if score(frame) >= 20:
                    candidates.append((score(frame), 'csv', path, '', frame))
            except Exception as exc:
                print(f'[WARN] {path}: {exc}')
    if not candidates:
        raise SystemExit(f'No completed round/ASR metrics found for {run_name}. Train this group first.')
    _, source_kind, source_path, table, frame = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    frame['round'] = pd.to_numeric(frame['round'], errors='coerce')
    frame = frame.dropna(subset=['round']).copy()
    frame['round'] = frame['round'].astype(int)
    for field in ('clean_acc', 'asr', 'psnr', 'ssim', 'linf'):
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors='coerce')
    preferred = ['round', 'clean_acc', 'asr', 'selected_clients', 'selected_malicious_clients',
                 'num_selected_malicious', 'attack_active', 'protocol_warning', 'average_train_loss',
                 'test_loss', 'num_attack_samples', 'num_successful_attacks', 'psnr', 'ssim',
                 'linf', 'recorded_at']
    frame = frame[[c for c in preferred if c in frame] + [c for c in frame if c not in preferred]]
    return frame.sort_values('round').drop_duplicates('round', keep='last'), source_kind, source_path, table


def value_at(frame, round_index, field):
    values = frame.loc[frame['round'] == round_index, field] if field in frame else pd.Series(dtype=float)
    return values.iloc[-1] if not values.empty else float('nan')


def normalized_auc(frame, start, end):
    points = frame[(frame['round'] >= start) & (frame['round'] <= end)][['round', 'asr']].dropna()
    if len(points) < 2 or end <= start:
        return float('nan')
    points = points.sort_values('round')
    area = sum((a + b) * (x2 - x1) / 2 for x1, a, x2, b in zip(
        points['round'], points['asr'], points['round'].iloc[1:], points['asr'].iloc[1:]))
    return area / (end - start)


SAVE.mkdir(parents=True, exist_ok=True)
rows, sources = [], []
for name, (start, end, total) in SPECS.items():
    frame, kind, source, table = load_metrics(name)
    keys = [end, end + 1, end + 10, end + 20, end + 40, end + 70, end + 90, end + 130, total]
    frame.to_csv(SAVE / f'FINAL_all_round_metrics_{name}.csv', index=False, encoding='utf-8-sig')
    frame[frame['round'].isin(keys)].to_csv(SAVE / f'FINAL_key_rounds_{name}.csv', index=False, encoding='utf-8-sig')
    final = frame[frame['round'] == total].tail(1)
    rows.append({
        'experiment': name, 'attack_start': start, 'attack_end': end, 'total_rounds': total,
        'attack_length': end - start + 1, 'post_attack_length': total - end,
        'CleanACC@final': value_at(frame, total, 'clean_acc'),
        'ASR@attack_end': value_at(frame, end, 'asr'),
        'ASR@post1': value_at(frame, end + 1, 'asr'), 'ASR@post10': value_at(frame, end + 10, 'asr'),
        'ASR@post20': value_at(frame, end + 20, 'asr'), 'ASR@post40': value_at(frame, end + 40, 'asr'),
        'ASR@post70': value_at(frame, end + 70, 'asr'), 'ASR@post90': value_at(frame, end + 90, 'asr'),
        'ASR@post130': value_at(frame, end + 130, 'asr'), 'ASR@final': value_at(frame, total, 'asr'),
        'PA-AUC@post': normalized_auc(frame, end + 1, total),
        'PSNR': final['psnr'].iloc[0] if not final.empty and 'psnr' in final else float('nan'),
        'SSIM': final['ssim'].iloc[0] if not final.empty and 'ssim' in final else float('nan'),
        'Linf': final['linf'].iloc[0] if not final.empty and 'linf' in final else float('nan'),
        'source_file': str(source),
    })
    sources.append(f'{name}: {kind} {source}' + (f' table={table}' if table else ''))

compare_path = SAVE / 'FINAL_compare_5groups.csv'
pd.DataFrame(rows).to_csv(compare_path, index=False, encoding='utf-8-sig')
(SAVE / 'README_where_to_open.txt').write_text(
    '训练完成后请优先打开：\n' + str(compare_path) +
    '\n\n每组关键轮次：FINAL_key_rounds_<group>.csv\n每组完整轮次：FINAL_all_round_metrics_<group>.csv\n\n来源：\n' +
    '\n'.join(sources) + '\n', encoding='utf-8')
print(f'Exported comparison: {compare_path}')
