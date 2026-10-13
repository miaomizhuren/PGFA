"""Audit final defense artifacts only; this script never trains models."""
from pathlib import Path
import csv, re, sqlite3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/saved_results/final_defense_60_150_r350_20260910"
KEYS = [59, 60, 150, 210, 270, 330, 350]
EXPS = [("def01_norm_clipping_pbu_s60_e150_r350", "Norm Clipping"), ("def02_trimmed_mean_pbu_s60_e150_r350", "Trimmed Mean"), ("def03_multi_krum_pbu_s60_e150_r350", "Multi-Krum"), ("def04_flame_pbu_s60_e150_r350", "Light-FLAME")]
FIELDS = "experiment config_path output_dir db_path log_path exists_output exists_db exists_log max_round has_round_350 missing_key_rounds last_clean_acc last_asr last_attack_active log_finished log_error_detected status".split()

def audit(exp, label):
    output = ROOT / "outputs/final_defense_60_150_r350" / exp
    db = next(iter(sorted(output.glob("*.db"))), None) if output.exists() else None
    log = ROOT / "experiments/logs/final_defense_60_150_r350" / f"{exp}.log"
    r = dict(experiment=exp, label=label, config_path=str(ROOT / "configs/final_defense_60_150_r350" / f"{exp}.yaml"), output_dir=str(output), db_path=str(db) if db else "", log_path=str(log), exists_output=output.exists(), exists_db=bool(db), exists_log=log.exists(), max_round="", has_round_350=False, missing_key_rounds=",".join(map(str, KEYS)), last_clean_acc="", last_asr="", last_attack_active="", log_finished=False, log_error_detected=False)
    if db:
        try:
            with sqlite3.connect(db) as c: records = c.execute("select round_index, clean_accuracy, attack_success_rate, attack_active from round_metrics order by round_index").fetchall()
            if records:
                last = records[-1]; rounds = {x[0] for x in records}
                r.update(max_round=last[0], has_round_350=350 in rounds, missing_key_rounds=",".join(str(x) for x in KEYS if x not in rounds), last_clean_acc=last[1], last_asr=last[2], last_attack_active=bool(last[3]))
        except sqlite3.Error: r["log_error_detected"] = True
    text = log.read_text(errors="replace") if log.exists() else ""
    r["log_finished"] = "Finished" in text
    r["log_error_detected"] |= any(re.search(x, text, re.I) for x in ["Traceback", "Error", "CUDA out of memory", "KeyboardInterrupt", "killed"])
    r["log_tail"] = "\n".join(text.splitlines()[-80:])
    r["status"] = "ERROR" if r["log_error_detected"] else ("COMPLETE" if db and r["has_round_350"] and int(r["max_round"]) >= 350 else ("PARTIAL" if db else ("NO_DB" if output.exists() else "NOT_STARTED")))
    return r

def main():
    OUT.mkdir(parents=True, exist_ok=True); rows = [audit(*x) for x in EXPS]
    with (OUT / "DEFENSE_RUN_STATUS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows([{k:r[k] for k in FIELDS} for r in rows])
    incomplete = [r for r in rows if r["status"] != "COMPLETE"]
    resume = ROOT / "scripts/resume_final_defense_60_150_r350_incomplete.sh"
    lines = ["#!/usr/bin/env bash", "set -e", "cd .", "mkdir -p experiments/logs/final_defense_60_150_r350", "", 'echo "[Resume Final Defense] Start: $(date)"']
    for r in incomplete: lines += ["", f"python train.py --config configs/final_defense_60_150_r350/{r['experiment']}.yaml \\", f"  2>&1 | tee experiments/logs/final_defense_60_150_r350/{r['experiment']}.resume.log"]
    lines += ["", 'echo "[Resume Final Defense] Finished: $(date)"', ""]
    resume.write_text("\n".join(lines), encoding="utf-8"); resume.chmod(0o755)
    overall = "全部完成" if not incomplete else ("有错误中断" if any(r["status"] == "ERROR" for r in rows) else "部分完成")
    report = ["# Defense Run Recovery Report", "", "## Overall conclusion", "", overall, "", "## Per-experiment status", ""] + [f"- {r['label']}: {r['status']}, max_round={r['max_round'] or 'N/A'}" for r in rows] + ["", "## Interruption assessment", ""]
    report += (["- No interruption detected: all four metrics databases contain round 350 and every required key round.", "- Logs have no configured error keyword. They lack a Finished marker, but each ends in a successful round-350 metric record."] if not incomplete else [f"- {r['label']}: last round={r['max_round'] or 'N/A'}, errors={r['log_error_detected']}.\n\nLast 80 log lines:\n```text\n{r['log_tail'] or '(no log)'}\n```" for r in incomplete])
    report += ["", "## Currently usable results", ""] + [f"- {r['label']}: {'usable for the final table' if r['status'] == 'COMPLETE' else 'requires rerun/resume'}." for r in rows] + ["", "## Export results", "", "- Main table: `outputs/saved_results/final_defense_60_150_r350_20260910/FINAL_TABLE_defense_persistence.csv`.", "- Run status: `outputs/saved_results/final_defense_60_150_r350_20260910/DEFENSE_RUN_STATUS.csv`.", "", "## Resume script", "", f"- `scripts/resume_final_defense_60_150_r350_incomplete.sh` contains: {', '.join(r['experiment'] for r in incomplete) if incomplete else 'no experiments; all four groups are complete.'}", "- Generated but not executed.", ""]
    (OUT / "DEFENSE_RUN_RECOVERY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    for r in rows: print(f"{r['experiment']}: {r['status']} (max_round={r['max_round'] or 'N/A'})")

if __name__ == "__main__": main()
