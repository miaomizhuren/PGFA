#!/usr/bin/env python3
"""Create data-driven client-search-fast comparison plots."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/summaries/client_search_fast"
ROUNDS=(100,110,120,130,140,150)
def save(fig,stem):
    for ext in ("png","pdf","svg"): fig.savefig(OUT/f"{stem}.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
def number(row,key):
    try: return float(row.get(key,""))
    except ValueError: return None
def ranked(rows):
    return sorted((row for row in rows if row.get("status")=="complete" and number(row,"score") is not None),key=lambda row:number(row,"score"),reverse=True)
def bar(rows,key,stem,ylabel):
    fig,ax=plt.subplots(figsize=(8,4.5)); labels=[r["qid"] for r in rows]; values=[number(r,key) for r in rows]
    ax.bar(labels,values); ax.set(xlabel="Configuration",ylabel=ylabel); ax.grid(axis="y",alpha=.25); fig.tight_layout(); save(fig,stem)
def main():
    source=OUT/"client_search_fast_summary.csv"
    if not source.exists(): print("Summary missing; run summarize_client_search_fast_results.py first."); return
    with source.open(encoding="utf-8",newline="") as handle: rows=list(csv.DictReader(handle))
    top=ranked(rows)[:10]
    if not top: print("No completed scored configurations; no plots generated."); return
    bar(top,"score","top10_score_bar","Score"); bar(top,"ASR@150","top10_asr150_bar","ASR@150 (%)"); bar(top,"CleanACC@150","top10_cleanacc150_bar","Clean ACC@150 (%)")
    fig,ax=plt.subplots(figsize=(8,4.5))
    for row in top[:5]:
        pts=[(r,number(row,f"ASR@{r}")) for r in ROUNDS if number(row,f"ASR@{r}") is not None]
        ax.plot([r for r,_ in pts],[v for _,v in pts],marker="o",label=row["qid"])
    ax.set(xlabel="Round",ylabel="ASR (%)",xlim=(100,150)); ax.grid(alpha=.25); ax.legend(title="Top 5 score",fontsize=8); fig.tight_layout(); save(fig,"best_configs_asr_curve")
if __name__=="__main__": main()
