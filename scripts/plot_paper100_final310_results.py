#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/summaries/paper100_final310"
p=OUT/"paper100_final310_summary_all.csv";rows=[r for r in csv.DictReader(p.open()) if r["status"]=="complete"] if p.exists() else []
if not rows: print("No completed paper100_final310 metrics available for plotting.")
else:
 fig,ax=plt.subplots()
 for r in rows:
  pts=[(x,r.get(f"ASR@{x}","")) for x in (110,150,200,220,240,260,280,300,310)]
  pts=[(x,float(y)) for x,y in pts if y]
  if pts:ax.plot([x for x,_ in pts],[y for _,y in pts],label=r["experiment_id"])
 ax.axvline(110,color="black",linestyle="--");ax.set(xlabel="Round",ylabel="ASR (%)");ax.legend(fontsize=6);fig.tight_layout()
 for ext in ("png","pdf","svg"):fig.savefig(OUT/f"p100f_all_asr_curve.{ext}",dpi=180)
 print(f"Plotted {len(rows)} completed paper100_final310 rows.")
