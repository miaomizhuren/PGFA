#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"experiments/summaries/paper100_epoch_final310";ROUNDS=(100,110,120,130,140,150,160,170,180,190,200,220,240,260,280,300,310)
def save(fig,n):
 for e in ("png","pdf","svg"):fig.savefig(OUT/f"{n}.{e}",dpi=180,bbox_inches="tight")
 plt.close(fig)
def main():
 p=OUT/"paper100_epoch_final310_summary.csv"
 rows=[r for r in csv.DictReader(p.open()) if r["status"]=="complete"] if p.exists() else []
 if not rows:print("No completed epoch-final310 results; no plots generated.");return
 for metric,name,label,start in [("ASR","p100ef_local_epochs_asr_curve","ASR (%)",110),("CleanACC","p100ef_local_epochs_cleanacc_curve","Clean ACC (%)",100)]:
  fig,ax=plt.subplots()
  for r in rows:
   x=[i for i in ROUNDS if i>=start and r.get(f"{metric}@{i}")];ax.plot(x,[float(r[f"{metric}@{i}"]) for i in x],marker="o",label=r["method"])
  ax.axvline(110,color="black",linestyle="--");ax.set(xlabel="Round",ylabel=label);ax.grid(alpha=.25);ax.legend();save(fig,name)
 for key,name,label in [("ASR@310","p100ef_local_epochs_asr310_bar","ASR@310 (%)"),("AUC@110-310","p100ef_local_epochs_auc110_310_bar","AUC@110-310"),("CleanACC@310","p100ef_local_epochs_cleanacc310_bar","Clean ACC@310 (%)")]:
  fig,ax=plt.subplots();ax.bar([r["method"] for r in rows],[float(r[key]) for r in rows]);ax.tick_params(axis="x",rotation=25);ax.set(ylabel=label);save(fig,name)
if __name__=="__main__":main()
