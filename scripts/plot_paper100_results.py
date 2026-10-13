#!/usr/bin/env python3
"""Paper100 plot entry point; reads generated summaries when results are available."""
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parents[1]; source=ROOT/"experiments/summaries/paper100/paper100_summary_all.csv"
if __name__=="__main__":
 rows=list(csv.DictReader(source.open())) if source.exists() else []
 print(f"paper100 plotting deferred: {sum(r['status']=='complete' for r in rows)} completed rows available.")
