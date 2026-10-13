#!/usr/bin/env python3
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parents[1];p=ROOT/"experiments/summaries/paper100_poison_long200/paper100_poison_long200_summary.csv"
if __name__=="__main__":print(f"completed poison-long200 rows: {sum(r['status']=='complete' for r in csv.DictReader(p.open())) if p.exists() else 0}")
