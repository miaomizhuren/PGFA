"""Print Pearson correlations between saved feature margins and ASR."""

import argparse
import csv
import math


def correlation(values, outcomes):
    if len(values) < 2:
        return None
    mean_x, mean_y = sum(values) / len(values), sum(outcomes) / len(outcomes)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, outcomes))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in values) * sum((y - mean_y) ** 2 for y in outcomes))
    return None if denominator == 0 else numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for column in ("fixed_margin", "dynamic_margin"):
        pairs = [(float(row[column]), float(row["asr"])) for row in rows
                 if row.get(column) not in (None, "") and row.get("asr") not in (None, "")]
        print(f"Corr({column}, ASR) = {correlation(*zip(*pairs)) if pairs else None}")


if __name__ == "__main__":
    main()
