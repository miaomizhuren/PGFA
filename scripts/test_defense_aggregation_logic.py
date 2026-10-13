"""Pure in-memory pre-run checks for server aggregation defenses; no training/data."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from federated.aggregation import aggregate_with_defense

OUT = ROOT / "outputs/saved_results/final_defense_60_150_r350_20260910/DEFENSE_AGGREGATION_UNIT_TEST_REPORT.md"

def run(name, config, states, base):
    result, diagnostic = aggregate_with_defense(states, [1] * 10, base, ["weight", "bias"], config, client_ids=list(range(10)), return_diagnostics=True)
    assert result["weight"].shape == base["weight"].shape and result["bias"].shape == base["bias"].shape
    return diagnostic

def main():
    base = {"weight": torch.zeros(2, 2), "bias": torch.zeros(2)}
    states = []
    for index in range(10):
        value = 20.0 if index in [0, 1, 2] else 0.1 + index * .01
        states.append({"weight": torch.full((2, 2), value), "bias": torch.full((2,), value)})
    none = run("none", {"enabled": False, "name": "none"}, states, base)
    clip = run("clip", {"enabled": True, "name": "norm_clipping", "params": {"clip_norm": 1.0}}, states, base)
    assert clip["clipped_clients"] > 0
    trim = run("trim", {"enabled": True, "name": "trimmed_mean", "params": {"trim_ratio": .2}}, states, base)
    assert trim["trim_k"] == 2 and trim["kept"] == 6
    krum = run("krum", {"enabled": True, "name": "multi_krum", "params": {"f": 3, "m": 5}}, states, base)
    assert krum["nearest_count"] == 5 and len(krum["selected_client_ids"]) == 5
    flame0 = run("flame0", {"enabled": True, "name": "flame", "params": {"noise_std": 0.0}}, states, base)
    flame = run("flame", {"enabled": True, "name": "flame", "params": {"noise_std": .001}}, states, base)
    assert flame0["kept_client_ids"] and flame0["clip_norm"] > 0 and flame["noise_std"] == .001
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# Defense Aggregation Unit Test Report\n\nPASS: none, norm clipping, trimmed mean, Multi-Krum (n=10, f=3, m=5, nearest_count=5), and Light-FLAME (clustering, median-norm clipping, noise=0/0.001) passed in-memory shape and diagnostic checks.\n", encoding="utf-8")
    print("PASS: defense aggregation unit tests")

if __name__ == "__main__": main()
