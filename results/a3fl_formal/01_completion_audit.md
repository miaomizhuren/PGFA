# A3FL-v2-Relaxed-Strong completion audit

- Formal database: `SOURCE_ROOT/outputs/a3fl_v2_relaxed_strong_seed2026/metrics_a3fl_v2_relaxed_strong_seed2026_resume3000.db`
- Configuration snapshot: `SOURCE_ROOT/outputs/a3fl_v2_relaxed_strong_seed2026/config_snapshot_a3fl_v2_relaxed_strong_seed2026_resume3000.yaml`
- Log: `SOURCE_ROOT/experiments/logs/a3fl_v2_relaxed_strong/a3fl_v2_relaxed_strong_seed2026_resume3000.log`
- Exit-code record: `SOURCE_ROOT/experiments/logs/a3fl_v2_relaxed_strong/a3fl_v2_relaxed_strong_seed2026_resume3000.exitcode` → `0`.

## Completion checks

| Check | Result | Evidence |
|---|---|---|
| Covers R3001–R3700 | PASS | 700 rows, 700 unique; min=3001, max=3700 |
| Attack rows | PASS | R3001–R3200 all `attack_active=1` |
| Post rows | PASS | R3201–R3700 all `attack_active=0` |
| Finite saved metrics | PASS | 0 non-finite values in ASR/CleanAcc/PSNR/SSIM/L∞ |
| Exit status | PASS | wrapper wrote `0` after Python exited |
| R3000 clean checkpoint | PASS | config snapshot specifies `outputs/clean_resume_force_clean500_to3000_bs64_e3_save500/checkpoints/global_round_3000.pt` |
| UAP refresh during attack | PASS | 200 `[A3FLv2Refresh]` records with `attack_active=True, outer_steps=200`; expected 200 |
| Frozen UAP post-attack | PASS | 500 post `[A3FLv2Round]` records say `delta_refreshed=False`, `delta_source_round=3200`, `using_delta_round=3200` |
| Post attack state | PASS | 500 post `LocalEpochPolicy` records say `attack_active=False`; no forced malicious clients |
| Runtime error signatures | PASS | no Traceback/OOM/CUDA OOM/KeyboardInterrupt in log |

R3200 and R3201 log records explicitly show the boundary: R3200 is refreshed using delta source 3200; R3201 is not refreshed and evaluates source 3200. The global checkpoints/delta files saved later retain this source (e.g. `delta_round_3700.pt` is logged with source round 3200).

## Protocol comparability with historical v2

The recorded client selection lists match historical stitched v2 in **700/700** rounds; recorded selected-malicious lists match in **700/700** rounds. The new configuration and historical v2 both use the same clean-3000 checkpoint, IID CIFAR-10/ResNet18, 100 clients, 10 selected per round, IDs [0,1,2], 3 local epochs, target 0, 3001–3200 attack and 3201–3700 observation. The attack resource budget differs substantially; this is a common federation protocol match, not a fully resource-matched comparison.
