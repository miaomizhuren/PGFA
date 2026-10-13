# Final public-release review

- Rebuilt `results/paper_summaries/01_main_results_cifar10.csv` directly from the five public formal roundwise CSVs.  Its A3FL row is `A3FL-v2-Relaxed-Strong` displayed as `A3FL`; its PA-AUC is 80.042822222222, N80 is 240, and ASR@+500 is 65.244444444444.
- Removed the misleading release-copy row/table provenance for historical `A3FL-v4 faithful UAP`; the original project was not changed.
- Reclassified the prior `formal_perturbation_audit.csv` as `roundwise_metric_snapshot_R3200.csv` and created a provenance-indexed formal audit table plus method note.  The old historical A3FL offline-audit row is excluded.
- Removed `results/five_seeds/verified_summaries/all_9groups_final_metrics.csv` from public results.  It mixed five-seed, Non-IID, and loss-ablation entries under a misleading five-seed location; all underlying verified per-experiment results remain present.
- Removed its standalone historical generator `scripts/export_pgfa_final_9groups_merged.py`; it was not imported by the formal training or release-export code.
- Removed standalone historical/diagnostic smoke scripts and Chameleon/F3BA-only tests that no formal released YAML invokes.  Generic optional Chameleon/Focused-Flip branches remain in the shared framework because `train.py` still imports them; they are not presented as PGFA-paper results.
- Corrected the formal expansion of PGFA to **Prototype-Guided Feature Anchoring**.  `Ordinary Feature Alignment` remains only as its ablation label.
- Preserved all 34 verified formal experiment exports, 34 YAML configurations, A3FL full trajectory, five-seed, Non-IID, ablation, defense, cross-dataset, sensitivity, and feature-analysis materials.
