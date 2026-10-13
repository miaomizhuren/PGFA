# PGFA Research Code and Results Release

This is a research-code and formal-results release for PGFA (Prototype-Guided Feature Anchoring) persistent-backdoor experiments in federated learning. It includes the actual PGFA implementation, the shared FedAvg framework and baselines, selected formal YAML configurations, and verified lightweight per-round metrics.

## What is included

- PGFA prototype-bank construction, target-versus-source feature loss, fixed UAP generation, and malicious-client feature training.
- FedAvg server/client code, ResNet-18, data loading, non-IID partitioning, evaluation, defenses, and experiment tracking.
- CE-only UAP, DBA, Neurotoxin-UAP, and the formal A3FL-v2-Relaxed-Strong implementation (displayed as A3FL in the paper).
- 34 verified SQLite-derived roundwise metric exports, each paired with its copied formal configuration and source metadata.
- Formal main, five-seed, stage-ablation, loss-form-ablation, cross-dataset, non-IID, defense, and attack-duration result groups.
- Feature-persistence summaries, selection indices, formal perturbation metrics, and original analysis/export/plot Python scripts.

docs/experiment_manifest.csv is the authority for result provenance. Each row records the original configuration, original SQLite source(s), copied configuration, exported result path, calculated metrics, and verification status. docs/reused_experiment_roles.csv records the single verified PGFA/CE source reused by multiple paper tables.

## Main-result verification

The main A3FL result is the formal A3FL-v2-Relaxed-Strong run, exported from:

outputs/a3fl_v2_relaxed_strong_seed2026/metrics_a3fl_v2_relaxed_strong_seed2026_resume3000.db

It verifies PA-AUC 80.04%, ASR@+500 65.24%, N80=240, N90=136, and its five reported post-attack checkpoints. Historical A3FL variants and their result files are not included.

## Data and checkpoint policy

No checkpoint, model weight, optimizer state, raw CIFAR-10/GTSRB/SVHN data, feature cache, virtual environment, or training log is included. The copied YAML files retain experiment settings but require users to provide public data and any needed initialization checkpoint themselves. This release therefore supports audit of released metrics and code inspection; it is not a turnkey end-to-end reproduction environment.

## Layout

- attacks/, federated/, models/, data/, evaluation/, utils/: implementation and framework.
- scripts/: original Python analysis, CSV-export, and plotting utilities, plus the read-only release exporter.
- configs/: formal result-associated YAML files.
- results/: lightweight exported metrics and selected formal summaries, organized by paper result group.
- docs/: experiment and reuse manifests plus code-path documentation.

## Environment

Install PyTorch and torchvision appropriate for the desired CPU/CUDA environment, then install the remaining original dependencies:

pip install -r requirements.txt

No license has been added; choose one only after confirming the intended terms and third-party baseline status.

## Configuration validation note

The formal main PGFA and A3FL YAML files are syntactically present and provenance-checked, but invoking train.py --dry-run still checks their historical R3000 resume checkpoint. That checkpoint is intentionally excluded, so the command stops with a missing-checkpoint message before training.
