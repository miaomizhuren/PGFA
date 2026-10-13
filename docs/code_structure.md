# Code structure

The training entry point is train.py. It loads YAML, builds AttackConfig, partitions the requested public dataset, creates clients, and runs the server loop.

The formal PGFA path is:

1. models/resnet18.py provides forward_features, including the layer-4 representation.
2. attacks/prototype.py collects clean local class statistics and constructs prototype banks.
3. attacks/adversarial_trigger.py computes target-versus-source prototype loss and fixed targeted UAP updates.
4. federated/malicious_client.py shares the UAP and applies PGFA feature loss during malicious local training.
5. federated/server.py and federated/aggregation.py coordinate clients and FedAvg/defense aggregation.
6. evaluation/evaluator.py reports clean accuracy, ASR, and support metrics.

Result files in results/ are exports of existing SQLite round_metrics tables. They contain real recorded values and a source_sqlite field; no inference, interpolation, model loading, training, or re-evaluation was performed during release construction.

For DBA, the formal R3001--R3700 trajectory merges the original R3001--R3590 database with the verified R3501--R3700 continuation database. Later continuation rows take precedence for overlapping rounds; both source paths are preserved in the manifest.
