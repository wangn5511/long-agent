# rightArm Agentic MLOps jobs

Install the thin job adapter from this checkout:

```bash
python -m pip install -e .
rightarm-mlops --help
```

Every command writes exactly one terminal manifest to its `--output-dir`:

- success: `result.json` using `rightarm.mlops.job-result/v1`;
- failure: `failure.json` with a stable failure class and retryability flag.

The first production-facing contract is a sealed batch directory containing
`batch.manifest.json`, `annotations.csv`, videos, and a final `_SUCCESS` marker.
The producer contract is machine-readable at
`mlops/contracts/rightarm-gesture-batch-v1.schema.json`.
`inspect-batch` verifies every declared size/hash and the 11-class CSV contract.
`land-batch` creates a canonical content-addressed raw snapshot and a
deterministic `raw-dataset.tar`; the tar, not a pointer to a temporary directory,
is the artifact passed to later runs. Cloud credentials stay in the TOS/COS
materializer outside the Agent; these commands consume its read-only local
materialization.
Landing also emits the exact `source-manifest` snapshot. The outer loop already
sealed the watcher-discovered copy and accepts the raw artifact only when its
lineage starts at the same content identity.

`evaluate` consumes JSONL rows with `truth` and `prediction` and emits versioned
accuracy, macro-F1, per-class recall, and `NONE -> gesture` false-accept metrics.

`build-dataset` can consume either precomputed RTM roots or a sealed
`raw-dataset.tar`. Raw mode runs the repository's fixed clip and RTM scripts,
splits by subject/device/session to prevent leakage, and packages videos plus a
relative-path PoseDataset index into a portable tar. `train` materializes that
tar on the executor, resolves the base checkpoint, resource count, seed and
schedule, and refuses success unless best/last checkpoints, scalar metrics,
the log, and the immutable command record all exist.

The hardware-specific steps (`export-onnx`, `build-calibration`, `export-rknn`,
`verify-conversion`, `verify-board`) wrap an explicit JSON argv without a shell
and verify all declared `KIND=PATH` outputs. They must run in the configured
isolated executor; the local OpenHands task runner disables them unless
`MLOPS_ALLOW_EXTERNAL_JOBS=1` is explicitly set for a trusted staging host.

`rightarm-docker-image-catalog.example.json` is the operator-owned map from job
kind to pinned container digest. Replace the zero digests before use. Container
jobs receive only the read-only repository, explicitly configured read-only
business materialization roots, their own output directory, and the current
attempt's materialized inputs. Network and GPU access are per-kind allowlists.

## Trigger ownership

The integrated distribution uses `openhands.mlops_loop` as the only durable
trigger, scheduler and lifecycle controller. The older
`rightarm_mlops.cos_trigger` module remains only as migration compatibility and
must not be started beside the Loop Controller. Cloud discovery is configured
through the controller's `command` watcher; its default polling period in the
packaged deployment is 24 hours.
