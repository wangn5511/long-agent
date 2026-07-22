# rightArmPerception model workspace

This directory is the model-specific workspace bundled by `model-agent`.
OpenHands reasons about and repairs this code; the central Loop Controller
launches the deterministic jobs under `rightarm_mlops/`, validates their
structured results and records artifacts and lineage.

For cloning, Docker startup, VSCode SSH port forwarding and LLM configuration,
read [`../../MODEL_AGENT.md`](../../MODEL_AGENT.md).

## Pipeline coverage

```text
Tencent COS sealed batch
  -> inventory and integrity validation
  -> safe extraction, annotation cleaning and leakage-safe split
  -> new PKL generation
  -> merge with the trusted 11-class baseline PKL
  -> smoke training and full training
  -> evaluation
  -> ONNX / RKNN export and consistency checks
  -> report and artifact archival
```

Production deployment remains a separate gated task and requires explicit
human confirmation.

## Layout

```text
rightArmPerception/
├── data/                     # data preparation and conversion
├── training/                 # MMAction2, Ultralytics and model training
├── deployment/               # MMDeploy, ONNX and RKNN tooling
├── inference/                # inference code
├── rightarm_mlops/           # deterministic MLOps job adapters
├── mlops/                    # domain contracts and allowlisted wrappers
└── tests/                    # focused MLOps tests
```

## Local adapter checks

From this directory:

```bash
python3 -m pip install -e .
rightarm-mlops --help
python3 -m pytest -q tests/test_cos_trigger.py tests/test_rightarm_mlops.py
```

The packaged deployment mounts this directory at
`/workspace/rightArmPerception` for the controller and at
`/opt/workspace_base/rightArmPerception` for OpenHands.

Do not add COS or LLM credentials, datasets, PKLs, checkpoints, ONNX/RKNN
exports, logs or run reports to this source tree. Those values belong in the
operator secret mounts and the content-addressed artifact store.
