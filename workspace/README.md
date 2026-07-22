# Bundled model workspace

`rightArmPerception/` is a vendored model workspace, not a nested Git checkout.
OpenHands receives it as its writable workspace; the Loop Controller mounts the
same source for deterministic `rightarm-mlops` jobs.

Generated datasets, checkpoints, model exports, logs and local environments are
ignored by the root `.gitignore`. They belong in the controller artifact store
or the per-user `.model-agent` state directory, not in this source tree.
