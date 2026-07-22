# Agentic MLOps V1

This implementation follows a two-loop contract:

- Loop Engineering discovers a sealed data watermark, creates a Run and tasks,
  dispatches one attempt per OpenHands Conversation, independently validates
  results, persists events/artifacts/lineage, and decides retry or stop.
- An OpenHands task Agent reasons and acts inside one attempt, launches
  allowlisted jobs, stages candidates, proposes lineage, and submits a typed
  `TaskResult`. Its final answer is never a lifecycle transition.

## Implemented path

`openhands.mlops_loop` contains the durable domain kernel, SQLite event ledger,
local or S3-compatible content-addressed store, source watcher, retry/recovery reconciler,
release/deployment invariants, REST boundary, and OpenHands dispatcher.
`openhands.app_server.mlops_task` contains the fenced task client and MCP tools.

SQLite is the local/staging ledger. The S3 adapter seals objects with a
conditional create under `objects/sha256`, verifies remote metadata, and
re-hashes controlled task-local downloads. Local subprocess and isolated
Docker executors share the same job protocol; the Docker executor requires
operator-pinned image digests and is read-only, capability-free, and offline by
default.

The right-arm repository supplies the model-specific `rightarm-mlops` jobs:
sealed batch inspection, verified landing, product evaluation, reproducible
external command wrappers, and release-manifest construction.

## Run the local continuous loop

```bash
python -m openhands.mlops_loop serve \
  --database /var/lib/rightarm-mlops/ledger.db \
  --artifact-root /var/lib/rightarm-mlops/cas \
  --openhands-url http://openhands:3000 \
  --task-mcp-url http://loop-controller:8765/mcp/ \
  --job-workspace /workspace/rightArmPerception \
  --job-root /var/lib/rightarm-mlops/jobs \
  --job-python /app/.venv/bin/python \
  --allow-external-jobs \
  --external-job-catalog /app-config/jobs/external-jobs.json \
  --operator-token-file /run/secrets/rightarm-mlops-operator-token \
  --operator-identity rightarm-release-owner \
  --incoming /var/lib/rightarm-mlops/incoming \
  --loop-spec openhands/mlops_loop/specs/right-arm-gesture-loop-v1-loop.yaml \
  --run-template openhands/mlops_loop/specs/right-arm-gesture-loop-v1-run.yaml \
  --set champion-base-model-artifact=base-model:sha256:REPLACE_ME \
  --set frozen-test-set-artifact=frozen-test-set:sha256:REPLACE_ME \
  --set rtm-detector-artifact=rtm-detector:sha256:REPLACE_ME \
  --set rtm-person-pose-artifact=rtm-person-pose:sha256:REPLACE_ME \
  --set rtm-hand-pose-artifact=rtm-hand-pose:sha256:REPLACE_ME \
  --set champion-evaluation-artifact=evaluation-report:sha256:REPLACE_ME \
  --set code-artifact=code:sha256:REPLACE_ME \
  --set container-image-artifact=container-image:sha256:REPLACE_ME \
  --set label-mapping-artifact=label-mapping:sha256:REPLACE_ME \
  --set tensor-contract-artifact=tensor-contract:sha256:REPLACE_ME \
  --set image-digest=REPLACE_ME \
  --poll-interval-seconds 86400 \
  --host 0.0.0.0
```

This single process mounts the fenced task MCP gateway at `/mcp`, runs the
continuous reconciler, and starts one fresh OpenHands Conversation for every
task attempt. Job requests, status changes, log hashes, artifacts, lineage,
release/deployment state, and rollback are included in the Run audit endpoint.
The supplied V1 LoopSpec uses the caller's active OpenHands Agent Profile
(`agentProfile: default`); replace it with stored profile UUIDs when specialized
profiles have been registered.
`--allow-external-jobs` is appropriate only on an isolated executor; even when
enabled, the gateway accepts only checked-in workspace scripts whose argv
prefix is pinned by the operator-owned catalog, not arbitrary host commands.
The repository includes a partial conversion catalog example; board and fleet
entries remain absent until their real adapters are checked in.
When `--operator-token-file` is configured, operator reads and mutations require
the `X-MLOps-Operator-Token` header. A non-loopback bind refuses to start
without it. Keep this token outside Agent workspaces; task results continue to
use their separate expiring attempt lease. Approval audit identity comes from
`--operator-identity`, not from Agent or request content.

For an S3-compatible CAS and Docker jobs, replace the local storage/executor
arguments with:

```bash
  --artifact-backend s3 \
  --s3-bucket rightarm-mlops \
  --s3-prefix gesture-v1 \
  --s3-endpoint-url https://S3_ENDPOINT \
  --s3-region-name REGION \
  --executor docker \
  --docker-image-catalog /app-config/generated/docker-images.json \
  --docker-source-root /var/lib/rightarm-mlops/cloud-materializations \
  --docker-gpu-count build-dataset=1 \
  --docker-gpu-count train=4
```

Replace every zero digest in the example image catalog with the registry's
actual immutable digest. Add `--docker-network-job-kind` only for the named
board/fleet adapter that truly needs network access. The Agent cannot choose an
image, mount, network mode, GPU entitlement, or S3 credential.

Register each pre-existing immutable input once with `bootstrap-artifact`; it
goes through the same hash validator and event ledger and prints the artifact id
used by the RunSpec replacement, for example:

```bash
python -m openhands.mlops_loop bootstrap-artifact \
  --database /var/lib/rightarm-mlops/ledger.db \
  --artifact-root /var/lib/rightarm-mlops/cas \
  --kind base-model \
  --path /srv/bootstrap/champion.pth
```

For TOS/COS, replace `--incoming` with `--watch-command-json '[...]'`. The
operator-owned command receives cloud credentials outside the Agent and prints
a JSON array of `{watermark, source_uri, manifest_uri}` records. It is executed
as an argv without a shell. In V1, `source_uri` must identify the adapter's
read-only local materialization (`file://...`); the Agent never receives cloud
credentials. `poll-local-once` remains available for one-shot staging
diagnostics.

At discovery time the outer loop copies and seals `batch.manifest.json`, checks
that its `batch_id` equals the watermark, and injects its artifact id into the
Run. The accepted raw dataset must have an explicit lineage edge from that
snapshot. Artifact staging is allowed only when a successful structured job
manifest declared the same output path, kind, size, and hash.

## Production inputs still required

No implementation can safely infer the TOS/COS region and prefix, current
champion artifact, GPU executor, RK3588 lab endpoint, or fleet/OTA API. Until
those adapters and credentials references are supplied, the system can prove
the staging lifecycle and domain invariants but must not claim a production
robot deployment.
