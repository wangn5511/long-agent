# Agentic MLOps V1 completion audit

Audit date: 2026-07-20

Scope: the OpenHands outer/inner-loop integration in this checkout and the
deterministic rightArmPerception job adapters. “Verified” below means the
SQLite/local-CAS path, S3 adapter contract, and isolated Docker executor were
executed with local/fake infrastructure as stated; it does not claim access to
the production TOS, S3 bucket, GPU cluster, RK3588 lab, registry, or fleet APIs.

## Mentor boundary

| Responsibility | Owner | V1 evidence |
|---|---|---|
| discover/trigger | Loop Engineering | continuous local watcher plus operator-owned cloud command watcher |
| dispatch | Loop Engineering | one deterministic OpenHands Conversation id per TaskAttempt |
| action/reasoning | OpenHands Agent | launch-scoped fenced MCP tools, task-kind capabilities, controlled input materialization and immutable TaskContext/RunSpec |
| validation | Loop Engineering | report/hash/contract validators ignore prose success claims |
| persistence | Loop Engineering | append-only events, optimistic version, leases, jobs, artifacts, lineage, release/deployment and rollback projections |
| retry/stop | Loop Engineering | classified bounded attempt retry, fresh Conversation, expired-attempt recovery and persistent consecutive-failure circuit breaker |

The Agent cannot create an attempt, seal a candidate, write a lineage edge,
promote a release, shift traffic, select a rollback target, or reopen a stopped
Loop. Task tools stop accepting calls as soon as the attempt is no longer
RUNNING. A candidate is bound to its attempt when staged, and the production
controller rejects candidates that were not registered through that attempt's
leased gateway.

## rightArm acceptance matrix

| # | Acceptance condition | Status | Executable evidence |
|---|---|---|---|
| 1 | Three watermarks execute exactly once | verified | `test_three_watermarks_are_discovered_exactly_once` |
| 2 | Insufficient data is retained for a later Run | verified | `test_data_inspection_can_finish_without_training_and_preserve_pending_data` |
| 3 | Deployment traces to release/model/eval/code/image/raw data | verified in staging | release provenance closure plus `audit_run`; incomplete closure is rejected |
| 4 | GPU/OOM failure produces an outer retry and spec diff | verified | failed Job observation and exact failure class are required; invented retry evidence is rejected |
| 5 | Identical content has one artifact identity | verified | global `(kind, sha256)` identity and idempotent bootstrap smoke/test |
| 6 | Bad metrics cannot release | verified | evaluation, parity and board validators recompute/read machine reports |
| 7 | Repeated deployment request has one deployment revision | verified | deployment idempotency assertion in the release/deployment lifecycle test |
| 8 | Failed canary rolls back to the proven champion without replacing its pointer | verified | the sealed `UNHEALTHY` observation is persisted before rollback scheduling |
| 9 | Restart/lease expiry creates a fresh attempt and Conversation | verified | expired Conversation recovery plus durable Job launch recovery |
| 10 | Audit answers version, data, failures, jobs, release and rollback | verified | `/api/v1/mlops/runs/{id}/audit` includes replayed projection and all domain records |
| 11 | A multi-task Run uses multiple Conversations | verified | `test_two_tasks_use_two_conversations_and_replay_projection` |
| 12 | Agent prose success without artifacts/metrics is rejected | verified | `test_agent_success_claim_without_artifact_cannot_advance` |

Additional verified boundaries include current-Run-only lineage inputs,
content-hash revalidation, safe tar extraction, grouped train/validation splits,
portable dataset paths, watcher-manifest snapshotting, declared Job output
kind/hash enforcement, managed attempt-owned inputs/outputs, URL-only MCP
injection, task-kind job capabilities, checked-in external scripts,
authenticated human confirmation before production canary, immutable release
sealing, cost/duration/artifact budgets, single-active scheduler lease, and
persistent Loop stop after consecutive failures.

## Business data-to-model path

1. A cloud materializer or local watcher exposes a sealed batch with
   `batch.manifest.json` and `_SUCCESS`.
2. The outer loop snapshots the manifest, checks `batch_id == watermark`, and
   seals it as the Run's source root. `inspect-batch` verifies object size/hash,
   annotation schema and the fixed 11-class ontology; `land-batch` creates a
   deterministic `raw-dataset.tar` whose contents are re-hashed independently.
3. `build-dataset --raw-dataset` safely materializes the snapshot, groups the
   split by subject/device/session, invokes the repository's fixed segment and
   RTM scripts, compiles MMAction V3 data, and emits a portable dataset tar.
4. `train` materializes that tar and freezes dataset/base/config/resources/seed
   into the MMAction command. Success requires best and last checkpoints,
   scalar metrics, log and resolved config.
5. Evaluation/package/board/deploy jobs run only through the fenced executor.
   Each kind is authorized by the current TaskSpec, and only outputs declared by
   a successful structured Job manifest may be staged. Independent validators
   control Release sealing, canary promotion and rollback.

## Verification performed

- OpenHands MLOps, API/model and app-conversation service tests: 281 passed.
- rightArmPerception adapter tests: 8 passed.
- OpenHands pre-commit checks: whitespace, debug checks, Ruff format/lint and
  mypy passed.
- LoopSpec and RunSpec both validate against their JSON Schemas.
- The real `bootstrap-artifact` CLI was invoked twice against one SQLite/CAS
  instance and returned the same sealed artifact id both times.
- An actual local Docker daemon launched the pinned
  `stgcn_openmmlab@sha256:ea6836...` image with the V1 read-only/offline
  sandbox, persisted the expected `release-lineage-incomplete` failure, and the
  test container was removed. Unit tests separately verify digest, mount,
  network, capability, GPU and recovery policy.
- S3 conditional create, remote metadata verification, CAS-prefix enforcement,
  idempotent sealing and hash-verified materialization were exercised against a
  deterministic fake S3 client; no production bucket was contacted.

## Production work that is intentionally not claimed complete

These are implementation or infrastructure inputs, not decisions the Agent may
invent:

- an operator-owned TOS/COS discovery and read-only materialization command;
- registered base checkpoint, frozen test set, RTM ONNX models, champion
  evaluation, code bundle, image descriptor, label mapping and tensor contract;
- an OpenHands deployment with a configured LLM/Agent Profile and network reach
  from its sandbox to the task MCP gateway;
- registry-published pinned production images and a checked-in frozen
  evaluation inference script; the Docker executor implementation itself is
  present and smoke-tested locally;
- checked-in ONNX/RKNN/parity/board scripts plus RK3588 lab access;
- checked-in deploy/observe/rollback adapters plus fleet credentials and policy
  owner approval;
- a production PostgreSQL ledger adapter and migration, S3 bucket
  versioning/Object Lock and write-once bucket policy, OpenTelemetry export,
  and an automatic OpenHands/GPU billing feed. The S3 adapter, durable scheduler
  lease, and cost ledger/budget enforcement are implemented, but production
  endpoints were not supplied or exercised.

Until those items are supplied and exercised, the correct claim is “staging
closed loop implemented and verified,” not “production gesture deployment is
live.”
