# OpenHands task: Tencent COS to a trained right-arm gesture model

This is the rightArm domain execution contract used by the bundled OpenHands
MLOps loop. Complete the pipeline in order and without routine human
confirmation. The central Loop Controller owns scheduling, durable state,
validation, retry and release decisions; an individual task attempt must not
create a second scheduler or bypass that controller.

This file is not a trigger by itself. The Loop Controller creates an OpenHands
conversation for a fenced task attempt. Once started, continue through
recoverable errors, make bounded fixes, and submit structured evidence through
the task MCP tools. Stop only for an evidenced external blocker such as invalid
credentials, insufficient disk after safe cleanup, unavailable GPU, an
ambiguous data ontology, or corrupt source objects that cannot be recovered.
Never invent success.

## Objective and authorized scope

Read the documented right-arm gesture source from Tencent COS, create a verified
local snapshot, safely extract and clean it, generate a new PKL, merge that PKL
with the existing trusted 11-class baseline, then smoke-test, train, evaluate,
and export a candidate model.

The task authorizes read-only COS operations under this exact prefix:

`cos://cytoderm-ai-shanghai-1389948562/datasets/dynamic_gestures_2603/`

Full download of that prefix is authorized after the inventory and disk-space
gates pass. Do not list unrelated prefixes, upload, overwrite, move, or delete
any COS object. Do not deploy the resulting model or restart robot/perception
services.

## Fixed inputs and paths

- Repository: `/workspace/rightArmPerception`
- Private COSCLI configuration: `/run/secrets/tencent-cos.yaml`
- COSCLI executable: `coscli` from the operator-owned execution image
- Output root: the attempt-specific `/mlops-output` directory
- Baseline dataset: the immutable dataset artifact materialized for the attempt
- Baseline checkpoint: the immutable checkpoint artifact materialized for the
  attempt
- Baseline training config:
  `training/mmaction2/handoff/stgcn_hand21_3d/configs/stgcn_hand21-rgb-bs64-gated-v3-ft-rtm_add-11cls.py`
- MMAction2 root in the training container:
  `/workspace/rightArmPerception/training/mmaction2`
- GPU wrapper: `/workspace/rightArmPerception/mlops/run_training_container.sh`
- Training image: `stgcn_openmmlab:v1`
- Detection ONNX: immutable task input `rtm-detector-artifact`
- Upper-body pose ONNX: immutable task input `rtm-person-pose-artifact`
- Hand pose ONNX: immutable task input `rtm-hand-pose-artifact`

Create one UTC timestamped run directory below the output root and keep every
mutable artifact there. Use this layout:

```text
<run>/
  source/remote-inventory.json
  source/download/
  source/download-inventory.json
  source/extracted/
  source/extraction-report.json
  data/coverage/
  data/splits/
  data/prepared/
  data/new-batch.pkl
  data/new-batch-report.json
  data/merged-gesture-11.pkl
  data/merge-report.json
  training/smoke/
  training/full/
  evaluation/
  export/
  resolved_task.md
  run_manifest.json
  training_report.md
```

## Secrets and safety rules

1. Never print, copy, summarize, commit, or place Tencent credentials in a
   command argument, environment dump, task report, repository file, or training
   container. Use COSCLI only with `--config-path` pointing at the private file.
2. Verify that the private configuration is owned by the current user and has
   mode `0600`; fail closed if it is more permissive. Never run `coscli config
   show` because it could disclose secrets.
3. Start with `git status --short`. The checkout already contains user changes;
   never reset, discard, overwrite, or commit unrelated changes.
4. Treat COS, `/home/datasets`, the baseline PKL, and the baseline checkpoint as
   read-only. Never overwrite `rtm_add.pkl`.
5. Before downloading, record the complete recursive object inventory and total
   bytes. Require free bytes on the destination filesystem to be at least three
   times the remote total plus 100 GiB. If this gate fails, report the numbers
   and stop without deleting user data.
6. Never load a PKL obtained from COS. Python pickle can execute code. Only load
   the fixed trusted local baseline and PKLs generated locally by this run.
7. Use GPU 0 by default. Check `nvidia-smi` before GPU work and do not interrupt
   an existing process. Do not install or upgrade host GPU drivers.
8. Verify files and reports, not console prose. Record size and SHA-256 for every
   final PKL, checkpoint, ONNX/RKNN export, resolved config, and report.

## Canonical data contract

The merged dataset must use this exact ordered ontology:

```text
WAVE_IN WAVE_OUT FIVE CLICK ROTATE SLIDE_DOWN SLIDE_UP OK CURSOR GUN NONE
```

The documented COS batch currently contains a nine-class subset. Missing
`WAVE_OUT` or `GUN` samples in the new batch are allowed because the baseline may
provide them. Unknown labels are never allowed. Map numeric labels through each
input PKL's category names and then into the canonical order; never assume that
numeric IDs from two PKLs mean the same thing.

The source annotation format is `compact_4col`, UTF-8/UTF-8-SIG, one-based frame
numbers, with columns video name, start frame, end frame, and label. A blank
video-name cell inherits the preceding non-empty video name.

## Required execution order

### 1. Preflight, remote inventory, and run manifest

- Record the repository revision and dirty status, host/container environment,
  COSCLI version, UTC start time, and all resolved input/output paths.
- Confirm that the fixed inputs and three ONNX files are readable.
- Perform a small read-only COS listing using the private config, followed by a
  recursive listing of only the authorized prefix. Save parsed object key, size,
  modification time, and ETag when available into `remote-inventory.json`.
- Reject unexpected object types. The expected inputs are `annotations.csv`,
  video archives, and documented metadata; no remote PKL or executable is an
  input to this task.
- Apply the disk-space gate before copying anything large.

### 2. Resumable, read-only download

- Download into `source/download/` without `--move` and without any upload or
  synchronization mode that can delete either side. Use COSCLI's resumable copy
  behavior when available.
- Write incomplete objects to temporary/partial paths. Promote them only after
  COSCLI succeeds and the local size matches the remote inventory.
- Retry transient network failures up to three times with bounded backoff. Do
  not retry authentication/authorization failures indefinitely.
- Compute local SHA-256 values and create `download-inventory.json`. Require the
  remote and local object counts, relative keys, and sizes to match before
  extraction. ETag is supporting evidence only and must not be treated as MD5
  for multipart objects.

### 3. Safe extraction and immutable source snapshot

- Never run an unguarded archive extraction. Reuse the repository's safe archive
  logic or implement an equivalent checked extractor that rejects absolute
  paths, `..` traversal, symlinks, hardlinks, devices, FIFOs, and expansion
  outside the run directory.
- Extract each `.tar.gz` into its own directory under `source/extracted/` so that
  same-named files from different archives cannot silently overwrite each other.
- Pre-compute an expanded-size limit and continuously check free space. On any
  collision: deduplicate only byte-identical files; quarantine and hard-fail on
  different content with the same normalized target identity.
- Keep downloaded archives immutable. Record archive SHA-256, member counts,
  extracted bytes, rejected members, duplicates, and destination paths in
  `extraction-report.json`.

### 4. Annotation validation, cleaning, and leakage-safe split

- Use the checked-in scripts under `data/hand21/` and the validation code under
  `rightarm_mlops/`; do not create an unrelated replacement pipeline.
- Parse annotations and validate required fields, inherited video names, integer
  bounds, `start <= end`, one-based frame semantics, known labels, duplicate
  spans, and conflicting labels. Validate each span against the real video frame
  count where decoding permits.
- Run the annotation/video coverage report. Missing annotated videos, unreadable
  videos, conflicting labels, and unknown labels are hard failures. Unannotated
  videos are reported and excluded unless they belong to an explicitly
  documented `NONE` pool.
- Split by source video and, when available, by subject/session/device group;
  never split clips from the same source group across train and validation. Use
  seed 42 and validation ratio 0.2. Force the documented low-light validation
  archive into validation. Preserve any frozen test set and never train on it.
- Prepare RTM clips/keypoints with the three fixed ONNX models. Start with a
  bounded sample, then process the full accepted set. Retain per-sample failure
  reasons and quality measurements.
- Apply the checked-in thresholds: minimum 8 frames, minimum 6 valid frames,
  primary valid ratio at least 0.5, low-quality threshold 0.9, other-track
  coverage threshold 0.3, maximum 5 consecutive missing frames, and maximum
  interpolation gap 5. Run label-whitelist cleanup first as a dry run, record
  the proposed removals, then apply it only to run-local prepared data.
- Mine `NONE` clips only from documented `NONE` pools. Keep their source groups
  in exactly one split and report the resulting class balance.

### 5. Generate and validate the new PKL

- Use `data/hand21/convert_hand21_json_to_pkl.py` or the repository's
  `rightarm-mlops build-dataset` path, depending on the normalized source format.
  Reuse checked-in conversion code; make only bounded, tested fixes if required.
- Write first to a temporary file and atomically rename it to
  `data/new-batch.pkl`. Reopen the locally generated PKL and validate `split`,
  `annotations`, `categories`, label ranges, tensor/keypoint shapes, referenced
  media, sample IDs, and disjoint train/validation membership.
- Produce `new-batch-report.json` with total/accepted/rejected counts, rejection
  reasons, counts per split and class, quality distributions, source groups, and
  the PKL size and SHA-256.

### 6. Merge with the trusted baseline PKL

- If no checked-in merge helper satisfies this section, add one focused helper
  under `rightarm_mlops/` plus tests. Do not edit either input PKL.
- Load only the fixed trusted baseline and the locally generated new PKL. Validate
  both before merging. Remap labels by category name into the canonical 11-class
  order.
- Preserve all baseline split assignments. Preserve new-data group assignments.
  Never move baseline validation/test samples into training.
- Define duplicate identity from stable source video hash plus clip frame range
  and canonical label where available, with a deterministic annotation-content
  fallback. Deduplicate byte/content-identical samples. Hard-fail when the same
  identity has conflicting labels, different content, or appears across splits.
- Namespace colliding display/sample IDs deterministically. Require all final
  IDs to be unique, train/validation/test sets to be disjoint, every annotation
  to appear in exactly one declared split, and every label ID to resolve to the
  canonical category list.
- Atomically write `data/merged-gesture-11.pkl`, reopen it, rerun all invariants,
  and write `merge-report.json` with before/after counts, class/split counts,
  remappings, deduplications, conflicts, provenance, size, and SHA-256.

### 7. Dataset acceptance gate

Do not train until all of these pass:

- Remote/local inventory and extraction reports are complete.
- Annotation coverage has no missing referenced video and no label conflict.
- The merged PKL has exactly the canonical 11 categories and valid label IDs.
- Split intersections are empty at sample, source-video, and known group levels.
- Both train and validation are non-empty; the combined dataset retains all 11
  classes in training unless the baseline itself is proven otherwise.
- Per-class changes relative to the baseline and all rejected/low-quality samples
  are visible in the report. Do not silently accept a major class collapse.

### 8. Container check and one-epoch smoke run

- Check PyTorch, CUDA, MMAction2 imports, GPU visibility, config/checkpoint
  compatibility, and readability of the merged PKL inside the container.
- Create a run-specific config or command override that points to
  `merged-gesture-11.pkl`; never modify the baseline config in place.
- Run one epoch on GPU 0 with conservative batch size/workers. Confirm a real
  checkpoint, scalar metrics, and logs. Diagnose bounded code/config failures,
  record each fix, and repeat the smoke run until it passes or an external
  blocker is proven.

### 9. Full fine-tuning and evaluation

- Only after the smoke gate passes, fine-tune from the fixed baseline checkpoint.
  Preserve the exact resolved config, command, environment, stdout/stderr, and
  every checkpoint. Resume from the latest valid run-local checkpoint after an
  interruption rather than restarting from scratch.
- Evaluate the best candidate on the fixed validation/test split. Report top-1
  accuracy, macro-F1, per-class precision/recall/confusion matrix when supported,
  and especially the `NONE -> gesture` false-accept rate. Compare against the
  baseline on the same split; do not claim improvement from incomparable runs.

### 10. Export and final evidence

- Export the accepted checkpoint to ONNX using the checked-in handoff tool and
  run ONNX inference validation on representative samples.
- Attempt RKNN only when the checked-in tool and `rknn.api` are available. An
  unavailable dependency is an explicit blocker, not a fabricated success.
- Complete `run_manifest.json` and `training_report.md` with inputs, inventories,
  code revision/dirty status, commands with secrets redacted, environment,
  dataset statistics, fixes/retries, metrics, outputs, sizes, SHA-256 hashes,
  elapsed times, and remaining blockers.

## Autonomous behavior and stop conditions

- Continue automatically between stages; ordinary stage completion does not
  require a person to click “continue”.
- For a recoverable code/config/data-path error, preserve evidence, make the
  smallest in-scope change, run focused checks, and retry the failed stage.
- Never weaken a validation gate merely to make the pipeline pass.
- Stop and report one precise blocker when proceeding would require broader COS
  permissions, deleting user data, changing the label ontology, selecting among
  genuinely ambiguous source layouts, or deploying/replacing a production
  model.
- This task does not provide scheduled or event-driven wake-up. A separate
  external trigger is required to start a new OpenHands conversation when new
  COS data arrives or after the OpenHands service itself restarts.

## Done when

- The authorized COS prefix has a verified local snapshot and extraction report.
- Cleaning, quality, coverage, split, new-PKL, and merge reports all pass.
- `merged-gesture-11.pkl` exists, reopens successfully, and has a recorded hash.
- The GPU smoke run passes.
- A full training run completes, or a precise external blocker with evidence is
  recorded after all safe in-scope recovery attempts.
- Evaluation and ONNX validation are recorded; RKNN is either validated or
  explicitly blocked.
- All artifacts are present under one timestamped run directory, and no source,
  baseline, cloud object, or production service was modified.
