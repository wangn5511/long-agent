#!/usr/bin/env bash
set -euo pipefail

: "${MLOPS_DATABASE:=/var/lib/rightarm-mlops/ledger.db}"
: "${MLOPS_ARTIFACT_ROOT:=/var/lib/rightarm-mlops/cas}"
: "${MLOPS_JOB_ROOT:=/var/lib/rightarm-mlops/jobs}"
: "${MLOPS_JOB_WORKSPACE:=/workspace/rightArmPerception}"
: "${MLOPS_EXTERNAL_JOB_CATALOG:=/app-config/jobs/external-jobs.json}"
: "${MLOPS_OPERATOR_TOKEN_FILE:=/run/secrets/mlops_operator_token}"
: "${MLOPS_OPERATOR_IDENTITY:=model-agent-operator}"
: "${MLOPS_OPERATOR_IMPORT_ROOT:=/var/lib/rightarm-mlops/imports}"
: "${MLOPS_TRIGGER_MODE:=manual}"
: "${MLOPS_POLL_INTERVAL_SECONDS:=86400}"
: "${MLOPS_RECONCILE_INTERVAL_SECONDS:=5}"
: "${MLOPS_HOST:=0.0.0.0}"
: "${MLOPS_PORT:=8765}"
: "${OPENHANDS_INTERNAL_URL:=http://openhands:3000}"
: "${MLOPS_TASK_MCP_URL:=http://loop-controller:8765/mcp/}"
: "${MLOPS_EXECUTOR:=local}"

job_python="${MLOPS_JOB_PYTHON:-$(command -v python)}"
if [[ -z "${job_python}" || ! -x "${job_python}" ]]; then
  echo "MLOPS_JOB_PYTHON must resolve to an executable Python interpreter" >&2
  exit 1
fi

args=(
  python -m openhands.mlops_loop serve
  --database "${MLOPS_DATABASE}"
  --artifact-root "${MLOPS_ARTIFACT_ROOT}"
  --openhands-url "${OPENHANDS_INTERNAL_URL}"
  --task-mcp-url "${MLOPS_TASK_MCP_URL}"
  --job-workspace "${MLOPS_JOB_WORKSPACE}"
  --job-root "${MLOPS_JOB_ROOT}"
  --job-python "${job_python}"
  --executor "${MLOPS_EXECUTOR}"
  --allow-external-jobs
  --external-job-catalog "${MLOPS_EXTERNAL_JOB_CATALOG}"
  --operator-token-file "${MLOPS_OPERATOR_TOKEN_FILE}"
  --operator-identity "${MLOPS_OPERATOR_IDENTITY}"
  --operator-import-root "${MLOPS_OPERATOR_IMPORT_ROOT}"
  --reconcile-interval-seconds "${MLOPS_RECONCILE_INTERVAL_SECONDS}"
  --host "${MLOPS_HOST}"
  --port "${MLOPS_PORT}"
)

if [[ -n "${RIGHTARM_REPOSITORY:-}" ]]; then
  args+=(--repository "${RIGHTARM_REPOSITORY}")
fi
if [[ -n "${RIGHTARM_BRANCH:-}" ]]; then
  args+=(--branch "${RIGHTARM_BRANCH}")
fi

case "${MLOPS_EXECUTOR}" in
  local)
    ;;
  docker|hybrid)
    : "${MLOPS_DOCKER_IMAGE_CATALOG:?required when MLOPS_EXECUTOR is docker or hybrid}"
    args+=(--docker-image-catalog "${MLOPS_DOCKER_IMAGE_CATALOG}")
    if [[ -n "${MLOPS_DOCKER_HOST_JOB_ROOT:-}" ]]; then
      : "${MLOPS_DOCKER_HOST_WORKSPACE:?required with MLOPS_DOCKER_HOST_JOB_ROOT}"
      args+=(
        --docker-host-job-root "${MLOPS_DOCKER_HOST_JOB_ROOT}"
        --docker-host-workspace "${MLOPS_DOCKER_HOST_WORKSPACE}"
      )
    elif [[ -n "${MLOPS_DOCKER_HOST_WORKSPACE:-}" ]]; then
      echo "MLOPS_DOCKER_HOST_JOB_ROOT is required with MLOPS_DOCKER_HOST_WORKSPACE" >&2
      exit 1
    fi
    if [[ -n "${MLOPS_DOCKER_SOURCE_ROOT:-}" ]]; then
      args+=(--docker-source-root "${MLOPS_DOCKER_SOURCE_ROOT}")
    fi
    if [[ -n "${MLOPS_TRAIN_GPU_COUNT:-}" ]]; then
      for gpu_job_kind in train benchmark-checkpoint train-eval-trial; do
        args+=(
          --docker-gpu-count
          "${gpu_job_kind}=${MLOPS_TRAIN_GPU_COUNT}"
        )
        if [[ -n "${MLOPS_TRAIN_GPU_DEVICES:-}" ]]; then
          args+=(
            --docker-gpu-devices
            "${gpu_job_kind}=${MLOPS_TRAIN_GPU_DEVICES}"
          )
        fi
      done
    elif [[ -n "${MLOPS_TRAIN_GPU_DEVICES:-}" ]]; then
      echo "MLOPS_TRAIN_GPU_COUNT is required with MLOPS_TRAIN_GPU_DEVICES" >&2
      exit 1
    fi
    if [[ "${MLOPS_EXECUTOR}" == "hybrid" ]]; then
      for docker_job_kind in train benchmark-checkpoint train-eval-trial; do
        args+=(--docker-job-kind "${docker_job_kind}")
      done
    fi
    ;;
  *)
    echo "MLOPS_EXECUTOR must be local, docker, or hybrid" >&2
    exit 1
    ;;
esac

case "${MLOPS_TRIGGER_MODE}" in
  manual)
    ;;
  local)
    : "${MLOPS_INCOMING:=/var/lib/rightarm-mlops/incoming}"
    : "${MLOPS_LOOP_SPEC:=/app/openhands/mlops_loop/specs/right-arm-gesture-loop-v1-loop.yaml}"
    : "${MLOPS_RUN_TEMPLATE:=/app/openhands/mlops_loop/specs/right-arm-gesture-loop-v1-run.yaml}"
    args+=(
      --incoming "${MLOPS_INCOMING}"
      --loop-spec "${MLOPS_LOOP_SPEC}"
      --run-template "${MLOPS_RUN_TEMPLATE}"
      --poll-interval-seconds "${MLOPS_POLL_INTERVAL_SECONDS}"
    )
    ;;
  command)
    : "${MLOPS_WATCH_COMMAND_JSON:?required when MLOPS_TRIGGER_MODE=command}"
    : "${MLOPS_LOOP_SPEC:=/app/openhands/mlops_loop/specs/right-arm-gesture-loop-v1-loop.yaml}"
    : "${MLOPS_RUN_TEMPLATE:=/app/openhands/mlops_loop/specs/right-arm-gesture-loop-v1-run.yaml}"
    args+=(
      --watch-command-json "${MLOPS_WATCH_COMMAND_JSON}"
      --loop-spec "${MLOPS_LOOP_SPEC}"
      --run-template "${MLOPS_RUN_TEMPLATE}"
      --poll-interval-seconds "${MLOPS_POLL_INTERVAL_SECONDS}"
    )
    ;;
  *)
    echo "MLOPS_TRIGGER_MODE must be manual, local, or command" >&2
    exit 1
    ;;
esac

replacement_names=(
  baseline-mmaction-dataset-artifact
  champion-base-model-artifact
  frozen-test-set-artifact
  rtm-detector-artifact
  rtm-person-pose-artifact
  rtm-hand-pose-artifact
  champion-evaluation-artifact
  code-artifact
  container-image-artifact
  label-mapping-artifact
  tensor-contract-artifact
  image-digest
)

for name in "${replacement_names[@]}"; do
  env_name="MLOPS_$(printf '%s' "${name}" | tr '[:lower:]-' '[:upper:]_')"
  value="${!env_name:-}"
  if [[ -n "${value}" ]]; then
    args+=(--set "${name}=${value}")
  fi
done

exec "${args[@]}"
