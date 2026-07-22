#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_AGENT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_AGENT_ENV_FILE="${MODEL_AGENT_ROOT_DIR}/.env"
readonly MODEL_AGENT_ENV_TEMPLATE="${MODEL_AGENT_ROOT_DIR}/configs/model-agent.env.example"

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 0
}

find_available_port() {
  local candidate="$1"
  local attempts=0
  while ((attempts < 2000)); do
    if python3 - "${candidate}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$((candidate + 1))"
    if ((candidate > 65535)); then
      candidate=20000
    fi
    attempts="$((attempts + 1))"
  done
  echo "Could not find an available loopback port" >&2
  return 1
}

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "Docker is unavailable for $(id -un). Ask the server administrator to grant approved Docker access." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required." >&2
  exit 1
fi

if [[ ! -f "${MODEL_AGENT_ENV_FILE}" ]]; then
  uid="$(id -u)"
  gid="$(id -g)"
  docker_gid="$(stat -c '%g' /var/run/docker.sock)"
  instance="$(printf '%s' "$(id -un)" | tr -c '[:alnum:]_.-' '-')"
  openhands_port="$((20000 + uid))"
  mlops_port="$((30000 + uid))"
  if ((openhands_port > 65535)); then
    openhands_port="$((20000 + uid % 20000))"
  fi
  if ((mlops_port > 65535)); then
    mlops_port="$((40000 + uid % 20000))"
  fi
  openhands_port="$(find_available_port "${openhands_port}")"
  mlops_port="$(find_available_port "${mlops_port}")"
  if [[ "${mlops_port}" == "${openhands_port}" ]]; then
    mlops_port="$(find_available_port "$((mlops_port + 1))")"
  fi
  state_dir="${HOME}/.model-agent/${instance}"
  default_dataset="$(first_existing_file /home/results/6t1/gesture_data/rtm_add.pkl)"
  default_base_model="$(first_existing_file /home/results/6t1/stgcn_hand21_rgb_gated_class11_v3_ft_rtm_add/best_acc_top1_epoch_41.pth)"
  default_rtm_detector="$(first_existing_file /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmdet_personhand_320_192_v1.onnx)"
  default_rtm_person_pose="$(first_existing_file /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmpose3d-s-upperbody-face.onnx)"
  default_rtm_hand_pose="$(first_existing_file /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmpose3d-s-hand-v2.onnx)"
  default_cos_config="$(first_existing_file "${HOME}/.openhands/tencent-cos.yaml")"
  if [[ -z "${default_cos_config}" ]]; then
    default_cos_config="${state_dir}/secrets/tencent-cos.yaml"
  fi
  default_coscli="$(command -v coscli || true)"
  if [[ -z "${default_coscli}" ]]; then
    default_coscli=/bin/false
  fi

  sed \
    -e "s|REPLACE_INSTANCE|${instance}|g" \
    -e "s|REPLACE_ROOT|${MODEL_AGENT_ROOT_DIR}|g" \
    -e "s|REPLACE_STATE|${state_dir}|g" \
    -e "s|REPLACE_OPENHANDS_PORT|${openhands_port}|g" \
    -e "s|REPLACE_MLOPS_PORT|${mlops_port}|g" \
    -e "s|REPLACE_UID|${uid}|g" \
    -e "s|REPLACE_GID|${gid}|g" \
    -e "s|REPLACE_DOCKER_GID|${docker_gid}|g" \
    -e "s|REPLACE_DEFAULT_DATASET|${default_dataset}|g" \
    -e "s|REPLACE_BASE_MODEL|${default_base_model}|g" \
    -e "s|REPLACE_RTM_DETECTOR|${default_rtm_detector}|g" \
    -e "s|REPLACE_RTM_PERSON_POSE|${default_rtm_person_pose}|g" \
    -e "s|REPLACE_RTM_HAND_POSE|${default_rtm_hand_pose}|g" \
    -e "s|REPLACE_COS_CONFIG|${default_cos_config}|g" \
    -e "s|REPLACE_COSCLI|${default_coscli}|g" \
    "${MODEL_AGENT_ENV_TEMPLATE}" >"${MODEL_AGENT_ENV_FILE}"
  chmod 600 "${MODEL_AGENT_ENV_FILE}"
  echo "Created private runtime configuration: ${MODEL_AGENT_ENV_FILE}"
fi

set -a
# shellcheck disable=SC1090
source "${MODEL_AGENT_ENV_FILE}"
set +a

ensure_env_path_default() {
  local name="$1"
  shift
  if grep -q "^${name}=" "${MODEL_AGENT_ENV_FILE}"; then
    if grep -q "^${name}=$" "${MODEL_AGENT_ENV_FILE}"; then
      local discovered
      discovered="$(first_existing_file "$@")"
      if [[ -n "${discovered}" ]]; then
        sed -i "s|^${name}=$|${name}=${discovered}|" "${MODEL_AGENT_ENV_FILE}"
        export "${name}=${discovered}"
      fi
    fi
    return 0
  fi
  local value
  value="$(first_existing_file "$@")"
  printf '%s=%s\n' "${name}" "${value}" >>"${MODEL_AGENT_ENV_FILE}"
  export "${name}=${value}"
}

# Migrate configurations created by earlier model-agent revisions without
# overwriting any operator-selected artifact path.
ensure_env_path_default MLOPS_DEFAULT_DATASET_PATH \
  /home/results/6t1/gesture_data/rtm_add.pkl
ensure_env_path_default MLOPS_BASE_MODEL_PATH \
  /home/results/6t1/stgcn_hand21_rgb_gated_class11_v3_ft_rtm_add/best_acc_top1_epoch_41.pth
ensure_env_path_default MLOPS_RTM_DETECTOR_PATH \
  /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmdet_personhand_320_192_v1.onnx
ensure_env_path_default MLOPS_RTM_PERSON_POSE_PATH \
  /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmpose3d-s-upperbody-face.onnx
ensure_env_path_default MLOPS_RTM_HAND_POSE_PATH \
  /home/wangning/.cache/gesture-loop-engine/rightarm_rtm_models/rtmpose3d-s-hand-v2.onnx
if ! grep -q '^TENCENT_COS_CONFIG_PATH=' "${MODEL_AGENT_ENV_FILE}"; then
  cos_config="$(first_existing_file "${HOME}/.openhands/tencent-cos.yaml")"
  if [[ -z "${cos_config}" ]]; then
    cos_config="${MODEL_AGENT_STATE_DIR}/secrets/tencent-cos.yaml"
  fi
  printf 'TENCENT_COS_CONFIG_PATH=%s\n' "${cos_config}" >>"${MODEL_AGENT_ENV_FILE}"
  export TENCENT_COS_CONFIG_PATH="${cos_config}"
fi
if ! grep -q '^COSCLI_HOST_PATH=' "${MODEL_AGENT_ENV_FILE}"; then
  coscli_path="$(command -v coscli || true)"
  printf 'COSCLI_HOST_PATH=%s\n' "${coscli_path:-/bin/false}" >>"${MODEL_AGENT_ENV_FILE}"
  export COSCLI_HOST_PATH="${coscli_path:-/bin/false}"
fi
if ! grep -q '^RIGHTARM_COS_ALLOWED_PREFIX=' "${MODEL_AGENT_ENV_FILE}"; then
  printf 'RIGHTARM_COS_ALLOWED_PREFIX=\n' >>"${MODEL_AGENT_ENV_FILE}"
fi
if ! grep -q '^RIGHTARM_COS_DISK_RESERVE_BYTES=' "${MODEL_AGENT_ENV_FILE}"; then
  printf 'RIGHTARM_COS_DISK_RESERVE_BYTES=21474836480\n' >>"${MODEL_AGENT_ENV_FILE}"
fi
if ! grep -q '^MLOPS_DEFAULT_DATASET_ARTIFACT=' "${MODEL_AGENT_ENV_FILE}"; then
  printf 'MLOPS_DEFAULT_DATASET_ARTIFACT=\n' >>"${MODEL_AGENT_ENV_FILE}"
fi

: "${MODEL_AGENT_INSTANCE_NAME:?missing from .env}"
: "${MODEL_AGENT_ROOT:?missing from .env}"
: "${MODEL_AGENT_STATE_DIR:?missing from .env}"
: "${MODEL_AGENT_IMAGE:?missing from .env}"
: "${MODEL_AGENT_BASE_IMAGE:?missing from .env}"
: "${HOST_UID:?missing from .env}"
: "${HOST_GID:?missing from .env}"
: "${DOCKER_GID:?missing from .env}"

if [[ "${MODEL_AGENT_ROOT}" != "${MODEL_AGENT_ROOT_DIR}" ]]; then
  echo "This clone moved after setup. Remove ${MODEL_AGENT_ENV_FILE} and run setup.sh again." >&2
  exit 1
fi
if [[ "${HOST_UID}" != "$(id -u)" ]] || [[ "${HOST_GID}" != "$(id -g)" ]]; then
  echo "This .env belongs to a different Linux user. Remove it and run setup.sh again." >&2
  exit 1
fi

mkdir -p \
  "${MODEL_AGENT_STATE_DIR}/openhands" \
  "${MODEL_AGENT_STATE_DIR}/controller-home" \
  "${MODEL_AGENT_STATE_DIR}/mlops/incoming" \
  "${MODEL_AGENT_STATE_DIR}/mlops/imports" \
  "${MODEL_AGENT_STATE_DIR}/secrets" \
  "${MODEL_AGENT_ROOT_DIR}/configs/generated"
chmod 700 \
  "${MODEL_AGENT_STATE_DIR}" \
  "${MODEL_AGENT_STATE_DIR}/openhands" \
  "${MODEL_AGENT_STATE_DIR}/controller-home" \
  "${MODEL_AGENT_STATE_DIR}/mlops" \
  "${MODEL_AGENT_STATE_DIR}/mlops/imports" \
  "${MODEL_AGENT_STATE_DIR}/secrets"

if [[ ! -f "${TENCENT_COS_CONFIG_PATH}" ]]; then
  umask 077
  printf '{}\n' >"${TENCENT_COS_CONFIG_PATH}"
fi
chmod 600 "${TENCENT_COS_CONFIG_PATH}"

operator_token="${MODEL_AGENT_STATE_DIR}/secrets/mlops-operator-token"
if [[ ! -f "${operator_token}" ]]; then
  umask 077
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32 >"${operator_token}"
  else
    python3 -c 'import secrets; print(secrets.token_hex(32))' >"${operator_token}"
  fi
fi
chmod 600 "${operator_token}"

if [[ "${MODEL_AGENT_SKIP_BUILD:-0}" == "1" ]]; then
  echo "Skipping image build because MODEL_AGENT_SKIP_BUILD=1."
elif [[ "${MODEL_AGENT_FORCE_BUILD:-0}" == "1" ]] || ! docker image inspect "${MODEL_AGENT_IMAGE}" >/dev/null 2>&1; then
  echo "Building the bundled OpenHands image. The first build can take several minutes."
  docker build \
    --file "${MODEL_AGENT_ROOT_DIR}/containers/app/Dockerfile" \
    --build-arg OPENHANDS_BUILD_VERSION=model-agent-local \
    --tag "${MODEL_AGENT_BASE_IMAGE}" \
    "${MODEL_AGENT_ROOT_DIR}"
  docker build \
    --file "${MODEL_AGENT_ROOT_DIR}/containers/model-agent/Dockerfile" \
    --build-arg "BASE_IMAGE=${MODEL_AGENT_BASE_IMAGE}" \
    --tag "${MODEL_AGENT_IMAGE}" \
    "${MODEL_AGENT_ROOT_DIR}"
fi

docker compose \
  --env-file "${MODEL_AGENT_ENV_FILE}" \
  --file "${MODEL_AGENT_ROOT_DIR}/docker-compose.yml" \
  config --quiet

echo "model-agent setup is complete."
echo "LLM credentials are not written to this repository; configure them in the OpenHands Settings page after start.sh."
