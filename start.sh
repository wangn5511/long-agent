#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_AGENT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_AGENT_ENV_FILE="${MODEL_AGENT_ROOT_DIR}/.env"

"${MODEL_AGENT_ROOT_DIR}/setup.sh"

set -a
# shellcheck disable=SC1090
source "${MODEL_AGENT_ENV_FILE}"
set +a

compose=(
  docker compose
  --env-file "${MODEL_AGENT_ENV_FILE}"
  --file "${MODEL_AGENT_ROOT_DIR}/docker-compose.yml"
)

"${compose[@]}" up -d

openhands_ready=0
controller_ready=0
for _ in $(seq 1 90); do
  if curl --fail --silent --max-time 2 \
    "http://127.0.0.1:${OPENHANDS_PORT}/ready" | grep --quiet 'OK'; then
    openhands_ready=1
  fi
  if curl --fail --silent --max-time 2 \
    "http://127.0.0.1:${MLOPS_PORT}/openapi.json" >/dev/null; then
    controller_ready=1
  fi
  if ((openhands_ready && controller_ready)); then
    echo "OpenHands and the MLOps loop controller are ready."
    echo "Forward server port ${OPENHANDS_PORT} in VSCode SSH, then open http://localhost:${OPENHANDS_PORT}"
    echo "Configure your LLM model, base URL, and API Key in OpenHands Settings."
    exit 0
  fi
  sleep 2
done

echo "model-agent did not become ready in 180 seconds. Recent logs:" >&2
"${compose[@]}" logs --tail 120 >&2
exit 1
