#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_AGENT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MODEL_AGENT_ENV_FILE="${MODEL_AGENT_ROOT_DIR}/.env"

if [[ ! -f "${MODEL_AGENT_ENV_FILE}" ]]; then
  echo "model-agent has not been set up in this clone."
  exit 0
fi

exec docker compose \
  --env-file "${MODEL_AGENT_ENV_FILE}" \
  --file "${MODEL_AGENT_ROOT_DIR}/docker-compose.yml" \
  down
