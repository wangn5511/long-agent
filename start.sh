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

controller_storage_ready() {
  "${compose[@]}" exec -T loop-controller /app/.venv/bin/python -c '
import os
import tempfile
from pathlib import Path

for raw in (
    "/var/lib/rightarm-mlops/imports",
    "/var/lib/rightarm-mlops/cas/staging",
    "/var/lib/rightarm-mlops/cas/objects",
    "/var/lib/rightarm-mlops/jobs",
):
    directory = Path(raw)
    if not directory.is_dir():
        raise SystemExit(f"missing runtime directory: {directory}")
    descriptor, probe = tempfile.mkstemp(prefix=".readiness-", dir=str(directory))
    try:
        os.write(descriptor, b"ready")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        Path(probe).unlink(missing_ok=True)
' >/dev/null 2>&1
}

"${compose[@]}" up -d

for _ in $(seq 1 90); do
  openhands_ready=0
  controller_ready=0
  if curl --fail --silent --max-time 2 \
    "http://127.0.0.1:${OPENHANDS_PORT}/ready" | grep --quiet 'OK'; then
    openhands_ready=1
  fi
  if "${MODEL_AGENT_ROOT_DIR}/model-agent" status \
    --limit 1 --timeout 2 --json >/dev/null 2>&1 \
    && controller_storage_ready; then
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
