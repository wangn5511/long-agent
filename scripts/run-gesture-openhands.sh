#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for operators who used the old host-local name.
# The integrated distribution now starts OpenHands and the Loop Controller
# together through the repository-level launcher.
readonly MODEL_AGENT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${MODEL_AGENT_ROOT_DIR}/start.sh"
