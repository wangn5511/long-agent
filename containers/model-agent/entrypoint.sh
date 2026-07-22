#!/usr/bin/env bash
set -euo pipefail

: "${HOST_UID:?HOST_UID is required}"
: "${HOST_GID:?HOST_GID is required}"

if [[ ! "${HOST_UID}" =~ ^[0-9]+$ ]] || ((HOST_UID == 0)); then
  echo "HOST_UID must be a non-root numeric uid" >&2
  exit 1
fi
if [[ ! "${HOST_GID}" =~ ^[0-9]+$ ]] || ((HOST_GID == 0)); then
  echo "HOST_GID must be a non-root numeric gid" >&2
  exit 1
fi
if [[ "$(id -u)" != "${HOST_UID}" ]] || [[ "$(id -g)" != "${HOST_GID}" ]]; then
  echo "container uid/gid do not match HOST_UID:HOST_GID" >&2
  exit 1
fi

export HOME=/home/model-agent
umask 077
mkdir -p "${HOME}/.openhands"
if [[ -w /opt/workspace_base ]]; then
  mkdir -p /opt/workspace_base/rightArmPerception
fi
if [[ -w /var/lib ]] || [[ -w /var/lib/rightarm-mlops ]]; then
  mkdir -p /var/lib/rightarm-mlops
fi
unset SESSION_API_KEY

exec "$@"
