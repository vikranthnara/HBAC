#!/usr/bin/env bash
# Load Rivanna SSH credentials from repo .env (SSH_KEY_USER, SSH_KEY_PASSWORD).
set -euo pipefail

RIVANNA_SSH_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ROOT="$(cd "${RIVANNA_SSH_DIR}/../.." && pwd)"
ENV_FILE="${LOCAL_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  _riv_creds="$(python3 - <<'PY'
from pathlib import Path
env = {}
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
print(env.get("SSH_KEY_USER", "eyu8ps"))
print(env.get("SSH_KEY_PASSWORD", ""))
PY
)"
  RIVANNA_USER="$(echo "${_riv_creds}" | sed -n '1p')"
  SSH_KEY_PASSWORD="$(echo "${_riv_creds}" | sed -n '2p')"
fi

RIVANNA_USER="${RIVANNA_USER:-eyu8ps}"
RIVANNA_HOST="${RIVANNA_HOST:-${RIVANNA_USER}@login.hpc.virginia.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/standard/liverobotics/hbac-run-20260630T183941Z}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o PreferredAuthentications=password -o PubkeyAuthentication=no)
if [[ -n "${SSH_KEY_PASSWORD:-}" ]] && command -v sshpass >/dev/null; then
  export SSHPASS="${SSH_KEY_PASSWORD}"
  SSH=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP=(sshpass -e scp "${SSH_OPTS[@]}")
  RSYNC=(sshpass -e rsync -avz -e "ssh ${SSH_OPTS[*]}")
else
  SSH=(ssh "${SSH_OPTS[@]}")
  SCP=(scp "${SSH_OPTS[@]}")
  RSYNC=(rsync -avz)
fi

rivanna_ssh() { "${SSH[@]}" "${RIVANNA_HOST}" "$@"; }
rivanna_scp() { "${SCP[@]}" "$@"; }
