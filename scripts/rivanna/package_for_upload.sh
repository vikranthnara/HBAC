#!/usr/bin/env bash
# Package HBAC for manual upload to Rivanna when SSH keys are not configured.
# Creates /tmp/hbac_deploy.tar.gz — copy to cluster with:
#   scp /tmp/hbac_deploy.tar.gz eyu8ps@login.hpc.virginia.edu:/standard/liverobotics/
# On Rivanna:
#   cd /standard/liverobotics && tar xzf hbac_deploy.tar.gz && cd hbac
#   bash scripts/rivanna/on_cluster_setup_and_submit.sh
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE="/tmp/hbac_deploy.tar.gz"

echo "Packaging ${LOCAL_ROOT} -> ${ARCHIVE}"
tar czf "${ARCHIVE}" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.env' \
  -C "${LOCAL_ROOT}" \
  .

echo ""
echo "Archive: ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1))"
echo ""
echo "=== Step 1: laptop (new terminal) ==="
echo "scp ${ARCHIVE} eyu8ps@login.hpc.virginia.edu:/standard/liverobotics/"
echo ""
echo "=== Step 2: Rivanna (login node) ==="
echo "mkdir -p /standard/liverobotics/hbac"
echo "tar xzf /standard/liverobotics/hbac_deploy.tar.gz -C /standard/liverobotics/hbac"
echo "cd /standard/liverobotics/hbac && bash scripts/rivanna/on_cluster_setup_and_submit.sh"
