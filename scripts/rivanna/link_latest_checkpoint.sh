#!/usr/bin/env bash
# Create checkpoints/variant_a/latest -> most recent stage1_stop_controller.npz parent run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LATEST=$(find checkpoints/variant_a -name stage1_stop_controller.npz -print0 | xargs -0 ls -t | head -1)
RUN_DIR=$(dirname "$LATEST")
RUN_NAME=$(basename "$RUN_DIR")
mkdir -p checkpoints/variant_a
ln -sfn "$RUN_NAME" checkpoints/variant_a/latest
echo "latest -> $RUN_NAME ($RUN_DIR)"
