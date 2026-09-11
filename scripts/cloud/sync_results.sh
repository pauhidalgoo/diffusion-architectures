#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:?Usage: sync_results.sh <rclone-destination>}"
rclone copy runs "$DESTINATION/runs" --checksum --progress
rclone copy reports "$DESTINATION/reports" --checksum --progress
rclone copy configs "$DESTINATION/configs" --checksum --progress
rclone copy data/manifests "$DESTINATION/data/manifests" --checksum --progress
rclone copy evaluation "$DESTINATION/evaluation" --checksum --progress
rclone copy requirements-lock.txt "$DESTINATION/" --checksum --progress
rclone copy README.md "$DESTINATION/" --checksum --progress
rclone copy RESEARCH_PLAN.md "$DESTINATION/" --checksum --progress
rclone copy MODEL_CARD.md "$DESTINATION/" --checksum --progress
