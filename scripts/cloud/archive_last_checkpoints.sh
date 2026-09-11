#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hemera
archive=/workspace/hemera-checkpoint-last.tar
log=reports/checkpoint-archive.log
status_file=reports/checkpoint-archive-exit-code

rm -f "${archive}" "${archive}.sha256" "${status_file}"
find runs -type f -name checkpoint-last.pt -print0 \
  | sort -z \
  | tar --null --files-from=- --create --file="${archive}"
sha256sum "${archive}" > "${archive}.sha256"
printf '0\n' > "${status_file}"
du -h "${archive}" | tee "${log}"
