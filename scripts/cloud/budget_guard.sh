#!/usr/bin/env bash

# Source this file; do not execute it directly.

if [[ -d /venv/main/bin ]]; then
  export PATH="/venv/main/bin:$PATH"
fi
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

hemera_budget_init() {
  HEMERA_MAX_EUR="$1"
  HEMERA_HOURLY_EUR="$2"
  HEMERA_PHASE="$3"
  HEMERA_START_FILE=".hemera-${HEMERA_PHASE}-start"
  if [[ ! -f "$HEMERA_START_FILE" ]]; then
    date +%s > "$HEMERA_START_FILE"
  fi
  HEMERA_STARTED_AT="$(cat "$HEMERA_START_FILE")"
  HEMERA_ALLOWED_SECONDS="$(python -c "print(int(float('$HEMERA_MAX_EUR') / float('$HEMERA_HOURLY_EUR') * 3600))")"
  export HEMERA_MAX_EUR HEMERA_HOURLY_EUR HEMERA_PHASE HEMERA_STARTED_AT HEMERA_ALLOWED_SECONDS
}

hemera_remaining_seconds() {
  local now elapsed
  now="$(date +%s)"
  elapsed="$((now - HEMERA_STARTED_AT))"
  echo "$((HEMERA_ALLOWED_SECONDS - elapsed))"
}

hemera_run() {
  local remaining
  remaining="$(hemera_remaining_seconds)"
  if (( remaining <= 120 )); then
    echo "Hemera ${HEMERA_PHASE} budget exhausted; refusing to start: $*" >&2
    return 75
  fi
  timeout --signal=INT --kill-after=60 "$((remaining - 60))s" "$@"
}

hemera_budget_report() {
  mkdir -p reports
  python -c "import json,time,pathlib; start=int('$HEMERA_STARTED_AT'); elapsed=max(0,time.time()-start); hourly=float('$HEMERA_HOURLY_EUR'); payload={'phase':'$HEMERA_PHASE','started_at':start,'finished_at':time.time(),'elapsed_seconds':elapsed,'hourly_eur':hourly,'estimated_eur':elapsed*hourly/3600,'cap_eur':float('$HEMERA_MAX_EUR')}; pathlib.Path('reports/cost-$HEMERA_PHASE.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')"
}
