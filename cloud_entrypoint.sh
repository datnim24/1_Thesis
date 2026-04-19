#!/usr/bin/env bash
# Cloud-run entrypoint for master_eval.
# Env vars (all optional):
#   RUN_NAME   - run label (default: cloud_run_<UTC timestamp>)
#   TIME_SEC   - seconds per method (default: 3600)
#   SEED       - RNG seed (default: 42)
#   SKIP       - space-separated method list to skip: cpsat ql ppo rlhh
#   EXTRA_ARGS - extra flags appended to master_eval.py
set -euo pipefail

: "${RUN_NAME:=cloud_run_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${TIME_SEC:=3600}"
: "${SEED:=42}"
: "${SKIP:=}"
: "${EXTRA_ARGS:=}"

skip_flags=()
for m in $SKIP; do
    skip_flags+=("--skip" "$m")
done

cd /app
mkdir -p /app/results

echo "[cloud_entrypoint] RUN_NAME=$RUN_NAME TIME_SEC=$TIME_SEC SEED=$SEED SKIP='$SKIP'"
exec python master_eval.py \
    --name "$RUN_NAME" \
    --time "$TIME_SEC" \
    --seed "$SEED" \
    "${skip_flags[@]}" \
    $EXTRA_ARGS
