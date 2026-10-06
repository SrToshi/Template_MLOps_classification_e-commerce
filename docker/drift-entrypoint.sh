#!/bin/sh
# Periodic Evidently drift check.
#
# DRIFT_INTERVAL_SECONDS controls the cadence (default: hourly). Set
# DRIFT_RUN_ONCE=1 to run a single check and exit, which is what
# `docker compose run drift` does during a demo.
set -eu

INTERVAL="${DRIFT_INTERVAL_SECONDS:-3600}"
CURRENT_LIMIT="${DRIFT_CURRENT_LIMIT:-500}"
REFERENCE_LIMIT="${DRIFT_REFERENCE_LIMIT:-1000}"

ARGS="--current-limit ${CURRENT_LIMIT} --reference-limit ${REFERENCE_LIMIT}"
if [ -n "${DRIFT_RETRAIN_API_URL:-}" ]; then
    ARGS="${ARGS} --retrain-on-drift ${DRIFT_RETRAIN_API_URL}"
fi

if [ "${DRIFT_RUN_ONCE:-0}" = "1" ]; then
    exec python src/drift_detection.py ${ARGS}
fi

while true; do
    echo "[drift] running check at $(date -u +%FT%TZ)"
    python src/drift_detection.py ${ARGS} || echo "[drift] check failed, continuing"
    sleep "${INTERVAL}"
done
