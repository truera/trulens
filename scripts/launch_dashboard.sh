#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_PATH="${REPO_ROOT}/test_perf.sqlite"

echo "=== Launching TruLens Dashboard (optimized) ==="
echo "DB: ${DB_PATH}"
echo "Port: 8502"
echo ""

cd "${REPO_ROOT}"
TRULENS_OTEL_TRACING=1 poetry run streamlit run \
  src/dashboard/trulens/dashboard/main.py \
  --server.port 8502 \
  --server.headless true \
  --theme.base dark \
  --theme.primaryColor "#E0735C" \
  -- \
  --database-url "sqlite:///${DB_PATH}" \
  --otel-tracing
