#!/usr/bin/env bash
set -euo pipefail
RELEASE_URL="https://github.com/MeaFew/attributor/releases/download/v1.0-data/data.zip"
DEST_DIR="$(dirname "$0")"
echo "Downloading data for attributor..."
curl -L -o "${DEST_DIR}/data.zip" "${RELEASE_URL}"
echo "Extracting..."
unzip -o "${DEST_DIR}/data.zip" -d "${DEST_DIR}/data/raw/"
mkdir -p "${DEST_DIR}/data/processed"
# MMM includes processed parquet
unzip -o "${DEST_DIR}/data.zip" "mmm_cleaned.parquet" -d "${DEST_DIR}/data/processed/" 2>/dev/null || true
rm "${DEST_DIR}/data.zip"
echo "Done. Run 'make all' to run the full pipeline."
