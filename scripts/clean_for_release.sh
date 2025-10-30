#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[INFO] Cleaning artifacts for release from $ROOT_DIR"

rm -rf "$ROOT_DIR/logs" || true
rm -rf "$ROOT_DIR/python/logs" || true
rm -rf "$ROOT_DIR/.knowledge" || true
rm -rf "$ROOT_DIR/lancedb" || true
rm -f  "$ROOT_DIR/valuecell.db" || true

echo "[OK] Cleaned: logs, python/logs, .knowledge, lancedb, valuecell.db"
echo "[HINT] Ensure .env is NOT committed. Use .env.example for placeholders."

