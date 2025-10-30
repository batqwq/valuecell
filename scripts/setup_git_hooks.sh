#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
echo "core.hooksPath set to .githooks"
chmod +x .githooks/pre-commit
echo "pre-commit hook enabled"

