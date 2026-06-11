#!/usr/bin/env bash
# Clean install script — removes known conflicting packages before installing
# from requirements.txt.
#
# Usage:
#   ./install.sh              # full install
#   ./install.sh webapp       # webapp-only install (webapp_requirements.txt)
set -euo pipefail

REQUIREMENTS="requirements.txt"
if [[ "${1:-}" == "webapp" ]]; then
    REQUIREMENTS="webapp_requirements.txt"
fi

echo "==> Removing known conflicting packages..."
# pandas-ta (any version) conflicts with our pinned numpy/pandas/tqdm stack.
# The codebase uses the 'ta' library instead; pandas-ta should not be present.
pip uninstall -y pandas-ta 2>/dev/null || true

echo "==> Installing from ${REQUIREMENTS}..."
pip install -r "${REQUIREMENTS}"

echo "==> Done."
