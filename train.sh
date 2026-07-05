#!/usr/bin/env bash
# train.sh — Launch training with correct CUDA library environment.
#
# System CUDA 12.0 and TF 2.18's bundled CUDA 12.3 (via nvidia-*-cu12 pip packages)
# both try to register cuDNN/cuBLAS/cuFFT plugins, causing a SIGBUS crash.
# This script resolves it by prepending the pip CUDA lib paths so the dynamic
# linker loads only the pip-installed versions.
#
# Usage:
#   ./train.sh                                  # runs both jobs in parallel
#   ./train.sh training/close.py                # close price only
#   ./train.sh training/adjusted_close.py       # adj-close only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $SCRIPT_DIR/venv — run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Prevent ~/.local packages from shadowing venv packages (avoids duplicate TF load)
export PYTHONNOUSERSITE=1

# Locate pip-installed CUDA lib directories inside the venv (nvidia-*-cu12 packages)
NV_LIB_PATHS=$(find "$SCRIPT_DIR/venv/lib" -path "*/nvidia/*/lib" -type d 2>/dev/null | sort -u | paste -sd: -)

if [ -n "$NV_LIB_PATHS" ]; then
    # Prepend pip CUDA paths; strip /usr/local/cuda-* system toolkit to avoid duplication
    CLEAN_LD=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -Ev '^(/usr/local/cuda|$)' | paste -sd: -)
    export LD_LIBRARY_PATH="${NV_LIB_PATHS}${CLEAN_LD:+:$CLEAN_LD}"
fi

if [ $# -eq 0 ]; then
    # No args — run both training jobs in parallel
    echo "[train.sh] Starting close + adjusted-close training in parallel..."
    "$VENV_PYTHON" training/close.py &
    PID_CLOSE=$!
    "$VENV_PYTHON" training/adjusted_close.py &
    PID_ADJ=$!
    wait $PID_CLOSE
    wait $PID_ADJ
    echo "[train.sh] Both training jobs complete."
else
    exec "$VENV_PYTHON" "$@"
fi
