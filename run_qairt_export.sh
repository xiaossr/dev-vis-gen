#!/bin/bash
# Run the QAIRT export with correct environment variables
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QAIRT_SDK_ROOT="$SCRIPT_DIR/qairt/2.45.0.260326"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
UV_PYTHON_LIB="$HOME/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/lib"
LOCAL_LIBS="$SCRIPT_DIR/.local-libs-jammy/extracted/usr/lib/llvm-14/lib"

export QAIRT_SDK_ROOT
export QNN_SDK_ROOT="$QAIRT_SDK_ROOT"
export PYTHONPATH="$QAIRT_SDK_ROOT/lib/python/:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:$UV_PYTHON_LIB:$LOCAL_LIBS:${LD_LIBRARY_PATH:-}"
export PATH="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang:$PATH"

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] QAIRT_SDK_ROOT=$QAIRT_SDK_ROOT"
echo "[INFO] Python: $VENV_PYTHON"
echo "[INFO] Running: export_flux2_klein_qairt.py $@"

exec "$VENV_PYTHON" "$SCRIPT_DIR/export_flux2_klein_qairt.py" "$@"
