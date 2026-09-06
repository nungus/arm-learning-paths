#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_BUILD="${EXECUTORCH_BUILD_DIR:-$ROOT/toolchain/executorch/cmake-out-xnnpack}"
BUILD_DIR="${SMOLVLA_RUNNER_BUILD_DIR:-$ROOT/build/runner}"

[[ -f "$RUNTIME_BUILD/lib/cmake/ExecuTorch/executorch-config.cmake" ]] || {
    echo "Build the ExecuTorch runtime first: scripts/build_runtime.sh" >&2
    exit 1
}
env -u EXECUTORCH_ROOT cmake -S "$ROOT/runtime" -B "$BUILD_DIR" \
    --log-level=WARNING \
    -DCMAKE_BUILD_TYPE=Release \
    -Dexecutorch_DIR="$RUNTIME_BUILD/lib/cmake/ExecuTorch"
cmake --build "$BUILD_DIR" -j"$(nproc)" --config Release
echo "Native runner: $BUILD_DIR/smolvla_runner"
