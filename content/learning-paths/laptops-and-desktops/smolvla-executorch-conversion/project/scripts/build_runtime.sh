#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-$ROOT/toolchain/executorch}"
BUILD_DIR="${EXECUTORCH_BUILD_DIR:-$EXECUTORCH_ROOT/cmake-out-xnnpack}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$ROOT/.venv/bin/python}"
EXPECTED_COMMIT="e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"

[[ -d "$EXECUTORCH_ROOT/.git" ]] || { echo "ExecuTorch source not found" >&2; exit 1; }
[[ "$(git -C "$EXECUTORCH_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
    echo "ExecuTorch must be at $EXPECTED_COMMIT" >&2; exit 1;
}

cmake -S "$EXECUTORCH_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON \
    -DXNNPACK_ENABLE_KLEIDIAI=ON \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE"
cmake --build "$BUILD_DIR" -j"$(nproc)" --target install --config Release

for setting in EXECUTORCH_XNNPACK_ENABLE_KLEIDI XNNPACK_ENABLE_KLEIDIAI; do
    [[ "$(sed -n "s/^${setting}:BOOL=//p" "$BUILD_DIR/CMakeCache.txt")" == "ON" ]] || {
        echo "$setting was not enabled" >&2; exit 1;
    }
done
echo "ExecuTorch/XNNPACK runtime: $BUILD_DIR"
