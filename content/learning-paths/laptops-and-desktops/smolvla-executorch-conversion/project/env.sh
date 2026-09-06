#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file: source env.sh" >&2
    exit 2
fi

SMOLVLA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOLVLA_PROJECT_VENV="${SMOLVLA_ARM_VENV:-$SMOLVLA_PROJECT_ROOT/.venv}"
if [[ ! -f "$SMOLVLA_PROJECT_VENV/bin/activate" ]]; then
    echo "Project virtual environment not found; run ./scripts/setup.sh first" >&2
    return 1
fi
# shellcheck disable=SC1090
source "$SMOLVLA_PROJECT_VENV/bin/activate"
export EXECUTORCH_ROOT="${SMOLVLA_ARM_EXECUTORCH_ROOT:-$SMOLVLA_PROJECT_ROOT/toolchain/executorch}"
export EXECUTORCH_BUILD_DIR="${SMOLVLA_ARM_EXECUTORCH_BUILD_DIR:-$EXECUTORCH_ROOT/cmake-out-xnnpack}"
export SMOLVLA_RUNNER_BUILD_DIR="${SMOLVLA_ARM_RUNNER_BUILD_DIR:-$SMOLVLA_PROJECT_ROOT/build/runner}"
export SMOLVLA_CHECKPOINT="${SMOLVLA_ARM_CHECKPOINT:-$SMOLVLA_PROJECT_ROOT/checkpoints/smolvla_base}"
export SMOLVLA_INPUT_SUITE="${SMOLVLA_ARM_INPUT_SUITE:-$SMOLVLA_PROJECT_ROOT/artifacts/input_suite_512_l48}"
export SMOLVLA_ARTIFACTS_DIR="${SMOLVLA_ARM_ARTIFACTS_DIR:-$SMOLVLA_PROJECT_ROOT/artifacts/fp32}"
export PYTHON_EXECUTABLE="$SMOLVLA_PROJECT_VENV/bin/python"
export PYTHONPATH="$SMOLVLA_PROJECT_ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset SMOLVLA_PROJECT_ROOT SMOLVLA_PROJECT_VENV
