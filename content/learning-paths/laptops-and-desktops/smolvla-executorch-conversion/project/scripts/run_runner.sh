#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$ROOT/.venv/bin/python}"
ARTIFACTS="${SMOLVLA_ARTIFACTS_DIR:-$ROOT/artifacts/fp32}"
RUNNER="${SMOLVLA_RUNNER_BUILD_DIR:-$ROOT/build/runner}/smolvla_runner"
OUTPUT="${SMOLVLA_OUTPUT_FILE:-$ARTIFACTS/native_orchestrator_output.bin}"

[[ -x "$RUNNER" ]] || { echo "Build the runner first: scripts/build_runner.sh" >&2; exit 1; }
mapfile -t CONTRACT < <("$PYTHON_EXECUTABLE" "$ROOT/scripts/native_contract.py" "$ARTIFACTS")
ARGS=(
    --artifacts_dir="$ARTIFACTS"
    --output_file="$OUTPUT"
    --cpu_threads="${SMOLVLA_CPU_THREADS:-5}"
    --precision_label="${SMOLVLA_VARIANT:-fp32}"
)
[[ -z "${SMOLVLA_CPU_AFFINITY:-}" ]] || ARGS+=(--cpu_affinity="$SMOLVLA_CPU_AFFINITY")
[[ -z "${SMOLVLA_VISION_CPU_THREADS:-}" ]] || ARGS+=(--vision_cpu_threads="$SMOLVLA_VISION_CPU_THREADS")
[[ -z "${SMOLVLA_VISION_CPU_AFFINITY:-}" ]] || ARGS+=(--vision_cpu_affinity="$SMOLVLA_VISION_CPU_AFFINITY")

"$RUNNER" "${ARGS[@]}" "${CONTRACT[@]}" "$@"
