#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOLVLA_PROJECT_VENV="${SMOLVLA_ARM_VENV:-$ROOT/.venv}"
PYTHON_EXECUTABLE="$SMOLVLA_PROJECT_VENV/bin/python"
export PYTHONPATH="$ROOT/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXECUTORCH_ROOT="${SMOLVLA_ARM_EXECUTORCH_ROOT:-$ROOT/toolchain/executorch}"
export EXECUTORCH_BUILD_DIR="${SMOLVLA_ARM_EXECUTORCH_BUILD_DIR:-$EXECUTORCH_ROOT/cmake-out-xnnpack}"
export SMOLVLA_RUNNER_BUILD_DIR="${SMOLVLA_ARM_RUNNER_BUILD_DIR:-$ROOT/build/runner}"
CHECKPOINT="${SMOLVLA_ARM_CHECKPOINT:-$ROOT/checkpoints/smolvla_base}"
INPUT_SUITE="${SMOLVLA_ARM_INPUT_SUITE:-}"
VARIANT="fp32"
OUTPUT_DIR=""
THREADS="${SMOLVLA_CPU_THREADS:-5}"
AFFINITY="${SMOLVLA_CPU_AFFINITY:-}"
VISION_THREADS="${SMOLVLA_VISION_CPU_THREADS:-}"
VISION_AFFINITY="${SMOLVLA_VISION_CPU_AFFINITY:-}"
IMAGE_SIZE=512
LANGUAGE_LENGTH=48
NUM_STEPS=10
REUSE_EXPORT=0
SKIP_BUILD=0

require_value() { (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }; }
while (($#)); do
    case "$1" in
        --variant) require_value "$@"; VARIANT="$2"; shift 2 ;;
        --checkpoint) require_value "$@"; CHECKPOINT="$2"; shift 2 ;;
        --input-suite) require_value "$@"; INPUT_SUITE="$2"; shift 2 ;;
        --output-dir) require_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
        --threads) require_value "$@"; THREADS="$2"; shift 2 ;;
        --cpu-affinity) require_value "$@"; AFFINITY="$2"; shift 2 ;;
        --vision-threads) require_value "$@"; VISION_THREADS="$2"; shift 2 ;;
        --vision-cpu-affinity) require_value "$@"; VISION_AFFINITY="$2"; shift 2 ;;
        --image-size) require_value "$@"; IMAGE_SIZE="$2"; shift 2 ;;
        --language-length) require_value "$@"; LANGUAGE_LENGTH="$2"; shift 2 ;;
        --num-steps) require_value "$@"; NUM_STEPS="$2"; shift 2 ;;
        --reuse-export) REUSE_EXPORT=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help)
            echo "Usage: scripts/pipeline.sh [--variant fp32|int8] [--checkpoint PATH] [--input-suite PATH] [--output-dir PATH] [--threads N] [--cpu-affinity LIST] [--vision-threads N] [--vision-cpu-affinity LIST] [--reuse-export] [--skip-build]"
            echo "Defaults: variant=fp32, image-size=512, language-length=48, num-steps=10"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$VARIANT" == "fp32" || "$VARIANT" == "int8" ]] || { echo "Variant must be fp32 or int8" >&2; exit 2; }
[[ -x "$PYTHON_EXECUTABLE" ]] || { echo "Run scripts/setup.sh first" >&2; exit 1; }
[[ -d "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
INPUT_SUITE="${INPUT_SUITE:-$ROOT/artifacts/input_suite_${IMAGE_SIZE}_l${LANGUAGE_LENGTH}}"
echo "[preflight] Verify the pinned Python and ExecuTorch environment"
"$PYTHON_EXECUTABLE" "$ROOT/scripts/check_environment.py"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/artifacts/$VARIANT}"
export SMOLVLA_CHECKPOINT="$CHECKPOINT"
export SMOLVLA_INPUT_SUITE="$INPUT_SUITE"
export SMOLVLA_ARTIFACTS_DIR="$OUTPUT_DIR"
export SMOLVLA_VARIANT="$VARIANT"
export SMOLVLA_CPU_THREADS="$THREADS"
export SMOLVLA_CPU_AFFINITY="$AFFINITY"
export SMOLVLA_VISION_CPU_THREADS="$VISION_THREADS"
export SMOLVLA_VISION_CPU_AFFINITY="$VISION_AFFINITY"

if [[ ! -f "$INPUT_SUITE/manifest.json" || ! -f "$INPUT_SUITE/input_suite.pt" ]]; then
    echo "[1/8] Generate deterministic inputs and PyTorch reference"
    "$PYTHON_EXECUTABLE" "$ROOT/scripts/generate_inputs.py" \
        --checkpoint "$CHECKPOINT" --output-dir "$INPUT_SUITE" \
        --synthetic-inputs --image-size "$IMAGE_SIZE" \
        --language-length "$LANGUAGE_LENGTH" --num-steps "$NUM_STEPS" \
        --threads "$THREADS"
else
    echo "[1/8] Reuse input/reference suite: $INPUT_SUITE"
    "$PYTHON_EXECUTABLE" - "$INPUT_SUITE/manifest.json" \
        "$IMAGE_SIZE" "$LANGUAGE_LENGTH" "$NUM_STEPS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "image_size": int(sys.argv[2]),
    "language_length": int(sys.argv[3]),
    "num_steps": int(sys.argv[4]),
}
mismatches = [
    f"{name}={manifest.get(name)!r}, requested {value!r}"
    for name, value in expected.items()
    if manifest.get(name) != value
]
if mismatches:
    raise SystemExit(
        "Existing input suite has a different fixed configuration: "
        + ", ".join(mismatches)
        + ". Choose another --input-suite directory."
    )
PY
fi

if ((REUSE_EXPORT == 0)); then
    [[ ! -e "$OUTPUT_DIR/manifest.json" ]] || {
        echo "Output already exists; use --reuse-export or choose another --output-dir" >&2
        exit 1
    }
    echo "[2/8] Export $VARIANT split programs"
    "$PYTHON_EXECUTABLE" "$ROOT/scripts/export.py" \
        --checkpoint "$CHECKPOINT" --input-suite "$INPUT_SUITE" \
        --output-dir "$OUTPUT_DIR" --variant "$VARIANT" --threads "$THREADS"
else
    echo "[2/8] Reuse exported programs"
    "$PYTHON_EXECUTABLE" "$ROOT/scripts/native_contract.py" "$OUTPUT_DIR" \
        --manifest-only --expect-variant "$VARIANT"
fi

echo "[3/8] Lower to ExecuTorch/XNNPACK"
"$PYTHON_EXECUTABLE" "$ROOT/scripts/lower.py" --artifacts-dir "$OUTPUT_DIR"
echo "[4/8] Prepare native tensor inputs"
"$PYTHON_EXECUTABLE" "$ROOT/scripts/prepare_inputs.py" --artifacts-dir "$OUTPUT_DIR"
echo "[5/8] Validate every PTE boundary and end-to-end actions"
VALIDATE_ARGS=(--artifacts-dir "$OUTPUT_DIR" --cpu-threads "$THREADS")
[[ -z "$AFFINITY" ]] || VALIDATE_ARGS+=(--cpu-affinity "$AFFINITY")
"$PYTHON_EXECUTABLE" "$ROOT/scripts/validate_pte.py" "${VALIDATE_ARGS[@]}"
if ((SKIP_BUILD == 0)); then
    echo "[6/8] Build the native split orchestrator"
    "$ROOT/scripts/build_runner.sh"
else
    echo "[6/8] Reuse native split orchestrator"
fi
echo "[7/8] Run a native latency smoke test"
BENCHMARK_ARGS=(--artifacts-dir "$OUTPUT_DIR" --warmup-runs 1 --benchmark-runs 1 --cpu-threads "$THREADS")
[[ -z "$AFFINITY" ]] || BENCHMARK_ARGS+=(--cpu-affinity "$AFFINITY")
[[ -z "$VISION_THREADS" ]] || BENCHMARK_ARGS+=(--vision-cpu-threads "$VISION_THREADS")
[[ -z "$VISION_AFFINITY" ]] || BENCHMARK_ARGS+=(--vision-cpu-affinity "$VISION_AFFINITY")
"$PYTHON_EXECUTABLE" "$ROOT/scripts/benchmark.py" "${BENCHMARK_ARGS[@]}"
echo "[8/8] Native accuracy gate passed"
echo "Artifacts: $OUTPUT_DIR"
