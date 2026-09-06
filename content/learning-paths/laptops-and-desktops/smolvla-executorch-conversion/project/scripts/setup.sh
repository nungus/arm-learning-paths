#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTORCH_COMMIT="e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"
SMOLVLA_REVISION="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SMOLVLM_REVISION="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
VENV="$ROOT/.venv"
EXECUTORCH_ROOT="$ROOT/toolchain/executorch"
CHECKPOINT="$ROOT/checkpoints/smolvla_base"
SKIP_MODEL=0
SKIP_RUNTIME=0

require_value() { (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }; }
while (($#)); do
    case "$1" in
        --venv) require_value "$@"; VENV="$2"; shift 2 ;;
        --executorch-root) require_value "$@"; EXECUTORCH_ROOT="$2"; shift 2 ;;
        --checkpoint) require_value "$@"; CHECKPOINT="$2"; shift 2 ;;
        --skip-model) SKIP_MODEL=1; shift ;;
        --skip-runtime-build) SKIP_RUNTIME=1; shift ;;
        -h|--help)
            echo "Usage: scripts/setup.sh [--venv PATH] [--executorch-root PATH] [--checkpoint PATH] [--skip-model] [--skip-runtime-build]"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This learning-path runner currently requires Linux." >&2
    exit 1
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Warning: this workflow targets AArch64; detected $(uname -m)." >&2
fi

VENV="$(realpath -m "$VENV")"
EXECUTORCH_ROOT="$(realpath -m "$EXECUTORCH_ROOT")"
CHECKPOINT="$(realpath -m "$CHECKPOINT")"
printf 'Setup locations:\n  virtual environment: %s\n  ExecuTorch source: %s\n  checkpoint: %s\n' \
    "$VENV" "$EXECUTORCH_ROOT" "$CHECKPOINT"

unset PYTHONHOME PYTHONPATH
python3.12 -m venv "$VENV"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
"$VENV/bin/python" -m pip install --upgrade \
    pip 'cmake>=3.24,<4' ninja wheel zstd certifi

if [[ ! -d "$EXECUTORCH_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$EXECUTORCH_ROOT")"
    git clone --no-tags https://github.com/pytorch/executorch.git "$EXECUTORCH_ROOT"
fi
git -C "$EXECUTORCH_ROOT" fetch --no-tags origin "$EXECUTORCH_COMMIT"
git -C "$EXECUTORCH_ROOT" checkout --detach "$EXECUTORCH_COMMIT"
git -C "$EXECUTORCH_ROOT" submodule sync --recursive
git -C "$EXECUTORCH_ROOT" submodule update --init --recursive

# Force the CPU wheel set even on Arm developer systems that also have CUDA.
"$VENV/bin/python" -m pip install \
    torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/test/cpu
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
"$VENV/bin/python" -m pip install \
    torchao==0.18.0 flatbuffers==25.12.19 ruamel.yaml==0.19.1 tabulate==0.10.0

if "$VENV/bin/python" -c \
    'import importlib.metadata as m; assert m.version("executorch") == "1.4.1+e4d02f4"; from executorch.extension.pybindings import portable_lib' \
    >/dev/null 2>&1; then
    echo "ExecuTorch Python package and portable binding already installed"
else
    (
        cd "$EXECUTORCH_ROOT"
        env -u DEBUG \
            CMAKE_ARGS="-DEXECUTORCH_BUILD_CMSIS_NN_PYBINDS=OFF -DEXECUTORCH_BUILD_COREML=OFF -DEXECUTORCH_BUILD_CUDA=OFF -DEXECUTORCH_BUILD_DEVTOOLS=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=OFF -DEXECUTORCH_BUILD_MLX=OFF -DEXECUTORCH_BUILD_OPENVINO=OFF -DEXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_QNN=OFF -DEXECUTORCH_BUILD_VULKAN=OFF -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=ON -DXNNPACK_ENABLE_KLEIDIAI=ON" \
            "$VENV/bin/python" -m pip install . \
                --no-build-isolation --no-deps
    )
fi
"$VENV/bin/python" -m pip install --pre --no-deps \
    torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/test/cpu
"$VENV/bin/python" -m pip install --no-deps lerobot==0.6.0
"$VENV/bin/python" -m pip install -e "$ROOT"

if ((SKIP_MODEL == 0)); then
    "$VENV/bin/hf" download lerobot/smolvla_base \
        config.json model.safetensors policy_preprocessor.json \
        policy_preprocessor_step_5_normalizer_processor.safetensors \
        policy_postprocessor.json \
        policy_postprocessor_step_0_unnormalizer_processor.safetensors \
        --revision "$SMOLVLA_REVISION" --local-dir "$CHECKPOINT"
    "$VENV/bin/hf" download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        merges.txt tokenizer.json tokenizer_config.json added_tokens.json \
        vocab.json special_tokens_map.json chat_template.json config.json \
        processor_config.json preprocessor_config.json generation_config.json \
        --revision "$SMOLVLM_REVISION"
fi

export EXECUTORCH_ROOT
export EXECUTORCH_BUILD_DIR="$EXECUTORCH_ROOT/cmake-out-xnnpack"
export PYTHON_EXECUTABLE="$VENV/bin/python"
export SMOLVLA_CHECKPOINT="$CHECKPOINT"
if ((SKIP_RUNTIME == 0)); then
    "$ROOT/scripts/build_runtime.sh"
fi

PYTHONPATH="$ROOT/src" "$VENV/bin/python" "$ROOT/scripts/check_environment.py"
echo "Setup complete. Run: source env.sh"
