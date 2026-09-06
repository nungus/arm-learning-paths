#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: bash download-project.sh [destination]" >&2
    exit 2
fi

DESTINATION="${1:-smolvla-executorch-arm}"
BASE_URL="${SMOLVLA_LP_BASE_URL:-https://raw.githubusercontent.com/ArmDeveloperEcosystem/arm-learning-paths/main/content/learning-paths/laptops-and-desktops/smolvla-executorch-conversion/project}"

if [[ -e "$DESTINATION" ]]; then
    echo "Destination already exists: $DESTINATION" >&2
    echo "Choose another destination or remove the existing directory." >&2
    exit 1
fi

FILES=(
    env.sh
    pyproject.toml
    requirements-real-data.txt
    requirements.txt
    runtime/CMakeLists.txt
    runtime/main.cpp
    scripts/benchmark.py
    scripts/build_runner.sh
    scripts/build_runtime.sh
    scripts/check_environment.py
    scripts/compare.py
    scripts/export.py
    scripts/generate_inputs.py
    scripts/lower.py
    scripts/native_contract.py
    scripts/pipeline.sh
    scripts/prepare_inputs.py
    scripts/run_runner.sh
    scripts/setup.sh
    scripts/validate_pte.py
    scripts/validate_runner.py
    scripts/variants.py
    src/sitecustomize.py
    src/smolvla_et/__init__.py
    src/smolvla_et/data.py
    src/smolvla_et/input_suite.py
    src/smolvla_et/optimized_split_policy.py
    src/smolvla_et/quantization.py
    src/smolvla_et/smolvla_policy.py
    src/smolvla_et/split_policy.py
    tests/test_workflow.py
)

for file in "${FILES[@]}"; do
    mkdir -p "$DESTINATION/$(dirname "$file")"
    echo "Downloading $file"
    curl -fL --retry 3 --silent --show-error \
        --output "$DESTINATION/$file" "$BASE_URL/$file"
done

chmod +x "$DESTINATION/env.sh" \
    "$DESTINATION"/scripts/*.sh "$DESTINATION"/scripts/*.py

echo "Downloaded the project to $DESTINATION"
echo "Next: cd $DESTINATION"
