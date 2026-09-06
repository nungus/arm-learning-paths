#!/usr/bin/env python3
"""Verify the pinned software, checkpoint, and ExecuTorch runtime."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED_EXECUTORCH = "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"
EXPECTED = {
    "executorch": "1.4.1+e4d02f4",
    "lerobot": "0.6.0",
    "torch": "2.13.0",
    "torchao": "0.18.0",
    "transformers": "5.5.4",
}
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def require_files(root: Path, relative_paths: tuple[str, ...], label: str) -> None:
    missing = [name for name in relative_paths if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"{label} is missing: {', '.join(missing)}")


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("The native runner requires Linux")
    if platform.machine() != "aarch64":
        raise RuntimeError(
            f"This workflow requires AArch64; detected {platform.machine()}"
        )
    project_root = Path(__file__).resolve().parents[1]
    root = Path(
        os.environ.get("EXECUTORCH_ROOT", project_root / "toolchain/executorch")
    ).resolve()
    runtime_build = Path(
        os.environ.get("EXECUTORCH_BUILD_DIR", root / "cmake-out-xnnpack")
    ).resolve()
    checkpoint = Path(
        os.environ.get("SMOLVLA_CHECKPOINT", project_root / "checkpoints/smolvla_base")
    ).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_EXECUTORCH:
        raise RuntimeError(f"ExecuTorch revision is {revision}, expected {EXPECTED_EXECUTORCH}")
    for package, expected in EXPECTED.items():
        actual = importlib.metadata.version(package)
        if package == "torch":
            actual = actual.split("+")[0]
        if actual != expected:
            raise RuntimeError(f"{package} is {actual}, expected {expected}")
    import torch

    if "+cpu" not in torch.__version__ or torch.version.cuda is not None:
        raise RuntimeError(
            f"CPU-only PyTorch is required; found torch {torch.__version__}, "
            f"CUDA {torch.version.cuda}"
        )
    from executorch.extension.pybindings import portable_lib  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401

    require_files(checkpoint, CHECKPOINT_FILES, "SmolVLA checkpoint")
    require_files(
        runtime_build,
        (
            "CMakeCache.txt",
            "lib/cmake/ExecuTorch/executorch-config.cmake",
        ),
        "ExecuTorch runtime build",
    )
    cache = (runtime_build / "CMakeCache.txt").read_text(errors="replace")
    for setting in (
        "EXECUTORCH_BUILD_XNNPACK:BOOL=ON",
        "EXECUTORCH_XNNPACK_ENABLE_KLEIDI:BOOL=ON",
        "XNNPACK_ENABLE_KLEIDIAI:BOOL=ON",
    ):
        if setting not in cache:
            raise RuntimeError(f"Runtime build does not contain {setting}")

    print(f"Environment OK: {platform.machine()}, ExecuTorch {revision[:8]}")
    print(f"  Python: {sys.executable}")
    print(f"  ExecuTorch: {root}")
    print(f"  Runtime: {runtime_build}")
    print(f"  Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
