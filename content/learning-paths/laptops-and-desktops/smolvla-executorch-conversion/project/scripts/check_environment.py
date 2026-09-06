#!/usr/bin/env python3
"""Verify the pinned Python and ExecuTorch environment."""

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


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("The native runner requires Linux")
    root = Path(os.environ.get("EXECUTORCH_ROOT", "toolchain/executorch")).resolve()
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

    print(f"Environment OK: {platform.machine()}, ExecuTorch {revision[:8]}")
    print(f"  Python: {sys.executable}")
    print(f"  ExecuTorch: {root}")


if __name__ == "__main__":
    main()
