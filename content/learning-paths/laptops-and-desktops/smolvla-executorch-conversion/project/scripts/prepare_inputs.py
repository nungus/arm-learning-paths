"""Write optimized split test tensors as native-runner binary inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fp32"

COMPONENT_INPUTS = {
    "vision_encoder": ("images",),
    "prefix_forward": (
        "image_embeddings",
        "image_masks",
        "language_tokens",
        "language_mask",
        "state",
    ),
    "denoise_step": (
        "prefix_pad_masks",
        "flat_cache",
        "actions",
        "initial_timestep",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "SMOLVLA_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR
            )
        ),
    )
    return parser.parse_args()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    manifest_path = artifacts_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", 0)) < 3:
        raise RuntimeError(
            "Manifest predates hashed test data; re-run export.py"
        )
    test_data_path = artifacts_dir / "test_data.pt"
    expected_test_data_hash = manifest.get("test_data_sha256")
    actual_test_data_hash = sha256_file(test_data_path)
    if not expected_test_data_hash or actual_test_data_hash != expected_test_data_hash:
        raise RuntimeError(
            f"test_data.pt hash mismatch: manifest {expected_test_data_hash}, "
            f"file {actual_test_data_hash}"
        )
    test_data = torch.load(
        test_data_path, map_location="cpu", weights_only=True
    )
    native_dir = artifacts_dir / "native_runner"
    values = dict(test_data)
    values["actions"] = test_data["noise"]
    metadata = {}

    for component, names in COMPONENT_INPUTS.items():
        component_dir = native_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preparing {component} inputs:")
        metadata[component] = {}
        for name in names:
            tensor = values[name].detach().cpu().contiguous()
            path = component_dir / f"{name}.bin"
            tensor.numpy().tofile(path)
            metadata[component][name] = {
                "path": str(path.resolve()),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")

    # Compatibility with the legacy runner name while the optimized runner
    # consumes actions.bin directly.
    actions = values["actions"].detach().cpu().contiguous()
    actions.numpy().tofile(native_dir / "denoise_step" / "noise.bin")
    (native_dir / "inputs.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    manifest["native_inputs"] = metadata
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Native inputs written beneath {native_dir}")


if __name__ == "__main__":
    main()
