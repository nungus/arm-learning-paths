"""Reusable, variant-independent inputs for optimized SmolVLA export."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


SUITE_SCHEMA_VERSION = 1
SUITE_DATA_NAME = "input_suite.pt"
SUITE_MANIFEST_NAME = "manifest.json"
DEFAULT_TASK = (
    "pick up the red die and place it in the player's tray\n"
)
REQUIRED_TENSORS = (
    "images",
    "image_masks",
    "language_tokens",
    "language_mask",
    "state",
    "noise",
    "initial_timestep",
    "reference_output",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_field(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def sha256_directory(path: Path) -> str:
    """Hash a checkpoint tree independently of its path and mtimes."""

    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {path}")
    digest = hashlib.sha256(b"smolvla-checkpoint-tree-v1\0")
    entries = sorted(
        path.rglob("*"), key=lambda entry: entry.relative_to(path).as_posix()
    )
    for entry in entries:
        relative = entry.relative_to(path).as_posix().encode("utf-8")
        _hash_field(digest, relative)
        if entry.is_symlink():
            digest.update(b"L")
            _hash_field(digest, os.readlink(entry).encode("utf-8"))
        elif entry.is_dir():
            digest.update(b"D")
        elif entry.is_file():
            digest.update(b"F")
            digest.update(
                entry.stat().st_size.to_bytes(8, byteorder="big", signed=False)
            )
            with entry.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            raise RuntimeError(f"Unsupported checkpoint entry type: {entry}")
    return digest.hexdigest()


def load_policy(checkpoint: Path):
    """Load the eager FP32 policy used for references and export."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = checkpoint.expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = "cpu"
    config.compile_model = False
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=False,
    ).eval()
    policy.model.float()
    return policy


def tokenize(
    model, task: str, language_length: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = model.vlm_with_expert.processor.tokenizer
    untruncated = tokenizer(
        [task], padding=False, truncation=False, return_tensors="pt"
    )
    required_tokens = int(untruncated["attention_mask"].sum())
    encoded = tokenizer(
        [task],
        max_length=language_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(torch.int64),
        encoded["attention_mask"].to(torch.bool),
        required_tokens,
    )


def fixed_contract(policy) -> dict[str, int]:
    return {
        "camera_count": len(policy.config.image_features),
        "chunk_size": int(policy.config.chunk_size),
        "padded_state_dim": int(policy.config.max_state_dim),
        "padded_action_dim": int(policy.config.max_action_dim),
        "action_dim": int(policy.config.action_feature.shape[0]),
        "state_dim": int(policy.config.robot_state_feature.shape[0]),
    }


def validate_native_contract(contract: dict[str, int]) -> None:
    required = {
        "chunk_size": 50,
        "padded_state_dim": 32,
        "padded_action_dim": 32,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in required.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint does not match the native runner contract: "
            + ", ".join(
                f"{key}={actual!r}, required {expected!r}"
                for key, (actual, expected) in mismatches.items()
            )
        )
    if contract.get("camera_count", 0) < 1:
        raise RuntimeError("Checkpoint must define at least one camera input")
    action_dim = contract.get("action_dim", 0)
    if not 1 <= action_dim <= contract["padded_action_dim"]:
        raise RuntimeError(
            f"action_dim={action_dim!r} is incompatible with the native runner"
        )


def save_input_suite(
    output_dir: Path,
    tensors: dict[str, Any],
    manifest: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Save a suite without silently replacing an existing comparison set."""

    output_dir = output_dir.expanduser().resolve()
    data_path = output_dir / SUITE_DATA_NAME
    manifest_path = output_dir / SUITE_MANIFEST_NAME
    existing = [path for path in (data_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Input suite already exists; reuse it or pass --force explicitly: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, data_path)
    complete_manifest = {
        **manifest,
        "schema_version": SUITE_SCHEMA_VERSION,
        "input_suite": True,
        "data_file": SUITE_DATA_NAME,
        "data_sha256": sha256_file(data_path),
    }
    manifest_path.write_text(json.dumps(complete_manifest, indent=2) + "\n")
    return data_path, manifest_path


def load_input_suite(
    suite_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    suite_dir = suite_dir.expanduser().resolve()
    manifest_path = suite_dir / SUITE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Input-suite manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("input_suite") is not True
        or int(manifest.get("schema_version", 0)) != SUITE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"Unsupported input-suite manifest: {manifest_path}")
    data_path = suite_dir / manifest.get("data_file", SUITE_DATA_NAME)
    if not data_path.is_file():
        raise FileNotFoundError(f"Input-suite tensors not found: {data_path}")
    actual_data_hash = sha256_file(data_path)
    if actual_data_hash != manifest.get("data_sha256"):
        raise RuntimeError(
            f"Input-suite tensor hash mismatch: manifest "
            f"{manifest.get('data_sha256')}, file {actual_data_hash}"
        )
    tensors = torch.load(data_path, map_location="cpu", weights_only=True)
    missing = [name for name in REQUIRED_TENSORS if name not in tensors]
    if missing:
        raise RuntimeError(f"Input suite lacks tensors: {missing}")
    hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "data_sha256": actual_data_hash,
    }
    return manifest, tensors, hashes
