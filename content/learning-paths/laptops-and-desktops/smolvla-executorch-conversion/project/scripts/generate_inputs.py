#!/usr/bin/env python3
"""Generate a reusable SmolVLA input and reference suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from smolvla_et.input_suite import (
    DEFAULT_TASK,
    fixed_contract,
    load_policy,
    save_input_suite,
    sha256_directory,
    tokenize,
    validate_native_contract,
)
from smolvla_et.smolvla_policy import SmolVLAWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "artifacts" / "input_suites"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT")
    dataset = os.environ.get("SMOLVLA_DATASET")
    configured_suite = os.environ.get("SMOLVLA_INPUT_SUITE")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(checkpoint) if checkpoint else None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(configured_suite) if configured_suite else None,
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--language-length", type=int, default=48)
    parser.add_argument("--task")
    parser.add_argument("--num-steps", type=int, default=10)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-root",
        type=Path,
        default=Path(dataset) if dataset else None,
    )
    source.add_argument("--synthetic-inputs", action="store_true")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace an existing suite instead of reusing it.",
    )
    args = parser.parse_args()
    if args.synthetic_inputs:
        args.data_root = None
    elif args.data_root is None:
        parser.error(
            "--data-root or SMOLVLA_DATASET is required unless "
            "--synthetic-inputs is used"
        )
    if args.checkpoint is None:
        parser.error("--checkpoint or SMOLVLA_CHECKPOINT is required")
    positive = (
        args.image_size,
        args.language_length,
        args.num_steps,
        args.threads,
    )
    if any(value < 1 for value in positive):
        parser.error("all sizes, steps, and threads must be positive")
    if args.output_dir is None:
        source_name = (
            f"synthetic-seed-{args.seed}"
            if args.synthetic_inputs
            else "real-data"
        )
        args.output_dir = (
            DEFAULT_INPUT_ROOT
            / f"{args.image_size}-l{args.language_length}"
            / source_name
        )
    return args


def synthetic_tensors(
    *,
    camera_count: int,
    image_size: int,
    padded_state_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    evaluation_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    images = torch.rand(
        (1, camera_count, 3, image_size, image_size),
        generator=evaluation_generator,
    ).mul(2.0).sub(1.0)
    state = torch.randn(
        (1, padded_state_dim), generator=evaluation_generator
    ).mul(0.25)
    return images, state


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not args.force and any(
        (output_dir / name).exists() for name in ("input_suite.pt", "manifest.json")
    ):
        raise FileExistsError(
            f"Input suite already exists at {output_dir}; reuse it rather than "
            "regenerating it, or pass --force explicitly"
        )

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = sha256_directory(checkpoint)
    policy = load_policy(checkpoint)
    policy.model.config.num_steps = args.num_steps
    contract = fixed_contract(policy)
    validate_native_contract(contract)
    camera_names = tuple(policy.config.image_features)

    if args.synthetic_inputs:
        images, state = synthetic_tensors(
            camera_count=contract["camera_count"],
            image_size=args.image_size,
            padded_state_dim=contract["padded_state_dim"],
            seed=args.seed,
        )
        task = args.task if args.task is not None else DEFAULT_TASK
        input_source = "synthetic"
    else:
        from smolvla_et.data import load_real_sample

        sample = load_real_sample(
            args.data_root,
            camera_names,
            args.image_size,
            contract["padded_state_dim"],
        )
        images = sample.images
        state = sample.state
        task = args.task if args.task is not None else sample.task
        input_source = "real-dataset"

    language_tokens, language_mask, required_tokens = tokenize(
        policy.model, task, args.language_length
    )
    if required_tokens > args.language_length:
        raise RuntimeError(
            f"Task needs {required_tokens} tokens but language length is "
            f"{args.language_length}; refusing to truncate it"
        )
    image_masks = torch.ones(
        (1, contract["camera_count"]), dtype=torch.bool
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    noise = torch.randn(
        (
            1,
            contract["chunk_size"],
            contract["padded_action_dim"],
        ),
        generator=noise_generator,
    )
    initial_timestep = torch.ones((1,), dtype=torch.float32)
    full_policy = SmolVLAWrapper(
        policy.model, contract["action_dim"], contract["camera_count"]
    ).eval()
    upstream_inputs = (
        *(images[:, camera] for camera in range(contract["camera_count"])),
        *(image_masks[:, camera] for camera in range(contract["camera_count"])),
        language_tokens,
        language_mask,
        state,
        noise,
    )
    reference_output = full_policy(*upstream_inputs).detach()

    tensors = {
        "images": images.contiguous(),
        "image_masks": image_masks,
        "language_tokens": language_tokens,
        "language_mask": language_mask,
        "state": state.contiguous(),
        "noise": noise,
        "initial_timestep": initial_timestep,
        "reference_output": reference_output,
        "num_steps": args.num_steps,
        "image_size": args.image_size,
        "language_length": args.language_length,
    }
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "image_size": args.image_size,
        "camera_names": camera_names,
        **contract,
        "language_length": args.language_length,
        "active_language_tokens": int(language_mask.sum()),
        "task_token_count": required_tokens,
        "task": task.rstrip("\n"),
        "num_steps": args.num_steps,
        "input_source": input_source,
        "synthetic_seed": args.seed if args.synthetic_inputs else None,
        "data_root": (
            str(args.data_root.expanduser().resolve())
            if args.data_root is not None
            else None
        ),
        "generation_seconds": time.perf_counter() - started,
    }
    data_path, manifest_path = save_input_suite(
        output_dir, tensors, manifest, force=args.force
    )
    saved_manifest = json.loads(manifest_path.read_text())
    print(
        f"Saved reusable {input_source} suite to {output_dir}\n"
        f"  evaluation task: {task.rstrip()} ({required_tokens} active tokens)\n"
        f"  tensors: {data_path.name} {data_path.stat().st_size / 1e6:.1f} MB\n"
        f"  data SHA256: {saved_manifest['data_sha256']}"
    )


if __name__ == "__main__":
    main()
