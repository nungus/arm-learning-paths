"""Export the recommended fixed-shape split SmolVLA programs."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import platform
import time
from pathlib import Path

import torch

from smolvla_et.input_suite import (
    fixed_contract,
    load_input_suite,
    load_policy,
    sha256_directory,
    sha256_file,
    validate_native_contract,
)
from smolvla_et.optimized_split_policy import (
    BatchedSmolVLAVisionEncoder,
    OptimizedSmolVLADenoiseStep,
    OptimizedSmolVLAPrefix,
    install_exportable_rope,
)
from smolvla_et.quantization import (
    quantize_dynamic_int8,
)
from variants import QUANTIZATION_PLANS, component_quantization


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fp32"
COMPONENTS = ("vision_encoder", "prefix_forward", "denoise_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = os.environ.get("SMOLVLA_CHECKPOINT")
    input_suite = os.environ.get("SMOLVLA_INPUT_SUITE")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(checkpoint) if checkpoint else None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("SMOLVLA_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
        ),
    )
    parser.add_argument(
        "--variant", choices=tuple(QUANTIZATION_PLANS), default="fp32"
    )
    parser.add_argument(
        "--input-suite",
        type=Path,
        default=Path(input_suite) if input_suite else None,
    )
    parser.add_argument("--expect-image-size", type=int)
    parser.add_argument("--expect-language-length", type=int)
    parser.add_argument("--expect-num-steps", type=int)
    parser.add_argument("--expect-task")
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint or SMOLVLA_CHECKPOINT is required")
    if args.input_suite is None:
        parser.error("--input-suite or SMOLVLA_INPUT_SUITE is required")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().to(torch.float64).flatten()
    actual = actual.detach().to(torch.float64).flatten()
    difference = actual - reference
    signal = reference.square().sum()
    noise = difference.square().sum()
    return {
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(reference, actual, dim=0)
        ),
        "sqnr_db": float(10 * torch.log10(signal / noise.clamp_min(1e-300))),
        "mae": float(difference.abs().mean()),
        "max_abs_error": float(difference.abs().max()),
    }


def environment_metadata(policy) -> dict:
    packages = {}
    for name in ("torch", "executorch", "lerobot", "transformers", "torchao"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "packages": packages,
        "parameter_count": sum(
            parameter.numel() for parameter in policy.model.parameters()
        ),
        "parameter_bytes_fp32": sum(
            parameter.numel() * parameter.element_size()
            for parameter in policy.model.parameters()
        ),
    }


def save_program(
    output_dir: Path, name: str, program: torch.export.ExportedProgram
) -> dict:
    path = output_dir / f"{name}.pt2"
    torch.export.save(program, path)
    report = {
        "component": name,
        "pt2": str(path.resolve()),
        "pt2_bytes": path.stat().st_size,
        "pt2_sha256": sha256_file(path),
        "graph_nodes": len(list(program.graph.nodes)),
    }
    print(
        f"Saved {name}.pt2: {report['pt2_bytes'] / 1e6:.1f} MB, "
        f"{report['graph_nodes']} nodes"
    )
    return report


def export_component(
    variant: str,
    name: str,
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.export.ExportedProgram, str]:
    quantization = component_quantization(variant, name)
    if quantization == "none":
        return torch.export.export(module, example_inputs, strict=False), "none"
    if quantization == "dynamic-per-channel-int8":
        return quantize_dynamic_int8(module, example_inputs), quantization
    raise AssertionError(f"Unhandled quantization mode: {quantization}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = sha256_directory(checkpoint)
    suite_dir = args.input_suite.expanduser().resolve()
    suite_manifest, suite, suite_hashes = load_input_suite(suite_dir)
    if suite_manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(
            "Input suite was generated from a different checkpoint: "
            f"suite {suite_manifest.get('checkpoint_sha256')}, "
            f"export {checkpoint_sha256}"
        )
    torch.manual_seed(int(suite_manifest.get("synthetic_seed") or 0))
    policy = load_policy(checkpoint)
    contract = fixed_contract(policy)
    validate_native_contract(contract)
    for name, actual in contract.items():
        if suite_manifest.get(name) != actual:
            raise RuntimeError(
                f"Input suite records {name}={suite_manifest.get(name)!r}, "
                f"but checkpoint requires {actual!r}"
            )
    image_size = int(suite_manifest["image_size"])
    language_length = int(suite_manifest["language_length"])
    num_steps = int(suite_manifest["num_steps"])
    expected_configuration = {
        "image_size": args.expect_image_size,
        "language_length": args.expect_language_length,
        "num_steps": args.expect_num_steps,
    }
    for name, expected in expected_configuration.items():
        if expected is not None and int(suite_manifest[name]) != expected:
            raise RuntimeError(
                f"Requested {name}={expected}, but input suite records "
                f"{suite_manifest[name]}"
            )
    if (
        args.expect_task is not None
        and suite_manifest["task"] != args.expect_task.rstrip("\n")
    ):
        raise RuntimeError(
            f"Requested task {args.expect_task.rstrip()!r}, but input suite "
            f"records {suite_manifest['task']!r}"
        )
    policy.model.config.num_steps = num_steps
    install_exportable_rope()
    camera_names = tuple(policy.config.image_features)
    if tuple(suite_manifest.get("camera_names", ())) != camera_names:
        raise RuntimeError(
            "Input-suite camera order differs from the checkpoint: "
            f"suite {suite_manifest.get('camera_names')}, checkpoint {camera_names}"
        )
    scalar_data = {
        "image_size": image_size,
        "language_length": language_length,
        "num_steps": num_steps,
    }
    for name, expected in scalar_data.items():
        if int(suite[name]) != expected:
            raise RuntimeError(
                f"Input-suite tensor file has {name}={suite[name]!r}, "
                f"manifest records {expected}"
            )
    images = suite["images"]
    image_masks = suite["image_masks"]
    language_tokens = suite["language_tokens"]
    language_masks = suite["language_mask"]
    state = suite["state"]
    noise = suite["noise"]
    initial_timestep = suite["initial_timestep"]
    reference_output = suite["reference_output"]
    expected_image_shape = (1, len(camera_names), 3, image_size, image_size)
    if tuple(images.shape) != expected_image_shape:
        raise RuntimeError(
            f"Input suite images have shape {tuple(images.shape)}, expected "
            f"{expected_image_shape}"
        )
    print(
        f"Configuration: {len(camera_names)} cameras, {image_size}px, "
        f"language L={language_length} "
        f"({suite_manifest['active_language_tokens']} active), "
        f"variant={args.variant}, suite={suite_dir}, FP32 graph boundaries"
    )

    model = policy.model.eval()
    vision = BatchedSmolVLAVisionEncoder(
        model, image_size, len(camera_names)
    ).eval()
    prefix = OptimizedSmolVLAPrefix(model).eval()
    denoise = OptimizedSmolVLADenoiseStep(model).eval()
    float_image_embeddings = vision(images).detach()
    float_prefix_masks, float_cache = prefix(
        float_image_embeddings,
        image_masks,
        language_tokens,
        language_masks,
        state,
    )
    float_output = noise.clone()
    for step in range(num_steps):
        timestep = torch.full((1,), 1.0 - step / num_steps)
        float_output = denoise(
            float_prefix_masks, float_cache, float_output, timestep
        )
    split_metrics = tensor_metrics(
        reference_output,
        float_output[:, :, : reference_output.shape[-1]],
    )
    print(f"Optimized split vs upstream: {json.dumps(split_metrics)}")
    if split_metrics["cosine_similarity"] < 0.999999:
        raise RuntimeError("Optimized split wrapper does not preserve upstream output")

    vision_inputs = (images,)
    vision_program, vision_quant = export_component(
        args.variant,
        "vision_encoder",
        vision,
        vision_inputs,
    )
    exported_vision = vision_program.module()
    image_embeddings = exported_vision(*vision_inputs).detach()
    vision_metrics = tensor_metrics(float_image_embeddings, image_embeddings)

    prefix_inputs = (
        image_embeddings,
        image_masks,
        language_tokens,
        language_masks,
        state,
    )
    prefix_program, prefix_quant = export_component(
        args.variant,
        "prefix_forward",
        prefix,
        prefix_inputs,
    )
    exported_prefix = prefix_program.module()
    prefix_masks, flat_cache = (
        value.detach() for value in exported_prefix(*prefix_inputs)
    )
    _, float_cache_at_boundary = prefix(*prefix_inputs)
    prefix_cache_metrics = tensor_metrics(float_cache_at_boundary, flat_cache)

    denoise_inputs = (prefix_masks, flat_cache, noise, initial_timestep)
    denoise_program, denoise_quant = export_component(
        args.variant,
        "denoise_step",
        denoise,
        denoise_inputs,
    )
    exported_denoise = denoise_program.module()
    initial_actions = exported_denoise(*denoise_inputs).detach()
    float_initial_actions = denoise(*denoise_inputs).detach()
    denoise_step_metrics = tensor_metrics(float_initial_actions, initial_actions)
    exported_output = noise.clone()
    for step in range(num_steps):
        timestep = torch.full((1,), 1.0 - step / num_steps)
        exported_output = exported_denoise(
            prefix_masks, flat_cache, exported_output, timestep
        )
    final_output = exported_output[:, :, : reference_output.shape[-1]]
    quantized_metrics = tensor_metrics(reference_output, final_output)
    print(f"Exported pipeline vs upstream: {json.dumps(quantized_metrics)}")
    if not torch.isfinite(final_output).all():
        raise RuntimeError("Exported pipeline produced non-finite output")
    accuracy_thresholds = (
        {"min_cosine": 0.999999, "min_sqnr_db": 80.0, "max_mae": 1e-5}
        if args.variant == "fp32"
        else {"min_cosine": 0.997, "min_sqnr_db": 22.0, "max_mae": 0.03}
    )
    failures = []
    if quantized_metrics["cosine_similarity"] < accuracy_thresholds["min_cosine"]:
        failures.append("cosine similarity")
    if quantized_metrics["sqnr_db"] < accuracy_thresholds["min_sqnr_db"]:
        failures.append("SQNR")
    if quantized_metrics["mae"] > accuracy_thresholds["max_mae"]:
        failures.append("MAE")
    if failures:
        raise RuntimeError(
            "Exported action accuracy gate failed (" + ", ".join(failures) + "): "
            + json.dumps(quantized_metrics)
        )

    component_reports = {}
    for name, program, quantization in (
        ("vision_encoder", vision_program, vision_quant),
        ("prefix_forward", prefix_program, prefix_quant),
        ("denoise_step", denoise_program, denoise_quant),
    ):
        report = save_program(args.output_dir, name, program)
        report["quantization"] = quantization
        component_reports[name] = report

    test_data_path = args.output_dir / "test_data.pt"
    torch.save(
        {
            "images": images,
            "image_masks": image_masks,
            "language_tokens": language_tokens,
            "language_mask": language_masks,
            "state": state,
            "noise": noise,
            "initial_timestep": initial_timestep,
            "image_embeddings": image_embeddings,
            "prefix_pad_masks": prefix_masks,
            "flat_cache": flat_cache,
            "initial_actions": initial_actions,
            "float_split_output": float_output,
            "exported_output": exported_output,
            "reference_output": reference_output,
            "num_steps": num_steps,
            "image_size": image_size,
            "language_length": language_length,
            "input_suite_data_sha256": suite_hashes["data_sha256"],
            "variant": args.variant,
            "quantization_mode": (
                "none" if args.variant == "fp32" else args.variant
            ),
        },
        test_data_path,
    )
    manifest = {
        "schema_version": 4,
        "optimized_split": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "input_suite": {
            "path": str(suite_dir),
            "schema_version": suite_manifest["schema_version"],
            "manifest_sha256": suite_hashes["manifest_sha256"],
            "data_sha256": suite_hashes["data_sha256"],
        },
        "input_suite_manifest_sha256": suite_hashes["manifest_sha256"],
        "input_suite_sha256": suite_hashes["data_sha256"],
        "environment": environment_metadata(policy),
        "variant": args.variant,
        "precision": "fp32",
        "image_size": image_size,
        "camera_names": camera_names,
        "camera_count": len(camera_names),
        "language_length": language_length,
        "active_language_tokens": suite_manifest["active_language_tokens"],
        "task_token_count": suite_manifest["task_token_count"],
        "task": suite_manifest["task"],
        "num_steps": num_steps,
        "state_dim": contract["state_dim"],
        "padded_state_dim": contract["padded_state_dim"],
        "action_dim": contract["action_dim"],
        "padded_action_dim": contract["padded_action_dim"],
        "chunk_size": contract["chunk_size"],
        "image_tokens_per_camera": float_image_embeddings.shape[2],
        "prefix_length": float_prefix_masks.shape[1],
        "kv_cache_shape": list(float_cache.shape),
        "input_source": suite_manifest["input_source"],
        "synthetic_seed": suite_manifest.get("synthetic_seed"),
        "data_root": suite_manifest.get("data_root"),
        "split_vs_upstream": split_metrics,
        "exported_vs_upstream": quantized_metrics,
        "accuracy_thresholds": accuracy_thresholds,
        "component_metrics": {
            "vision_vs_float": vision_metrics,
            "prefix_cache_vs_float_at_quantized_boundary": prefix_cache_metrics,
            "denoise_step_vs_float_at_quantized_boundary": denoise_step_metrics,
        },
        "components": component_reports,
        "test_data_sha256": sha256_file(test_data_path),
        "export_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved optimized split export to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
