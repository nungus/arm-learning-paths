"""Validate optimized ExecuTorch PTEs and enforce action-accuracy gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fp32"
COMPONENTS = ("vision_encoder", "prefix_forward", "denoise_step")


def parse_cpu_set(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            if last < first:
                raise ValueError(f"invalid descending CPU range: {part}")
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result:
        raise ValueError("CPU affinity cannot be empty")
    return result


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
    parser.add_argument("--cpu-threads", type=int, default=5)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--min-sqnr-db", type=float)
    parser.add_argument("--max-mae", type=float)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    try:
        args.cpu_set = parse_cpu_set(args.cpu_affinity) if args.cpu_affinity else None
    except ValueError as error:
        parser.error(str(error))
    return args


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().to(torch.float64).flatten()
    actual = actual.detach().to(torch.float64).flatten()
    difference = actual - reference
    absolute_difference = difference.abs()
    signal = reference.square().sum()
    noise = difference.square().sum()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0)
    return {
        "cosine_similarity": float(cosine),
        "sqnr_db": float(10.0 * torch.log10(signal / noise.clamp_min(1e-300))),
        "mae": float(absolute_difference.mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "p95_abs_error": float(torch.quantile(absolute_difference, 0.95)),
        "p99_abs_error": float(torch.quantile(absolute_difference, 0.99)),
        "max_abs_error": float(absolute_difference.max()),
        "reference_rms": float(reference.square().mean().sqrt()),
    }


def load_module(portable_lib, artifacts_dir: Path, name: str):
    path = artifacts_dir / f"{name}_xnnpack.pte"
    if not path.is_file():
        raise FileNotFoundError(f"Missing PTE; run lower.py first: {path}")
    print(f"Loading {name}: {path.name}")
    return portable_lib._load_for_executorch(str(path))


def run(module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    # Clone binding outputs in this validator because later calls may reuse the
    # module's allocator. The deployment runner instead retains EValue owners.
    return tuple(value.clone() for value in module.run_method("forward", inputs))


def thresholds(variant: str, args: argparse.Namespace) -> dict[str, float]:
    if variant == "fp32":
        defaults = {"min_cosine": 0.9999, "min_sqnr_db": 40.0, "max_mae": 0.005}
    else:
        defaults = {"min_cosine": 0.997, "min_sqnr_db": 22.0, "max_mae": 0.03}
    for name in tuple(defaults):
        override = getattr(args, name)
        if override is not None:
            defaults[name] = override
    return defaults


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from executorch.extension.pybindings import portable_lib

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "native_contract.py"),
            str(artifacts_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = artifacts_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = torch.load(
        artifacts_dir / "test_data.pt", map_location="cpu", weights_only=True
    )

    available = os.sched_getaffinity(0)
    if args.cpu_set is not None:
        missing = args.cpu_set - available
        if missing:
            raise RuntimeError(
                f"Requested CPUs {sorted(missing)} are outside process affinity "
                f"{sorted(available)}"
            )
        os.sched_setaffinity(0, args.cpu_set)
    portable_lib._unsafe_reset_threadpool(args.cpu_threads)
    print(
        f"Runtime: {args.cpu_threads} XNNPACK threads, "
        f"affinity={args.cpu_affinity or 'unrestricted'}"
    )

    vision = load_module(portable_lib, artifacts_dir, "vision_encoder")
    prefix = load_module(portable_lib, artifacts_dir, "prefix_forward")
    denoise = load_module(portable_lib, artifacts_dir, "denoise_step")

    image_direct = run(vision, (data["images"],))[0]
    direct_prefix_mask, direct_cache = run(
        prefix,
        (
            data["image_embeddings"],
            data["image_masks"],
            data["language_tokens"],
            data["language_mask"],
            data["state"],
        ),
    )
    direct_actions = run(
        denoise,
        (
            data["prefix_pad_masks"],
            data["flat_cache"],
            data["noise"],
            data["initial_timestep"],
        ),
    )[0]

    prefix_mask, flat_cache = run(
        prefix,
        (
            image_direct,
            data["image_masks"],
            data["language_tokens"],
            data["language_mask"],
            data["state"],
        ),
    )
    actions = data["noise"].clone()
    for step in range(int(data["num_steps"])):
        timestep = torch.full(
            (actions.shape[0],),
            1.0 - step / int(data["num_steps"]),
            dtype=torch.float32,
        )
        actions = run(denoise, (prefix_mask, flat_cache, actions, timestep))[0]

    action_dim = int(manifest["action_dim"])
    real_actions = actions[:, :, :action_dim]
    reference = data["reference_output"]
    metrics = {
        "pte_vision_vs_exported": tensor_metrics(
            data["image_embeddings"], image_direct
        ),
        "pte_prefix_mask_vs_exported": tensor_metrics(
            data["prefix_pad_masks"].to(torch.float32),
            direct_prefix_mask.to(torch.float32),
        ),
        "pte_cache_vs_exported": tensor_metrics(data["flat_cache"], direct_cache),
        "pte_first_step_vs_exported": tensor_metrics(
            data["initial_actions"], direct_actions
        ),
        "pte_pipeline_vs_exported": tensor_metrics(
            data["exported_output"], actions
        ),
        "pte_actions_padded_vs_float_split": tensor_metrics(
            data["float_split_output"], actions
        ),
        "pte_actions_real_dims_vs_upstream": tensor_metrics(reference, real_actions),
    }
    for name, values in metrics.items():
        print(
            f"{name}: cosine={values['cosine_similarity']:.9f}, "
            f"SQNR={values['sqnr_db']:.2f} dB, "
            f"MAE={values['mae']:.6g}, RMSE={values['rmse']:.6g}, "
            f"p95={values['p95_abs_error']:.6g}, "
            f"p99={values['p99_abs_error']:.6g}, "
            f"max={values['max_abs_error']:.6g}"
        )

    final_metrics = metrics["pte_actions_real_dims_vs_upstream"]
    limits = thresholds(manifest["variant"], args)
    failures = []
    if not torch.isfinite(actions).all():
        failures.append("non-finite action output")
    if final_metrics["cosine_similarity"] < limits["min_cosine"]:
        failures.append(
            f"cosine {final_metrics['cosine_similarity']:.6f} "
            f"< {limits['min_cosine']:.6f}"
        )
    if final_metrics["sqnr_db"] < limits["min_sqnr_db"]:
        failures.append(
            f"SQNR {final_metrics['sqnr_db']:.2f} dB "
            f"< {limits['min_sqnr_db']:.2f} dB"
        )
    if final_metrics["mae"] > limits["max_mae"]:
        failures.append(
            f"MAE {final_metrics['mae']:.6g} > {limits['max_mae']:.6g}"
        )

    report = {
        "artifacts_dir": str(artifacts_dir),
        "variant": manifest["variant"],
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": sorted(args.cpu_set) if args.cpu_set is not None else None,
        "thresholds": limits,
        "metrics": metrics,
        "passed": not failures,
        "failures": failures,
    }
    report_path = artifacts_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest["validation"] = report
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise RuntimeError("Accuracy gate failed: " + "; ".join(failures))
    print(f"Accuracy gate passed; report: {report_path}")


if __name__ == "__main__":
    main()
