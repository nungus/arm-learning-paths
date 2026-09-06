import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fp32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate native split-runner actions against PyTorch."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("SMOLVLA_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)),
    )
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--min-sqnr-db", type=float)
    parser.add_argument("--max-mae", type=float)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(metadata + b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def output_metrics(
    reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, float | int]:
    """Compute FP32-reference error metrics over every executable action value."""
    if reference.shape != actual.shape:
        raise ValueError(
            f"output metric shapes differ: {tuple(reference.shape)} vs "
            f"{tuple(actual.shape)}"
        )
    reference = reference.detach().to(torch.float64).flatten()
    actual = actual.detach().to(torch.float64).flatten()
    error = actual - reference
    absolute_error = error.abs()
    signal = reference.square().sum()
    noise = error.square().sum()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0)
    return {
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "rmse": float(error.square().mean().sqrt()),
        "p95_abs_error": float(torch.quantile(absolute_error, 0.95)),
        "p99_abs_error": float(torch.quantile(absolute_error, 0.99)),
        "cosine_similarity": float(cosine),
        "sqnr_db": float(10.0 * torch.log10(signal / noise.clamp_min(1e-300))),
        "evaluated_elements": absolute_error.numel(),
    }


def action_names(manifest: dict, action_dim: int) -> list[str]:
    """Return presentation-friendly action names when dataset metadata exists."""
    names: list[str] = []
    dataset_root = manifest.get("data_root")
    if dataset_root:
        info_path = Path(dataset_root) / "meta" / "info.json"
        if info_path.is_file():
            try:
                raw_names = json.loads(info_path.read_text())["features"]["action"][
                    "names"
                ]
                if isinstance(raw_names, list) and len(raw_names) == action_dim:
                    names = [str(name) for name in raw_names]
            except (KeyError, TypeError, json.JSONDecodeError):
                names = []
    if not names:
        names = [f"Action {index + 1}" for index in range(action_dim)]
    return [name.removesuffix(".pos").replace("_", " ") for name in names]


def action_dimension_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    names: list[str],
) -> list[dict[str, float | int | str]]:
    if reference.shape != actual.shape:
        raise ValueError(
            f"action metric shapes differ: {tuple(reference.shape)} vs "
            f"{tuple(actual.shape)}"
        )
    if reference.shape[-1] != len(names):
        raise ValueError(
            f"received {len(names)} action names for {reference.shape[-1]} dimensions"
        )
    reports = []
    for index, name in enumerate(names):
        error = (
            (actual[..., index].double() - reference[..., index].double())
            .abs()
            .flatten()
        )
        reports.append(
            {
                "index": index,
                "name": name,
                "evaluated_elements": error.numel(),
                "mae": float(error.mean()),
                "rmse": float(error.square().mean().sqrt()),
                "p95_abs_error": float(torch.quantile(error, 0.95)),
                "p99_abs_error": float(torch.quantile(error, 0.99)),
                "max_abs_error": float(error.max()),
            }
        )
    return reports


def thresholds(variant: str, args: argparse.Namespace) -> dict[str, float]:
    defaults = {"min_cosine": 0.997, "min_sqnr_db": 22.0, "max_mae": 0.03}
    for name in tuple(defaults):
        override = getattr(args, name)
        if override is not None:
            defaults[name] = override
    return defaults


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    test_data_path = artifacts_dir / "test_data.pt"
    test_data_file_sha256 = file_sha256(test_data_path)
    test_data = torch.load(
        test_data_path,
        map_location="cpu",
        weights_only=True,
    )
    reference_output = test_data["reference_output"]
    reference_output_sha256 = tensor_sha256(reference_output)
    manifest_path = artifacts_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        expected_test_data_hash = manifest.get("test_data_sha256")
        if (
            not expected_test_data_hash
            or test_data_file_sha256 != expected_test_data_hash
        ):
            raise RuntimeError("test_data.pt does not match the artifact manifest")
        is_quantized = manifest.get("variant") not in (None, "fp32")
        variant = str(manifest.get("variant", "unknown"))
    else:
        manifest = {}
        is_quantized = test_data.get(
            "quantization_mode", "none"
        ) != "none" or test_data.get("int8", False)
        variant = str(test_data.get("variant", "unknown"))
    output_path = (
        Path(
            os.environ.get(
                "SMOLVLA_OUTPUT_FILE",
                artifacts_dir / "native_orchestrator_output.bin",
            )
        )
        .expanduser()
        .resolve()
    )
    raw_output = np.fromfile(output_path, dtype=np.float32)
    native_output_sha256 = file_sha256(output_path)
    if raw_output.size != reference_output.numel():
        raise RuntimeError(
            f"{output_path} contains {raw_output.size} elements, but the reference "
            f"requires {reference_output.numel()}."
        )

    actual_output = torch.from_numpy(raw_output.reshape(reference_output.shape))
    metrics = output_metrics(reference_output, actual_output)
    per_dimension = action_dimension_metrics(
        reference_output,
        actual_output,
        action_names(manifest, reference_output.shape[-1]),
    )
    failures = []
    limits = thresholds(variant, args)
    if not torch.isfinite(actual_output).all():
        failures.append("native inference produced non-finite actions")
    if is_quantized:
        if metrics["cosine_similarity"] < limits["min_cosine"]:
            failures.append(
                f"cosine {metrics['cosine_similarity']:.6f} "
                f"< {limits['min_cosine']:.6f}"
            )
        if metrics["sqnr_db"] < limits["min_sqnr_db"]:
            failures.append(
                f"SQNR {metrics['sqnr_db']:.2f} dB < {limits['min_sqnr_db']:.2f} dB"
            )
        if metrics["mean_absolute_error"] > limits["max_mae"]:
            failures.append(
                f"MAE {metrics['mean_absolute_error']:.6g} > {limits['max_mae']:.6g}"
            )
    else:
        torch.testing.assert_close(
            actual_output,
            reference_output,
            atol=2e-2,
            rtol=1e-3,
        )

    print(f"Maximum absolute error: {metrics['maximum_absolute_error']}")
    print(f"Mean absolute error: {metrics['mean_absolute_error']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"P95 absolute error: {metrics['p95_abs_error']}")
    print(f"P99 absolute error: {metrics['p99_abs_error']}")
    print(f"Cosine similarity: {metrics['cosine_similarity']}")
    print(f"SQNR: {metrics['sqnr_db']} dB")
    report = {
        "schema_version": 2,
        "output": str(output_path),
        "quantized": is_quantized,
        "test_data_sha256": test_data_file_sha256,
        "reference_output_sha256": reference_output_sha256,
        "native_output_sha256": native_output_sha256,
        "output_shape": list(reference_output.shape),
        **metrics,
        "thresholds": limits if is_quantized else None,
        "action_dimensions": per_dimension,
        "passed": not failures,
        "failures": failures,
    }
    (artifacts_dir / "native_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if failures:
        raise RuntimeError("Native accuracy gate failed: " + "; ".join(failures))
    if is_quantized:
        print(
            "Native quantized accuracy gate passed against the floating-point reference."
        )
    else:
        print("Native split orchestrator output matches the full PyTorch reference.")


if __name__ == "__main__":
    main()
