"""Lower fixed-shape split exports to ExecuTorch/XNNPACK programs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fp32"
COMPONENTS = ("vision_encoder", "prefix_forward", "denoise_step")


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
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        default=list(COMPONENTS),
    )
    return parser.parse_args()


def configure_flatc() -> Path:
    configured = os.environ.get("FLATC_EXECUTABLE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"FLATC_EXECUTABLE does not exist: {candidate}")

    executable = shutil.which("flatc")
    candidates = (
        Path(executable) if executable else None,
        Path(sys.prefix) / "bin" / "flatc",
        Path(os.environ["EXECUTORCH_BUILD_DIR"])
        / "third-party/flatc_ep/bin/flatc"
        if os.environ.get("EXECUTORCH_BUILD_DIR")
        else None,
    )
    candidate = next(
        (path.resolve() for path in candidates if path and path.is_file()), None
    )
    if candidate is None:
        raise RuntimeError(
            "FlatBuffers compiler not found; activate the project environment "
            "or set FLATC_EXECUTABLE"
        )
    os.environ["FLATC_EXECUTABLE"] = str(candidate)
    return candidate


def component_args(name: str, data: dict) -> tuple[torch.Tensor, ...]:
    if name == "vision_encoder":
        return (data["images"],)
    if name == "prefix_forward":
        return (
            data["image_embeddings"],
            data["image_masks"],
            data["language_tokens"],
            data["language_mask"],
            data["state"],
        )
    if name == "denoise_step":
        return (
            data["prefix_pad_masks"],
            data["flat_cache"],
            data["noise"],
            data["initial_timestep"],
        )
    raise ValueError(f"Unknown component: {name}")


def expected_outputs(name: str, data: dict) -> tuple[torch.Tensor, ...]:
    if name == "vision_encoder":
        return (data["image_embeddings"],)
    if name == "prefix_forward":
        return (data["prefix_pad_masks"], data["flat_cache"])
    if name == "denoise_step":
        return (data["initial_actions"],)
    raise ValueError(f"Unknown component: {name}")


def as_outputs(value) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def target_name(target) -> str:
    try:
        return target.name()
    except (AttributeError, TypeError):
        return str(target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_export_sources(
    artifacts_dir: Path,
    manifest: dict,
    test_data_path: Path,
) -> dict[str, str]:
    """Validate the complete exported-program source set before lowering."""
    if int(manifest.get("schema_version", 0)) < 3:
        raise RuntimeError(
            "Manifest predates content-addressed PT2 provenance; re-run "
            "export.py"
        )
    if not manifest.get("checkpoint_sha256"):
        raise RuntimeError("Manifest has no checkpoint_sha256")

    expected_test_data = manifest.get("test_data_sha256")
    if not expected_test_data:
        raise RuntimeError("Manifest has no test_data_sha256")
    actual_test_data = sha256(test_data_path)
    if actual_test_data != expected_test_data:
        raise RuntimeError(
            "test_data.pt hash mismatch: manifest "
            f"{expected_test_data}, file {actual_test_data}"
        )

    component_reports = manifest.get("components", {})
    source_hashes: dict[str, str] = {}
    for name in COMPONENTS:
        pt2_path = artifacts_dir / f"{name}.pt2"
        if not pt2_path.is_file():
            raise FileNotFoundError(f"Missing export: {pt2_path}")
        expected_hash = component_reports.get(name, {}).get("pt2_sha256")
        if not expected_hash:
            raise RuntimeError(f"Manifest has no PT2 hash for {name}")
        actual_hash = sha256(pt2_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"PT2 hash mismatch for {name}: manifest {expected_hash}, "
                f"file {actual_hash}"
            )
        source_hashes[name] = actual_hash
    return source_hashes


def validate_untouched_pte_reports(
    artifacts_dir: Path,
    manifest: dict,
    selected_components: set[str],
    source_hashes: dict[str, str],
) -> None:
    """Prove that every PTE preserved by selective lowering is current."""
    lower_reports = manifest.get("lowering", {}).get("components", {})
    for name in COMPONENTS:
        if name in selected_components:
            continue
        report = lower_reports.get(name)
        if not isinstance(report, dict):
            raise RuntimeError(
                f"Selective lowering cannot preserve {name}: no lower report"
            )
        if report.get("source_pt2_sha256") != source_hashes[name]:
            raise RuntimeError(
                f"Selective lowering cannot preserve {name}: its lower report "
                "does not correspond to the current PT2"
            )

        report_path = artifacts_dir / f"{name}.lower.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Selective lowering cannot preserve {name}: missing {report_path}"
            )
        if json.loads(report_path.read_text()) != report:
            raise RuntimeError(
                f"Selective lowering cannot preserve {name}: {report_path.name} "
                "does not match the manifest lower report"
            )

        pte_path = artifacts_dir / f"{name}_xnnpack.pte"
        if not pte_path.is_file():
            raise FileNotFoundError(
                f"Selective lowering cannot preserve {name}: missing {pte_path}"
            )
        expected_pte_hash = report.get("pte_sha256")
        actual_pte_hash = sha256(pte_path)
        if not expected_pte_hash or actual_pte_hash != expected_pte_hash:
            raise RuntimeError(
                f"Selective lowering cannot preserve {name}: PTE hash mismatch "
                f"(manifest {expected_pte_hash}, file {actual_pte_hash})"
            )


@torch.no_grad()
def lower_component(
    name: str,
    artifacts_dir: Path,
    test_data: dict,
    quantization: str,
    expected_source_pt2_sha256: str,
) -> dict:
    pt2_path = artifacts_dir / f"{name}.pt2"
    if not pt2_path.is_file():
        raise FileNotFoundError(f"Missing export: {pt2_path}")
    source_pt2_sha256 = sha256(pt2_path)
    if source_pt2_sha256 != expected_source_pt2_sha256:
        raise RuntimeError(
            f"PT2 changed before lowering {name}: expected "
            f"{expected_source_pt2_sha256}, file {source_pt2_sha256}"
        )
    print(f"\nLoading {name}: {pt2_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        exported = torch.export.load(pt2_path)

    inputs = component_args(name, test_data)
    actual_outputs = as_outputs(exported.module()(*inputs))
    for actual, expected in zip(
        actual_outputs, expected_outputs(name, test_data), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    print("Saved PT2 output matches the recorded component boundary.")

    started = time.perf_counter()
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner(verbose=False)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=quantization == "none",
            # Layout propagation generated hundreds of copies in the legacy
            # graph. XNNPACK already owns layout inside each delegate segment.
            _skip_dim_order=True,
        ),
    )
    lower_seconds = time.perf_counter() - started
    graph = edge.exported_program().graph
    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    delegate_nodes = [
        node
        for node in call_nodes
        if "executorch_call_delegate" in target_name(node.target)
    ]
    remaining = Counter(
        target_name(node.target)
        for node in call_nodes
        if "executorch_call_delegate" not in target_name(node.target)
    )
    if not delegate_nodes:
        raise RuntimeError(f"{name} did not delegate any work to XNNPACK")

    started = time.perf_counter()
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    pte_path = artifacts_dir / f"{name}_xnnpack.pte"
    with pte_path.open("wb") as handle:
        program.write_to_file(handle)
    serialize_seconds = time.perf_counter() - started
    edge_path = artifacts_dir / f"{name}.edge.txt"
    edge_path.write_text(str(graph))

    report = {
        "component": name,
        "quantization": quantization,
        "pte": str(pte_path.resolve()),
        "pte_bytes": pte_path.stat().st_size,
        "pte_sha256": sha256(pte_path),
        "source_pt2_sha256": source_pt2_sha256,
        "delegate_calls": len(delegate_nodes),
        "remaining_call_function_nodes": len(call_nodes) - len(delegate_nodes),
        "remaining_ops": dict(sorted(remaining.items())),
        "skip_dim_order": True,
        "lower_seconds": lower_seconds,
        "serialize_seconds": serialize_seconds,
    }
    report_path = artifacts_dir / f"{name}.lower.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {pte_path.name}: {report['pte_bytes'] / 1e6:.1f} MB, "
        f"{report['delegate_calls']} delegate calls, "
        f"{report['remaining_call_function_nodes']} portable call nodes"
    )
    del program, edge, exported
    gc.collect()
    return report


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    manifest_path = artifacts_dir / "manifest.json"
    test_data_path = artifacts_dir / "test_data.pt"
    if not manifest_path.is_file() or not test_data_path.is_file():
        raise FileNotFoundError(
            f"Run export.py first; missing manifest/test data in {artifacts_dir}"
        )
    flatc = configure_flatc()
    print(f"Using FlatBuffers compiler: {flatc}")
    manifest = json.loads(manifest_path.read_text())
    source_hashes = validate_export_sources(
        artifacts_dir,
        manifest,
        test_data_path,
    )
    selected_components = set(args.components)
    if selected_components != set(COMPONENTS):
        validate_untouched_pte_reports(
            artifacts_dir,
            manifest,
            selected_components,
            source_hashes,
        )
    test_data = torch.load(test_data_path, map_location="cpu", weights_only=True)
    reports = {}
    for name in args.components:
        quantization = manifest["components"][name]["quantization"]
        reports[name] = lower_component(
            name, artifacts_dir, test_data, quantization, source_hashes[name]
        )

    lowering = manifest.setdefault("lowering", {})
    component_reports = dict(lowering.get("components", {}))
    component_reports.update(reports)
    manifest["lowering"].update(
        {
            "backend": "XNNPACK",
            "skip_dim_order": True,
            "flatc": str(flatc),
            "components": component_reports,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nLowered optimized split programs in {artifacts_dir}")


if __name__ == "__main__":
    main()
