"""Validate an optimized artifact and print its native runner shape arguments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from variants import canonical_variant


COMPONENTS = ("vision_encoder", "prefix_forward", "denoise_step")
FIXED_CHECKPOINT_CONTRACT = {
    "chunk_size": 50,
    "padded_state_dim": 32,
    "padded_action_dim": 32,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--expect-variant")
    parser.add_argument("--expect-image-size", type=int)
    parser.add_argument("--expect-language-length", type=int)
    parser.add_argument("--expect-num-steps", type=int)
    args = parser.parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Optimized manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("schema_version", 0)) < 3:
        raise RuntimeError(
            "Manifest predates content-addressed artifact provenance; re-run export"
        )
    test_data_path = artifacts_dir / "test_data.pt"
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Missing optimized test data: {test_data_path}")
    expected_test_data_hash = manifest.get("test_data_sha256")
    actual_test_data_hash = sha256(test_data_path)
    if not expected_test_data_hash or actual_test_data_hash != expected_test_data_hash:
        raise RuntimeError(
            f"test_data.pt hash mismatch: manifest {expected_test_data_hash}, "
            f"file {actual_test_data_hash}"
        )
    if int(manifest.get("schema_version", 0)) >= 4:
        suite = manifest.get("input_suite") or {}
        if (
            not manifest.get("input_suite_manifest_sha256")
            or not manifest.get("input_suite_sha256")
            or suite.get("manifest_sha256")
            != manifest.get("input_suite_manifest_sha256")
            or suite.get("data_sha256") != manifest.get("input_suite_sha256")
        ):
            raise RuntimeError("Manifest has inconsistent input-suite provenance")

    if not manifest.get("optimized_split"):
        raise RuntimeError(
            f"Artifact does not use the optimized native contract: {artifacts_dir}"
        )
    mismatches = {
        name: (manifest.get(name), expected)
        for name, expected in FIXED_CHECKPOINT_CONTRACT.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (runner requires {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(f"Unsupported native shape contract: {details}")
    if int(manifest.get("camera_count", 0)) < 1:
        raise RuntimeError("The native runner requires at least one camera")
    expected_manifest_values = {
        "image_size": args.expect_image_size,
        "language_length": args.expect_language_length,
        "num_steps": args.expect_num_steps,
    }
    if args.expect_variant is not None:
        actual_variant = canonical_variant(manifest)
        if actual_variant != args.expect_variant:
            raise RuntimeError(
                f"Requested variant={args.expect_variant!r}, but component "
                f"quantization records {actual_variant!r} "
                f"(manifest label {manifest.get('variant')!r})"
            )
    for name, expected in expected_manifest_values.items():
        if expected is not None and manifest.get(name) != expected:
            raise RuntimeError(
                f"Requested {name}={expected!r}, but reused manifest records "
                f"{manifest.get(name)!r}"
            )
    if args.manifest_only:
        return


    lower_reports = manifest.get("lowering", {}).get("components", {})
    export_reports = manifest.get("components", {})
    for component in COMPONENTS:
        expected_source_hash = export_reports.get(component, {}).get("pt2_sha256")
        reported_source_hash = lower_reports.get(component, {}).get(
            "source_pt2_sha256"
        )
        if not expected_source_hash:
            raise RuntimeError(f"Manifest has no PT2 hash for {component}")
        if reported_source_hash != expected_source_hash:
            raise RuntimeError(
                f"Lower report source mismatch for {component}: exported PT2 "
                f"{expected_source_hash}, lowered source {reported_source_hash}"
            )
        pte_path = artifacts_dir / f"{component}_xnnpack.pte"
        if not pte_path.is_file():
            raise FileNotFoundError(f"Missing native PTE: {pte_path}")
        expected_hash = lower_reports.get(component, {}).get("pte_sha256")
        if not expected_hash:
            raise RuntimeError(f"Manifest has no PTE hash for {component}")
        actual_hash = sha256(pte_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"PTE hash mismatch for {component}: manifest {expected_hash}, "
                f"file {actual_hash}"
            )

    required_inputs = (
        "native_runner/vision_encoder/images.bin",
        "native_runner/prefix_forward/image_masks.bin",
        "native_runner/prefix_forward/language_tokens.bin",
        "native_runner/prefix_forward/language_mask.bin",
        "native_runner/prefix_forward/state.bin",
        "native_runner/denoise_step/actions.bin",
    )
    for relative in required_inputs:
        path = artifacts_dir / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing native input; run prepare_inputs.py: {path}"
            )

    input_reports = manifest.get("native_inputs")
    if not input_reports:
        raise RuntimeError(
            "Manifest has no native input hashes; run prepare_inputs.py"
        )
    for component, tensors in input_reports.items():
        for name, report in tensors.items():
            path = artifacts_dir / "native_runner" / component / f"{name}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"Missing native input: {path}")
            expected_hash = report.get("sha256")
            actual_hash = sha256(path)
            if not expected_hash or actual_hash != expected_hash:
                raise RuntimeError(
                    f"Native input hash mismatch for {component}/{name}: "
                    f"manifest {expected_hash}, file {actual_hash}"
                )

    print(f"--num_steps={int(manifest['num_steps'])}")
    print(f"--camera_count={int(manifest['camera_count'])}")
    print(f"--image_size={int(manifest['image_size'])}")
    print(f"--language_length={int(manifest['language_length'])}")
    print(f"--action_dim={int(manifest['action_dim'])}")


if __name__ == "__main__":
    main()
