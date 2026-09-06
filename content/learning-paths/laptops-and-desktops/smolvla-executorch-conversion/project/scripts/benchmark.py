#!/usr/bin/env python3
"""Run and record a small native latency and accuracy benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATISTIC = re.compile(
    r"^(?P<stage>Vision, \d+ cameras|Prefix|Denoise, all steps|Total inference): "
    r"mean (?P<mean>[0-9.]+) \+/- (?P<stddev>[0-9.]+), "
    r"median (?P<median>[0-9.]+), p95 (?P<p95>[0-9.]+), "
    r"min (?P<minimum>[0-9.]+), max (?P<maximum>[0-9.]+)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("SMOLVLA_ARTIFACTS_DIR", ROOT / "artifacts/fp32")),
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=int(os.environ.get("SMOLVLA_CPU_THREADS", 5)))
    parser.add_argument("--cpu-affinity", default=os.environ.get("SMOLVLA_CPU_AFFINITY", ""))
    parser.add_argument("--vision-cpu-threads", type=int)
    parser.add_argument("--vision-cpu-affinity", default="")
    args = parser.parse_args()
    if min(args.warmup_runs, args.benchmark_runs, args.cpu_threads) < 0 or args.benchmark_runs < 1 or args.cpu_threads < 1:
        parser.error("run counts must be non-negative and threads/runs must be positive")
    return args


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts_dir.expanduser().resolve()
    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runner = Path(
        os.environ.get(
            "SMOLVLA_RUNNER",
            Path(os.environ.get("SMOLVLA_RUNNER_BUILD_DIR", ROOT / "build/runner"))
            / "smolvla_runner",
        )
    ).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"Native runner not found: {runner}")

    contract = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/native_contract.py"), str(artifacts)],
        text=True,
    ).splitlines()
    command = [
        str(runner),
        f"--artifacts_dir={artifacts}",
        f"--output_file={artifacts / 'native_orchestrator_output.bin'}",
        f"--precision_label={manifest['variant']}",
        f"--cpu_threads={args.cpu_threads}",
        f"--warmup_runs={args.warmup_runs}",
        f"--benchmark_runs={args.benchmark_runs}",
        *contract,
    ]
    if args.cpu_affinity:
        command.append(f"--cpu_affinity={args.cpu_affinity}")
    if args.vision_cpu_threads is not None:
        command.append(f"--vision_cpu_threads={args.vision_cpu_threads}")
    if args.vision_cpu_affinity:
        command.append(f"--vision_cpu_affinity={args.vision_cpu_affinity}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    run_stdout = "".join(output_lines)
    statistics = {}
    for line in run_stdout.splitlines():
        match = STATISTIC.fullmatch(line)
        if match:
            stage = match.group("stage")
            if stage.startswith("Vision"):
                stage = "vision"
            elif stage.startswith("Denoise"):
                stage = "denoise"
            elif stage.startswith("Total"):
                stage = "total"
            else:
                stage = stage.lower()
            statistics[stage] = {
                key: float(match.group(key))
                for key in ("mean", "stddev", "median", "p95", "minimum", "maximum")
            }
    if set(statistics) != {"vision", "prefix", "denoise", "total"}:
        raise RuntimeError(f"Could not parse all runner statistics: {statistics.keys()}")

    validation = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_runner.py"), "--artifacts-dir", str(artifacts)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(validation.stdout, end="")
    native_validation = json.loads((artifacts / "native_validation.json").read_text())
    runtime_build = Path(os.environ.get("EXECUTORCH_BUILD_DIR", ""))
    runtime_cache = runtime_build / "CMakeCache.txt"
    executorch_root = Path(os.environ.get("EXECUTORCH_ROOT", ""))
    revision = (
        subprocess.check_output(["git", "-C", str(executorch_root), "rev-parse", "HEAD"], text=True).strip()
        if (executorch_root / ".git").is_dir()
        else "unknown"
    )
    lower = manifest["lowering"]["components"]
    report = {
        "schema_version": 1,
        "variant": manifest["variant"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "input_suite_sha256": manifest["input_suite_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "runner_sha256": sha256(runner),
        "executorch_commit": revision,
        "kleidi_enabled": (
            "EXECUTORCH_XNNPACK_ENABLE_KLEIDI:BOOL=ON" in runtime_cache.read_text()
            if runtime_cache.is_file()
            else None
        ),
        "system": {"machine": platform.machine(), "platform": platform.platform()},
        "configuration": {
            "camera_count": manifest["camera_count"],
            "image_size": manifest["image_size"],
            "language_length": manifest["language_length"],
            "num_steps": manifest["num_steps"],
            "cpu_threads": args.cpu_threads,
            "cpu_affinity": args.cpu_affinity or None,
            "vision_cpu_threads": args.vision_cpu_threads,
            "vision_cpu_affinity": args.vision_cpu_affinity or None,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
        },
        "pte_total_bytes": sum(lower[name]["pte_bytes"] for name in lower),
        "delegate_calls": {name: lower[name]["delegate_calls"] for name in lower},
        "latency_ms": statistics,
        "accuracy": native_validation,
    }
    (artifacts / "benchmark.log").write_text(run_stdout + "\n" + validation.stdout)
    (artifacts / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Benchmark report: {artifacts / 'benchmark.json'}")


if __name__ == "__main__":
    main()
