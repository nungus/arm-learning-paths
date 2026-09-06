#!/usr/bin/env python3
"""Compare matched FP32 and INT8 benchmark reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path: Path) -> dict:
    report = json.loads((path / "benchmark.json").read_text())
    if not report["accuracy"]["passed"]:
        raise RuntimeError(f"Accuracy gate failed for {path}")
    return report


def load_actions(path: Path, report: dict) -> np.ndarray:
    shape = tuple(int(value) for value in report["accuracy"]["output_shape"])
    output = np.fromfile(path / "native_orchestrator_output.bin", dtype=np.float32)
    if output.size != math.prod(shape):
        raise RuntimeError(
            f"Native output in {path} has {output.size} values; expected {math.prod(shape)}"
        )
    return output.reshape(shape)


def font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def save_action_figure(
    output: Path,
    fp32_actions: np.ndarray,
    int8_actions: np.ndarray,
    dimension_names: list[str],
) -> list[dict[str, float | int | str]]:
    if fp32_actions.shape != int8_actions.shape:
        raise RuntimeError(
            f"FP32 and INT8 output shapes differ: {fp32_actions.shape} vs {int8_actions.shape}"
        )
    if fp32_actions.ndim != 3 or fp32_actions.shape[0] != 1:
        raise RuntimeError(
            f"Expected action shape [1, steps, dimensions], got {fp32_actions.shape}"
        )
    steps, dimensions = fp32_actions.shape[1:]
    if len(dimension_names) != dimensions:
        raise RuntimeError(
            f"Received {len(dimension_names)} action names for {dimensions} dimensions"
        )

    columns = 2
    rows = math.ceil(dimensions / columns)
    canvas_width = 1200
    header_height = 105
    panel_height = 245
    canvas_height = header_height + rows * panel_height + 35
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(26, bold=True)
    subtitle_font = font(15)
    panel_title_font = font(16, bold=True)
    label_font = font(12)
    metric_font = font(12, bold=True)
    fp32_color = "#0072B2"
    int8_color = "#D55E00"
    grid_color = "#D9DEE3"
    axis_color = "#4B5563"

    draw.text(
        (48, 18), "FP32 and INT8 action trajectories", fill="#17212B", font=title_font
    )
    draw.text(
        (48, 55),
        "Normalized action output across the 50-step prediction chunk",
        fill=axis_color,
        font=subtitle_font,
    )
    legend_y = 80
    draw.line((48, legend_y, 82, legend_y), fill=fp32_color, width=4)
    draw.text((90, legend_y - 9), "FP32", fill=axis_color, font=label_font)
    draw.line((158, legend_y, 192, legend_y), fill=int8_color, width=4)
    draw.text((200, legend_y - 9), "INT8", fill=axis_color, font=label_font)

    reports: list[dict[str, float | int | str]] = []
    for dimension, name in enumerate(dimension_names):
        row, column = divmod(dimension, columns)
        panel_x = 38 + column * 590
        panel_y = header_height + row * panel_height
        plot_left, plot_top = panel_x + 74, panel_y + 42
        plot_right, plot_bottom = panel_x + 560, panel_y + 205
        fp32_values = fp32_actions[0, :, dimension].astype(np.float64)
        int8_values = int8_actions[0, :, dimension].astype(np.float64)
        difference = np.abs(int8_values - fp32_values)
        reports.append(
            {
                "index": dimension,
                "name": name,
                "mae": float(difference.mean()),
                "max_abs_error": float(difference.max()),
            }
        )

        low = float(min(fp32_values.min(), int8_values.min()))
        high = float(max(fp32_values.max(), int8_values.max()))
        span = high - low
        padding = max(span * 0.08, 1e-4)
        low -= padding
        high += padding

        for grid_index in range(5):
            y = plot_top + grid_index * (plot_bottom - plot_top) / 4
            draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        draw.line(
            (plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=1
        )
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=1
        )

        def points(values: np.ndarray) -> list[tuple[float, float]]:
            return [
                (
                    plot_left + index * (plot_right - plot_left) / max(steps - 1, 1),
                    plot_bottom
                    - (float(value) - low) * (plot_bottom - plot_top) / (high - low),
                )
                for index, value in enumerate(values)
            ]

        draw.line(points(fp32_values), fill=fp32_color, width=3, joint="curve")
        draw.line(points(int8_values), fill=int8_color, width=2, joint="curve")
        draw.text(
            (panel_x + 4, panel_y + 6),
            name.title(),
            fill="#17212B",
            font=panel_title_font,
        )
        metric = f"MAE {difference.mean():.4f}"
        metric_width = draw.textbbox((0, 0), metric, font=metric_font)[2]
        draw.text(
            (plot_right - metric_width, panel_y + 10),
            metric,
            fill=int8_color,
            font=metric_font,
        )
        draw.text(
            (plot_left - 66, plot_top - 7),
            f"{high:.2f}",
            fill=axis_color,
            font=label_font,
        )
        draw.text(
            (plot_left - 66, plot_bottom - 7),
            f"{low:.2f}",
            fill=axis_color,
            font=label_font,
        )
        draw.text(
            (plot_left - 3, plot_bottom + 7), "0", fill=axis_color, font=label_font
        )
        draw.text(
            (plot_right - 17, plot_bottom + 7),
            str(steps - 1),
            fill=axis_color,
            font=label_font,
        )
        draw.text(
            (plot_left + 205, plot_bottom + 7),
            "Action step",
            fill=axis_color,
            font=label_font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/comparison.json")
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("artifacts/action_dimension_comparison.png"),
    )
    args = parser.parse_args()
    fp32 = load(args.fp32)
    int8 = load(args.int8)
    for field in ("checkpoint_sha256", "input_suite_sha256"):
        if fp32[field] != int8[field]:
            raise RuntimeError(f"Reports do not use the same {field}")
    fp32_ms = fp32["latency_ms"]["total"]["median"]
    int8_ms = int8["latency_ms"]["total"]["median"]
    fp32_actions = load_actions(args.fp32, fp32)
    int8_actions = load_actions(args.int8, int8)
    names = [f"Action {index + 1}" for index in range(fp32_actions.shape[-1])]
    action_dimensions = save_action_figure(
        args.figure, fp32_actions, int8_actions, names
    )
    comparison = {
        "fp32_median_ms": fp32_ms,
        "int8_median_ms": int8_ms,
        "speedup": fp32_ms / int8_ms,
        "fp32_pte_mb": fp32["pte_total_bytes"] / 1_000_000,
        "int8_pte_mb": int8["pte_total_bytes"] / 1_000_000,
        "int8_cosine_similarity": int8["accuracy"]["cosine_similarity"],
        "int8_mae": int8["accuracy"]["mean_absolute_error"],
        "int8_sqnr_db": int8["accuracy"]["sqnr_db"],
        "fp32_vs_int8_action_dimensions": action_dimensions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print("variant  median_ms  PTE_MB  cosine      MAE       SQNR_dB")
    print(
        f"FP32     {fp32_ms:9.2f}  {comparison['fp32_pte_mb']:6.1f}  {fp32['accuracy']['cosine_similarity']:.9f}  {fp32['accuracy']['mean_absolute_error']:.6f}  {fp32['accuracy']['sqnr_db']:.2f}"
    )
    print(
        f"INT8     {int8_ms:9.2f}  {comparison['int8_pte_mb']:6.1f}  {comparison['int8_cosine_similarity']:.9f}  {comparison['int8_mae']:.6f}  {comparison['int8_sqnr_db']:.2f}"
    )
    print(f"INT8 speedup: {comparison['speedup']:.2f}x")
    print(f"Action comparison figure: {args.figure}")


if __name__ == "__main__":
    main()
