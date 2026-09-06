"""XNNPACK PT2E quantization recipes used by the learning path."""

from __future__ import annotations

import torch
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def quantize_int8(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.export.ExportedProgram:
    """Dynamically quantize Linear activations with per-channel INT8 weights."""

    captured = torch.export.export(module, example_inputs, strict=False)
    quantizer = XNNPACKQuantizer().set_operator_type(
        torch.ops.aten.linear.default,
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,
        ),
    )
    prepared = prepare_pt2e(captured.module(), quantizer)
    with torch.inference_mode():
        prepared(*example_inputs)
    converted = convert_pt2e(prepared)
    return torch.export.export(converted, example_inputs, strict=False)


def quantize_dynamic_int8(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.export.ExportedProgram:
    return quantize_int8(module, example_inputs)
