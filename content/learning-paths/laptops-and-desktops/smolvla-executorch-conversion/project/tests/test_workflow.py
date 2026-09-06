from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from smolvla_et.input_suite import validate_native_contract  # noqa: E402
from validate_pte import parse_cpu_set  # noqa: E402
from variants import QUANTIZATION_PLANS, canonical_variant  # noqa: E402


class WorkflowTests(unittest.TestCase):
    def test_public_checkpoint_camera_count_is_supported(self) -> None:
        validate_native_contract(
            {
                "camera_count": 3,
                "chunk_size": 50,
                "padded_state_dim": 32,
                "padded_action_dim": 32,
                "action_dim": 6,
            }
        )

    def test_other_positive_camera_counts_are_supported(self) -> None:
        for camera_count in (1, 2, 4):
            with self.subTest(camera_count=camera_count):
                validate_native_contract(
                    {
                        "camera_count": camera_count,
                        "chunk_size": 50,
                        "padded_state_dim": 32,
                        "padded_action_dim": 32,
                        "action_dim": 6,
                    }
                )

    def test_missing_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "camera"):
            validate_native_contract(
                {
                    "camera_count": 0,
                    "chunk_size": 50,
                    "padded_state_dim": 32,
                    "padded_action_dim": 32,
                    "action_dim": 6,
                }
            )

    def test_cpu_set_parser(self) -> None:
        self.assertEqual(parse_cpu_set("2-4,7"), {2, 3, 4, 7})
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_cpu_set("4-2")

    def test_int8_plan_is_accuracy_preserving_selection(self) -> None:
        self.assertEqual(
            QUANTIZATION_PLANS["int8"],
            {
                "vision_encoder": "dynamic-per-channel-int8",
                "prefix_forward": "none",
                "denoise_step": "dynamic-per-channel-int8",
            },
        )
        manifest = {
            "variant": "anything",
            "components": {
                name: {"quantization": value}
                for name, value in QUANTIZATION_PLANS["int8"].items()
            },
        }
        self.assertEqual(canonical_variant(manifest), "int8")


if __name__ == "__main__":
    unittest.main()
