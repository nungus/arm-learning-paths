"""The two precision variants used by the learning path."""

COMPONENTS = ("vision_encoder", "prefix_forward", "denoise_step")
QUANTIZATION_PLANS = {
    "fp32": {name: "none" for name in COMPONENTS},
    "int8": {
        "vision_encoder": "dynamic-per-channel-int8",
        "prefix_forward": "none",
        "denoise_step": "dynamic-per-channel-int8",
    },
}


def component_quantization(variant: str, component: str) -> str:
    try:
        return QUANTIZATION_PLANS[variant][component]
    except KeyError as error:
        raise ValueError(f"Unknown variant/component: {variant}/{component}") from error


def canonical_variant(manifest: dict) -> str:
    actual = {
        name: manifest.get("components", {}).get(name, {}).get("quantization")
        for name in COMPONENTS
    }
    for variant, expected in QUANTIZATION_PLANS.items():
        if actual == expected:
            return variant
    return str(manifest.get("variant", "unknown"))
