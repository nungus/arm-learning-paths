"""Fixed-shape, export-friendly adapters for low-latency split SmolVLA."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

from .split_policy import _StageVLMAdapter


def exportable_apply_rope(
    x: Tensor, positions: Tensor, max_wavelength: float = 10_000
) -> Tensor:
    """Apply RoPE without empty_permuted/select_scatter export artifacts."""
    half = x.shape[-1] // 2
    original_dtype = x.dtype
    x_float = x.to(torch.float32)
    exponents = (2.0 / x.shape[-1]) * torch.arange(
        half, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    sin = torch.sin(radians[..., None, :])
    cos = torch.cos(radians[..., None, :])
    first, second = x_float.split(half, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    ).to(original_dtype)


def install_exportable_rope() -> None:
    """Patch the local LeRobot module global used by its copied attention methods."""
    from lerobot.policies.smolvla import smolvlm_with_expert

    smolvlm_with_expert.apply_rope = exportable_apply_rope


class BatchedSmolVLAVisionEncoder(nn.Module):
    """Encode every camera in one fixed-shape vision batch."""

    def __init__(self, model: nn.Module, image_size: int, camera_count: int) -> None:
        super().__init__()
        vlm_model = model.vlm_with_expert.get_vlm_model()
        self.vision_embeddings = vlm_model.vision_model.embeddings
        self.vision_layers = vlm_model.vision_model.encoder.layers
        self.post_layernorm = vlm_model.vision_model.post_layernorm
        self.connector = vlm_model.connector
        self.camera_count = camera_count
        self.num_heads = self.vision_layers[0].self_attn.num_heads
        self.head_dim = self.vision_layers[0].self_attn.head_dim
        self.attention_scale = self.vision_layers[0].self_attn.scale

        patch_size = self.vision_embeddings.patch_size
        required_multiple = patch_size * model.vlm_with_expert.config.scale_factor
        if image_size % required_multiple:
            raise ValueError(
                f"image_size must be divisible by patch_size * scale_factor "
                f"({required_multiple})"
            )
        patch_height = image_size // patch_size
        patch_width = image_size // patch_size
        position_grid = self.vision_embeddings.num_patches_per_side
        boundaries = torch.arange(
            1 / position_grid, 1.0, 1 / position_grid, dtype=torch.float32
        )
        buckets_h = torch.bucketize(
            torch.arange(patch_height, dtype=torch.float32) / patch_height,
            boundaries,
            right=True,
        )
        buckets_w = torch.bucketize(
            torch.arange(patch_width, dtype=torch.float32) / patch_width,
            boundaries,
            right=True,
        )
        position_ids = (
            buckets_h[:, None] * position_grid + buckets_w[None, :]
        ).reshape(1, -1)
        self.register_buffer("fixed_position_ids", position_ids)

    def forward(self, images: Tensor) -> Tensor:
        batch, cameras, channels, height, width = images.shape
        flat_batch = batch * cameras
        flat_images = images.reshape(flat_batch, channels, height, width)
        patch_embeddings = self.vision_embeddings.patch_embedding(flat_images)
        embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.vision_embeddings.position_embedding(
            self.fixed_position_ids
        )

        sequence_length = embeddings.shape[1]
        head_shape = (flat_batch, sequence_length, self.num_heads, self.head_dim)
        flat_head_shape = (
            flat_batch * self.num_heads,
            sequence_length,
            self.head_dim,
        )
        for layer in self.vision_layers:
            residual = embeddings
            normalized = layer.layer_norm1(embeddings)
            attention = layer.self_attn
            queries = (
                attention.q_proj(normalized)
                .reshape(head_shape)
                .permute(0, 2, 1, 3)
                .reshape(flat_head_shape)
            )
            keys = (
                attention.k_proj(normalized)
                .reshape(head_shape)
                .permute(0, 2, 1, 3)
                .reshape(flat_head_shape)
            )
            values = (
                attention.v_proj(normalized)
                .reshape(head_shape)
                .permute(0, 2, 1, 3)
                .reshape(flat_head_shape)
            )
            scores = torch.bmm(queries, keys.transpose(1, 2)) * self.attention_scale
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            attention_output = torch.bmm(weights, values)
            attention_output = (
                attention_output.reshape(
                    flat_batch,
                    self.num_heads,
                    sequence_length,
                    self.head_dim,
                )
                .permute(0, 2, 1, 3)
                .reshape(
                    flat_batch,
                    sequence_length,
                    self.num_heads * self.head_dim,
                )
            )
            embeddings = residual + attention.out_proj(attention_output)
            residual = embeddings
            embeddings = residual + layer.mlp(layer.layer_norm2(embeddings))

        embeddings = self.connector(self.post_layernorm(embeddings))
        return embeddings.reshape(
            batch, cameras, embeddings.shape[1], embeddings.shape[2]
        )


class OptimizedSmolVLAPrefix(nn.Module):
    """Build a reusable prefix mask and interleaved, flattened KV cache."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if model.add_image_special_tokens:
            raise ValueError(
                "The optimized checkpoint-specific prefix expects no image special tokens"
            )
        self.vlm = _StageVLMAdapter(model.vlm_with_expert, stage="prefix")
        self.state_proj = model.state_proj
        self.hidden_scale = math.sqrt(model.vlm_with_expert.config.text_config.hidden_size)
        self.vlm_layers = model.vlm_with_expert.num_vlm_layers

    def forward(
        self,
        image_embeddings: Tensor,
        image_masks: Tensor,
        language_tokens: Tensor,
        language_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, cameras, image_tokens, hidden = image_embeddings.shape
        image_embeddings = (image_embeddings * self.hidden_scale).reshape(
            batch, cameras * image_tokens, hidden
        )
        image_pad_masks = image_masks[:, :, None].expand(
            batch, cameras, image_tokens
        ).reshape(batch, cameras * image_tokens)

        language_embeddings = self.vlm.embed_language_tokens(language_tokens)
        language_embeddings = language_embeddings * self.hidden_scale
        state_embeddings = self.state_proj(state)[:, None, :]
        state_masks = torch.ones_like(state_embeddings[:, :, 0], dtype=torch.bool)

        prefix_embeddings = torch.cat(
            (image_embeddings, language_embeddings, state_embeddings), dim=1
        )
        prefix_pad_masks = torch.cat(
            (image_pad_masks, language_masks, state_masks), dim=1
        )
        prefix_attention_ar = torch.cat(
            (
                torch.zeros_like(image_pad_masks, dtype=torch.bool),
                torch.zeros_like(language_masks, dtype=torch.bool),
                state_masks,
            ),
            dim=1,
        )
        cumulative = torch.cumsum(prefix_attention_ar, dim=1)
        attention_mask = cumulative[:, None, :] <= cumulative[:, :, None]
        attention_mask = attention_mask & (
            prefix_pad_masks[:, None, :] & prefix_pad_masks[:, :, None]
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, cache = self.vlm.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        flat_cache = torch.stack(
            tuple(
                cache[layer][kind]
                for layer in range(self.vlm_layers)
                for kind in ("key_states", "value_states")
            ),
            dim=0,
        )
        return prefix_pad_masks, flat_cache


class OptimizedSmolVLADenoiseStep(nn.Module):
    """Execute one expert evaluation and Euler update inside the PTE."""

    embed_suffix = VLAFlowMatching.embed_suffix

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.vlm_with_expert = _StageVLMAdapter(
            model.vlm_with_expert, stage="denoise"
        )
        self.config = model.config
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.action_time_mlp_in = model.action_time_mlp_in
        self.action_time_mlp_out = model.action_time_mlp_out
        self.vlm_layers = model.vlm_with_expert.num_vlm_layers
        self.dt = -1.0 / model.config.num_steps

    def forward(
        self,
        prefix_pad_masks: Tensor,
        flat_cache: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        past_key_values = {
            layer: {
                "key_states": flat_cache[layer * 2],
                "value_states": flat_cache[layer * 2 + 1],
            }
            for layer in range(self.vlm_layers)
        }
        velocity = VLAFlowMatching.denoise_step(
            self,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noisy_actions,
            timestep=timestep,
        )
        return noisy_actions + self.dt * velocity
