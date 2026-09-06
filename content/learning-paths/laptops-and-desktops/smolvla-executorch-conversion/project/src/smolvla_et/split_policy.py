"""Tensor-only adapters for split SmolVLA inference."""

import math
from types import SimpleNamespace

import torch
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


class _StageVLMAdapter(torch.nn.Module):
    forward = SmolVLMWithExpertModel.forward
    forward_attn_layer = SmolVLMWithExpertModel.forward_attn_layer
    forward_cross_attn_layer = SmolVLMWithExpertModel.forward_cross_attn_layer
    get_model_layers = SmolVLMWithExpertModel.get_model_layers
    get_attention_interface = SmolVLMWithExpertModel.get_attention_interface
    eager_attention_forward = SmolVLMWithExpertModel.eager_attention_forward

    def __init__(self, source: SmolVLMWithExpertModel, stage: str) -> None:
        super().__init__()
        self.num_vlm_layers = source.num_vlm_layers
        self.num_expert_layers = source.num_expert_layers
        self.self_attn_every_n_layers = source.self_attn_every_n_layers
        self.attention_mode = source.attention_mode
        self.num_attention_heads = source.num_attention_heads
        self.num_key_value_heads = source.num_key_value_heads
        self.expert_hidden_size = source.expert_hidden_size
        self.vlm = SimpleNamespace(config=source.vlm.config)

        empty_layers = [None] * self.num_vlm_layers
        if stage == "prefix":
            self.text_model = source.get_vlm_model().text_model
            self.lm_expert = SimpleNamespace(layers=empty_layers, norm=None)
        elif stage == "denoise":
            self.text_model = SimpleNamespace(layers=empty_layers, norm=None)
            self.lm_expert = source.lm_expert
        else:
            raise ValueError(f"Unknown SmolVLA stage: {stage}")

    def get_vlm_model(self) -> SimpleNamespace:
        return SimpleNamespace(text_model=self.text_model)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_model.get_input_embeddings()(tokens)


class SmolVLAVisionEncoderWrapper(torch.nn.Module):
    """Encode one preprocessed camera image into connected VLM embeddings."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        vlm_model = model.vlm_with_expert.get_vlm_model()
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.vision_model(
            pixel_values=image.to(dtype=self.vision_model.dtype),
            patch_attention_mask=None,
        ).last_hidden_state
        return self.connector(image_hidden_states)


class SmolVLAPrefixWrapper(torch.nn.Module):
    """Build the reusable prefix masks and stacked per-layer KV cache."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.vlm = _StageVLMAdapter(model.vlm_with_expert, stage="prefix")
        self.state_proj = model.state_proj
        self.add_image_special_tokens = model.add_image_special_tokens
        self.use_cache = model.config.use_cache
        self.register_buffer(
            "global_image_start_token", model.global_image_start_token.clone()
        )
        self.register_buffer("image_end_token", model.image_end_token.clone())

    def _image_prefix_parts(
        self,
        image_embeddings: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        embeddings = []
        pad_masks = []
        attention_masks = []
        batch_size = image_embeddings.shape[0]

        if self.add_image_special_tokens:
            image_start = self.vlm.embed_language_tokens(
                self.global_image_start_token.to(device=image_embeddings.device)
            )
            image_start = image_start.unsqueeze(0).expand(batch_size, -1, -1)
            embeddings.append(image_start)
            pad_masks.append(torch.ones_like(image_start[:, :, 0], dtype=torch.bool))
            attention_masks += [0] * image_start.shape[1]

        embedding_dim = image_embeddings.shape[-1]
        image_embeddings = image_embeddings * math.sqrt(embedding_dim)
        image_sequence_length = image_embeddings.shape[1]
        embeddings.append(image_embeddings)
        pad_masks.append(image_mask[:, None].expand(batch_size, image_sequence_length))
        attention_masks += [0] * image_sequence_length

        if self.add_image_special_tokens:
            image_end = self.vlm.embed_language_tokens(
                self.image_end_token.to(device=image_embeddings.device)
            )
            image_end = image_end.unsqueeze(0).expand(batch_size, -1, -1)
            embeddings.append(image_end)
            pad_masks.append(torch.ones_like(image_end[:, :, 0], dtype=torch.bool))
            attention_masks += [0] * image_end.shape[1]

        return embeddings, pad_masks, attention_masks

    def forward(
        self,
        overhead_image_embeddings: torch.Tensor,
        robot_image_embeddings: torch.Tensor,
        overhead_mask: torch.Tensor,
        robot_mask: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = []
        pad_masks = []
        attention_masks = []

        for image_embeddings, image_mask in (
            (overhead_image_embeddings, overhead_mask),
            (robot_image_embeddings, robot_mask),
        ):
            image_parts = self._image_prefix_parts(image_embeddings, image_mask)
            embeddings.extend(image_parts[0])
            pad_masks.extend(image_parts[1])
            attention_masks.extend(image_parts[2])

        language_embeddings = self.vlm.embed_language_tokens(language_tokens)
        language_embeddings = language_embeddings * math.sqrt(
            language_embeddings.shape[-1]
        )
        embeddings.append(language_embeddings)
        pad_masks.append(language_mask)
        attention_masks += [0] * language_embeddings.shape[1]

        state_embeddings = self.state_proj(state)
        if state_embeddings.ndim == 2:
            state_embeddings = state_embeddings[:, None, :]
        embeddings.append(state_embeddings)
        state_mask = torch.ones(
            state_embeddings.shape[:2],
            dtype=torch.bool,
            device=state_embeddings.device,
        )
        pad_masks.append(state_mask)
        attention_masks += [1] * state_embeddings.shape[1]

        prefix_embeddings = torch.cat(embeddings, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_attention_masks = torch.tensor(
            attention_masks,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )[None, :].expand(prefix_pad_masks.shape[0], -1)

        prefix_attention_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_attention_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm.forward(
            attention_mask=prefix_attention_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
        )

        cache_keys = torch.stack(
            [
                past_key_values[index]["key_states"]
                for index in range(len(past_key_values))
            ]
        )
        cache_values = torch.stack(
            [
                past_key_values[index]["value_states"]
                for index in range(len(past_key_values))
            ]
        )
        return prefix_pad_masks, cache_keys, cache_values


class SmolVLADenoiseStepWrapper(torch.nn.Module):
    """Run one denoising evaluation using a flattened tensor KV cache."""

    embed_suffix = VLAFlowMatching.embed_suffix

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.vlm_with_expert = _StageVLMAdapter(
            model.vlm_with_expert, stage="denoise"
        )
        self.config = model.config
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.action_time_mlp_in = model.action_time_mlp_in
        self.action_time_mlp_out = model.action_time_mlp_out
        self.num_cache_layers = model.vlm_with_expert.num_vlm_layers

    def forward(
        self,
        prefix_pad_masks: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        past_key_values = {
            index: {
                "key_states": cache_keys[index],
                "value_states": cache_values[index],
            }
            for index in range(self.num_cache_layers)
        }
        return VLAFlowMatching.denoise_step(
            self,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noisy_actions,
            timestep=timestep,
        )
