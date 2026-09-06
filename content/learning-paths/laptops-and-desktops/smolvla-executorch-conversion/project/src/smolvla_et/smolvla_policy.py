"""SmolVLA adapter used by the ExecuTorch export pipeline."""

import torch


class SmolVLAWrapper(torch.nn.Module):
    def __init__(self, model, action_dim, camera_count):
        super().__init__()
        self.model = model
        self.action_dim = action_dim
        self.camera_count = camera_count

    # Preprocessing would be required to use actual camera images. For now,
    # going with the transformed 512x512 images (i.e. not the outermost I/O).
    def forward(self, *inputs):
        images = list(inputs[: self.camera_count])
        masks = list(inputs[self.camera_count : 2 * self.camera_count])
        language_tokens, language_mask, state, noise = inputs[2 * self.camera_count :]
        actions = self.model.sample_actions(
            images=images,
            img_masks=masks,
            lang_tokens=language_tokens,
            lang_masks=language_mask,
            state=state,
            noise=noise,
        )
        # One dimension per joint. Get rid of padded entries.
        return actions[:, :, :self.action_dim]
