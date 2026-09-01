"""Image encoders for the VLTA policy.

This module provides encoders for the two image-based modalities consumed by the
model:

* **Camera images** (``CameraImageEncoder``): four fixed camera views (head
  left/right and wrist left/right) each encoded by a shared pretrained ViT
  (e.g. DINOv2) whose patch tokens are concatenated into a single token
  sequence.

* **Tactile images** (``TactileImageEncoder`` / ``TinyViT_TactileImageEncoder`` /
  ``PerFingerSingleTokenTactileEncoder``): high-resolution skin-contact images
  split into per-hand / per-finger regions. Two strategies are provided:

  * ``TinyViT_TactileImageEncoder`` (the default) treats all fingers as extra
    input channels of a single "fat" image.
  * ``PerFingerSingleTokenTactileEncoder`` (an alternative, not used by default)
    encodes each finger image independently and collapses it to a single token.
"""

from __future__ import annotations

import stable_pretraining as spt
import torch
import torch.nn as nn
from einops import rearrange
from typing import override
from transformers import AutoImageProcessor, AutoModel

from origami_iros._typing import Image
from origami_iros.models._typing import ImageObservation
from origami_iros.models.base import (
    BaseEncoder,
    BaseImageEncoder,
    BaseTactileImageEncoder,
)


class CameraImageEncoder(BaseEncoder[ImageObservation, torch.Tensor]):
    """Encode the four camera views into a shared token sequence.

    The head and wrist stereo pairs are stacked along a view dimension, encoded
    by a shared :class:`BaseImageEncoder` (one batch element per view), and the
    resulting patch tokens are concatenated back into a single sequence per
    batch element.

    Attributes:
        _encoder: The shared image encoder applied to every view.
    """

    def __init__(self, encoder: BaseImageEncoder[torch.Tensor]) -> None:
        """Initialise the encoder.

        Args:
            encoder: A single image encoder shared across all four views.
        """
        super().__init__()
        self._encoder: BaseImageEncoder[torch.Tensor] = encoder

    @override
    def forward(self, x: ImageObservation) -> torch.Tensor:
        """Encode the four camera views.

        Args:
            x: The image observation containing the head and wrist stereo pairs.

        Returns:
            Concatenated patch tokens of shape ``(batch, 4 * n_patches, token_dim)``.
        """
        images = torch.stack(
            [x.head.left, x.head.right, x.wrist.left, x.wrist.right], dim=1
        )
        images = rearrange(images, "b n_views c h w -> (b n_views) c h w")
        encodings: torch.Tensor = self._encoder(images)
        encodings = rearrange(
            encodings, "(b n_views) n d -> b (n_views n) d", n_views=4
        )
        return encodings


class TactileImageEncoder(BaseEncoder[ImageObservation, torch.Tensor]):
    """Encode tactile images, fusing a primary and (optionally) a secondary stream.

    The dataset provides two tactile streams: ``raw`` (the direct sensor image,
    which carries the actual touch signal) and ``deform`` (a derived
    deformation-visualisation stream). In the recorded data the deform stream is
    all zeros, so **``raw`` is treated as the primary stream** by default.

    The primary stream is always encoded. If a secondary encoder is supplied and
    the observation provides the secondary stream, its tokens are concatenated
    with a per-batch dropout so the model does not over-rely on the (optionally
    absent) secondary signal.

    Attributes:
        _primary_key: ``"raw"`` or ``"deform"`` selected as the primary stream.
        _secondary_key: The other stream, if a secondary encoder is configured.
        _primary_encoder: Encoder for the primary stream.
        _secondary_encoder: Optional encoder for the secondary stream.
        _secondary_dropout: Probability of dropping the secondary stream (training).
    """

    def __init__(
        self,
        primary_encoder: BaseImageEncoder,
        primary_key: str = "raw",
        secondary_encoder: BaseImageEncoder | None = None,
        secondary_key: str = "deform",
        secondary_dropout: float = 0.2,
    ) -> None:
        """Initialise the encoder.

        Args:
            primary_encoder: Encoder for the primary tactile stream.
            primary_key: Attribute name of the primary stream on the
                :class:`TactileImageObservation` (``"raw"`` or ``"deform"``).
            secondary_encoder: Optional encoder for the secondary stream.
            secondary_key: Attribute name of the secondary stream.
            secondary_dropout: Dropout probability applied to the secondary stream.
        """
        super().__init__()
        self._primary_key = primary_key
        self._secondary_key = secondary_key
        self._primary_encoder = primary_encoder
        self._secondary_encoder = secondary_encoder
        self._secondary_dropout = secondary_dropout

    @override
    def forward(self, x: ImageObservation) -> torch.Tensor:
        """Encode the tactile image streams.

        Args:
            x: The image observation containing the tactile streams.

        Returns:
            Concatenated tactile tokens of shape ``(batch, n_tokens, token_dim)``.
        """
        tokens = self._primary_encoder(getattr(x.tactile, self._primary_key))
        secondary = getattr(x.tactile, self._secondary_key, None)
        use_secondary = (
            self._secondary_encoder is not None
            and secondary is not None
            and (not self.training or torch.rand(1).item() > self._secondary_dropout)
        )
        if use_secondary:
            tokens = torch.cat([tokens, self._secondary_encoder(secondary)], dim=1)
        return tokens


class PretrainedHF_ViT_Encoder(BaseImageEncoder[torch.Tensor]):
    """A pretrained Hugging Face Vision Transformer used as a frozen feature extractor.

    Wraps ``transformers.AutoModel`` with its ``AutoImageProcessor``. By default
    the CLS token is dropped (``last_hidden_state[:, 1:]``) so the returned tokens
    are per-patch patch encodings, which downstream cross-attention can use.

    Attributes:
        _model_name: Hugging Face model identifier, e.g. ``"facebook/dinov2-small"``.
        _inference_only: If ``True``, run the backbone under ``torch.inference_mode``.
        _processor: The image processor of the model.
        _model: The wrapped Hugging Face model.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | int,
        model_name: str,
        inference_only: bool = False,
    ) -> None:
        """Initialise the encoder.

        Args:
            image_size: Input image size, either a scalar or ``(h, w)``.
            model_name: Hugging Face model identifier.
            inference_only: Whether to run under ``torch.inference_mode``.
        """
        super().__init__(image_size=image_size)
        self._model_name: str = model_name
        self._inference_only: bool = inference_only
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_name, use_fast=True
        )
        self._model = AutoModel.from_pretrained(self._model_name)

    @override
    def forward(self, x: Image) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            x: A batch of images of shape ``(batch, channels, h, w)``.

        Returns:
            Patch tokens (excluding the CLS token) of shape
            ``(batch, n_patches, token_dim)``.
        """
        inputs = self._processor(images=x, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = inputs.to(device=device)
        with torch.inference_mode(mode=self._inference_only):
            outputs = self._model(**inputs)
        return outputs.last_hidden_state[:, 1:]


class TinyViT_TactileImageEncoder(BaseTactileImageEncoder[torch.Tensor]):
    """Default tactile image encoder using a tiny ViT over a "fat" image.

    All tactile fingers are rearranged into a single image whose channels equal
    ``n_hands * n_fingers`` (after collapsing the per-finger height to one row).
    This is the default strategy retained in the policy.

    Attributes:
        _patch_size: Patch size fed to the tiny ViT.
        _model: A ``stable_pretraining`` tiny ViT backbone.
    """

    def __init__(
        self, image_size: tuple[int, int], patch_size: int, n_hands: int, n_fingers: int
    ) -> None:
        """Initialise the encoder.

        Args:
            image_size: Size of the full tactile image as ``(h, w)``.
            patch_size: Side length of each patch.
            n_hands: Number of hands in the image.
            n_fingers: Number of fingers per hand.
        """
        super().__init__(image_size=image_size, n_hands=n_hands, n_fingers=n_fingers)
        self._patch_size = patch_size
        h = image_size[0] // n_hands
        w = image_size[1] // n_fingers
        h_patches = max(1, h // patch_size)
        w_patches = max(1, w // patch_size)
        side_patches = max(1, round((h_patches * w_patches) ** 0.5))
        ref_side = side_patches * patch_size
        self._model: nn.Module = spt.backbone.utils.vit_hf(
            size="tiny",
            image_size=(ref_side, ref_side),
            patch_size=patch_size,
            num_channels=n_hands * n_fingers,
        )

    @override
    def forward(self, x: Image) -> torch.Tensor:
        """Encode a batch of full tactile images.

        Args:
            x: Full tactile image batch of shape ``(batch, c, h, w)``.

        Returns:
            Tactile patch tokens of shape ``(batch, n_patches, token_dim)``.
        """
        x = self._reshape_image(image=x)
        x = x.mean(dim=3)
        x = rearrange(x, "b n_hands n_fingers h w -> b (n_hands n_fingers) h w")
        x = self._pad_to_patch_multiple(x)
        outputs = self._model(x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 1:]

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the spatial dimensions so they are a multiple of the patch size.

        Args:
            x: Image batch of shape ``(batch, c, h, w)``.

        Returns:
            The padded image batch.
        """
        _, _, h, w = x.shape
        pad_h = (-h) % self._patch_size
        pad_w = (-w) % self._patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))


class PerFingerSingleTokenTactileEncoder(BaseTactileImageEncoder[torch.Tensor]):
    """Alternative tactile encoder: one token per finger image.

    Unlike :class:`TinyViT_TactileImageEncoder` (which fuses all fingers into a
    single "fat" image at the channel dimension), this encoder processes each
    finger image independently with a small shared ViT and mean-pools the patch
    tokens to produce a single token per finger. The result is
    ``n_hands * n_fingers`` tokens per batch element.

    This encoder is an alternative implementation and is **not** used by the
    default policy. It is provided to explore a spatially localised, finger-level
    tactile representation.

    Attributes:
        _patch_size: Patch size fed to the per-finger ViT.
        _model: A small ViT shared across all fingers.
        _token_dim: Output embedding dimension per finger token.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        n_hands: int,
        n_fingers: int,
        token_dim: int = 192,
        size: str = "tiny",
    ) -> None:
        """Initialise the encoder.

        Args:
            image_size: Size of the full tactile image as ``(h, w)``.
            patch_size: Side length of each patch within a single finger image.
            n_hands: Number of hands in the image.
            n_fingers: Number of fingers per hand.
            token_dim: Output embedding dimension for each per-finger token.
            size: Backbone size string passed to ``stable_pretraining``.
        """
        super().__init__(image_size=image_size, n_hands=n_hands, n_fingers=n_fingers)
        self._patch_size = patch_size
        self._token_dim = token_dim

        # A single finger image occupies h/n_hands x w/n_fingers.
        h_image = image_size[0] // n_hands
        w_image = image_size[1] // n_fingers

        # The HF ViT requires a square number of patch positions, so pad each
        # finger image to a square that is a multiple of the patch size.
        h_patches = (h_image + patch_size - 1) // patch_size
        w_patches = (w_image + patch_size - 1) // patch_size
        side_patches = max(h_patches, w_patches)
        self._finger_h = h_image
        self._finger_w = w_image
        self._finger_pad = (side_patches * patch_size, side_patches * patch_size)

        self._model: nn.Module = spt.backbone.utils.vit_hf(
            size=size,
            image_size=self._finger_pad,
            patch_size=patch_size,
            num_channels=3,
        )
        # Map the backbone embedding dimension to the requested token dim.
        backbone_dim = self._model.config.hidden_size
        self._token_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, token_dim),
        )

    @override
    def forward(self, x: Image) -> torch.Tensor:
        """Encode a batch of full tactile images into per-finger tokens.

        Args:
            x: Full tactile image batch of shape ``(batch, c, h, w)``.

        Returns:
            Per-finger tokens of shape ``(batch, n_hands * n_fingers, token_dim)``.
        """
        # Reshape into individual finger images: (b, n_hands, n_fingers, c, h, w).
        fingers = self._reshape_image(image=x)
        b, n_hands, n_fingers, c, fh, fw = fingers.shape

        # Pad each finger image to the square target size.
        pad_h = self._finger_pad[0] - fh
        pad_w = self._finger_pad[1] - fw
        if pad_h > 0 or pad_w > 0:
            fingers = torch.nn.functional.pad(fingers, (0, pad_w, 0, pad_h))

        # Flatten batch and fingers so every finger image is processed independently.
        fingers = fingers.reshape(b * n_hands * n_fingers, c, self._finger_pad[0], self._finger_pad[1])

        outputs = self._model(fingers, interpolate_pos_encoding=True)
        # Mean-pool the patch tokens (excluding CLS if present) to a single token.
        tokens = outputs.last_hidden_state[:, 1:].mean(dim=1)
        tokens = self._token_head(tokens)

        return tokens.reshape(b, n_hands * n_fingers, self._token_dim)
