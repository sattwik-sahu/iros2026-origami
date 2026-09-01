import stable_pretraining as spt
import torch
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel

from origami_iros._typing import Image
from origami_iros.modules._typing import ImageObservation
from origami_iros.modules.base import (
    BaseEncoder,
    BaseImageEncoder,
    BaseTactileImageEncoder,
)


class CameraImageEncoder(BaseEncoder[ImageObservation, torch.Tensor]):
    def __init__(self, encoder: BaseImageEncoder[torch.Tensor]) -> None:
        super().__init__()
        self._encoder: BaseImageEncoder[torch.Tensor] = encoder

    def forward(self, x: ImageObservation) -> torch.Tensor:
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
    def __init__(self, deform_encoder, raw_encoder=None, raw_dropout: float = 0.2):
        super().__init__()
        self._deform_encoder = deform_encoder
        self._raw_encoder = raw_encoder
        self._raw_dropout = raw_dropout

    def forward(self, x: ImageObservation) -> torch.Tensor:
        tokens = self._deform_encoder(x.tactile.deform)
        use_raw = self._raw_encoder is not None and x.tactile.raw is not None
        if use_raw and (not self.training or torch.rand(1).item() > self._raw_dropout):
            tokens = torch.cat([tokens, self._raw_encoder(x.tactile.raw)], dim=1)
        return tokens


class PretrainedHF_ViT_Encoder(BaseImageEncoder[torch.Tensor]):
    def __init__(
        self,
        image_size: tuple[int, int] | int,
        model_name: str,
        inference_only: bool = False,
    ) -> None:
        super().__init__(image_size=image_size)
        self._model_name: str = model_name
        self._inference_only: bool = inference_only
        self._processor = AutoImageProcessor.from_pretrained(
            self._model_name, use_fast=True
        )
        self._model = AutoModel.from_pretrained(self._model_name)

    def forward(self, x: Image) -> torch.Tensor:
        inputs = self._processor(images=x, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = inputs.to(device=device)
        with torch.inference_mode(mode=self._inference_only):
            outputs = self._model(**inputs)
        return outputs.last_hidden_state[:, 1:]


class TinyViT_TactileImageEncoder(BaseTactileImageEncoder[torch.Tensor]):
    def __init__(
        self, image_size: tuple[int, int], patch_size: int, n_hands: int, n_fingers: int
    ) -> None:
        super().__init__(image_size=image_size, n_hands=n_hands, n_fingers=n_fingers)
        self._patch_size = patch_size
        h = image_size[0] // n_hands
        w = image_size[1] // n_fingers
        h_patches = max(1, h // patch_size)
        w_patches = max(1, w // patch_size)
        side_patches = max(1, round((h_patches * w_patches) ** 0.5))
        ref_side = side_patches * patch_size
        self._model: torch.nn.Module = spt.backbone.utils.vit_hf(
            size="tiny",
            image_size=(ref_side, ref_side),
            patch_size=patch_size,
            num_channels=n_hands * n_fingers,
        )

    def forward(self, x: Image) -> torch.Tensor:
        x = self._reshape_image(image=x)
        x = x.mean(dim=3)
        x = rearrange(x, "b n_hands n_fingers h w -> b (n_hands n_fingers) h w")
        x = self._pad_to_patch_multiple(x)
        outputs = self._model(x, interpolate_pos_encoding=True)
        return outputs.last_hidden_state[:, 1:]

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (-h) % self._patch_size
        pad_w = (-w) % self._patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))