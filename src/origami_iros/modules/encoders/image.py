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
    """Encoder for the camera images."""

    def __init__(self, encoder: BaseImageEncoder[torch.Tensor]) -> None:
        super().__init__()

        self._encoder: BaseImageEncoder[torch.Tensor] = encoder

    def forward(self, x: ImageObservation) -> torch.Tensor:
        # Concatenate images for batching
        images = torch.stack(
            [x.head.left, x.head.right, x.wrist.left, x.wrist.right], dim=1
        )
        images = rearrange(images, "b n_views c h w -> (b n_views) c h w")

        # Pass batched through the encoder
        encodings: torch.Tensor = self._encoder(images)

        # Reshape (n_views=4 because 4 cameras)
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
    """Any ViT encoder from Huggingface."""

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

        self._model: torch.nn.Module = spt.backbone.utils.vit_hf(
            size="tiny",
            image_size=(image_size[0] // n_hands),
            patch_size=patch_size,
            num_channels=n_hands * n_fingers,
        )

    def forward(self, x: Image) -> torch.Tensor:
        x = self._reshape_image(image=x)
        x = x.mean(dim=3)  # Mean along RGB channels (since grayscale)
        x = rearrange(x, "b n_hands n_fingers h w -> b (n_hands n_fingers) h w")
        outputs = self._model(x)
        return outputs.last_hidden_state[:, 1:]
