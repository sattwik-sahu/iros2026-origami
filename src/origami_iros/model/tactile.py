import torch
from einops import rearrange

from origami_iros._typing import TactileImage
from origami_iros.model._typing import VLTA_Input


class TactileEncoder(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        image_encoder: torch.nn.Module,
        tactile_state_encoder: torch.nn.Module,
    ) -> None:
        super().__init__()

        self._dim: int = dim
        self._n_layers: int = n_layers
        self._n_heads: int = n_heads

        self._image_encoder: torch.nn.Module = image_encoder
        self._tactile_state_encoder: torch.nn.Module = tactile_state_encoder

        self._tactile_state_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(self._dim)
        )
        self._tactile_image_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(self._dim)
        )

        self._fusion_former: torch.nn.TransformerEncoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self._dim,
                nhead=self._n_heads,
                batch_first=True,
                activation=torch.nn.GELU(),
            ),
            num_layers=self._n_layers,
        )

    def _encode_image(self, x: TactileImage) -> torch.Tensor:
        x = rearrange(
            x,
            "b c (n_hand h) (n_finger w) -> b c (n_hand n_finger) h w",
            n_hand=2,
            n_finger=5,
        )
        x = x.mean(dim=1)  # Average along the channel dim, because grayscale images
        x_enc = self._image_encoder(x)  # 10-channel image in, patch encodings out
        return x_enc

    def forward(self, x: VLTA_Input) -> torch.Tensor:
        tactile_image_encoding = self._encode_image(
            x=x.observation.image.tactile.raw
        )  # Some patch encodings
        tactile_state_encoding = self._tactile_state_encoder(
            x.observation.state.tactile
        )  # State encoding(s)

        # Apply embedding to recognize format of token (image/state)
        tactile_image_encoding = (
            tactile_image_encoding
            + self._tactile_image_embedding.expand_as(tactile_image_encoding)
        )
        tactile_state_encoding = (
            tactile_state_encoding
            + self._tactile_state_embedding.expand_as(tactile_state_encoding)
        )

        # Concat the encoding tokens and pass through the fusion module
        fusion_input = torch.cat(
            [tactile_image_encoding, tactile_state_encoding], dim=1
        )

        return self._fusion_former(fusion_input)
