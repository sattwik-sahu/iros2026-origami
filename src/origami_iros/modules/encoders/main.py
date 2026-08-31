import torch

from origami_iros.modules._typing import Observation, ObservationEncoding
from origami_iros.modules.base import (
    BaseEncoder,
    BaseImageEncoder,
    BaseProprioceptiveStateEncoder,
    BaseTactileImageEncoder,
)


class VLTA_Encoder(BaseEncoder[Observation, ObservationEncoding]):
    """Unified encoder module for all observation modalities."""

    def __init__(
        self,
        camera_image_encoder: BaseImageEncoder,
        tactile_image_encoder: BaseTactileImageEncoder,
        state_encoder: BaseProprioceptiveStateEncoder,
    ) -> None:
        super().__init__()

        self._camera_image_encoder: BaseImageEncoder = camera_image_encoder
        self._tactile_image_encoder: BaseTactileImageEncoder = tactile_image_encoder
        self._state_encoder: BaseProprioceptiveStateEncoder = state_encoder

    def forward(self, x: Observation) -> ObservationEncoding:
        camera_image_encoding: torch.Tensor = self._camera_image_encoder(x.image)
        tactile_image_encoding: torch.Tensor = self._tactile_image_encoder(x.image)
        state_encoding: torch.Tensor = self._state_encoder(x.state)

        observation_encoding: ObservationEncoding = ObservationEncoding(
            camera_image=camera_image_encoding,
            tactile_image=tactile_image_encoding,
            state=state_encoding,
        )

        return observation_encoding
