from abc import ABC, abstractmethod

import torch
from einops import rearrange

from origami_iros._typing import Action, Image, TensorData
from origami_iros.modules._typing import RobotStateObservation


class BaseEncoder[TInput, TEncoding: TensorData](torch.nn.Module, ABC):
    """The base encoder module."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: TInput) -> TEncoding:
        pass


class BaseProprioceptiveStateEncoder[TEncoding: TensorData](
    BaseEncoder[RobotStateObservation, TEncoding], ABC
):
    """The base state encoder module."""

    def __init__(self) -> None:
        super().__init__()


class BaseImageEncoder[TEncoding: TensorData](BaseEncoder[Image, TEncoding], ABC):
    """The base image encoder module."""

    def __init__(self, image_size: tuple[int, int] | int) -> None:
        super().__init__()

        self._image_size: tuple[int, int] = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )

    @property
    def image_size(self) -> tuple[int, int]:
        """The image size that the model accepts as input."""
        return self._image_size


class BaseTactileImageEncoder[TEncoding: TensorData](BaseImageEncoder[TEncoding]):
    """The base tactile image encoder module."""

    def __init__(
        self, image_size: tuple[int, int], n_hands: int, n_fingers: int
    ) -> None:
        super().__init__(image_size=image_size)

        self._n_hands: int = n_hands
        self._n_fingers: int = n_fingers

    def _reshape_image(self, image: Image) -> torch.Tensor:
        return rearrange(
            image,
            "b c (n_hands h) (n_fingers w) -> b n_hands n_fingers c h w",
            n_hands=self._n_hands,
            n_fingers=self._n_fingers,
        )


class BaseActionModule[TInput: TensorData](torch.nn.Module, ABC):
    """The base action module."""

    def __init__(self, dim_action: int) -> None:
        super().__init__()

        self._dim_action: int = dim_action

    @property
    def dim_action(self) -> int:
        return self._dim_action

    @abstractmethod
    def forward(self, x: TInput) -> Action:
        pass
