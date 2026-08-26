import torch
from transformers import AutoImageProcessor, AutoModel

from origami_iros._typing import Image
from origami_iros.modules.base import BaseImageEncoder


class HuggingfaceViT_Encoder(BaseImageEncoder[torch.Tensor]):
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

        with torch.inference_mode(mode=self._inference_only):
            outputs = self._model(**inputs)

        outputs = outputs.last_hidden_state

        return outputs
