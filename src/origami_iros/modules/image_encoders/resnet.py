import torch
from einops import rearrange
from torchvision.models import resnet18


class ResNet18Backbone(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self._dim: int = dim

        _proj_enc: torch.nn.Conv2d = torch.nn.Conv2d(
            in_channels=512, out_channels=self._dim, kernel_size=1, stride=1
        )
        self._backbone: torch.nn.Module = torch.nn.Sequential(
            *list(resnet18().children())[:-2] + [self._proj_enc]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encodings = self._backbone(image)
        encodings = rearrange(encodings, "b c h w -> b (h w) c")
        return encodings
