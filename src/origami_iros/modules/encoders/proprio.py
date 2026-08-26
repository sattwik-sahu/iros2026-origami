import torch

from origami_iros.modules.base import BaseProprioceptiveStateEncoder


class MLP_Encoder(BaseProprioceptiveStateEncoder[torch.Tensor]):
    """MLP-based proprioceptive state encoder."""

    def __init__(
        self,
        dim_input: int,
        hidden_sizes: list[int],
        activation: type[torch.nn.Module],
    ) -> None:
        super().__init__()

        self._dim: int = dim_input
        self._hidden_sizes: list[int] = hidden_sizes
        self._activation: type[torch.nn.Module] = activation
        self._mlp: torch.nn.Sequential = self._build_mlp()

    def _build_mlp(self) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(in_features=self._dim, out_features=self._hidden_sizes[0])
        ]
        for s1, s2 in zip(self._hidden_sizes[:-1], self._hidden_sizes[:-1]):
            layers.append(torch.nn.Linear(in_features=s1, out_features=s2))
            layers.append(self._activation())

        layers.pop(-1)

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)
