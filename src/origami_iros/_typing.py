import numpy as np
import torch
from numpy import typing as npt
from tensordict import TensorClass, TensorDict

type Image = torch.Tensor
type Action = torch.Tensor


type DictData = dict[str, npt.NDArray[np.uint8 | np.float32] | str]
type TensorData = TensorClass | torch.Tensor | TensorDict
