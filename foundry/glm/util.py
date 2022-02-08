from typing import Optional

import torch


class NoWeightModule(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(dim))
        self.weight = torch.empty(0)

    def forward(self, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.bias
        if input is not None:
            out = out.expand(len(input), -1)
        return out
