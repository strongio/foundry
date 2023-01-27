from typing import Optional

import torch
from torch.distributions import NegativeBinomial as NegativeBinomialTorch
from torch.distributions.utils import broadcast_all


class NegativeBinomial(NegativeBinomialTorch):
    def __init__(self,
                 loc: torch.Tensor,
                 dispersion: torch.Tensor,
                 validate_args: Optional[bool] = None):
        loc, dispersion = broadcast_all(loc, dispersion)
        super().__init__(
            total_count=dispersion,
            logits=loc.log() - dispersion.log(),
            validate_args=validate_args
        )
