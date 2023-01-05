from typing import Optional

import numpy as np
import torch
from foundry.util import transpose_last_dims

class Covariance:
    def __init__(self, low_rank: Optional[int] = None):
        self.low_rank = low_rank

    def get_num_params(self, rank: int) -> int:
        if self.low_rank:
            raise NotImplementedError('todo')
        else:
            return int(rank + (rank * (rank - 1) / 2))

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        if self.low_rank:
            raise NotImplementedError('todo')
        else:
            nparams = params.shape[-1]
            rank = .5 * (np.sqrt(8 * nparams + 1) - 1)
            log_diag, off_diag = torch.split(params, [rank, nparams - rank], dim=-1)
            L = log_chol_to_chol(log_diag, off_diag)
            return L @ transpose_last_dims(L)


def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    assert log_diag.shape[:-1] == off_diag.shape[:-1]
    rank = log_diag.shape[-1]
    L1 = torch.diag_embed(torch.exp(log_diag))
    L2 = torch.zeros_like(L1)
    mask = torch.tril_indices(rank, rank, offset=-1)
    L2[mask[0], mask[1]] = off_diag
    return L1 + L2
