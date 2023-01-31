from typing import Optional

import torch
from torch.distributions import transforms


class SoftmaxKp1:
    """
    Given logits corresponding to the probabilities of K classes, convert to class-probabilities for K+1 classes.

    :param x: A tensor of logits whose final dim indexes the class
    :return: A tensor of probs with the same shape as the input, except the last dim is one longer.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        *leading_dims, n_classes = x.shape
        x_p1 = torch.cat([x, torch.zeros(*leading_dims, 1, dtype=x.dtype, device=x.device)], -1)
        return torch.softmax(x_p1, -1)

    def get_param_dim(self, y_dim: int) -> int:
        return y_dim - 1


class SigmoidTransformForClassification(transforms.SigmoidTransform):
    def get_param_dim(self, y_dim: int) -> int:
        return y_dim - 1


class NoWeightModule(torch.nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(dim))
        self.weight = torch.empty(0)

    def forward(self, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.bias
        if input is not None:
            out = out.expand(len(input), -1)
        else:
            out = out.expand(1, -1)
        return out


class Stopping:
    def __init__(self,
                 abstol: Optional[float] = .001,
                 reltol: Optional[float] = None,
                 patience: int = 2,
                 type: str = 'params_and_loss',
                 optimizer: Optional[torch.optim.Optimizer] = None):

        self.type = type
        self.optimizer = optimizer
        self.abstol = abstol
        self.reltol = reltol
        self.old_value = None
        self.patience = patience
        self._patience_counter = 0
        self._info = (float('nan'), min(self.abstol or float('inf'), self.reltol or float('inf')))

    @torch.no_grad()
    def get_info(self, fmt: str = "{:.4}/{:}") -> str:
        return fmt.format(*self._info)

    @torch.no_grad()
    def get_new_value(self, loss: Optional[float]):
        flat_params = []
        if 'params' in self.type:
            assert self.optimizer is not None
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    flat_params.append(p.view(-1))
        if 'loss' in self.type:
            flat_params.append(torch.as_tensor([loss]))
        return torch.cat(flat_params)

    @torch.no_grad()
    def __call__(self, loss: Optional[float]) -> bool:
        new_value = self.get_new_value(loss)
        if self.old_value is None:
            self.old_value = new_value
            return False
        old_value = self.old_value
        self.old_value = new_value
        abs_change = (new_value - old_value).abs()
        assert not (self.abstol is None and self.reltol is None)
        if self.abstol is not None and (abs_change > self.abstol).any():
            self._info = (abs_change.max(), self.abstol)
            self._patience_counter = 0
            return False
        if self.reltol is None:
            self._info = (abs_change.max(), self.abstol)  # even if we've converged, print up to date info
        else:
            rel_change = abs_change / old_value.abs()
            self._info = (rel_change.max(), self.reltol)
            if (rel_change > self.reltol).any():
                self._patience_counter = 0
                return False
        self._patience_counter += 1
        return self._patience_counter >= self.patience
