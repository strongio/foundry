from typing import Union, Optional

import torch
from torch.distributions import Weibull, Exponential, constraints
from torch.distributions.utils import broadcast_all

from ..util import log1mexp


def exp_log_surv(self: Exponential, value: torch.Tensor) -> torch.Tensor:
    value, rate = broadcast_all(value, self.rate)
    log_surv = torch.zeros_like(value)
    nonzero = ~torch.isclose(value, torch.zeros_like(value))
    log_surv[nonzero] = -value[nonzero] * rate[nonzero]
    return log_surv


Exponential.log_surv = exp_log_surv


def weibull_log_surv(self: Weibull, value: torch.Tensor) -> torch.Tensor:
    value, scale, concentration = broadcast_all(value, self.scale, self.concentration)
    log_surv = torch.zeros_like(value)
    nonzero = ~torch.isclose(value, torch.zeros_like(value))
    log_surv[nonzero] = -((value[nonzero] / scale[nonzero]) ** concentration[nonzero])
    return log_surv


Weibull.log_surv = weibull_log_surv


class CeilingWeibull(Weibull):
    arg_constraints = {
        'scale': constraints.positive,
        'concentration': constraints.positive,
        'ceiling': constraints.unit_interval
    }

    def __init__(self,
                 scale: torch.Tensor,
                 concentration: torch.Tensor,
                 ceiling: Union[torch.Tensor, float] = 1.,
                 validate_args: Optional[bool] = None):
        scale, concentration, self.ceiling = broadcast_all(scale, concentration, ceiling)
        super().__init__(
            scale=scale,
            concentration=concentration,
            validate_args=validate_args
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return super().log_prob(value) + ceiling.log()

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return ceiling * super().cdf(value)

    def log_cdf(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return log1mexp(super().log_surv(value)) + ceiling.log()

    def log_surv(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    @property
    def mean(self):
        if (self.ceiling < 1).any():
            raise NotImplementedError("Mean not implemented when ceiling < 1")
        return super().mean

    @property
    def variance(self):
        if (self.ceiling < 1).any():
            raise NotImplementedError("Variance not implemented when ceiling < 1")
        return super().variance
