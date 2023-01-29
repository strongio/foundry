from typing import Optional, Union

import torch
from torch import distributions
from torch.distributions import constraints, Exponential, Weibull
from torch.distributions.utils import broadcast_all

from foundry.util import log1mexp


class NegativeBinomial(distributions.NegativeBinomial):
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


class _MultinomialStrict(constraints.Constraint):
    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound: Optional[torch.Tensor]):
        self.upper_bound = upper_bound

    def check(self, x: torch.Tensor) -> bool:
        check = (x >= 0).all(dim=-1)
        if self.upper_bound is not None:
            check &= (x.sum(dim=-1) == self.upper_bound).all()
        return check


class Multinomial(distributions.Multinomial):
    """
    In torch's Multinomial, ``total_count`` is used:

    1) Only for validation in log_prob (which unnecessarily enforces a limitation that total_count be a single value).
    2) Critically in ``sample``, where sampling for a tensor of total-counts is limited by
     https://github.com/pytorch/pytorch/issues/42407.
    3) In mean, variance, etc. where total_count can be a tensor without issue.

    In this implementation, ``total_count`` is instead allowed to be a tensor, except when using ``sample``.
    """

    def __init__(self,
                 total_count: Optional[torch.Tensor] = None,
                 probs: Optional[torch.Tensor] = None,
                 logits: Optional[torch.Tensor] = None,
                 validate_args: Optional[bool] = None):
        self._categorical = torch.distributions.Categorical(probs=probs, logits=logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        if isinstance(total_count, torch.Tensor) and total_count.shape != batch_shape:
            raise ValueError(
                f"If ``total_count`` is a tensor, it must have shape==probs/logits.shape[:-1] ({batch_shape})"
            )
        self.total_count: Optional[Union[torch.Tensor, int]] = total_count
        super(distributions.Multinomial, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return _MultinomialStrict(self.total_count)

    def sample(self, sample_shape=torch.Size(), total_count: Optional[int] = None):
        old_total_count = self.total_count
        try:
            if total_count is not None:
                self.total_count = total_count
            if not isinstance(self.total_count, int):
                raise NotImplementedError('inhomogeneous total_count is not supported')
            out = super().sample(sample_shape=sample_shape)
        finally:
            self.total_count = old_total_count
        return out


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
