from typing import Optional
from warnings import warn

import torch
from torch import distributions
from torch.distributions import Weibull
from torch.distributions.utils import broadcast_all

from .base import GlmDistribution
from .util import log1mexp, subset_distribution


def weibull_log_surv(self: Weibull, value: torch.Tensor) -> torch.Tensor:
    value, scale, concentration = broadcast_all(value, self.scale, self.concentration)
    log_surv = torch.zeros_like(value)
    nonzero = ~torch.isclose(value, torch.zeros_like(value))
    log_surv[nonzero] = -((value[nonzero] / scale[nonzero]) ** concentration[nonzero])
    return log_surv


Weibull.log_surv = weibull_log_surv


class SurvivalGlmDistribution(GlmDistribution):

    def log_prob(self,
                 distribution: distributions.Distribution,
                 value: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 is_right_censored: Optional[torch.Tensor] = None,
                 is_left_censored: Optional[torch.Tensor] = None,
                 right_truncation: Optional[torch.Tensor] = None,
                 left_truncation: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        :param distribution: Torch distribution.
        :param value: Value for log-prob.
        :param weights: Optional weights, same shape as value.
        :param is_right_censored: Optional bool tensor indicating whether the corresponding value is right-censored.
        :param is_left_censored: Optional bool tensor indicating whether the corresponding value is left-censored.
        :param right_truncation: Optional tensor indicating the value of right-truncation for each value.
        :param left_truncation: Optional tensor indicating the value of left-truncation for each value.
        :param kwargs: TODO
        :return: The log-prob tensor.
        """
        value, weights = self._validate_shapes(value, weights, distribution)

        log_probs = self._get_censored_log_prob(
            distribution=distribution,
            value=value,
            is_right_censored=is_right_censored,
            is_left_censored=is_left_censored
        )

        log_probs = self._truncate_log_probs(
            log_probs=log_probs,
            distribution=distribution,
            right_truncation=right_truncation,
            left_truncation=left_truncation
        )

        log_probs = log_probs.unsqueeze(-1)

        return log_probs * weights

    def _truncate_log_probs(self,
                            log_probs: torch.Tensor,
                            distribution: distributions.Distribution,
                            right_truncation: Optional[torch.Tensor] = None,
                            left_truncation: Optional[torch.Tensor] = None,
                            ) -> torch.Tensor:
        trunc_values = torch.zeros_like(log_probs)

        # left truncation:
        if left_truncation is not None:
            raise NotImplementedError

        # right truncation:
        if right_truncation is not None:
            assert right_truncation.shape == log_probs.shape
            is_rtrunc = right_truncation > 0
            trunc_values[is_rtrunc] = self.log_cdf(
                subset_distribution(distribution, is_rtrunc), value=right_truncation[is_rtrunc], lower_tail=False
            )

        return log_probs - trunc_values

    def _get_censored_log_prob(self,
                               distribution: distributions.Distribution,
                               value: torch.Tensor,
                               is_right_censored: Optional[torch.Tensor] = None,
                               is_left_censored: Optional[torch.Tensor] = None) -> torch.Tensor:
        #
        log_probs = torch.zeros_like(value)
        is_uncens = torch.ones(value.shape[0], dtype=torch.bool)

        # left censoring
        if is_left_censored is not None:
            assert is_left_censored.shape == value.shape
            is_uncens[is_left_censored] = False
            log_probs[is_left_censored] = self.log_cdf(
                subset_distribution(distribution, is_left_censored), value=value[is_left_censored], lower_tail=True
            )

        # right censoring
        if is_right_censored is not None:
            assert is_right_censored.shape == value.shape
            is_uncens[is_right_censored] = False
            log_probs[is_left_censored] = self.log_cdf(
                subset_distribution(distribution, is_right_censored), value=value[is_right_censored], lower_tail=False
            )

        # no censoring:
        log_probs[is_uncens] = subset_distribution(distribution, is_uncens).log_prob(value[is_uncens])

        return log_probs

    @staticmethod
    def log_cdf(distribution: distributions.Distribution, value: torch.Tensor, lower_tail: bool) -> torch.Tensor:
        """
        Try to get the log(cdf) or log(1-cdf) using a stable method; if this fails fall back to unstable w/warning.

        :param distribution: The distribution.
        :param value: The value to plug into the log cdf.
        :param lower_tail: If True, this is the CDF. If False, this is 1-CDF.
        :return: Tensor with values.
        """
        # first look for the method that doesn't require flipping (log1mexp), then for the method that does:
        methods = ['log_cdf', 'log_surv']
        if not lower_tail:
            methods = list(reversed(methods))

        if hasattr(distribution, methods[0]):
            method = getattr(distribution, methods[0])
            out = method(value)
        elif hasattr(distribution, methods[1]):
            method = getattr(distribution, methods[1])
            out = log1mexp(method(value))
        else:
            warn(f"{type(distribution).__name__} does not implement `log_cdf` or `log_surv`, results may be unstable")
            if lower_tail:
                out = distribution.cdf(value).log()
            else:
                out = (1 - distribution.cdf(value)).log()
        return out
