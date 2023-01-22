from typing import Optional, Tuple

import torch
from torch import distributions

from foundry.util import to_1d, is_invalid, to_2d, log1mexp
from foundry.glm.family.family import Family
from foundry.glm.family.util import subset_distribution


class SurvivalFamily(Family):

    def log_prob(self,
                 distribution: distributions.Distribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None,
                 right_censoring: Optional[torch.Tensor] = None,
                 is_right_censored: Optional[torch.Tensor] = None,
                 left_censoring: Optional[torch.Tensor] = None,
                 is_left_censored: Optional[torch.Tensor] = None,
                 right_truncation: Optional[torch.Tensor] = None,
                 left_truncation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param distribution: Torch family.
        :param value: Value for log-prob.
        :param weight: Optional weights, same shape as value.
        :param is_right_censored: Optional bool tensor indicating whether the corresponding value is right-censored.
        :param right_censoring: Optional tensor indicating the value of right-censoring. Alternative to
         ``is_right_censored``, which cannot be used when an entry is both right and left censored.
        :param is_left_censored: Optional bool tensor indicating whether the corresponding value is left-censored.
        :param left_censoring: Optional tensor indicating the value of left-censoring. Alternative to
         ``is_left_censored``, which cannot be used when an entry is both right and left censored.
        :param right_truncation: Optional tensor indicating the value of right-truncation for each value.
        :param left_truncation: Optional tensor indicating the value of left-truncation for each value.
        :return: The log-prob tensor.
        """
        value, weight = self._validate_values(value, weight, distribution)

        left_censoring, right_censoring = self._validate_censoring(
            value=value,
            right_censoring=right_censoring,
            is_right_censored=is_right_censored,
            left_censoring=left_censoring,
            is_left_censored=is_left_censored
        )

        log_probs = self._get_censored_log_prob(
            distribution=distribution,
            value=value,
            left_censoring=left_censoring,
            right_censoring=right_censoring
        )

        log_probs = self._truncate_log_probs(
            log_probs=log_probs,
            distribution=distribution,
            right_truncation=right_truncation,
            left_truncation=left_truncation
        )

        return log_probs * weight

    @staticmethod
    def _validate_censoring(value: torch.Tensor,
                            right_censoring: Optional[torch.Tensor] = None,
                            is_right_censored: Optional[torch.Tensor] = None,
                            left_censoring: Optional[torch.Tensor] = None,
                            is_left_censored: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if (is_right_censored is not None) and (is_left_censored is not None):
            raise ValueError(
                "Cannot pass both `is_right_censored` and `is_left_censored`, use `right_censoring` and "
                "`left_censoring` kwargs instead."
            )

        if is_right_censored is not None:
            if right_censoring is not None:
                raise ValueError("Cannot pass both `is_right_censored` and `right_censoring`")
            is_right_censored = to_1d(_as_bool(is_right_censored))
            right_censoring = torch.full_like(value, fill_value=float('inf'))
            right_censoring[is_right_censored] = value[is_right_censored]
        if right_censoring is None:
            right_censoring = torch.full_like(value, fill_value=float('inf'))

        if is_left_censored is not None:
            if left_censoring is not None:
                raise ValueError("Cannot pass both `is_left_censored` and `left_censoring`")
            is_left_censored = to_1d(_as_bool(is_left_censored))
            left_censoring = torch.full_like(value, fill_value=-float('inf'))
            left_censoring[is_left_censored] = value[is_left_censored]
        if left_censoring is None:
            left_censoring = torch.full_like(value, fill_value=-float('inf'))

        return left_censoring, right_censoring

    @staticmethod
    def _raise_invalid_values(value: torch.Tensor):
        # invalid values permitted for censored observations
        pass

    def _validate_values(self,
                         value: torch.Tensor,
                         weight: Optional[torch.Tensor],
                         distribution: distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
        value, weight = super()._validate_values(
            value=value,
            weight=weight,
            distribution=distribution
        )
        if value.shape[1] > 1:
            raise ValueError(f"SurvivalFamily does not currently support y.shape[1]>1")
        return value, weight

    @classmethod
    def _truncate_log_probs(cls,
                            log_probs: torch.Tensor,
                            distribution: distributions.Distribution,
                            right_truncation: Optional[torch.Tensor] = None,
                            left_truncation: Optional[torch.Tensor] = None,
                            ) -> torch.Tensor:
        assert len(log_probs.shape) == 2 and log_probs.shape[1] == 1

        trunc_values = torch.zeros_like(log_probs).expand(-1, 2).clone()

        # left truncation:
        if left_truncation is not None:
            left_truncation = to_2d(left_truncation)
            assert left_truncation.shape == log_probs.shape
            trunc_values[:, 0:1] = cls.log_cdf(distribution, value=left_truncation, lower_tail=False)

        # right truncation:
        if right_truncation is not None:
            right_truncation = to_2d(right_truncation)
            assert right_truncation.shape == log_probs.shape
            trunc_values[:, 1:2] = cls.log_cdf(distribution, value=right_truncation, lower_tail=True)

        if (_near_zero(trunc_values).sum(1) < 1).any():
            # TODO: this can be supported. pdf(x) / [cdf(upper)-cdf(lower)]
            raise NotImplementedError("Simultaneous left and right truncation not yet supported.")
        return log_probs - trunc_values.sum(1, keepdim=True)

    @classmethod
    def _get_censored_log_prob(cls,
                               distribution: distributions.Distribution,
                               value: torch.Tensor,
                               right_censoring: torch.Tensor,
                               left_censoring: torch.Tensor) -> torch.Tensor:
        assert len(value.shape) == 2 and value.shape[1] == 1  # currently only 1d supported
        right_censoring = right_censoring.view(*value.shape)
        left_censoring = left_censoring.view(*value.shape)

        is_right_censored = to_1d(~torch.isinf(right_censoring))
        if not distribution.support.check(0):
            # if value's lower-bound is 0 on a distribution that does not include zero, that is the same
            # as no censoring; trying to mark this as right-censored will lead to -inf log-probs
            is_right_censored &= to_1d(~_near_zero(right_censoring))
        is_left_censored = to_1d(~torch.isinf(left_censoring))
        is_doubly_censored = is_left_censored & is_right_censored
        is_uncens = ~(is_right_censored | is_left_censored)
        log_probs = torch.zeros_like(value)

        # only right censored:
        log_probs[is_right_censored & ~is_left_censored] = cls.log_cdf(
            subset_distribution(distribution, is_right_censored & ~is_left_censored),
            value=right_censoring[is_right_censored & ~is_left_censored],
            lower_tail=False
        )

        # only left censored:
        log_probs[is_left_censored & ~is_right_censored] = cls.log_cdf(
            subset_distribution(distribution, is_left_censored & ~is_right_censored),
            value=left_censoring[is_left_censored & ~is_right_censored],
            lower_tail=True
        )

        # interval/double censored:
        log_probs[is_doubly_censored] = cls._interval_log_prob(
            subset_distribution(distribution, is_doubly_censored),
            upper_bound=left_censoring[is_doubly_censored],
            lower_bound=right_censoring[is_doubly_censored]
        )

        # no censoring:
        if is_invalid(value[is_uncens]):
            raise ValueError("Some uncensored entries in `value` are nan/inf")
        log_probs[is_uncens] = subset_distribution(distribution, is_uncens).log_prob(value[is_uncens])

        return log_probs

    @classmethod
    def _interval_log_prob(cls,
                           distribution: distributions.Distribution,
                           lower_bound: torch.Tensor,
                           upper_bound: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(lower_bound)
        if not lower_bound.numel():
            return out

        log_cdf1 = cls.log_cdf(distribution, value=upper_bound, lower_tail=True)
        log_cdf2 = cls.log_cdf(distribution, value=lower_bound, lower_tail=True)
        assert not torch.isinf(log_cdf2).any()

        # need to avoid evaluating for is_too_close because:
        # 1. interval cens prob vanishes, so log_prob is -inf
        # 2. double censored prob approaches 1, but got there via -inf interval_cens, so gradient is nan
        is_too_close = torch.isclose(log_cdf2, log_cdf1)

        # lower-bound < upper-bound means interval censoring:
        is_interval = (lower_bound < upper_bound) & ~is_too_close
        # more stable version of log_cdf1.exp() - log_cdf2.exp()
        out[is_interval] = log_cdf1[is_interval] + (1 - (log_cdf2[is_interval] - log_cdf1[is_interval]).exp()).log()

        # lower-bound > upper-bound means double censored (x is > lower-bound *or* x is < upper-bound)
        is_double = (lower_bound > upper_bound) & ~is_too_close
        out[is_double] = log1mexp(log_cdf2[is_double] + (1 - (log_cdf1[is_double] - log_cdf2[is_double]).exp()).log())

        # case (1) above:
        if (is_too_close & is_interval).any():
            raise ValueError("Interval-censoring where lower_bound ~= upper_bound")
        # case (2) above:
        out[is_too_close & is_double] = 1.
        return out


def _near_zero(tens: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.isclose(tens, torch.zeros_like(tens), **kwargs)


def _as_bool(tens: torch.Tensor) -> torch.Tensor:
    if tens.dtype is torch.bool:
        return tens
    if not ((tens == 0) | (tens == 1)).all():
        raise ValueError("Unable to convert tensor to bool")
    return tens.to(torch.bool)
