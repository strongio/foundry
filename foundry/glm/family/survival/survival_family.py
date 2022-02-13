from typing import Optional, Tuple

import torch
from torch import distributions
from torch.distributions import transforms

from foundry.util import to_1d, is_invalid, to_2d
from .weibull_distribution import CeilingWeibull
from ..family import Family
from ..util import subset_distribution, log1mexp


class SurvivalFamily(Family):
    aliases = Family.aliases.copy()
    aliases['ceiling_weibull'] = (
        CeilingWeibull,
        {
            'scale': transforms.ExpTransform(),
            'concentration': transforms.ExpTransform(),
            'ceiling': transforms.SigmoidTransform()
        }
    )

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
    def _raise_invalid_values(value: torch.Tensor):
        # invalid values permitted for censored observations
        pass

    @classmethod
    def _validate_values(cls,
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
            raise ValueError("Some values are both left and right truncated.")

        return log_probs - trunc_values.sum(1, keepdim=True)

    @classmethod
    def _get_censored_log_prob(cls,
                               distribution: distributions.Distribution,
                               value: torch.Tensor,
                               right_censoring: torch.Tensor,
                               left_censoring: torch.Tensor) -> torch.Tensor:
        assert len(value.shape) == 2 and value.shape[1] == 1  # currently only 1d supported

        is_right_censored = to_1d(~torch.isinf(right_censoring))
        is_left_censored = to_1d(~torch.isinf(left_censoring))
        is_interval_censored = to_1d(left_censoring > right_censoring)
        is_doubly_censored = is_left_censored & is_right_censored & ~is_interval_censored
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

        # interval censored:
        log_probs[is_interval_censored] = cls.interval_log_prob(
            subset_distribution(distribution, is_interval_censored),
            lower=left_censoring[is_interval_censored],
            upper=right_censoring[is_interval_censored]
        )

        # double censored:
        log_probs[is_doubly_censored] = log1mexp(cls.interval_log_prob(
            subset_distribution(distribution, is_doubly_censored),
            lower=left_censoring[is_doubly_censored],
            upper=right_censoring[is_doubly_censored]
        ))

        # no censoring:
        if is_invalid(value[is_uncens]):
            raise ValueError("Some uncensored entries in `value` are nan/inf")
        log_probs[is_uncens] = subset_distribution(distribution, is_uncens).log_prob(value[is_uncens])

        return log_probs

    @classmethod
    def interval_log_prob(cls,
                          distribution: distributions.Distribution,
                          lower: torch.Tensor,
                          upper: torch.Tensor) -> torch.Tensor:
        lower_log_cdf = cls.log_cdf(distribution, value=lower, lower_tail=True)
        upper_log_cdf = cls.log_cdf(distribution, value=upper, lower_tail=True)

        # more stable version of upper_log_cdf.exp() - lower_log_cdf.exp()
        return upper_log_cdf + (1 - (lower_log_cdf - upper_log_cdf).exp()).log()


def _near_zero(tens: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.isclose(tens, torch.zeros_like(tens), **kwargs)


def _as_bool(tens: torch.Tensor) -> torch.Tensor:
    if tens.dtype is torch.bool:
        return tens
    if not ((tens == 0) | (tens == 1)).all():
        raise ValueError("Unable to convert tensor to bool")
    return tens.to(torch.bool)
