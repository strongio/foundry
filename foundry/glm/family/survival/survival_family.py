from typing import Optional

import torch
from torch import distributions
from torch.distributions import transforms

from .weibull_distribution import CeilingWeibull
from ..family import Family
from ..util import subset_distribution


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
                 weights: Optional[torch.Tensor] = None,
                 is_right_censored: Optional[torch.Tensor] = None,
                 is_left_censored: Optional[torch.Tensor] = None,
                 right_truncation: Optional[torch.Tensor] = None,
                 left_truncation: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        :param distribution: Torch family.
        :param value: Value for log-prob.
        :param weights: Optional weights, same shape as value.
        :param is_right_censored: Optional bool tensor indicating whether the corresponding value is right-censored.
        :param is_left_censored: Optional bool tensor indicating whether the corresponding value is left-censored.
        :param right_truncation: Optional tensor indicating the value of right-truncation for each value.
        :param left_truncation: Optional tensor indicating the value of left-truncation for each value.
        :param kwargs: TODO
        :return: The log-prob tensor.
        """
        value, weights = self._validate_values(value, weights, distribution)

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

    @classmethod
    def _truncate_log_probs(cls,
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
            trunc_values[is_rtrunc] = cls.log_cdf(
                subset_distribution(distribution, is_rtrunc), value=right_truncation[is_rtrunc], lower_tail=False
            )

        return log_probs - trunc_values

    @classmethod
    def _get_censored_log_prob(cls,
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
            log_probs[is_left_censored] = cls.log_cdf(
                subset_distribution(distribution, is_left_censored), value=value[is_left_censored], lower_tail=True
            )

        # right censoring
        if is_right_censored is not None:
            assert is_right_censored.shape == value.shape
            is_uncens[is_right_censored] = False
            log_probs[is_right_censored] = cls.log_cdf(
                subset_distribution(distribution, is_right_censored), value=value[is_right_censored], lower_tail=False
            )

        # no censoring:
        log_probs[is_uncens] = subset_distribution(distribution, is_uncens).log_prob(value[is_uncens])

        return log_probs
