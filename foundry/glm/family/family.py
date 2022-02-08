from typing import Callable, Dict, Type, Optional, Tuple, Sequence
from warnings import warn

import torch
from torch import distributions
from torch.distributions import transforms

from .util import log1mexp
from foundry.util import to_2d


class Family:
    """
    Combines a link-function and a torch family for use in a GLM.
    """
    aliases = {
        'binomial': (
            distributions.Binomial,
            {'probs': transforms.SigmoidTransform()}  # TODO: support total_count?
        ),
        'weibull': (
            torch.distributions.Weibull,
            {
                'scale': transforms.ExpTransform(),
                'concentration': transforms.ExpTransform()
            }
        )
    }

    @classmethod
    def from_name(cls, name: str, **kwargs) -> 'Family':
        args = cls.aliases[name]
        return cls(*args, **kwargs)

    def __init__(self,
                 distribution_cls: Type[torch.distributions.Distribution],
                 params_and_links: Dict[str, Callable]):
        self.distribution_cls = distribution_cls
        self.params_and_links = params_and_links

    @property
    def params(self) -> Sequence[str]:
        return list(self.params_and_links)

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        dist_kwargs = {}
        for p, ilink in self.params_and_links.items():
            dist_kwargs[p] = ilink(kwargs.pop(p))
        dist_kwargs.update(kwargs)
        return self.distribution_cls(**dist_kwargs)

    @staticmethod
    def _validate_values(value: torch.Tensor,
                         weights: Optional[torch.Tensor],
                         distribution: distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
        value = to_2d(value)
        if weights is None:
            weights = torch.ones_like(value)
        else:
            if (weights <= 0).any():
                raise ValueError("Some weights <= 0")
            weights = to_2d(weights)
            if weights.shape[0] != value.shape[0]:
                raise ValueError(f"weights.shape[0] is {weights.shape[0]} but value.shape[0] is {value.shape[0]}")

        if len(distribution.batch_shape) != 2:
            raise ValueError(f"family.batch_shape should be 2D, but it's {distribution.batch_shape}")

        return value, weights

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        value, weights = self._validate_values(value, weights, distribution)

        # TODO: support discretized

        log_probs = distribution.log_prob(value)
        log_probs = weights * log_probs
        return log_probs

    @staticmethod
    def log_cdf(distribution: distributions.Distribution, value: torch.Tensor, lower_tail: bool) -> torch.Tensor:
        """
        Try to get the log(cdf) or log(1-cdf) using a stable method; if this fails fall back to unstable w/warning.

        :param distribution: The family.
        :param value: The value to plug into the log cdf.
        :param lower_tail: If True, this is the CDF. If False, this is 1-CDF.
        :return: Tensor with values.
        """
        # first look for the method that doesn't require flipping (log1mexp), then for the method that does:
        methods = ['log_cdf', 'log_surv']
        if not lower_tail:
            methods = list(reversed(methods))
        methods.append('cdf')

        result = _maybe_method(distribution, method_nm=methods[0])
        if result is None:
            # doesn't implement what we want, maybe we can get 1 minus what_we_want then flip it?
            result = _maybe_method(distribution, method_nm=methods[1])
            if result is not None:
                # yes! just need to flip it
                result = log1mexp(result)
            else:
                # no, fall back to unstable
                warn(
                    f"{type(distribution).__name__} does not implement `log_cdf` or `log_surv`, results may be unstable"
                )
                if lower_tail:
                    result = distribution.cdf(value).log()
                else:
                    result = (1 - distribution.cdf(value)).log()

        return result


def _maybe_method(obj: any, method_nm: str, fallback_value: any = None, *args, **kwargs) -> any:
    """
    Try to call a method of an object. If it's not a method of that object, or its a NotImplemented method, return a
     default value.

    :param obj: Object
    :param method_nm: Name of method
    :param fallback_value: What to return if non-existent or not implemented.
    :param args: Arguments to method
    :param kwargs: Kwargs to method.
    :return: Return value of method, or `fallback_value`.
    """
    method = getattr(obj, method_nm, False)
    if not method or not callable(method):
        return fallback_value
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return fallback_value
