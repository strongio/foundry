from typing import Callable, Dict, Type, Optional, Tuple, Sequence
from warnings import warn

import torch
from torch import distributions

from .util import log1mexp
from foundry.util import to_2d, is_invalid





class Family:
    """
    Combines a link-function and a torch family for use in a GLM.
    """

    def __init__(self,
                 distribution_cls: Type[torch.distributions.Distribution],
                 params_and_links: Dict[str, Callable],
                 is_classifier: Optional[bool] = None):
        self.distribution_cls = distribution_cls
        self.params_and_links = params_and_links
        has_probs_attr = hasattr(self.distribution_cls, 'probs')
        if is_classifier is None:
            is_classifier = has_probs_attr
        if is_classifier and not has_probs_attr:
            raise TypeError(f"`is_classifier=True`, but {self.distribution_cls.__name__} doesn't have a `probs` attr.")
        self.is_classifier = is_classifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.distribution_cls.__name__})"

    @property
    def params(self) -> Sequence[str]:
        return list(self.params_and_links)

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        dist_kwargs = {}
        for p, ilink in self.params_and_links.items():
            dist_kwargs[p] = ilink(kwargs.pop(p))
        dist_kwargs.update(kwargs)
        return self.distribution_cls(**dist_kwargs)

    @classmethod
    def _validate_values(cls,
                         value: torch.Tensor,
                         weight: Optional[torch.Tensor],
                         distribution: distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
        value = to_2d(value)

        cls._raise_invalid_values(value)

        if weight is None:
            weight = torch.ones_like(value)
        else:
            if (weight <= 0).any():
                raise ValueError("Some weight <= 0")
            weight = to_2d(weight)
            if weight.shape[0] != value.shape[0]:
                raise ValueError(f"weight.shape[0] is {weight.shape[0]} but value.shape[0] is {value.shape[0]}")

        if len(distribution.batch_shape) != 2:
            raise ValueError(f"distribution.batch_shape should be 2D, but it's {distribution.batch_shape}")

        if distribution.batch_shape != value.shape:
            raise ValueError(f"distribution.batch_shape is {distribution.batch_shape} but value.shape is {value.shape}")

        return value, weight

    @staticmethod
    def _raise_invalid_values(value: torch.Tensor):
        if is_invalid(value):
            raise ValueError("nans/infs in `value`")

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        value, weight = self._validate_values(value, weight, distribution)

        # TODO: support discretized

        log_probs = distribution.log_prob(value)
        log_probs = weight * log_probs
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

        result = _maybe_method(distribution, method_nm=methods[0], value=value)
        if result is None:
            # doesn't implement what we want, maybe we can get 1 minus what_we_want then flip it?
            result = _maybe_method(distribution, method_nm=methods[1], value=value)
            if result is not None:
                # yes! just need to flip it
                result = log1mexp(result)
            else:
                # no, fall back to unstable
                warn(
                    f"{type(distribution).__name__} does not implement `log_cdf` or `log_surv`, results may be unstable"
                )
                if lower_tail:
                    result = distribution.cdf(value=value).log()
                else:
                    result = (1 - distribution.cdf(value=value)).log()

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
