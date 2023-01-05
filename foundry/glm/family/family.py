from typing import Callable, Dict, Type, Optional, Tuple, Sequence
from warnings import warn

import torch
from torch import distributions

from .util import log1mexp
from foundry.util import is_invalid, to_1d, to_2d, to_tensor


class Family:
    """
    Combines a link-function and a torch family for use in a GLM.
    """

    def __init__(self,
                 distribution_cls: Type[torch.distributions.Distribution],
                 params_and_links: Dict[str, Callable],
                 supports_predict_proba: Optional[bool] = None):
        """

        :param distribution_cls: A distribution class.
        :param params_and_links: A dictionary whose keys are names of distribution-parameters (i.e. init-args for the
         distribution-class and values are (inverse-)link functions.
        :param supports_predict_proba: Does this distribution support predicting probabilities for its values? Default
         is determined by whether the distribution has a ``probs`` attribute.
        """
        self.distribution_cls = distribution_cls
        self.params_and_links = params_and_links

        # supports_predict_proba:
        has_probs_attr = hasattr(self.distribution_cls, 'probs')
        if supports_predict_proba is None:
            supports_predict_proba = has_probs_attr
        if supports_predict_proba and not has_probs_attr:
            raise TypeError(
                f"`supports_predict_proba=True`, but {self.distribution_cls.__name__} doesn't have a `probs` attr."
            )
        self.supports_predict_proba = supports_predict_proba

        # if distribution has `total_count` (binomial/multinomial) then we can't fully support classification;
        # e.g. it doesn't make sense to train on or predict classes b/c these don't capture heterogenous counts
        self.has_total_count = hasattr(self.distribution_cls, 'total_count')

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.distribution_cls.__name__})"

    @property
    def params(self) -> Sequence[str]:
        return list(self.params_and_links)

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        dist_kwargs = {}
        for p, inverse_link in self.params_and_links.items():
            dist_kwargs[p] = inverse_link(kwargs.pop(p))
        dist_kwargs.update(kwargs)
        return self.distribution_cls(**dist_kwargs)

    def _validate_values(self,
                         value: torch.Tensor,
                         weight: Optional[torch.Tensor],
                         distribution: distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standardize the shape of value and weight so they match the distribution

        The shape of the returned value and weight depend on the batch_shape of the distribution.
        """
        # Check for nans and infs
        self._raise_invalid_values(value)

        # Evenly weighted if not defined
        if weight is None:
            weight = torch.ones_like(value)

        distribution_shape = distribution.batch_shape + distribution.event_shape

        if len(distribution_shape) == 1:
            value = to_1d(value)
            weight = to_1d(weight)
        elif len(distribution_shape) == 2:
            value = to_2d(value)
            weight = to_2d(weight)
        else:
            raise NotImplementedError("distribution batch_shape has more than 2 dimensions")

        # Check that weights are strictly positive
        if (weight <= 0).any():
            raise ValueError("Some weight <= 0")
        # Check that value and weight shapes have the same length
        if weight.shape[0] != value.shape[0]:
            raise ValueError(f"weight.shape[0] is {weight.shape[0]} but value.shape[0] is {value.shape[0]}")

        return value, weight

    @staticmethod
    def _raise_invalid_values(value: torch.Tensor):
        if is_invalid(value):
            raise ValueError("nans/infs in `value`")

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return a 1d array of log-probs, one for each row in ``value``.
        """

        value, weight = self._validate_values(value, weight, distribution)

        # TODO: support discretized

        log_probs = distribution.log_prob(value)
        if len(log_probs.shape) > 1:
            # - if multi-output with some relationship between outputs (e.g. categorical, mvnorm), then
            #   log_prob will be 1D-like.
            # - but if multi-output with no relationship between outputs, then log-prob will be 2d; the prob-per-row is
            #   just the product of the individual probs. todo: handle nans in one but not all dims?
            assert len(log_probs.shape) == 2
            if log_probs.shape[-1] > 1:
                assert len(value.shape) > 1 and value.shape[-1] == log_probs.shape[-1]
                log_probs = log_probs.sum(1)
            else:
                log_probs = to_1d(log_probs)
        assert len(log_probs.shape) == 1
        log_probs = to_1d(weight) * log_probs
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
    """ Get a method called with *args, **kwargs, from an object.

    This function gets the method specified by method_nm from the obj object.
    If the method is not callable, or it is not a method of the object, the function returns the fallback_value.
    Otherwise, the function attempts to call the method with *args and **kwargs and returns the returned value.
    If there is a NotImplementedError, the function returns the fallback value.

    :param obj: any object
    :param method_nm: a string reperesenting the name of method
    :param fallback_value: what to return if non-existent or not implemented.
    :param args: a tuple of arguments to pass to the method
    :param kwargs: a dic of kwargs to pass to the method.
    :return: method(*args, **kwargs), or `fallback_value`.
    """
    method = getattr(obj, method_nm, False)
    if not method or not callable(method):
        return fallback_value
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return fallback_value
