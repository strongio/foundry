from typing import Union

import torch

from foundry.util import ArrayType


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Implement a numerically stable ``log(1 - exp(x))``, as described in
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    out_dtype = x.dtype if torch.is_tensor(x) and torch.is_floating_point(x) else torch.get_default_dtype()
    x = torch.as_tensor(x, dtype=torch.double)
    out = torch.empty_like(x)
    mask = -x < 0.693
    out[mask] = torch.log(-torch.expm1(x[mask]))
    out[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return out.to(dtype=out_dtype)


def subset_distribution(
        dist: torch.distributions.Distribution,
        item: Union[slice, ArrayType]
) -> torch.distributions.Distribution:
    # approach taken in torch.distributions.Distribution.__repr__:
    param_names = [k for k in dist.arg_constraints if k in dist.__dict__]
    new_kwargs = {par: getattr(dist, par)[item] for par in param_names}
    return type(dist)(**new_kwargs, validate_args=False)


def maybe_method(obj: any, method_nm: str, fallback_value: any = None, *args, **kwargs) -> any:
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
