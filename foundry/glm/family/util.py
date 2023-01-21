import inspect
from typing import Union, Type

import torch
from torch.distributions.utils import lazy_property

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


_dist2pars = {}


def _get_dist_pars(cls: Type[torch.distributions.Distribution]) -> list:
    if cls not in _dist2pars:
        # check that default approach is OK by making sure it captures __init__ kwargs:
        init_kwargs = set(
            p.name
            for p in inspect.signature(cls.__init__).parameters.values()
            if p.name not in ("self", "validate_args")
        )
        non_params = init_kwargs - set(cls.arg_constraints)
        if non_params:
            raise TypeError(
                f"Unable to subset {cls.__name__} using default approach, due to params in init that aren't in "
                f"`arg_constraints`: {non_params}. Please report this error to the package maintainer."
            )

        _dist2pars[cls] = list(cls.arg_constraints)
    return _dist2pars[cls]


def subset_distribution(
        dist: torch.distributions.Distribution,
        item: Union[slice, ArrayType]
) -> torch.distributions.Distribution:
    # safest:
    if hasattr(dist, '__getitem__'):
        return dist[item]
    klass = dist.__class__

    # approach in torch.distributions.Distribution:
    param_names = [
        k for k in _get_dist_pars(klass)
        if k in dist.__dict__ or not isinstance(getattr(klass, k), lazy_property)
    ]
    new_kwargs = {par: getattr(dist, par)[item] for par in param_names}
    new = klass(**new_kwargs, validate_args=False)
    if '_validate_args' in dist.__dict__:
        new._validate_args = dist._validate_args

    return new
