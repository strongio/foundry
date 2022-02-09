import inspect
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
    # safest:
    if hasattr(dist, '__getitem__'):
        return dist[item]

    # check that default approach is OK by making sure it captures __init__ kwargs:
    init_kwargs = set(
        p.name
        for p in inspect.signature(dist.__init__).parameters.values()
        if p.name not in ("self", "validate_args")
    )
    non_param = init_kwargs - set(dist.arg_constraints)
    if non_param:
        raise TypeError(
            f"Unable to subset {type(dist).__name__} using default approach, due to params in init that aren't in "
            f"`arg_constraints`: {non_param}. Please report this error to the package maintainer."
        )

    # approach in torch.distributions.Distribution:
    param_names = [k for k in list(dist.arg_constraints) if k in dist.__dict__]
    new_kwargs = {par: getattr(dist, par)[item] for par in param_names}
    new = type(dist)(**new_kwargs)
    if '_validate_args' in dist.__dict__:
        new._validate_args = dist._validate_args

    return new
