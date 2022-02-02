from typing import Union

import torch

from foundry.util import ArrayType


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Implement a numerically stable ``log(1 - exp(x))``, as described in
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    out = torch.empty_like(x)
    mask = -x < -0.693
    out[mask] = torch.log(-torch.expm1(x[mask]))
    out[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return out


def subset_distribution(
        dist: torch.distributions.Distribution,
        item: Union[slice, ArrayType]
) -> torch.distributions.Distribution:
    # approach taken in torch.distributions.Distribution.__repr__:
    param_names = [k for k in dist.arg_constraints if k in dist.__dict__]
    new_kwargs = {par: getattr(dist, par)[item] for par in param_names}
    return type(dist)(**new_kwargs, validate_args=False)
