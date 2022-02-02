from typing import Callable, Dict, Type, Optional, Tuple

import torch
from torch import distributions
from torch.distributions import transforms

from foundry.util import to_2d


class GlmDistribution:
    """
    Why does this exist?
    - Maps predicted params to arguments of the distribution
    - Makes assumptions about shape
    - Allows more complex use-cases (e.g survival analysis).
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
    def from_name(cls, name: str, **kwargs) -> 'GlmDistribution':
        args = cls.aliases.get(name, None)
        if args is None:
            raise NotImplementedError("TODO")
        return cls(*args, **kwargs)

    def __init__(self,
                 torch_distribution_cls: Type[torch.distributions.Distribution],
                 params_and_links: Dict[str, Callable]):
        self.torch_distribution_cls = torch_distribution_cls
        self.params_and_links = params_and_links

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        dist_kwargs = {}
        for p, ilink in self.params_and_links.items():
            dist_kwargs[p] = ilink(kwargs.pop(p))
        dist_kwargs.update(kwargs)
        return self.torch_distribution_cls(**dist_kwargs)

    @staticmethod
    def _validate_shapes(value: torch.Tensor,
                         weights: Optional[torch.Tensor],
                         distribution: distributions.Distribution) -> Tuple[torch.Tensor, torch.Tensor]:
        value = to_2d(value)
        if weights is None:
            weights = torch.ones_like(value)
        else:
            weights = to_2d(weights)
            if weights.shape[0] != value.shape[0]:
                raise ValueError(f"weights.shape[0] is {weights.shape[0]} but value.shape[0] is {value.shape[0]}")

        if len(distribution.batch_shape) != 2:
            raise ValueError(f"distribution.batch_shape should be 2D, but it's {distribution.batch_shape}")

        return value, weights

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weights: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        value, weights = self._validate_shapes(value, weights, distribution)

        log_probs = distribution.log_prob(value)
        log_probs = weights * log_probs
        return log_probs
