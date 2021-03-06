from typing import Union, Dict

import numpy as np
import torch

from foundry.util import get_to_kwargs


class Penalty:
    def __call__(self, module: torch.nn.Module, module_param_names: Dict[str, np.ndarray]) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class L2(Penalty):
    """
    Create a penalty that can be passed to ``Glm(penalty=)``.

    :param precision: Module weights will have penalties equivalent to gaussian priors; this is the precision for that
     gaussian distribution. Can also be a dictionary with names corresponding to feature-names.
    :param mean: See above; the mean of this gaussian (default zero). Can be a dictionary with feature-names.
    """

    def __init__(self, precision: Union[float, dict], mean: Union[float, dict] = 0.):
        if not isinstance(mean, dict):
            mean = {'_default': mean}
        self.mean = mean
        if not isinstance(precision, dict):
            precision = {'_default': precision}
        self.precision = precision

    def __call__(self, module: torch.nn.Module, module_param_names: Dict[str, np.ndarray]) -> torch.Tensor:
        if set(module_param_names) != {'bias', 'weight'}:
            raise NotImplementedError(f"{type(self)} not implemented for module with params!={'bias', 'weight'}")

        to = get_to_kwargs(module)

        feature_nms = list(module_param_names['weight'].reshape(-1))
        try:
            means = torch.tensor([self.mean.get(nm, self.mean['_default']) for nm in feature_nms], **to)
            precisions = torch.tensor([self.precision.get(nm, self.precision['_default']) for nm in feature_nms], **to)
        except KeyError as e:
            raise RuntimeError(
                f"mean/precision passed to {type(self)} should be dict with keys '_default' or:\n{feature_nms}"
            ) from e
        feature_dist = torch.distributions.Normal(loc=means, scale=1 / precisions ** .5, validate_args=False)
        return -feature_dist.log_prob(module.weight.view(-1)).sum()
