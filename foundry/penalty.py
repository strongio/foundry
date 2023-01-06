from typing import Union, Dict, Sequence, Tuple

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
     gaussian distribution. Can alternatively be a list of (callable, precision) tuples, which are treated like switch
     statements: the callable takes a feature-name and returns True/False, if it returns True that precision will be
     applied to that feature.
    :param mean: The mean of prior gaussian (see above) -- default zero. Like ``precision``, can be a list of tuples.
    """

    def __init__(self,
                 precision: Union[float, Sequence[Tuple[callable, float]]],
                 mean: Union[float, Sequence[Tuple[callable, float]]] = 0.):
        self.mean = mean
        self.precision = precision

    @staticmethod
    def _arg_to_tensor(arg: Union[float, Sequence[Tuple[callable, float]]],
                       feature_nms: np.ndarray,
                       **kwargs) -> torch.Tensor:
        if isinstance(arg, (float, int)):
            return torch.as_tensor([arg] * len(feature_nms), **kwargs)
        out = torch.zeros(len(feature_nms), **kwargs)
        for cond_fun, val in reversed(arg):
            mask = np.asarray([cond_fun(feat) for feat in feature_nms])
            out[mask] = val
        return out

    def __call__(self, module: torch.nn.Module, module_param_names: Dict[str, np.ndarray]) -> torch.Tensor:
        if set(module_param_names) != {'bias', 'weight'}:
            raise NotImplementedError(f"{type(self)} not implemented for module with params!={'bias', 'weight'}")

        feature_nms = module_param_names['weight'].reshape(-1)

        to_kwargs = get_to_kwargs(module)
        means = self._arg_to_tensor(self.mean, feature_nms, **to_kwargs)
        precisions = self._arg_to_tensor(self.precision, feature_nms, **to_kwargs)

        feature_dist = torch.distributions.Normal(loc=means, scale=1 / precisions ** .5, validate_args=False)
        return -feature_dist.log_prob(module.weight.view(-1)).sum()
