from typing import Union

import numpy as np
import pandas as pd
import torch

from foundry.util import to_1d


def make_discrete_target(value: Union[pd.Series, torch.Tensor],
                         is_right_censored: Union[pd.Series, torch.Tensor] = None,
                         interval: Union[pd.Series, torch.Tensor, float] = 1.) -> dict:
    """
    If a target is discrete -- e.g. can only take on integer values -- then it can be modelled as 'interval' censored.

    :param value: An array/series of values.
    :param is_right_censored: Optional indicator for right-censoring.
    :param interval: The minimum distance between values. Default 1.
    :return: A dictionary that can be fed to ``Glm.fit(y=)``.
    """
    if not isinstance(value, (pd.Series, torch.Tensor)):
        value = pd.Series(value)
    if is_right_censored is None:
        is_right_censored = np.zeros_like(value, dtype='bool')
    valuem1 = value - interval
    return {
        'value': value,
        'left_censoring': value.where(~is_right_censored, other=float('inf')),
        'right_censoring': valuem1.where(~is_right_censored, other=value)
    }
