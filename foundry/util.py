import copy
from typing import Union

import numpy as np
import pandas as pd
import torch

ArrayType = Union[np.ndarray, torch.Tensor]


def to_tensor(x, **kwargs) -> torch.Tensor:
    """
    Convert anything that ``np.asanyarray`` can handle into a tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = np.asarray(x)
    return torch.as_tensor(x, **kwargs)


def get_to_kwargs(x) -> dict:
    """
    Get kwargs for ``Tensor.to(dtype=?, device=?)``
    :param x: A tensor or Module.
    :return: Dictionary of kwargs.
    """
    if isinstance(x, torch.nn.Module):
        return get_to_kwargs(next(iter(x.parameters())))
    return {'dtype': x.dtype, 'device': x.device}


def is_invalid(x: torch.Tensor, reduce: bool = True) -> bool:
    if reduce:
        return torch.isinf(x).any() or torch.isnan(x).any()
    else:
        return torch.isinf(x) | torch.isnan(x)


class FitFailedException(RuntimeError):
    pass


class SliceDict(dict):
    """
    Adapted from https://github.com/skorch-dev/skorch/blob/baf0580/skorch/helper.py#L20
    """

    def __init__(self, **kwargs):
        lens = [v.shape[0] for v in kwargs.values()]
        num_lens = len(set(lens))
        if num_lens:
            if num_lens > 1:
                raise ValueError("XXX")
            self._len = lens[0]
        else:
            self._len = 0

        super().__init__(**kwargs)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, sl: Union[int, str, slice]) -> Union['SliceDict', ArrayType]:
        if isinstance(sl, int):
            raise ValueError(
                f"`{type(self).__name__}` can't be indexed with a single int; did you mean to wrap in a list?"
            )
        if isinstance(sl, str):
            return super(SliceDict, self).__getitem__(sl)
        return SliceDict(**{k: v[sl] for k, v in self.items()})

    def __setitem__(self, key: str, value: ArrayType):
        val_len = value.shape[0]
        if not len(self):
            self._len = val_len
        elif len(self) != val_len:
            raise ValueError("")

        super().__setitem__(key, value)

    def update(self, kwargs: dict):
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __repr__(self):
        out = super(SliceDict, self).__repr__()
        return "SliceDict(**{})".format(out)

    def copy(self) -> 'SliceDict':
        return type(self)(**self)

    def fromkeys(self, *args, **kwargs):
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        raise NotImplementedError


class ToSliceDict:
    """
    For use in sklearn pipelines (e.g. ``make_pipeline(ToSliceDict(['probs']), Glm(family='negative_binomial'))``).
    """

    def __init__(self, dist_params: list):
        # TODO: support dictionary w/keys as params and values as col-names/indices
        self.dist_params = dist_params

    def get_params(self, deep: bool = True) -> dict:
        return {'dist_params': copy.deepcopy(self.dist_params) if deep else self.dist_params}

    def transform(self, X: pd.DataFrame) -> SliceDict:
        return SliceDict(**{p: X for p in self.dist_params})

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self


def to_1d(arr: ArrayType) -> ArrayType:
    """
    - If 1d, return unchanged.
    - If 2d-nd, raise unless all but first dim is singleton.

    :param arr: Numpy array or tensor.
    :return: 1d of same type
    """
    if len(arr.shape) == 1:
        return arr
    arr = arr.squeeze()
    if len(arr.shape) > 1:
        # unable to squeeze
        raise ValueError("Unable to squeeze array to 1d.")
    if not arr.shape:
        # squeezed to scalar
        arr = arr[None, ...]
    return arr


def to_2d(arr: ArrayType) -> ArrayType:
    """
    - If 1d, unsqueeze 2nd dim.
    - If 2d, return unchanged
    - If 3d, raise unless all but first two dims are singleton.

    :param arr: Numpy array or tensor.
    :return: 2d of same type
    """
    ndim = len(arr.shape)
    if ndim == 2:
        return arr
    if ndim > 2:
        for _ in range(2, ndim + 1):
            arr = arr.squeeze(-1)  # this will fail if size > 1
        if len(arr.shape) > 2:
            raise ValueError("Unable to squeeze array to 2d.")
    else:
        if len(arr.shape) == 0:
            arr = arr[None, ...]
        if len(arr.shape) == 1:
            arr = arr[..., None]
    return arr
