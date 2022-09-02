import copy
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

ArrayType = Union[np.ndarray, torch.Tensor]
ModelMatrix = Union[np.ndarray, pd.DataFrame, Dict[str, Union[np.ndarray, pd.DataFrame]]]


# TODO: unit-tests

def is_array(x) -> bool:
    """
    Check if an object is an array, by checking if it has an `__array__` method. This is usually to distinguish between
    arguments that are meant to be converted to tensors, and those that are meant to be left as-is, e.g., when
    preparing arbitrary arguments for an arbitrary pytorch method in ``Glm.predict``.
    """
    return hasattr(x, '__array__')


def to_tensor(x, **kwargs) -> torch.Tensor:
    """
    Convert anything that ``np.asarray`` can handle into a tensor.
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


class ToSliceDict(TransformerMixin, BaseEstimator):
    """
    Many distributions have multiple parameters: for example the normal distribution has a location and scale
    parameter. We configure which predictors should handle which dist-params in :class:`foundry.glm.Glm.fit` by passing
    a dictionary for X, with keys being arguments to the distribution, and values being model-matrices. This class
    enables us to configure this behavior within a pipeline, for example:

    >>> make_pipeline(ToSliceDict(['loc','scale']), Glm(family='gaussian'))

    :param mapping: This can be either a list or a dictionary. For example, ``mapping=['loc','scale']`` will create a
     dict with the full model-matrix assigned to both params. A dictionary allows finer-grained control, e.g.:
     ``mapping={'loc':[col1,col2,col3],'scale':[col1]}``. Finally, instead of the dict-values being lists of columns,
     these can be functions that takes the data and return the relevant columns: e.g.
     ``mapping={'loc':sklearn.compose.make_column_selector('^col+.'), 'scale':[col1]}``.
    """

    def __init__(self, mapping: Union[list, dict, None]):
        self.mapping = mapping

    def transform(self, X: ModelMatrix, extras_ok: bool = True) -> SliceDict:
        if not isinstance(self.mapping, dict):
            raise RuntimeError("This instance is not fitted yet.")

        if isinstance(X, dict):
            # if it's already a dict, not much transforming to do, but need to validate
            keys_ok = set(X).issubset(set(self.mapping)) if extras_ok else (set(X) == set(self.mapping))
            if not keys_ok:
                raise RuntimeError(
                    f"`mapping` from ``ToSliceDict(mapping=)`` is `{self.mapping}`, but ``X.keys()`` is `{set(X)}`"
                )

            for k, selection in self.mapping.items():
                actual = _get_col_names_or_idx(X[k])
                expected = _get_col_names_or_idx(_select_col_names_or_idx(X[k], selection))
                if actual != expected:
                    raise RuntimeError(f"For {k}, expected cols {expected} but got {actual}")

        else:
            X = {k: _select_col_names_or_idx(X, selection) for k, selection in self.mapping.items()}

        if not isinstance(X, SliceDict):
            X = SliceDict(**X)
        return X

    def fit(self, X: ModelMatrix, y: np.ndarray = None) -> 'ToSliceDict':
        if self.mapping is None:
            if isinstance(X, dict):
                self.mapping = list(X)
            else:
                raise RuntimeError(f"Can only set mapping to None if X is a dict, but got {type(X)}")

        if isinstance(X, dict) and set(self.mapping) != set(X):
            raise RuntimeError(
                f"`mapping` from ``ToSliceDict(mapping=)`` is `{self.mapping}`, but ``X.keys()`` is `{set(X)}`"
            )

        if isinstance(self.mapping, (list, tuple)):
            if isinstance(X, dict):
                self.mapping = {k: _get_col_names_or_idx(thisX) for k, thisX in X.items()}
            else:
                self.mapping = {k: _get_col_names_or_idx(X) for k in self.mapping}

        if not isinstance(self.mapping, dict):
            raise ValueError("`mapping` must be a list, tuple or dict.")

        remainder_name = None
        for k in list(self.mapping):
            thisX = X[k] if isinstance(X, dict) else X
            if callable(self.mapping[k]):
                self.mapping[k] = list(self.mapping[k](thisX))
                if not self.mapping[k]:
                    raise RuntimeError(f"self.mapping['{k}'] is a callable that returned no columns.")
            elif isinstance(self.mapping[k], str):
                if self.mapping[k] == 'remainder':
                    if remainder_name is not None:
                        raise RuntimeError("Only one element of mapping can be 'remainder'.")
                    remainder_name = k
                else:
                    raise ValueError(f"mapping (for {k}) is a string not a list-- did you mean to wrap in []?")
            else:
                # will raise if missing
                _select_col_names_or_idx(thisX, self.mapping[k])

        if remainder_name:
            if isinstance(X, dict):
                raise RuntimeError("Cannot set any values of `mapping` to 'remainder' if X is a dict.")
            # one and only one of the selections can be the literal 'remainder': everything not selected by the others
            remainder = set(X.columns)
            for k, selection in self.mapping.items():
                if k == remainder_name:
                    continue
                remainder = remainder - set(selection)
            self.mapping[remainder_name] = list(remainder)

        return self


def _get_col_names_or_idx(x: Union[pd.DataFrame, np.ndarray]) -> list:
    if hasattr(x, 'columns'):
        return list(x.columns)
    else:
        return list(range(x.shape[1]))


def _select_col_names_or_idx(x: Union[pd.DataFrame, np.ndarray], selection: list) -> Union[pd.DataFrame, np.ndarray]:
    if hasattr(x, 'columns'):
        all_cols = set(x.columns)
        extra = set(selection) - all_cols
        if extra:
            msg = f"The following columns are not present:{extra}."
            if len(all_cols) < 25:
                msg += f"\nAvailable cols:{all_cols}"
            raise ValueError(msg)
        return x.loc[:, selection]
    else:
        return x[:, selection]


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
