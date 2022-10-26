import itertools
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer as FunctionTransformerBase

TransformerLike = Union[Iterable, Callable, TransformerMixin]


class DataFrameTransformer(ColumnTransformer):
    """
    Like ColumnTransformer, but returns a DataFrame. This is useful if the column-transformer is being used in a
    pipeline, since there is no other way to pass feature-names to the next step(s). It is also needed to pass
    categoricals to LGBM.
    """

    @classmethod
    def _convert_single_transform_to_df(cls, X) -> pd.DataFrame:
        if sparse.issparse(X):
            warn(f"sparse output not suported in {cls.__name__}")
            return X.toarray()
        if isinstance(X, pd.DataFrame):
            X.reset_index(drop=True, inplace=True)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X

    def _hstack(self, Xs: Sequence[Union[pd.DataFrame, np.ndarray]]) -> pd.DataFrame:
        Xs = [self._convert_single_transform_to_df(X) for X in Xs]
        out = pd.concat(Xs, axis=1)
        out.columns = self.get_feature_names_out()
        return out


class InteractionFeatures(TransformerMixin, BaseEstimator):
    """
    Take a model-matrix (dataframe) and return a copy with added interaction-terms.
    """

    def __init__(
            self,
            interactions: Sequence[Tuple[Union[str, Callable], ...]] = (),
            sep: str = ":",
    ):
        self.interactions = interactions
        self.sep = sep
        self.unrolled_interactions_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'InteractionFeatures':
        unrolled_interactions = []
        for int_idx, cols in enumerate(self.interactions):
            any_callables = False
            to_unroll = {}
            for i, col in enumerate(cols):
                if callable(col):
                    any_callables = True
                    to_unroll[i] = col(X)
                    if not to_unroll[i]:
                        warn(f"self.interactions[{int_idx}][{i}] is a callable that returned no columns")
                elif isinstance(col, str):
                    to_unroll[i] = [col]
                else:
                    raise ValueError(f"Expected {col} to be str or callable, got {type(col)}")

            for interaction in itertools.product(*to_unroll.values()):
                # e.g. interactions=[('my_col',make_column_selector('*'))]
                # will include a my_col*my_col interaction that should be dropped
                if len(interaction) != len(set(interaction)):
                    if len(interaction) > 3:
                        # for three-way interactions or less, any duplicate means the interaction can be dropped
                        # for >3, unclear whether we should drop the interaction or just drop the duplicates
                        raise NotImplementedError(f"n>3-way interaction with duplicates: {interaction}")
                    if not any_callables:
                        # if callable, no warning needed
                        warn(f"Dropping {interaction} because of duplicates.")
                    continue
                unrolled_interactions.append(interaction)
        self.unrolled_interactions_ = unrolled_interactions
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        available_cols = set(X.columns)

        new_cols = {}
        for interaction in self.unrolled_interactions_:
            if len(interaction) < 2:
                continue
            new_col = self.sep.join(interaction)
            if new_col in X.columns and new_col not in available_cols:
                warn(f"{new_col} is duplicated.")
            # TODO: support non-numeric
            new_cols[new_col] = 1.0
            for col in interaction:
                if col not in available_cols:
                    raise RuntimeError(f"{col} in `interactions`, but not in columns:\n{available_cols}")
                new_cols[new_col] *= X[col].values

        return X.assign(**new_cols)


class FunctionTransformer(FunctionTransformerBase):
    """
    Add ``get_feature_names_out()`` to ``FunctionTransformer``
    """
    _feature_names_out = None

    def fit(self, X, y=None) -> 'FunctionTransformer':
        super().fit(X=X, y=y)
        maybe_df = self.transform(X)
        if hasattr(maybe_df, 'columns'):
            self._feature_names_out = list(maybe_df.columns)
        else:
            if len(maybe_df.shape) == 1:
                # this will always (?) error out later but that error message is more understandable
                self._feature_names_out = ['x0']
            else:
                assert len(maybe_df.shape) == 2
                self._feature_names_out = [f'x{i}' for i in range(maybe_df.shape[1])]
        return self

    def get_feature_names_out(self, feature_names_in) -> Sequence[str]:
        if len(feature_names_in) == len(self._feature_names_out):
            return np.asarray([
                x if x == y else f'{x}_{y}' for x, y in zip(feature_names_in, self._feature_names_out)
            ], dtype='object')
        elif len(feature_names_in) == 1:
            if len(self._feature_names_out) == 1 and feature_names_in[0] == self._feature_names_out[0]:
                return np.asarray([f'{feature_names_in[0]}'], dtype='object')
            return np.asarray([f'{feature_names_in[0]}_{y}' for y in self._feature_names_out], dtype='object')
        else:
            return np.asarray(self._feature_names_out)


def identity(x):
    return x


def as_transformer(x: TransformerLike) -> TransformerMixin:
    """
    Standardize a transformer, function, or list of these into a transformer. Lists are converted to sklearn.pipelines.
    """
    if x is None:
        return as_transformer(identity)
    if hasattr(x, '__iter__') and not isinstance(x, str):
        return make_pipeline(*[as_transformer(xi) for xi in x])
    if hasattr(x, 'transform'):
        return x
    elif callable(x):
        return FunctionTransformer(x)
    else:
        raise TypeError(f"{type(x).__name__} does not have a `transform()` method.")



def make_column_dropper(*args, **kwargs):
    warn("make_column_dropper is deprecated, use make_drop_transformer")
    return make_drop_transformer(*args, **kwargs)

def make_drop_transformer(
    names: Optional[Union[str, Iterable[str]]]=None,
    pattern: Optional[str]=None
):
    """
    Returns a DataFrameTranformer (i.e. a ColumnTransformer) that drops a subset of features.

    Useful when you want certain downstream paths of the sklearn pipeline to use different features than each other.

    names: the name or names of features to drop
    regex: the pattern to match for features to drop
    """
    kwargs = {
        "remainder": "passthrough", # we specify which columns to drop, the rest stay.
        "verbose_feature_names_out": False, # don't add the "remainder__" prefix to the undropped columns
    }
    if names is None and pattern is None:
        return DataFrameTransformer(transformers = [], **kwargs)
    if names and pattern:
        raise ValueError("Both names and regex defined for drop_transformer. Only one may be defined.")

    # regex part
    if pattern is not None:
        return DataFrameTransformer(
            transformers = [
                ("drop", "drop", make_column_selector(pattern))
            ],
            **kwargs
        )
    # names part
    else:
        return DataFrameTransformer(
            transformers=[
                ("drop", "drop", names)
            ],
            **kwargs
        )
