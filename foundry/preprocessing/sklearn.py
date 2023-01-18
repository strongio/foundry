import itertools
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, Collection
from warnings import warn

import numpy as np
import pandas as pd
from pandas.core.arrays import SparseArray
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as make_column_selector_sklearn
from sklearn.exceptions import NotFittedError
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
            return pd.DataFrame.sparse.from_spmatrix(X)
        if isinstance(X, pd.DataFrame):
            X.reset_index(drop=True, inplace=True)
        else:
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

    :param interactions: A list of tuples. Each contains either (1) a column-name, (2) a 'column-selector' like
     ``sklearn.compose.make_column_selector()`` that takes a model-matrix and returns a list of column-names.
    :param sep: The separator between colnames in the outputted columns.
    :param no_selection_handling: During fit, if an element of ``interactions`` is not in the input (or is a
     column-selector and it returns no columns), how should this be handled? The default 'raise' will throw an
     exception, 'warn' will emit a warning, and 'ignore' will ignore.
    """

    def __init__(
            self,
            interactions: Sequence[Tuple[Union[str, Callable], ...]] = (),
            sep: str = ":",
            no_selection_handling: str = 'raise'
    ):
        self.interactions = interactions
        self.sep = sep
        self.unrolled_interactions_ = None
        self.no_selection_handling = no_selection_handling
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'InteractionFeatures':
        assert not X.empty
        self.feature_names_in_ = X.columns
        self.no_selection_handling = self.no_selection_handling.lower()
        assert self.no_selection_handling in {'raise', 'warn', 'ignore'}

        unrolled_interactions = []
        unrolled_interaction_sets = set()
        for int_idx, cols in enumerate(self.interactions):
            any_callables = False
            to_unroll = {}
            for i, col in enumerate(cols):
                if callable(col):
                    any_callables = True
                    to_unroll[i] = col(X)
                    if not to_unroll[i]:
                        _repr = f'({col}) ' if '<' not in repr(col) else ''
                        msg = f"`interactions[{int_idx}][{i}]` {_repr}is a callable that returned no columns"
                        if self.no_selection_handling == 'raise':
                            raise RuntimeError(msg)
                        if self.no_selection_handling == 'warn':
                            warn(msg)
                elif isinstance(col, str):
                    if col not in X.columns:
                        msg = f"{col} not in X.columns"
                        if self.no_selection_handling == 'raise':
                            raise RuntimeError(msg)
                        if self.no_selection_handling == 'warn':
                            warn(msg)
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

                frozen_interaction = frozenset(interaction)
                if frozen_interaction in unrolled_interaction_sets:
                    warn(f"{interaction} specified more than once, only keeping the first.")
                else:
                    unrolled_interactions.append(interaction)
                    unrolled_interaction_sets.add(frozen_interaction)

        self.unrolled_interactions_ = unrolled_interactions

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        orig_cols = set(X.columns)

        new_cols = {}
        for interaction_column_names in self.unrolled_interactions_:
            if len(interaction_column_names) < 2:
                continue
            if not set(interaction_column_names).issubset(orig_cols):
                raise RuntimeError(
                    f"Columns {interaction_column_names} in `interactions`, but not in columns:\n{orig_cols}"
                )

            new_colname = self.sep.join(interaction_column_names)
            if new_colname in orig_cols:
                warn(f"{new_colname} is already in X, overwriting.")

            new_cols[new_colname] = None
            for col in interaction_column_names:
                new_cols[new_colname] = _sparse_safe_multiply(new_cols[new_colname], X[col].values)

        df_new_cols = pd.DataFrame(new_cols, index=X.index)

        return X.drop(columns=X.columns[X.columns.isin(new_cols)]).join(df_new_cols)

    def get_feature_names_out(self):
        if self.feature_names_in_ is None:
            raise NotFittedError(f"This {type(self).__name__} is not fitted. Cannot get_feature_names_out.")

        feature_names_out = list(self.feature_names_in_)
        for interaction_column_names in self.unrolled_interactions_:
            if len(interaction_column_names) < 2:
                continue
            new_col = self.sep.join(interaction_column_names)
            if new_col in feature_names_out:
                # this follows the behavior above, where interactions are always placed in the same location
                # regardless of where they were in the original df
                feature_names_out.remove(new_col)
            feature_names_out.append(new_col)

        return feature_names_out


def _sparse_safe_multiply(old_vals: pd.Series, new_vals: pd.Series) -> Union[SparseArray, pd.Series]:
    if old_vals is None:
        return new_vals.copy()

    # because sparse-arrays allow for any fill-value, they don't leverage the fact that
    # sparse*sparse only needs to capture the intersection of the two; instead, they fill the union.
    old_is_sparse = pd.api.types.is_sparse(old_vals)
    new_is_sparse = pd.api.types.is_sparse(new_vals)
    if new_is_sparse and old_is_sparse:
        index_intersection = old_vals.sp_index.intersect(new_vals.sp_index)
        assert new_vals.fill_value == old_vals.fill_value == 0
    elif new_is_sparse:
        index_intersection = new_vals.sp_index
        assert new_vals.fill_value == 0
    elif old_is_sparse:
        index_intersection = old_vals.sp_index
        assert old_vals.fill_value == 0
    else:
        old_vals = old_vals * new_vals
        return old_vals
    product = (
            old_vals[index_intersection.to_int_index().indices] *
            new_vals[index_intersection.to_int_index().indices]
    )
    return SparseArray(data=product, sparse_index=index_intersection, fill_value=0.)


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


class ToCategorical(TransformerMixin, BaseEstimator):
    """
    This is useful for pipelines that end in a LightGBM estimator, which accept categoricals natively. It ensures we
    don't redefine categories when processing a new dataset at the time of prediction.
    """
    _feature_names_out = None

    def __init__(self):
        self.categories_ = None

    def get_feature_names_out(self, feature_names_in) -> Sequence[str]:
        return self._feature_names_out

    def fit(self, X: pd.DataFrame, y=None) -> 'ToCategorical':
        self.transform(X, fit_=True)
        return self

    def transform(self, X: pd.DataFrame, fit_: bool = False) -> pd.DataFrame:
        if not fit_ and self.categories_ is None:
            raise RuntimeError("This instance is not fitted yet.")

        out = {}
        for col in X.columns:
            if fit_:
                out[col] = X[col].astype('category')
                # TODO: lgbm doesn't like categories that are intervals (as in ``pd.cut``)
            else:
                out[col] = X[col].astype(pd.CategoricalDtype(categories=self.categories_[col]))

        if fit_:
            self._feature_names_out = list(X.columns)
            self.categories_ = {col: series.cat.categories.tolist() for col, series in out.items()}

        return pd.DataFrame(out, index=X.index)


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


class make_column_selector(make_column_selector_sklearn):
    def __repr__(self):
        return f"make_column_selector(pattern={self.pattern}, dtype_include={self.dtype_include}, dtype_exclude={self.dtype_exclude})"


class ColumnDropper(TransformerMixin, BaseEstimator):
    """
    Drop columns based on a name, pattern, and/or zero-variance
    """

    def __init__(self,
                 names: Collection[str] = (),
                 pattern: Optional[str] = None,
                 drop_zero_var: bool = True
                 ):
        self.names = names
        self.pattern = pattern
        self.drop_zero_var = drop_zero_var
        self.drop_cols_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'ColumnDropper':
        if self.names and self.pattern:
            raise ValueError("Both names and regex defined. Only one may be defined.")
        elif self.pattern:
            self.drop_cols_ = make_column_selector(pattern=self.pattern)(X)
            if not self.drop_cols_:
                warn(f"pattern `{self.pattern}` returned no columns.")
        else:
            # avoid weirdness if they passed a string or generator:
            self.names = [self.names] if isinstance(self.names, str) else list(self.names)
            unmatched = set(self.names) - set(X.columns)
            if unmatched:
                raise RuntimeError(f"Some `names` not in X: {unmatched}")
            self.drop_cols_ = list(self.names)

        if self.drop_zero_var:
            zero_var_cols = X.columns[(X == X.iloc[0]).all()]  # faster than `X.columns[X.nunique() <= 1]`
            self.drop_cols_.extend(col for col in zero_var_cols if col not in set(self.drop_cols_))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[col for col in X.columns if col not in self.drop_cols_]].copy(deep=False)


def make_drop_transformer(**kwargs) -> ColumnDropper:
    warn("``make_drop_transformer`` is deprecated, please use ``ColumnDropper()`` instead", category=DeprecationWarning)
    return ColumnDropper(**kwargs)
