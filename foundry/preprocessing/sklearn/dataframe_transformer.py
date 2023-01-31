from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer


class DataFrameTransformer(ColumnTransformer):
    """
    For sklearn <1.2, ``DataFrameTransformer`` is like ``ColumnTransformer`` except it returns a dataframe instead of
    an ndarray. This is useful for passing feature-names to downstream transformers in a pipeline (e.g.
    :class:`foundry.preprocessing.InteractionFeatures`), or preserving categorical dtypes for estimators that support
    these (e.g. LightGBM's).

    For sklearn >1.2, the newer ``set_output`` API means output-type no longer differentiates the two, and the
    main difference is how the two handle sparsity: ``ColumnTransformer`` does not support transformers with sparse
    outputs if pandas is the output-type, while ``DataFrameTransformer`` will permit sparse-outputs, encoding each as
    a ``SparseArray``.

    ``ColumnTransformer``'s lack of support for sparsity is an intentional design-decision -- performance degradation
    when the number of columns is large (~10k or more) -- so you should choose based on your use-case.
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
