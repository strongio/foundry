from typing import Sequence, Union
from warnings import warn

from sklearn.compose import ColumnTransformer
from scipy import sparse
import pandas as pd
import numpy as np


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
