from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer


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
