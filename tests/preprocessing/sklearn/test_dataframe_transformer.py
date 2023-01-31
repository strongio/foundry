from typing import Union

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import spmatrix
from sklearn.preprocessing import OneHotEncoder

from foundry.preprocessing import DataFrameTransformer


class TestDataFrameTransformer:
    @pytest.mark.parametrize(
        argnames=['X', 'expected'],
        argvalues=[
            # resets index:
            (pd.DataFrame({'x': [1, 2, 3]}, index=[1, 2, 3]), pd.DataFrame({'x': [1, 2, 3]})),
            # convert numpy:
            (np.zeros((3, 2)), pd.DataFrame(np.zeros((3, 2)))),
            # convert sparse:
            (
                    OneHotEncoder(sparse=True).fit_transform([['a'], ['b'], ['c'], ['d']]),
                    pd.DataFrame(np.eye(4))
            )
        ]
    )
    def test__convert_single_transform_to_df(self,
                                             X: Union[np.ndarray, pd.DataFrame, spmatrix],
                                             expected: pd.DataFrame):
        result = DataFrameTransformer._convert_single_transform_to_df(X)
        if isinstance(X, spmatrix):
            assert hasattr(result, 'sparse')
            assert result.sparse.density == .25
            result = result.sparse.to_dense()
        pd.testing.assert_frame_equal(result, expected)
