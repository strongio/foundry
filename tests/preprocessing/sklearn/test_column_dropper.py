from typing import Union, Type

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from foundry.preprocessing import ColumnDropper


def make_small_df():
    return pd.DataFrame({
        "A": [0., 1., 2., 3.],
        "B": [0.5, 0.5, 0.5, 0.5],
        "C": [-1., -2., -3., -4.],
    })


@pytest.mark.parametrize(
    ['df', 'kwargs', 'expected'],
    [
        pytest.param(
            make_small_df(),
            {'drop_zero_var': False},
            make_small_df(),
            id="no names/pattern"
        ),
        pytest.param(
            make_small_df(),
            {"names": "A", 'drop_zero_var': False},
            pd.DataFrame({"B": [0.5, 0.5, 0.5, 0.5], "C": [-1., -2., -3., -4.]}),
            id="drop based on name"
        ),
        pytest.param(
            make_small_df(),
            {"names": ["A", "B"], 'drop_zero_var': False},
            pd.DataFrame({"C": [-1., -2., -3., -4., ]}),
            id="drop based on names"
        ),
        pytest.param(
            make_small_df(),
            {"pattern": "[AB]", 'drop_zero_var': False},
            pd.DataFrame({"C": [-1., -2., -3., -4., ]}),
            id="drop based on pattern"
        ),
        pytest.param(
            make_small_df(),
            {'drop_zero_var': True},
            pd.DataFrame({"A": [0., 1., 2., 3., ], "C": [-1., -2., -3., -4.]}),
            id="drop zero-var columns"
        ),
        pytest.param(
            make_small_df(),
            {'names': 'A', 'drop_zero_var': True},
            pd.DataFrame({"C": [-1., -2., -3., -4.]}),
            id="drop zero-var columns and more"
        ),
        pytest.param(
            make_small_df(),
            {"names": "A", "pattern": ".*", 'drop_zero_var': False},
            ValueError,
            id="can't pass both name and pattern"
        ),
        pytest.param(
            make_small_df(),
            {"names": "D", 'drop_zero_var': False},
            RuntimeError,
            id="can't pass colname that's not in df"
        ),
        pytest.param(
            make_small_df().assign(A=lambda df: df['A'].astype('str').astype('category'),
                                   B=pd.arrays.SparseArray([0, 1, 0, 0])),
            {'drop_zero_var': True},
            make_small_df().assign(A=lambda df: df['A'].astype('str').astype('category'),
                                   B=pd.arrays.SparseArray([0, 1, 0, 0])),
            id="mixed dtypes"
        )
    ]
)
def test_column_dropper(df: pd.DataFrame,
                        kwargs: dict,
                        expected: Union[pd.DataFrame, Type[Exception]]):
    my_drop_transformer = ColumnDropper(**kwargs)
    if isinstance(expected, pd.DataFrame):
        test = my_drop_transformer.fit(df).transform(df)
        assert_frame_equal(expected, test)
    else:
        with pytest.raises(expected):
            my_drop_transformer.fit(df)
