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
    "kwargs, expected",
    [
        pytest.param(
            {'drop_zero_var': False},
            make_small_df(),
            id="no names/pattern"
        ),
        pytest.param(
            {"names": "A", 'drop_zero_var': False},
            pd.DataFrame({"B": [0.5, 0.5, 0.5, 0.5], "C": [-1., -2., -3., -4.]}),
            id="drop based on name"
        ),
        pytest.param(
            {"names": ["A", "B"], 'drop_zero_var': False},
            pd.DataFrame({"C": [-1., -2., -3., -4., ]}),
            id="drop based on names"
        ),
        pytest.param(
            {"pattern": "[AB]", 'drop_zero_var': False},
            pd.DataFrame({"C": [-1., -2., -3., -4., ]}),
            id="drop based on pattern"
        ),
        pytest.param(
            {'drop_zero_var': True},
            pd.DataFrame({"A": [0., 1., 2., 3., ], "C": [-1., -2., -3., -4.]}),
            id="drop zero-var columns"
        ),
        pytest.param(
            {'names': 'A', 'drop_zero_var': True},
            pd.DataFrame({"C": [-1., -2., -3., -4.]}),
            id="drop zero-var columns and more"
        ),
        pytest.param(
            {"names": "A", "pattern": ".*", 'drop_zero_var': False},
            ValueError,
            id="can't pass both name and pattern"
        ),
        pytest.param(
            {"names": "D", 'drop_zero_var': False},
            RuntimeError,
            id="can't pass colname that's not in df"
        ),
    ]
)
def test_column_dropper(kwargs: dict,
                        expected: Union[pd.DataFrame, Type[Exception]]):
    small_df = make_small_df()
    my_drop_transformer = ColumnDropper(**kwargs)
    if isinstance(expected, pd.DataFrame):
        test = my_drop_transformer.fit(small_df).transform(small_df)
        assert_frame_equal(expected, test)
    else:
        with pytest.raises(expected):
            my_drop_transformer.fit(small_df)
