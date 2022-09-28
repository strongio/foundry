from typing import Callable
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from pandas.testing import assert_series_equal, assert_frame_equal
from sklearn.pipeline import make_pipeline

from foundry.preprocessing import make_column_dropper

@pytest.fixture()
def small_dataframe():
    return pd.DataFrame({
        "A": [0., 1., 2., 3.],
        "B": [0.5, -0.5, 0.5, -0.5],
        "C": [-1., -2., -3., -4.],
    })


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(),
            pd.DataFrame({"A": [0., 1., 2., 3.,], "B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4.,]})
        ),
        (
            {"names": "A"},
            pd.DataFrame({"B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4.]})
        ),
        (
            {"names": ["A", "B"]},
            pd.DataFrame({"C": [-1., -2., -3., -4.,]})
        ),
        (
            {"pattern": "[AB]"},
            pd.DataFrame({"C": [-1., -2., -3., -4.,]})
        ),
        pytest.param(
            {"names": "A", "pattern": ".*"},
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
    ids=[
        "no_drop",
        "drop_by_name",
        "drop_by_names",
        "drop_by_regex",
        "err_too_many",
    ]
)
def test_make_column_dropper(small_dataframe, kwargs, expected):
    my_pipeline = make_pipeline(
        make_column_dropper(**kwargs)
    )

    test = my_pipeline.fit(small_dataframe).transform(small_dataframe)

    assert_frame_equal(expected, test)
