from typing import Callable
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from pandas.testing import assert_series_equal

from foundry.evaluation.marginal_effects import Binned, binned, raw

class TestBinned():
    @pytest.mark.parametrize(
        "bins",
        [
            (10),
            ("skip"),
            ([1, 3, 5]),
            (False)
        ],
        ids=[
            "bins=10",
            "bins=default",
            "bins=list",
            "bins=False"
        ]
    )
    def test_binned_init(self, bins):
        if bins == "skip":
            binned: Binned = Binned("feature_name")
        else:
            binned: Binned = Binned("feature_name", bins=bins)
        assert binned.orig_name == "feature_name"
        assert binned.bins == (bins if bins != "skip" else 20)

    @pytest.mark.parametrize(
        "bins, expected",
        [
            (
                2,
                pd.Series(
                    [
                        *([pd.Interval(-0.001, 9.5)] * 10),
                        *([pd.Interval(9.5, 19.0)] * 10),
                    ],
                    dtype=pd.CategoricalDtype(
                        categories=[pd.Interval(-0.001, 9.5), pd.Interval(9.5, 19.0)],
                        ordered=True
                    ),
                    name="my_feature"
                ),
            ),
            (
                [-np.inf, 1, 3, 5, np.inf],
                pd.Series(
                    [
                        *([pd.Interval(-np.inf, 1)] * 2),
                        *([pd.Interval(1, 3)] * 2),
                        *([pd.Interval(3, 5)] * 2),
                        *([pd.Interval(5, np.inf)] * 14)
                    ],
                    dtype=pd.CategoricalDtype(
                        categories=[
                            pd.Interval(-np.inf, 1),
                            pd.Interval(1, 3),
                            pd.Interval(3, 5),
                            pd.Interval(5, np.inf),
                        ],
                        ordered=True
                    ),
                    name="my_feature"
                )
            ),
            (
                False,
                pd.Series(list(range(20)), name="my_feature")
            )
        ],
        ids=[
            "bins=2",
            "bins=list",
            "bins=False"
        ]
    )
    def test_binned_call(self, bins, expected):
        binned = Binned("my_feature", bins=bins)

        test_dataframe = pd.DataFrame({
            "my_feature": list(range(20))
        })

        assert_series_equal(binned(test_dataframe), expected)


@patch("foundry.evaluation.marginal_effects.Binned", autospec=True)
def test_binned(fake_Binned):
    binned_feature = binned("my_feature", bins=20)
    fake_Binned.assert_called_once_with("my_feature", bins=20)


def test_raw():
    column_name = "my_feature"
    test_dataframe = pd.DataFrame({
        "my_feature": list(range(20))
    })

    raw_callable: Callable = raw(column_name)

    assert_series_equal(
        raw_callable(test_dataframe),
        test_dataframe[column_name]
    )
