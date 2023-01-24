from typing import Callable
import pandas as pd
import numpy as np
import pytest
from unittest.mock import create_autospec
from pandas.testing import assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from foundry.evaluation.marginal_effects import Binned, MarginalEffects, binned, raw

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


def test_binned():
    binned_feature = binned("my_feature", bins=20)
    assert isinstance(binned_feature, Binned)


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


class TestMarginalEffects:

    x_data: pd.DataFrame = pd.DataFrame({f"col{x}": [1,2,3] for x in ['A', 'B', 'C']})

    @pytest.mark.parametrize(
        argnames=["col_transformer__columns", "expected"],
        argvalues=[
            ([["colA"]], ["colA"]),
            ([["colA"], ["colA", "colB"]], ["colA", "colB"]),
            ([["colA"], ("colA", "colB")], ["colA", "colB"]),
        ]
    )
    def test_feature_names_in(self, col_transformer__columns, expected):
        fake_column_transformer = create_autospec(ColumnTransformer, instance=True)
        fake_column_transformer._columns = col_transformer__columns
        fake_column_transformer.remainder = 'drop'
        fake_pipeline = Pipeline([("preprocessing", fake_column_transformer)])

        me = MarginalEffects(fake_pipeline)

        assert isinstance(me.feature_names_in, list)
        assert list(sorted(expected)) == list(sorted(me.feature_names_in))
