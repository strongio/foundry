from typing import Callable
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from foundry.evaluation.marginal_effects import (Binned, MarginalEffects,
                                                 binned, raw)
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
                None,
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

    binned_col_A = pd.Series(
        [
            pd.Interval(0.999, 2.0),
            pd.Interval(2.0, 3.0),
        ],
        dtype=pd.CategoricalDtype(
            categories=[
                pd.Interval(0.999, 2.0),
                pd.Interval(2.0, 3.0)
            ],
            ordered=True
        ),
        name="binnedA"
    )

    @pytest.mark.parametrize(
        argnames=["aggfun", "expected"],
        argvalues=[
            (
                "mid",
                pd.DataFrame({"binnedA": binned_col_A, "colA": [1.4995, 2.5]})
            ),
            (
                "min",
                pd.DataFrame({"binnedA": binned_col_A, "colA": [1, 3]})
            ),
            (
                np.median,
                pd.DataFrame({"binnedA": binned_col_A, "colA": [1.5, 3.0]})
            ),
        ]
    )
    def test__get_binned_feature_map(self, aggfun, expected):
        df = (
            self.x_data
            .assign(
                **{
                    "binnedA": [
                        pd.Interval(0.999, 2.0),
                        pd.Interval(0.999, 2.0),
                        pd.Interval(2.0, 3.0),
                    ],
                },
            )
            .astype({"binnedA": self.binned_col_A.dtype})
        )

        test = MarginalEffects._get_binned_feature_map(
            df,
            "binnedA",
            "colA",
            aggfun=aggfun,
        )

        print(test.dtypes, expected.dtypes)
        assert_frame_equal(test, expected)

    def test__get_binned_feature_map_empty_bins(self):
        df = (
            self.x_data
            .assign(
                **{
                    "binnedA": pd.Categorical(
                        [
                            pd.Interval(0.999, 2.0),
                            pd.Interval(0.999, 2.0),
                            pd.Interval(2.0, 3.0),
                        ],
                        categories=[
                            pd.Interval(-np.inf, 0.999),
                            pd.Interval(0.999, 2.0),
                            pd.Interval(2.0, 3.0)
                        ],
                    )
                },
            )
        )

        with pytest.raises(ValueError):
            MarginalEffects._get_binned_feature_map(
                df,
                "binnedA",
                "colA",
                "median",
            )
