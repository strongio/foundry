from typing import Union, Type
from unittest.mock import create_autospec

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from foundry.preprocessing import make_drop_transformer, make_column_selector, InteractionFeatures


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
                pd.DataFrame({"A": [0., 1., 2., 3., ], "B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4., ]})
        ),
        (
                {"names": "A"},
                pd.DataFrame({"B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4.]})
        ),
        (
                {"names": ["A", "B"]},
                pd.DataFrame({"C": [-1., -2., -3., -4., ]})
        ),
        (
                {"pattern": "[AB]"},
                pd.DataFrame({"C": [-1., -2., -3., -4., ]})
        ),
        pytest.param(
            {"names": "A", "pattern": ".*"},
            ValueError
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
def test_make_drop_transformer(small_dataframe, kwargs, expected):
    if isinstance(expected, pd.DataFrame):
        my_drop_transformer = make_drop_transformer(**kwargs)
        test = my_drop_transformer.fit(small_dataframe).transform(small_dataframe)
        assert_frame_equal(expected, test)
    else:
        with pytest.raises(expected):
            make_drop_transformer(**kwargs)


def test_make_column_selector():
    small_dataframe = pd.DataFrame({
        "A": [0., 1., 2., 3., ],
        "B": [0.5, -0.5, 0.5, -0.5],
        "C": [-1., -2., -3., -4., ]
    })
    column_selector = make_column_selector("[AB]")

    assert column_selector(small_dataframe) == ["A", "B"]
    assert "[AB]" in repr(column_selector)


class TestInteractionFeatures:
    x_data = {f'col{i}': [1] for i in range(1, 5)}

    @pytest.mark.parametrize(
        argnames=['interactions', 'expected'],
        argvalues=[
            pytest.param(
                [('col1', 'col2')],
                [('col1', 'col2')],
                id='simple'
            ),
            pytest.param(
                [('col1', 'col50')],
                RuntimeError,
                id='not-present',
            ),
            pytest.param(
                [('col1', lambda x: ['col2', 'col3']), ('col1', 'col4')],
                [('col1', 'col2'), ('col1', 'col3'), ('col1', 'col4')],
                id='with_callable'
            ),
            pytest.param(
                [('col1', make_column_selector(pattern='unmatched'))],
                RuntimeError,
                id='callable_returns_nothing',
            ),
        ]
    )
    def test_fit(self, interactions: list, expected: Union[list, Type[Exception]]):
        X = pd.DataFrame(self.x_data)
        estimator = InteractionFeatures(interactions=interactions, no_selection_handling='raise')
        if isinstance(expected, list):
            assert estimator.fit(X=X).unrolled_interactions_ == expected
        else:
            with pytest.raises(expected):
                estimator.fit(X=X)

    @pytest.mark.parametrize(
        argnames=['unrolled_interactions', 'x_cols', 'expected'],
        argvalues=[
            pytest.param(
                [('col1', 'col2')],
                ['col1', 'col2', 'col3'],
                ['col1', 'col2', 'col3', 'col1:col2'],
                id='simple'
            ),
            pytest.param(
                [('col1', 'col2'), ('col1', 'col3'), ('col1', 'col4')],
                ['col1', 'col2', 'col3', 'col4'],
                ['col1', 'col2', 'col3', 'col4', 'col1:col2', 'col1:col3', 'col1:col4'],
                id='simple2'
            ),
            pytest.param(
                [('col1', 'colX')],
                ['col1', 'col2', 'col3', 'col4'],
                RuntimeError,
                id='missing'
            ),
            pytest.param(
                [('col1', 'col2')],
                ['col1', 'col2', 'col1:col2'],
                ['col1', 'col2', 'col1:col2'],
                id='pre-existing'
            ),
        ]
    )
    def test_transform(self, unrolled_interactions: list, x_cols: list, expected: Union[list, Type[Exception]]):
        instance = create_autospec(InteractionFeatures, instance=True)
        instance.sep = ":"
        instance.unrolled_interactions_ = unrolled_interactions

        X = pd.DataFrame(columns=x_cols)
        if isinstance(expected, list):
            assert InteractionFeatures.transform(instance, X=X).columns.tolist() == expected
        else:
            with pytest.raises(expected):
                InteractionFeatures.transform(instance, X=X)
