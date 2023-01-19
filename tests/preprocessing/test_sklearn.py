from typing import Union, Type
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from pandas.core.arrays import SparseArray
from pandas.testing import assert_frame_equal
from scipy.sparse import spmatrix, coo_matrix
from sklearn.preprocessing import OneHotEncoder

from foundry.preprocessing import make_column_selector, InteractionFeatures, DataFrameTransformer, ColumnDropper


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


def test_make_column_selector():
    small_df = make_small_df()
    column_selector = make_column_selector("[AB]")

    assert column_selector(small_df) == ["A", "B"]
    assert "[AB]" in repr(column_selector)


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
                [('col1', 'col1')],
                [],
                id='paired-feature',
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
        argnames=['unrolled_interactions', 'X', 'expected'],
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
            pytest.param(
                [('a', 'b')],
                pd.DataFrame({'a': SparseArray([1, 1, 0]),
                              'b': SparseArray([1, 0, 1])}),
                pd.DataFrame({'a': SparseArray([1, 1, 0]),
                              'b': SparseArray([1, 0, 1]),
                              'a:b': SparseArray([1, 0, 0])}),
                id='sparse*sparse'
            ),
            pytest.param(
                [('a', 'b')],
                pd.DataFrame({'a': SparseArray([1, 1, 1]),
                              'b': SparseArray([1, 0, 1])}),
                pd.DataFrame({'a': SparseArray([1, 1, 1]),
                              'b': SparseArray([1, 0, 1]),
                              'a:b': SparseArray([1, 0, 1])}),
                id='sparse*sparse2'
            ),
            pytest.param(
                [('a', 'b')],
                pd.DataFrame({'a': SparseArray([0, 1, 1]),
                              'b': [.5, .5, .5]}),
                pd.DataFrame({'a': SparseArray([0, 1, 1]),
                              'b': [.5, .5, .5],
                              'a:b': SparseArray([0, .5, .5], fill_value=0.)}),
                id='sparse*dense'
            )
        ]
    )
    def test_transform(self,
                       unrolled_interactions: list,
                       X: Union[list, pd.DataFrame],
                       expected: Union[list, pd.DataFrame, Type[Exception]]):
        instance = create_autospec(InteractionFeatures, instance=True)
        instance.sep = ":"
        instance.unrolled_interactions_ = unrolled_interactions

        if isinstance(X, list):
            X = pd.DataFrame(columns=X)
        if isinstance(expected, list):
            expected = pd.DataFrame(columns=expected)

        if isinstance(expected, pd.DataFrame):
            result = InteractionFeatures.transform(instance, X=X)
            pd.testing.assert_frame_equal(result, expected)
            if hasattr(X, 'sparse'):
                assert result['a:b'].sparse.density < 1
        else:
            with pytest.raises(expected):
                InteractionFeatures.transform(instance, X=X)
