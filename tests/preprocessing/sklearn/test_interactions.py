from typing import Union, Type
from unittest.mock import create_autospec

import pandas as pd
import pytest
from pandas.core.arrays import SparseArray
from sklearn.compose import make_column_selector
from sklearn.exceptions import NotFittedError

from foundry.preprocessing import InteractionFeatures


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
                [('col1', 'col2'), ('col2', 'col1')],
                [('col1', 'col2')],
                id='redundant'
            ),
            pytest.param(
                [('col1', 'col50')],
                RuntimeError,
                id='not-present',
            ),
            pytest.param(
                [('col1', lambda _: ['col2', 'col3']), ('col1', 'col4')],
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
            assert all(estimator.feature_names_in_ == X.columns)
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
                [('col1', 'col2')],
                ['col1', 'col1:col2', 'col2'],
                ['col1', 'col2', 'col1:col2'],
                id='pre-existing2'
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
        instance.feature_names_in_ = X if isinstance(X, list) else X.columns

        if isinstance(X, list):
            X = pd.DataFrame(columns=X)
        if isinstance(expected, list):
            expected = pd.DataFrame(columns=expected)

        if isinstance(expected, pd.DataFrame):
            result = InteractionFeatures.transform(instance, X=X)
            pd.testing.assert_frame_equal(result, expected)
            if hasattr(X, 'sparse'):
                assert result['a:b'].sparse.density < 1

            assert all(InteractionFeatures.get_feature_names_out(instance) == expected.columns)
        else:
            with pytest.raises(expected):
                InteractionFeatures.transform(instance, X=X)

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
                [('col1', 'col2')],
                None,
                NotFittedError,
                id='unfitted'
            ),
            pytest.param(
                [('col1', 'col2')],
                ['col1', 'col2', 'col1:col2'],
                ['col1', 'col2', 'col1:col2'],
                id='pre-existing'
            ),
            pytest.param(
                [('col1', 'col2')],
                ['col1', 'col1:col2', 'col2'],
                ['col1', 'col2', 'col1:col2'],
                id='pre-existing2'
            ),
        ]
    )
    def test_get_feature_names_out(self,
                                   unrolled_interactions: list,
                                   x_cols: list,
                                   expected: Union[list, Type[Exception]]):
        instance = create_autospec(InteractionFeatures, instance=True)
        instance.sep = ":"
        instance.unrolled_interactions_ = unrolled_interactions
        instance.feature_names_in_ = x_cols

        if isinstance(expected, list):
            assert InteractionFeatures.get_feature_names_out(instance) == expected
        else:
            with pytest.raises(expected):
                InteractionFeatures.get_feature_names_out(instance)
