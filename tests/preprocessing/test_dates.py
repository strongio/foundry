from typing import Union, Optional
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest

from foundry.preprocessing import FourierFeatures


class TestFourierFeatures:
    @pytest.mark.parametrize(
        argnames=['period', 'expected_period'],
        argvalues=[
            ('weekly', np.timedelta64(7, 'D')),
            (np.timedelta64(7, 'D'), np.timedelta64(7, 'D')),
        ]
    )
    def test_fit(self, period: str, expected_period: str):
        ff = FourierFeatures(K=1, period=period).fit(pd.DataFrame({'a': [1], 'b': [1]}))
        assert ff.period == expected_period
        assert ff.feature_names_in_ == ['a', 'b']

    @pytest.mark.parametrize(
        argnames=['output', 'expected'],
        argvalues=[
            (None, np.ones((3, 2))),
            ('pandas', pd.DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1]}))
        ]
    )
    def test_transform(self, output: Optional[str], expected: Union[np.ndarray, pd.DataFrame]):
        ff = FourierFeatures(K=None, period=None)
        ff._transform_datetimes = create_autospec(ff._transform_datetimes)
        ff._transform_datetimes.side_effect = lambda x: x
        X = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1]})
        ff.get_feature_names_out = create_autospec(ff.get_feature_names_out, return_value=X.columns)

        if output:
            ff.set_output(transform=output)
        result = ff.transform(X)
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(result, expected)
        else:
            np.testing.assert_array_equal(result, expected)
