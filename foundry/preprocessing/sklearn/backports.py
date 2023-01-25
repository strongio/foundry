"""
For the not-that-rare data-scientist who's stuck in python 3.7
"""
from typing import Sequence

import numpy as np

from sklearn.impute import SimpleImputer as SimpleImputerBase
from sklearn.preprocessing import FunctionTransformer as FunctionTransformerBase

from sklearn.utils.validation import _check_feature_names_in


class FunctionTransformer(FunctionTransformerBase):
    """
    Add ``get_feature_names_out()`` to ``FunctionTransformer``
    """
    _feature_names_out = None

    def fit(self, X, y=None) -> 'FunctionTransformer':
        super().fit(X=X, y=y)
        maybe_df = self.transform(X)
        if hasattr(maybe_df, 'columns'):
            self._feature_names_out = list(maybe_df.columns)
        else:
            if len(maybe_df.shape) == 1:
                # this will always (?) error out later but that error message is more understandable
                self._feature_names_out = ['x0']
            else:
                assert len(maybe_df.shape) == 2
                self._feature_names_out = [f'x{i}' for i in range(maybe_df.shape[1])]
        return self

    def get_feature_names_out(self, feature_names_in) -> Sequence[str]:
        if len(feature_names_in) == len(self._feature_names_out):
            return np.asarray([
                x if x == y else f'{x}_{y}' for x, y in zip(feature_names_in, self._feature_names_out)
            ], dtype='object')
        elif len(feature_names_in) == 1:
            if len(self._feature_names_out) == 1 and feature_names_in[0] == self._feature_names_out[0]:
                return np.asarray([f'{feature_names_in[0]}'], dtype='object')
            return np.asarray([f'{feature_names_in[0]}_{y}' for y in self._feature_names_out], dtype='object')
        else:
            return np.asarray(self._feature_names_out)


class SimpleImputer(SimpleImputerBase):
    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        names = input_features[~np.isnan(self.statistics_)]
        if not self.add_indicator:
            return names
        prefix = type(self.indicator_).__name__.lower()
        indicator_names = np.asarray([f"{prefix}_{nm}" for nm in names[self.indicator_.features_]])
        return np.concatenate([names, indicator_names])
