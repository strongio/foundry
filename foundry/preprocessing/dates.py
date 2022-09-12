from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import _check_feature_names_in


class FourierFeatures(TransformerMixin, BaseEstimator):
    feature_names_in_ = None

    def __init__(self, K: int, period: Union[int, np.timedelta64, str], output_dataframe: bool = True):
        self.K = K
        self.period = period
        self.output_dataframe = output_dataframe

    def fit(self, X, y=None) -> 'FourierFeatures':
        if isinstance(self.period, str):
            if self.period == 'weekly':
                self.period = np.timedelta64(7, 'D')
            elif self.period == 'yearly':
                self.period = np.timedelta64(int(365.25 * 24), 'h')
            elif self.period == 'daily':
                self.period = np.timedelta64(24, 'h')
            else:
                raise ValueError("Unrecognized `period`.")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = [f'x{i}' for i in range(X.shape[1])]

        return self

    def get_feature_names_out(self, feature_names_in) -> np.ndarray:
        feature_names_in = _check_feature_names_in(self, feature_names_in)

        if len(feature_names_in) == 1:
            return np.asarray([colname for _, _, colname in self._iter()])

        out = []
        for f in feature_names_in:
            for _, _, colname in self._iter():
                out.append(f'{f}_{colname}')
        return np.asarray(out)

    def transform(self, X, y=None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X = X.values
        out = [self._transform_datetimes(X[:, i]) for i in range(X.shape[1])]
        out = np.hstack(out)
        if self.output_dataframe:
            out = pd.DataFrame(out, columns=self.get_feature_names_out(None))
        return out

    def _iter(self):
        for idx in range(self.K):
            k = idx + 1
            for is_cos in range(2):
                col_idx = idx * 2 + is_cos
                colname = f"K{k}_{'cos' if is_cos else 'sin'}"
                yield k, col_idx, colname

    def _transform_datetimes(self, datetimes: np.ndarray) -> np.ndarray:
        datetimes = np.asanyarray(datetimes)
        if isinstance(self.period, int):
            if not datetimes.dtype.name.startswith('int'):
                raise ValueError("`period` is an int, but the values passed are not type int")
            period_int = self.period
            time_int = datetimes
        else:
            period_int = int(self.period / np.timedelta64(1, 'ns'))
            time_int = (datetimes.astype("datetime64[ns]") - np.datetime64(0, 'ns')).astype('int64')

        out_shape = tuple(datetimes.shape) + (self.K * 2,)
        out = np.empty(out_shape, dtype='float64')
        for k, col_idx, colname in self._iter():
            val = 2. * np.pi * k * time_int / period_int
            out[..., col_idx] = np.sin(val) if colname.endswith('sin') else np.cos(val)
        return out
