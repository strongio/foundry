from typing import Sequence

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class ToCategorical(TransformerMixin, BaseEstimator):
    """
    This is useful for pipelines that end in a LightGBM estimator, which accept categoricals natively. It ensures we
    don't redefine categories when processing a new dataset at the time of prediction.
    """
    _feature_names_out = None

    def __init__(self):
        self.categories_ = None

    def get_feature_names_out(self, feature_names_in=None) -> Sequence[str]:
        return self._feature_names_out

    def fit(self, X: pd.DataFrame, y=None) -> 'ToCategorical':
        self.transform(X, fit_=True)
        return self

    def transform(self, X: pd.DataFrame, fit_: bool = False) -> pd.DataFrame:
        if not fit_ and self.categories_ is None:
            raise RuntimeError("This instance is not fitted yet.")

        out = {}
        for col in X.columns:
            if fit_:
                out[col] = X[col].astype('category')
                # TODO: lgbm doesn't like categories that are intervals (as in ``pd.cut``)
            else:
                out[col] = X[col].astype(pd.CategoricalDtype(categories=self.categories_[col]))

        if fit_:
            self._feature_names_out = list(X.columns)
            self.categories_ = {col: series.cat.categories.tolist() for col, series in out.items()}

        return pd.DataFrame(out, index=X.index)
