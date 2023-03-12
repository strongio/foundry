from typing import Optional, Collection
from warnings import warn
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector


class ColumnDropper(TransformerMixin, BaseEstimator):
    """
    Drop columns based on a name, pattern, and/or zero-variance
    """

    def __init__(self,
                 names: Collection[str] = (),
                 pattern: Optional[str] = None,
                 drop_zero_var: bool = True
                 ):
        self.names = names
        self.pattern = pattern
        self.drop_zero_var = drop_zero_var
        self.drop_cols_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'ColumnDropper':
        if self.names and self.pattern:
            raise ValueError("Both names and regex defined. Only one may be defined.")
        elif self.pattern:
            self.drop_cols_ = make_column_selector(pattern=self.pattern)(X)
            if not self.drop_cols_:
                warn(f"pattern `{self.pattern}` returned no columns.")
        else:
            # avoid weirdness if they passed a string or generator:
            self.names = [self.names] if isinstance(self.names, str) else list(self.names)
            unmatched = set(self.names) - set(X.columns)
            if unmatched:
                raise RuntimeError(f"Some `names` not in X: {unmatched}")
            self.drop_cols_ = list(self.names)

        if self.drop_zero_var:
            # faster than `X.columns[X.nunique() <= 1]`
            zero_var_cols = X.columns[X.apply(lambda s: (s == s.iloc[0]).all())]
            self.drop_cols_.extend(col for col in zero_var_cols if col not in set(self.drop_cols_))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[[col for col in X.columns if col not in self.drop_cols_]].copy(deep=False)
