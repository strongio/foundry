from sklearn.base import TransformerMixin
from sklearn.compose import make_column_selector as make_column_selector_base
from sklearn.pipeline import make_pipeline

from .backports import FunctionTransformer


def identity(x):
    return x


def as_transformer(x) -> TransformerMixin:
    """
    Standardize a transformer, function, or list of these into a transformer. Lists are converted to sklearn.pipelines.

    # TODO: consider deprecating, only use-case is `as_transformer(x)` which has no real advantages
            over `FunctionTransformer(x)`
    """
    if x is None:
        return as_transformer(identity)
    if hasattr(x, '__iter__') and not isinstance(x, str):
        return make_pipeline(*[as_transformer(xi) for xi in x])
    if hasattr(x, 'transform'):
        return x
    elif callable(x):
        return FunctionTransformer(x)
    else:
        raise TypeError(f"{type(x).__name__} does not have a `transform()` method.")


class make_column_selector(make_column_selector_base):
    def __repr__(self):
        return f"make_column_selector(pattern={self.pattern}, " \
               f"dtype_include={self.dtype_include}, " \
               f"dtype_exclude={self.dtype_exclude})"
