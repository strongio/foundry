from sklearn.base import TransformerMixin
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
