from typing import Union

import torch

from foundry.util import ArrayType


def assert_arrays_equal(x: ArrayType, y: ArrayType, tol: float = 0.):
    assert x.shape == y.shape, "shape did not match"
    if not tol:
        assert (x == y).all(), "some values did not match"
    else:
        assert torch.allclose(x, y, atol=tol, rtol=0), "some values did not match"


def assert_scalars_equal(x: Union[ArrayType, float], y: Union[ArrayType, float], tol: float = 0.):
    x = float(x)
    y = float(y)
    if not tol:
        assert x == y
    else:
        assert abs(x - y) <= tol
