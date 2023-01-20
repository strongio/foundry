from typing import Union, Dict

import torch

from foundry.util import ArrayType


def assert_tensors_equal(x: torch.Tensor, y: torch.Tensor, tol: float = 0., prefix: str = ''):
    assert x.shape == y.shape, f"{prefix}shape did not match"
    assert x.is_sparse == y.is_sparse, f"{prefix}one but not both of the tensors is sparse"
    if x.is_sparse:
        x = x.to_dense()
        y = y.to_dense()
    if not tol:
        assert (x == y).all(), f"{prefix}some values did not match"
    else:
        assert torch.allclose(x, y, atol=tol, rtol=0), f"{prefix}some values did not match"
    assert x.dtype == y.dtype, f"{prefix}dtype did not match"
    assert x.device == y.device, f"{prefix}device did not match"


def assert_dict_of_tensors_equal(x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], tol: float = 0.):
    assert set(x) == set(y)
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            assert_tensors_equal(v, y[k], tol=tol, prefix=k + ': ')
        else:
            assert v == y[k], f"{k}: not equal"


def assert_scalars_equal(x: Union[ArrayType, float], y: Union[ArrayType, float], tol: float = 0.):
    x = float(x)
    y = float(y)
    if not tol:
        assert x == y
    else:
        assert abs(x - y) <= tol
