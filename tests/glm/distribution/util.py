import torch

from tests.conftest import assert_tensors_equal


def assert_dist_equal(x: torch.distributions.Distribution, y: torch.distributions.Distribution, tol: float = .001):
    assert x.arg_constraints == y.arg_constraints
    for nm in x.arg_constraints:
        x_attr = getattr(x, nm)
        if isinstance(x_attr, torch.Tensor):
            try:
                assert_tensors_equal(x_attr, getattr(y, nm), tol=tol)
            except AssertionError as e:
                raise AssertionError(f"{nm} was not equal") from e
        else:
            assert x == y, f"{nm} was not equal"
