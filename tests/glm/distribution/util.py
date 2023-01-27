import torch

from tests.conftest import assert_tensors_equal


def assert_dist_equal(x: torch.distributions.Distribution, y: torch.distributions.Distribution, tol: float = .001):
    if type(x) != type(y):
        if type(x).__name__ == type(y).__name__ and x.__module__ != y.__module__:
            # this is legit confusing
            raise AssertionError(f"type(x) ({x.__module__}) != type(y) ({y.__module__})")
        assert type(x) == type(y)  # otherwise not too confusing

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
