from math import log, exp
from typing import Optional, Type, Sequence, Callable
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.distributions import Binomial, Distribution

from foundry.glm.family.util import subset_distribution, log1mexp, maybe_method
from tests.glm.distribution.util import assert_dist_equal
from tests.util import assert_scalars_equal


@pytest.mark.parametrize(
    argnames=["x"],
    argvalues=[(x,) for x in np.linspace(-0.1, -9.0, 10)]
)
def test_log1mexp_equiv(x: float):
    assert_scalars_equal(log1mexp(x), log(1 - exp(x)), tol=.001)


@pytest.mark.parametrize(
    ids=['very small', 'very large'],
    argnames=["x"],
    argvalues=[(-3.3119e-17,), (-700.,)]
)
def test_log1mexp_stable(x: float):
    try:
        unstable = log(1 - exp(x))
        # very large
        assert unstable == 0
        assert log1mexp(torch.as_tensor(x, dtype=torch.double)).abs() > 0
    except ValueError as e:
        # very small
        assert 'domain' in str(e)
        assert not torch.isinf(log1mexp(x))


@pytest.mark.parametrize(
    argnames=["cls", "params", "idx", "expected_params", "expected_exception"],
    argvalues=[
        (Binomial, {'probs': torch.tensor([.5, .6])}, [1], {'probs': torch.tensor([.6])}, None),
        (Binomial, {'logits': torch.arange(5.)}, [1, 2], {'logits': torch.tensor([1., 2.])}, None),
        (Binomial, {'probs': torch.tensor([.5, .6])}, [True, False, True], None, IndexError)
    ]
)
def test_subset_distribution(cls: Type[Distribution],
                             params: dict,
                             idx: Sequence,
                             expected_params: dict,
                             expected_exception: Optional[Type[Exception]], ):
    if expected_exception:
        with pytest.raises(expected_exception):
            subset_distribution(cls(**params), idx)
    else:
        actual = subset_distribution(cls(**params), idx)
        expected = cls(**expected_params)
        assert_dist_equal(actual, expected)


def _notimplemented(x):
    raise NotImplementedError


def _identity(x):
    return x


@pytest.mark.parametrize(
    argnames=['a_method', 'expected_res'],
    argvalues=[
        (_notimplemented, False),
        (_identity, 1.)
    ]
)
def test_maybe_method(a_method: Callable, expected_res: any):
    obj = Mock()
    obj.fake_method = a_method
    res = maybe_method(obj, method_nm='fake_method', fallback_value=False, x=1.)
    assert res == expected_res
