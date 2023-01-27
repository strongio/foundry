from math import log, exp
from typing import Optional, Type, Sequence

import numpy as np
import pytest
import torch
from torch.distributions import Binomial, Distribution, Weibull, ContinuousBernoulli

from foundry.glm.family.util import subset_distribution, log1mexp
from tests.glm.distribution.util import assert_dist_equal
from tests.conftest import assert_scalars_equal


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
        (Binomial, {'probs': torch.tensor([.5, .6])}, [True, False, True], None, IndexError),
        (Weibull,
         {'scale': torch.arange(1, 4.), 'concentration': torch.ones(3)},
         [1, 2],
         {'scale': torch.arange(2, 4.), 'concentration': torch.ones(2)},
         None
         ),
        (ContinuousBernoulli, {'probs': torch.tensor([.5, .6])}, [1], None, TypeError)
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
