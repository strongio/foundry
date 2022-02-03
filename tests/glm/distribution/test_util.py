from math import log, exp
from typing import Optional, Type, Sequence

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
def test_log1mexp(x: float):
    x = torch.as_tensor(x)
    assert_scalars_equal(log1mexp(x), log(1 - exp(x)), tol=.001)


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


def test_maybe_method():
    pass  # TODO
