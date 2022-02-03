from dataclasses import dataclass

import pytest
import torch
from torch.distributions import Weibull

from foundry.glm.distribution.survival.weibull_distribution import weibull_log_surv, CeilingWeibull
from tests.util import assert_scalars_equal


@pytest.mark.parametrize(
    argnames=["params", "value"],
    argvalues=[
        ({'scale': 1, 'concentration': 1}, 1.),
        ({'scale': 1, 'concentration': 1}, 2.),
        ({'scale': 5, 'concentration': 1}, 1.),
        ({'scale': 5, 'concentration': 1}, 2.),
        ({'scale': 1, 'concentration': .5}, 1.),
        ({'scale': 1, 'concentration': 2}, 1.),
    ]
)
def test_weibull_log_surv(params: dict, value: float):
    weibull = Weibull(**params)
    value = torch.as_tensor(value)
    expected = 1 - weibull.cdf(value)
    actual1 = weibull_log_surv(weibull, value).exp()
    actual2 = weibull.log_surv(value).exp()
    assert_scalars_equal(actual1, expected, tol=.001)
    assert_scalars_equal(actual2, expected, tol=.001)


class TestCeilingWeibull:
    @dataclass
    class Params:
        scale: float
        concentration: float
        ceiling: float
        value: float

    @dataclass
    class Fixture:
        ceiling_weibull: CeilingWeibull
        weibull: Weibull
        value: torch.Tensor

    @pytest.fixture(
        params=[Params(scale=1., concentration=1., ceiling=1., value=1.),
                Params(scale=1., concentration=1., ceiling=1., value=10.),
                Params(scale=1., concentration=1., ceiling=.75, value=1.),
                Params(scale=1., concentration=1., ceiling=.75, value=10.)]
    )
    def setup(self, request):
        return self.Fixture(
            weibull=Weibull(
                scale=request.param.scale, concentration=request.param.concentration
            ),
            ceiling_weibull=CeilingWeibull(
                scale=request.param.scale, concentration=request.param.concentration, ceiling=request.param.ceiling
            ),
            value=torch.as_tensor(request.param.value)
        )

    def test_log_prob(self, setup: Fixture):
        expected = setup.weibull.log_prob(setup.value).exp() * setup.ceiling_weibull.ceiling
        actual = setup.ceiling_weibull.log_prob(setup.value).exp()
        assert_scalars_equal(expected, actual, tol=.001)
        assert round(expected.item(), 3) == round(actual.item(), 3)

    def test_cdf(self, setup: Fixture):
        expected = setup.weibull.cdf(setup.value) * setup.ceiling_weibull.ceiling
        actual = setup.ceiling_weibull.cdf(setup.value)
        assert_scalars_equal(expected, actual, tol=.001)

    def test_log_cdf(self, setup: Fixture):
        expected = setup.weibull.cdf(setup.value) * setup.ceiling_weibull.ceiling
        actual = setup.ceiling_weibull.log_cdf(setup.value).exp()
        assert_scalars_equal(expected, actual, tol=.001)
