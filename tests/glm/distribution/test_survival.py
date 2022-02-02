from dataclasses import dataclass

import pytest
import torch.distributions
from torch.distributions import Weibull

from foundry.glm.distribution.survival import weibull_log_surv, SurvivalGlmDistribution


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
    actual = weibull_log_surv(weibull, value).exp()
    assert round(actual.item(), 3) == round(expected.item(), 3)


class TestSurvivalGlmDistribution:
    @dataclass
    class Params:
        description: str
        alias: str

    @dataclass
    class Fixture:
        glm_distribution: SurvivalGlmDistribution

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description='weibull',
                alias='weibull',
            ),
        ]
    )
    def setup(self, request) -> Fixture:
        pass
