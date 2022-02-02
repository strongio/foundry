import math
from dataclasses import dataclass
from typing import Type, Optional
from unittest.mock import Mock

import pytest
import torch.distributions

from foundry.glm.distribution import GlmDistribution
from tests.glm.distribution.util import assert_dist_equal
from tests.util import assert_arrays_equal


class TestGlmDistribution:
    @dataclass
    class Params:
        description: str
        alias: str
        call_input: dict
        expected_call_output: torch.distributions.Distribution

    @dataclass
    class Fixture:
        glm_distribution: GlmDistribution
        call_output: torch.distributions.Distribution
        expected_call_output: torch.distributions.Distribution
        mock_log_prob: Mock

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description='vanilla binomial',
                alias='binomial',
                call_input={'probs': torch.tensor([0., 1.]).unsqueeze(-1)},
                expected_call_output=torch.distributions.Binomial(probs=torch.tensor([.5, 0.7311]).unsqueeze(-1)),
            ),
            # TODO: binomial with weights/counts
            Params(
                description='weibull',
                alias='weibull',
                call_input={
                    'scale': torch.tensor([0., 1.]).unsqueeze(-1),
                    'concentration': torch.tensor([0., 0.]).unsqueeze(-1)
                },
                expected_call_output=torch.distributions.Weibull(
                    scale=torch.tensor([1., math.exp(1.)]).unsqueeze(-1),
                    concentration=torch.ones(2).unsqueeze(-1)
                ),
            ),
        ]
    )
    def setup(self, request) -> Fixture:
        # create distribution:
        glm_distribution = GlmDistribution.from_name(name=request.param.alias)
        # __call__
        torch_distribution = glm_distribution(**request.param.call_input)
        # log_prob:
        torch_distribution.log_prob = Mock()
        torch_distribution.log_prob.return_value = torch.zeros(1)
        glm_distribution.log_prob(torch_distribution, value=torch.zeros(1))

        return self.Fixture(
            glm_distribution=glm_distribution,
            call_output=torch_distribution,
            expected_call_output=request.param.expected_call_output,
            mock_log_prob=torch_distribution.log_prob
        )

    def test_call(self, setup: Fixture):
        assert_dist_equal(setup.call_output, setup.expected_call_output)

    def test_log_prob(self, setup: Fixture):
        setup.mock_log_prob.assert_called_with(torch.zeros(1))

    @pytest.mark.parametrize(
        argnames=["input", "expected_output", "expected_exception"],
        argvalues=[
            ((torch.ones(3), None), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3, 1), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3, 1), torch.ones(2)), None, ValueError)
        ]
    )
    def test__validate_shapes(self,
                              input: tuple,
                              expected_output: tuple,
                              expected_exception: Optional[Type[Exception]],
                              setup: Fixture):
        torch_distribution = setup.expected_call_output
        if expected_exception:
            with pytest.raises(expected_exception):
                GlmDistribution._validate_shapes(*input, distribution=torch_distribution)
        else:
            value, weights = GlmDistribution._validate_shapes(*input, distribution=torch_distribution)
            assert_arrays_equal(value, expected_output[0])
            assert_arrays_equal(weights, expected_output[1])
