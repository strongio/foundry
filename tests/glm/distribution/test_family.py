import math
from dataclasses import dataclass
from typing import Type, Optional
from unittest.mock import Mock, patch

import pytest
import torch.distributions

from foundry.glm.family import Family
from tests.glm.distribution.util import assert_dist_equal
from tests.util import assert_arrays_equal


class TestFamily:
    @dataclass
    class Params:
        alias: str
        call_input: dict
        expected_call_output: torch.distributions.Distribution

    @dataclass
    class Fixture:
        family: Family
        call_input: dict
        expected_call_output: torch.distributions.Distribution

    @pytest.fixture(
        ids=lambda x: x.alias,
        params=[
            Params(
                alias='binomial',
                call_input={'probs': torch.tensor([0., 1.]).unsqueeze(-1)},
                expected_call_output=torch.distributions.Binomial(probs=torch.tensor([.5, 0.7311]).unsqueeze(-1)),
            ),
            Params(
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
        family = Family.from_name(name=request.param.alias)
        return self.Fixture(
            call_input=request.param.call_input,
            family=family,
            expected_call_output=request.param.expected_call_output,
        )

    def test_call(self, setup: Fixture):
        call_output = setup.family(**setup.call_input)
        assert_dist_equal(call_output, setup.expected_call_output)

    @patch('foundry.glm.family.Family._validate_values')
    def test_log_prob(self, _validate_values_mock: Mock, setup: Fixture):
        # todo: test weights

        # mock _validate_values
        _validate_values_mock.return_value = ('value', torch.ones(1))

        # mock log-prob:
        torch_distribution = Mock()
        torch_distribution.log_prob.return_value = torch.zeros(1)

        # call with value
        setup.family.log_prob(torch_distribution, value='value')

        # make sure family.log_prob was called with value:
        torch_distribution.log_prob.assert_called_with('value')


    @pytest.mark.parametrize(
        argnames=["input", "expected_output", "expected_exception"],
        argvalues=[
            ((torch.ones(3), None), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3, 1), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
            ((torch.ones(3, 1), torch.ones(2)), None, ValueError)
        ]
    )
    def test__validate_values(self,
                              input: tuple,
                              expected_output: tuple,
                              expected_exception: Optional[Type[Exception]],
                              setup: Fixture):
        torch_distribution = setup.expected_call_output
        if expected_exception:
            with pytest.raises(expected_exception):
                Family._validate_values(*input, distribution=torch_distribution)
        else:
            value, weights = Family._validate_values(*input, distribution=torch_distribution)
            assert_arrays_equal(value, expected_output[0])
            assert_arrays_equal(weights, expected_output[1])
