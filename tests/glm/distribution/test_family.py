import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Type, Optional, Sequence
from unittest.mock import Mock, patch

import pytest
import torch.distributions

from foundry.glm.family import Family

from tests.glm.distribution.util import assert_dist_equal
from tests.util import assert_arrays_equal


class TestFamilyCall:
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


class TestFamilyLogProb:

    @patch('foundry.glm.family.Family._validate_values', autospec=True)
    def test_log_prob(self, _validate_values_mock: Mock):
        # todo: test weights

        # mock _validate_values
        _validate_values_mock.return_value = ('value', torch.ones(1))

        # mock log-prob:
        family = Family(distribution_cls=Mock(), params_and_links={})
        torch_distribution = Mock()
        torch_distribution.log_prob.return_value = torch.zeros(1)

        # call with value
        family.log_prob(torch_distribution, value='value')

        # make sure family.log_prob was called with value:
        torch_distribution.log_prob.assert_called_with('value')


class TestFamilyLogCdf:
    @dataclass
    class Params:
        description: str
        lower_tail: bool
        implemented: Sequence[str]
        dist_cdf_return_value: Optional[torch.Tensor]
        expected_result: any

    @dataclass
    class Fixture:
        actual_result: any
        expected_result: any

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description="Want log cdf, implements neither log_*, calls log(cdf())",
                lower_tail=True,
                implemented=[],
                dist_cdf_return_value=torch.ones(1),
                expected_result=torch.log(torch.ones(1))
            ),
            Params(
                description="Want log 1-cdf, implements neither log_*, calls log(1-cdf())",
                lower_tail=False,
                implemented=[],
                dist_cdf_return_value=torch.zeros(1),
                expected_result=torch.log(torch.ones(1))
            ),
            Params(
                description="Want log cdf, implements log_surv, calls 1-log_surv",
                lower_tail=True,
                implemented=['log_surv'],
                dist_cdf_return_value=None,
                expected_result='log1mexp(log_surv)'
            ),
            Params(
                description="Want log surv, implements log_surv, calls log_surv",
                lower_tail=False,
                implemented=['log_surv'],
                dist_cdf_return_value=None,
                expected_result='log_surv'
            ),
            Params(
                description="Want log cdf, implements log_cdf, calls log_cdf",
                lower_tail=True,
                implemented=['log_cdf'],
                dist_cdf_return_value=None,
                expected_result='log_cdf'
            ),
            Params(
                description="Want log surv, implements log_cdf, calls 1-log_cdf",
                lower_tail=False,
                implemented=['log_cdf'],
                dist_cdf_return_value=None,
                expected_result='log1mexp(log_cdf)'
            ),
            Params(
                description="Want log cdf, implements both, calls log_cdf",
                lower_tail=True,
                implemented=['log_cdf', 'log_surv'],
                dist_cdf_return_value=None,
                expected_result='log_cdf'
            ),
            Params(
                description="Want log surv, implements both, calls log_surv",
                lower_tail=False,
                implemented=['log_cdf', 'log_surv'],
                dist_cdf_return_value=None,
                expected_result='log_surv'
            ),
        ]
    )
    @patch('foundry.glm.family.family.maybe_method', autospec=True)
    @patch('foundry.glm.family.family.log1mexp')
    def setup(self, mock_log1mexp: Mock, mock_maybe_method: Mock, request: 'FixtureRequest'):
        # like the real maybe_method, return None if not implemented
        # otherwise return the method that was used, so we know we used it
        mock_maybe_method.side_effect = \
            lambda *a, **k: k['method_nm'] if k['method_nm'] in request.param.implemented else None

        #
        mock_log1mexp.side_effect = lambda x: f'log1mexp({x})'

        #
        mock_distribution = Mock()
        mock_distribution.cdf.return_value = request.param.dist_cdf_return_value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Family.log_cdf(
                distribution=mock_distribution,
                value='value',
                lower_tail=request.param.lower_tail
            )

        return self.Fixture(
            actual_result=result,
            expected_result=request.param.expected_result,
        )

    def test_results(self, setup: Fixture):
        assert setup.expected_result == setup.actual_result


@pytest.mark.parametrize(
    argnames=["input", "expected_output", "expected_exception"],
    argvalues=[
        ((torch.ones(3), None), (torch.ones(3, 1), torch.ones(3, 1)), None),
        ((torch.ones(3), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
        ((torch.ones(3, 1), torch.ones(3)), (torch.ones(3, 1), torch.ones(3, 1)), None),
        ((torch.ones(3, 1), torch.ones(2)), None, ValueError)
    ]
)
def test__validate_values(input: tuple,
                          expected_output: tuple,
                          expected_exception: Optional[Type[Exception]]):
    torch_distribution = Mock()
    torch_distribution.batch_shape = (3, 1)
    if expected_exception:
        with pytest.raises(expected_exception):
            Family._validate_values(*input, distribution=torch_distribution)
    else:
        value, weights = Family._validate_values(*input, distribution=torch_distribution)
        assert_arrays_equal(value, expected_output[0])
        assert_arrays_equal(weights, expected_output[1])
