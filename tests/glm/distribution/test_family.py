import math
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Type, Optional, Collection
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch.distributions

from foundry.glm import Glm
from foundry.glm.family import Family
from foundry.glm.glm import _softmax_kp1

from tests.glm.distribution.util import assert_dist_equal
from tests.util import assert_tensors_equal


class TestFamily:
    @dataclass
    class Params:
        alias: str
        call_input: dict
        expected_is_classifier: bool
        expected_call_output: torch.distributions.Distribution

    @dataclass
    class Fixture:
        family: Family
        call_input: dict
        expected_is_classifier: bool
        expected_call_output: torch.distributions.Distribution

    @pytest.fixture(
        ids=lambda x: x.alias,
        params=[
            Params(
                alias='binomial',
                call_input={'probs': torch.tensor([0., 1.]).unsqueeze(-1)},
                expected_is_classifier=True,
                expected_call_output=torch.distributions.Binomial(probs=torch.tensor([.5, 0.7311]).unsqueeze(-1)),
            ),
            Params(
                alias='negative_binomial',
                call_input={
                    'probs': torch.tensor([0., 1.]).unsqueeze(-1),
                    'total_count': torch.tensor([0., 1.]).unsqueeze(-1)
                },
                expected_is_classifier=False,
                expected_call_output=torch.distributions.NegativeBinomial(
                    probs=torch.tensor([.5, 0.7311]).unsqueeze(-1),
                    total_count=torch.tensor([1., math.e]).unsqueeze(-1)
                ),
            ),
            Params(
                alias='weibull',
                call_input={
                    'scale': torch.tensor([0., 1.]).unsqueeze(-1),
                    'concentration': torch.tensor([0., 0.]).unsqueeze(-1)
                },
                expected_is_classifier=False,
                expected_call_output=torch.distributions.Weibull(
                    scale=torch.tensor([1., math.exp(1.)]).unsqueeze(-1),
                    concentration=torch.ones(2).unsqueeze(-1)
                ),
            ),
        ])
    def setup(self, request):
        return self.Fixture(
            family=Glm._init_family(Glm, request.param.alias),
            call_input=request.param.call_input,
            expected_call_output=request.param.expected_call_output,
            expected_is_classifier=request.param.expected_is_classifier
        )

    def test_is_classifier(self, setup: Fixture):
        assert setup.family.is_classifier == setup.expected_is_classifier

    def test_call_output(self, setup: Fixture):
        call_output = setup.family(**setup.call_input)
        assert_dist_equal(call_output, setup.expected_call_output)


class TestFamilyLogCdf:
    """
    Test that the result came from the desired method: either log_cdf, log_surv, or cdf.
    """

    Params = namedtuple('Params', ['lower_tail', 'implemented', 'expected_result'])

    @pytest.mark.parametrize(
        argnames=Params._fields,
        argvalues=[
            Params(
                lower_tail=True,
                implemented=[],
                expected_result='cdf().log()'
            ),
            Params(
                lower_tail=False,
                implemented=[],
                expected_result='(1-cdf()).log()'
            ),
            Params(
                lower_tail=True,
                implemented=['log_surv'],
                expected_result='log1mexp(log_surv())'
            ),
            Params(
                lower_tail=False,
                implemented=['log_surv'],
                expected_result='log_surv()'
            ),
            Params(
                lower_tail=True,
                implemented=['log_cdf'],
                expected_result='log_cdf()'
            ),
            Params(
                lower_tail=False,
                implemented=['log_cdf'],
                expected_result='log1mexp(log_cdf())'
            ),
            Params(
                lower_tail=True,
                implemented=['log_cdf', 'log_surv'],
                expected_result='log_cdf()'
            ),
            Params(
                lower_tail=False,
                implemented=['log_cdf', 'log_surv'],
                expected_result='log_surv()'
            ),
        ]
    )
    @patch('foundry.glm.family.family.log1mexp', autospec=True)
    def test_log_cdf(self,
                     mock_log1mexp: Mock,
                     lower_tail: bool,
                     implemented: Collection[str],
                     expected_result: str):
        mock_distribution = Mock()

        # log_surv:
        mock_distribution.log_surv = Mock()
        if 'log_surv' in implemented:
            mock_distribution.log_surv.return_value = 'log_surv()'
        else:
            mock_distribution.log_surv.side_effect = NotImplementedError

        # log_cdf:
        mock_distribution.log_cdf = Mock()
        if 'log_cdf' in implemented:
            mock_distribution.log_cdf.return_value = 'log_cdf()'
        else:
            mock_distribution.log_cdf.side_effect = NotImplementedError

        # log1mexp:
        mock_log1mexp.side_effect = lambda x: f'log1mexp({x})'

        # cdf:
        mock_distribution.cdf.return_value = MagicMock()
        mock_distribution.cdf.return_value.log.return_value = 'cdf().log()'
        mock_distribution.cdf.return_value.__rsub__.return_value.log.return_value = '(1-cdf()).log()'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Family.log_cdf(
                distribution=mock_distribution,
                value='value',
                lower_tail=lower_tail
            )
        assert result == expected_result

        if implemented:
            mock_distribution.cdf.assert_not_called()

            if not lower_tail and 'log_surv' in implemented:
                mock_distribution.log_cdf.assert_not_called()
            else:
                mock_distribution.log_cdf.assert_called_with(value='value')

            if lower_tail and 'log_cdf' in implemented:
                mock_distribution.log_surv.assert_not_called()
            else:
                mock_distribution.log_surv.assert_called_with(value='value')
        else:
            mock_distribution.cdf.assert_called_with(value='value')


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
        assert_tensors_equal(value, expected_output[0])
        assert_tensors_equal(weights, expected_output[1])


@pytest.mark.parametrize(
    argnames=["input", "expected_output"],
    argvalues=[
        (torch.tensor([0., 0.]), torch.tensor([.3333, .3333, 0.3333])),
        (torch.tensor([0.]), torch.tensor([.5, .5])),
        (torch.tensor([1.]), torch.tensor([.7311, .2689])),
    ]
)
def test__softmax_kp1(input: torch.Tensor, expected_output: torch.Tensor):
    assert_tensors_equal(_softmax_kp1(input), expected_output, tol=.0001)
