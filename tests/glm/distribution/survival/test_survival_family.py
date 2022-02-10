from dataclasses import dataclass, fields
from typing import Optional, Sequence
from unittest.mock import Mock, patch

import pytest
import torch

from foundry.glm.family.survival import SurvivalFamily
from foundry.util import to_2d
from tests.util import assert_tensors_equal


class TestSurvivalFamilyCensLogProb:
    @dataclass
    class Params:
        description: str
        value: torch.Tensor
        right_censoring: Optional[torch.Tensor]
        left_censoring: Optional[torch.Tensor]
        expected_input_cdf_upper_tail: Optional[torch.Tensor]
        expected_input_cdf_lower_tail: Optional[torch.Tensor]
        expected_input_log_prob: Optional[torch.Tensor]

    @dataclass
    class Fixture:
        expected_cdf_upper_tail_input: Optional[torch.Tensor]
        actual_cdf_upper_tail_inputs: Sequence[torch.Tensor]
        expected_cdf_lower_tail_input: Optional[torch.Tensor]
        actual_cdf_lower_tail_inputs: Sequence[torch.Tensor]
        expected_log_prob_input: torch.Tensor
        actual_log_prob_inputs: Sequence[torch.Tensor]

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description='Partial left and right censoring, exclusive',
                value=torch.tensor([4., 5., 6.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), float('inf'), 9.]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), 2., -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([[9.]]),
                expected_input_cdf_lower_tail=torch.tensor([[2.]]),
                expected_input_log_prob=torch.tensor([[4.]])
            ),
            Params(
                description='Partial left and right censoring, overlapping',
                value=torch.tensor([4., 5., 6.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), 8., 9.]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), 2., -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([8., 9.]).unsqueeze(-1),
                expected_input_cdf_lower_tail=torch.tensor([2.]).unsqueeze(-1),
                expected_input_log_prob=torch.tensor([4.]).unsqueeze(-1)
            ),
            Params(
                description='Total left and right censoring, overlapping',
                value=torch.tensor([4., 5., 6.]).unsqueeze(-1),
                right_censoring=torch.tensor([7., 8., 9.]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), 2., -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([7., 8., 9.]).unsqueeze(-1),
                expected_input_cdf_lower_tail=torch.tensor([2.]).unsqueeze(-1),
                expected_input_log_prob=torch.tensor([]).unsqueeze(-1)
            ),
            Params(
                description='No censoring',
                value=torch.tensor([4., 5., 6.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), float('inf'), float('inf')]).unsqueeze(-1),
                left_censoring=None,
                expected_input_cdf_upper_tail=torch.tensor([]).unsqueeze(-1),
                expected_input_cdf_lower_tail=None,
                expected_input_log_prob=torch.tensor([4., 5., 6.]).unsqueeze(-1)
            ),
        ]
    )
    @patch('foundry.glm.family.Family.log_cdf', autospec=True)
    @patch('torch.distributions.Weibull.log_prob', autospec=True)
    def setup(self, mock_weibull_log_prob: Mock, mock_log_cdf: Mock, request) -> Fixture:
        params: TestSurvivalFamilyCensLogProb.Params = request.param

        family = SurvivalFamily.from_name('weibull')
        torch_dist = family(
            scale=torch.zeros_like(params.value),
            concentration=torch.zeros_like(params.value)
        )

        mock_weibull_log_prob.return_value = 0.
        mock_log_cdf.return_value = 0.
        family._get_censored_log_prob(
            distribution=torch_dist,
            value=params.value,
            right_censoring=params.right_censoring,
            left_censoring=params.left_censoring,
        )
        actual_cdf_lower_tail_inputs = [
            call[1]['value'] for call in mock_log_cdf.call_args_list if call[1]['lower_tail']
        ]
        actual_cdf_upper_tail_inputs = [
            call[1]['value'] for call in mock_log_cdf.call_args_list if not call[1]['lower_tail']
        ]
        actual_log_prob_inputs = [call[0][1] for call in mock_weibull_log_prob.call_args_list]

        return self.Fixture(
            expected_cdf_upper_tail_input=params.expected_input_cdf_upper_tail,
            actual_cdf_upper_tail_inputs=actual_cdf_upper_tail_inputs,
            expected_cdf_lower_tail_input=params.expected_input_cdf_lower_tail,
            actual_cdf_lower_tail_inputs=actual_cdf_lower_tail_inputs,
            expected_log_prob_input=params.expected_input_log_prob,
            actual_log_prob_inputs=actual_log_prob_inputs
        )

    def test_left_cens(self, setup: Fixture):
        """
        Test that log_cdf is called with lower_tail=True for all/only left-censored inputs.
        """
        if setup.expected_cdf_lower_tail_input is None:
            assert not len(setup.actual_cdf_lower_tail_inputs), \
                "no expected_cdf_lower_tail_input, but log_cdf called with lower_tail=True"
        else:
            assert len(setup.actual_cdf_lower_tail_inputs) == 1, \
                "log_cdf called with lower_tail=True more than once"
            assert_tensors_equal(setup.actual_cdf_lower_tail_inputs[0],
                                 setup.expected_cdf_lower_tail_input)

    def test_right_cens(self, setup: Fixture):
        """
        Test that log_cdf is called with lower_tail=False for all/only right-censored inputs.
        """
        if setup.expected_cdf_upper_tail_input is None:
            assert not len(setup.actual_cdf_upper_tail_inputs), \
                "no expected_cdf_upper_tail_input, but log_cdf called with lower_tail=False"
        else:
            assert len(setup.actual_cdf_upper_tail_inputs) == 1, \
                "log_cdf called w/lower_tail=False more than once"
            assert_tensors_equal(setup.actual_cdf_upper_tail_inputs[0],
                                 setup.expected_cdf_upper_tail_input)

    def test_uncens(self, setup: Fixture):
        """
        Test that log_prob is called for all/only uncensored inputs.
        """
        assert len(setup.actual_log_prob_inputs) == 1, "log_prob called more than once"
        assert_tensors_equal(setup.actual_log_prob_inputs[0], setup.expected_log_prob_input)

# todo: trunc
