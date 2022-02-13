from dataclasses import dataclass
from typing import Sequence, Dict
from unittest.mock import Mock, patch

import pytest
import torch

from foundry.glm.family.survival import SurvivalFamily

from tests.util import assert_tensors_equal, assert_dict_of_tensors_equal


class TestSurvivalFamilyCensLogProb:
    @dataclass
    class Params:
        description: str
        value: torch.Tensor
        right_censoring: torch.Tensor
        left_censoring: torch.Tensor
        expected_input_cdf_upper_tail: torch.Tensor
        expected_input_cdf_lower_tail: torch.Tensor
        expected_input_interval_log_prob: Dict[str, torch.Tensor]
        expected_input_log_prob: torch.Tensor
        expected_output_log_prob: torch.Tensor

    @dataclass
    class Fixture:
        expected_cdf_upper_tail_input: torch.Tensor
        actual_cdf_upper_tail_inputs: Sequence[torch.Tensor]
        expected_cdf_lower_tail_input: torch.Tensor
        actual_cdf_lower_tail_inputs: Sequence[torch.Tensor]
        expected_log_prob_input: torch.Tensor
        actual_log_prob_inputs: Sequence[torch.Tensor]
        actual_output_log_prob: torch.Tensor
        expected_output_log_prob: torch.Tensor
        expected_interval_log_prob_input: Dict[str, torch.Tensor]
        actual_interval_log_prob_inputs: Sequence[Dict[str, torch.Tensor]]

    @pytest.fixture(
        ids=lambda x: x.description,
        params=[
            Params(
                description='Partial left and right censoring, exclusive',
                value=torch.tensor([10., 11., 12.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), float('inf'), 300.]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), 2., -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([[300.]]),
                expected_input_cdf_lower_tail=torch.tensor([[2.]]),
                expected_input_interval_log_prob={
                    'lower': torch.tensor([]).unsqueeze(-1),
                    'upper': torch.tensor([]).unsqueeze(-1)
                },
                expected_input_log_prob=torch.tensor([[10.]]),
                expected_output_log_prob=torch.tensor([10., 2., 300.]).unsqueeze(-1)
            ),
            Params(
                description='Partial left and right censoring, overlapping',
                value=torch.tensor([10., 11., 12.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), 200., 300.]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), 2., -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([300.]).unsqueeze(-1),
                expected_input_cdf_lower_tail=torch.tensor([]).unsqueeze(-1),
                expected_input_interval_log_prob={
                    'lower': torch.tensor([2.]).unsqueeze(-1),
                    'upper': torch.tensor([200.]).unsqueeze(-1)
                },
                expected_input_log_prob=torch.tensor([10.]).unsqueeze(-1),
                expected_output_log_prob=torch.tensor([10., -202., 300.]).unsqueeze(-1)
            ),
            Params(
                description='No censoring',
                value=torch.tensor([10., 11., 12.]).unsqueeze(-1),
                right_censoring=torch.tensor([float('inf'), float('inf'), float('inf')]).unsqueeze(-1),
                left_censoring=torch.tensor([-float('inf'), -float('inf'), -float('inf')]).unsqueeze(-1),
                expected_input_cdf_upper_tail=torch.tensor([]).unsqueeze(-1),
                expected_input_cdf_lower_tail=torch.tensor([]).unsqueeze(-1),
                expected_input_interval_log_prob={
                    'lower': torch.tensor([]).unsqueeze(-1),
                    'upper': torch.tensor([]).unsqueeze(-1)
                },
                expected_input_log_prob=torch.tensor([10., 11., 12.]).unsqueeze(-1),
                expected_output_log_prob=torch.tensor([10., 11., 12.]).unsqueeze(-1)
            ),
        ]
    )
    @patch('foundry.glm.family.survival.survival_family.log1mexp', autospec=True)
    @patch('foundry.glm.family.survival.SurvivalFamily.interval_log_prob', autospec=True)
    @patch('foundry.glm.family.Family.log_cdf', autospec=True)
    @patch('torch.distributions.Weibull.log_prob', autospec=True)
    def setup(self,
              mock_weibull_log_prob: Mock,
              mock_log_cdf: Mock,
              mock_interval_log_prob: Mock,
              mock_log1mexp: Mock,
              request) -> Fixture:
        params: TestSurvivalFamilyCensLogProb.Params = request.param

        family = SurvivalFamily.from_name('weibull')
        torch_dist = family(
            scale=torch.zeros_like(params.value),
            concentration=torch.zeros_like(params.value)
        )

        mock_weibull_log_prob.side_effect = lambda self, value: value
        mock_log_cdf.side_effect = lambda distribution, value, lower_tail: value
        mock_interval_log_prob.side_effect = lambda distribution, lower, upper: lower + upper
        mock_log1mexp.side_effect = lambda x: -x

        actual_log_prob = family._get_censored_log_prob(
            distribution=torch_dist,
            value=params.value,
            right_censoring=params.right_censoring,
            left_censoring=params.left_censoring,
        )
        actual_cdf_lower_tail_inputs = [
            kwargs['value'] for args, kwargs in mock_log_cdf.call_args_list if kwargs['lower_tail']
        ]
        actual_cdf_upper_tail_inputs = [
            kwargs['value'] for args, kwargs in mock_log_cdf.call_args_list if not kwargs['lower_tail']
        ]
        actual_interval_log_prob_inputs = [
            kwargs for args, kwargs in mock_interval_log_prob.call_args_list
        ]
        actual_log_prob_inputs = [args[1] for args, kwargs in mock_weibull_log_prob.call_args_list]

        return self.Fixture(
            expected_cdf_upper_tail_input=params.expected_input_cdf_upper_tail,
            actual_cdf_upper_tail_inputs=actual_cdf_upper_tail_inputs,
            expected_cdf_lower_tail_input=params.expected_input_cdf_lower_tail,
            actual_cdf_lower_tail_inputs=actual_cdf_lower_tail_inputs,
            expected_log_prob_input=params.expected_input_log_prob,
            actual_log_prob_inputs=actual_log_prob_inputs,
            expected_interval_log_prob_input=params.expected_input_interval_log_prob,
            actual_interval_log_prob_inputs=actual_interval_log_prob_inputs,
            actual_output_log_prob=actual_log_prob,
            expected_output_log_prob=params.expected_output_log_prob,
        )

    def test_interval_log_prob(self, setup: Fixture):
        """
        Test that interval_log_prob is called for doubly censored inputs
        """
        assert len(setup.actual_interval_log_prob_inputs) == 2
        # no tests currently use left > right
        # todo: don't hardcode this
        assert_dict_of_tensors_equal(
            setup.actual_interval_log_prob_inputs[0],
            {'lower': torch.tensor([]).unsqueeze(-1), 'upper': torch.tensor([]).unsqueeze(-1)}
        )
        # 2nd call is double censoring
        assert_dict_of_tensors_equal(
            setup.actual_interval_log_prob_inputs[1],
            setup.expected_interval_log_prob_input
        )

    def test_left_cens(self, setup: Fixture):
        """
        Test that log_cdf is called with lower_tail=True for all/only left-censored inputs.
        """
        assert len(setup.actual_cdf_lower_tail_inputs) == 1, \
            "log_cdf called with lower_tail=True more than once"
        assert_tensors_equal(setup.actual_cdf_lower_tail_inputs[0], setup.expected_cdf_lower_tail_input)

    def test_right_cens(self, setup: Fixture):
        """
        Test that log_cdf is called with lower_tail=False for all/only right-censored inputs.
        """
        assert len(setup.actual_cdf_upper_tail_inputs) == 1, \
            "log_cdf called w/lower_tail=False more than once"
        assert_tensors_equal(setup.actual_cdf_upper_tail_inputs[0], setup.expected_cdf_upper_tail_input)

    def test_uncens(self, setup: Fixture):
        """
        Test that log_prob is called for all/only uncensored inputs.
        """
        assert len(setup.actual_log_prob_inputs) == 1, "log_prob called more than once"
        assert_tensors_equal(setup.actual_log_prob_inputs[0], setup.expected_log_prob_input)

    def test_output(self, setup: Fixture):
        assert_tensors_equal(setup.expected_output_log_prob, setup.actual_output_log_prob)

# todo: trunc
