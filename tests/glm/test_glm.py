from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Type
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError
from torch.distributions import constraints, identity_transform

from foundry.glm.family import Family
from foundry.glm.glm import ModelMatrix, Glm
from foundry.glm.util import NoWeightModule
from foundry.util import to_2d
from tests.util import assert_dict_of_tensors_equal, assert_scalars_equal


class _FakeDist:
    arg_constraints = {
        'param1': constraints.real,
        'param2': constraints.real
    }

    def __init__(self, param1: torch.Tensor, param2: torch.Tensor):
        self.param1 = param1
        self.param2 = param2


@pytest.fixture()
def family() -> Family:
    return Family(
        distribution_cls=_FakeDist,
        params_and_links={k: identity_transform for k in _FakeDist.arg_constraints}
    )


class TestBuildModelMats:
    @pytest.fixture()
    def glm(self, family: Family):
        glm = Glm(family=family)

        # module is used to determine dtype/device:
        glm.module_ = Mock(spec_set=['dtype', 'device'])
        glm.module_.dtype = torch.get_default_dtype()
        glm.module_.device = torch.device('cpu')

        return glm

    @dataclass
    class Params:
        description: str
        mm_params: Sequence[str]
        X: ModelMatrix
        y: Optional[ModelMatrix]
        sample_weight: Optional[np.ndarray]
        expect_y: bool = True
        expected_xdict: Optional[Dict[str, torch.Tensor]] = None
        expected_ydict: Optional[Dict[str, torch.Tensor]] = None
        expected_exception: Optional[Type[Exception]] = None  # todo: match

    @dataclass
    class Fixture:
        params: 'Params'
        xdict: Dict[str, torch.Tensor]
        ydict: Dict[str, torch.Tensor]
        exception: Optional[Type[Exception]]

    @pytest.fixture(
        ids=lambda p: p.description,
        params=[
            Params(
                description="Pass X as array, get a dict of tensors for all params",
                mm_params=['param1', 'param2'],
                X=np.array([-np.arange(5), np.arange(5.)]).T,
                y=np.arange(5.),
                sample_weight=None,
                expected_xdict={k: torch.stack([-torch.arange(5), torch.arange(5.)], 1)
                                for k in _FakeDist.arg_constraints},
                expected_ydict={'value': to_2d(torch.arange(5.)), 'weight': torch.ones(5, 1)}
            ),
            Params(
                description="Pass X as dict, get a dict; pass weights w/sample_weight",
                mm_params=['param1'],
                X={'param1': np.array([-np.arange(5), np.arange(5.)]).T},
                y=np.arange(5.),
                sample_weight=np.arange(5.),
                expected_xdict={'param1': torch.stack([-torch.arange(5), torch.arange(5.)], 1)},
                expected_ydict={'value': to_2d(torch.arange(5.)), 'weight': to_2d(torch.arange(5.))}
            ),
            Params(
                description="Pass X as dict but wrong keys",
                mm_params=['param1', 'param2'],
                X={k: np.array([-np.arange(5), np.arange(5.)]).T for k in ['param1', 'parmesian']},
                y=np.arange(5.),
                sample_weight=None,
                expected_exception=ValueError
            ),
            Params(
                description="Pass X as dataframe, pass y as dict of dataframes w/extra key",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y={
                    'value': pd.DataFrame({'y': np.ones(5)}),
                    'weight': np.arange(5.),
                    'something_else': np.zeros(5)
                },
                sample_weight=None,
                expected_xdict={k: torch.stack([torch.arange(5), -torch.arange(5.)], 1)
                                for k in ['param1', 'param2']},
                expected_ydict={
                    'value': torch.ones(5, 1),
                    'weight': to_2d(torch.arange(5.)),
                    'something_else': torch.zeros(5, 1)
                }
            ),
            Params(
                description="y as dict, redundant sample_weight",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y={'value': pd.DataFrame({'y': np.ones(5)}), 'weight': np.arange(5.)},
                sample_weight=np.arange(5.),
                expected_exception=ValueError
            ),
            Params(
                description="y is None but expected",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=None,
                sample_weight=np.arange(5.),
                expected_exception=ValueError
            ),
            Params(
                description="y is None and not expected",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=None,
                sample_weight=np.arange(5.),
                expect_y=False,
                expected_xdict={k: torch.stack([torch.arange(5), -torch.arange(5.)], 1)
                                for k in ['param1', 'param2']},
                expected_ydict=None
            ),
            Params(
                description="weights are wrong shape",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y={'value': pd.DataFrame({'y': np.ones(5)})},
                sample_weight=np.arange(4.),
                expected_exception=ValueError
            ),
            Params(
                description="y has too many dims",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=np.ones((5, 2, 2)),
                sample_weight=None,
                expected_exception=ValueError
            ),
            Params(
                description="y and X have different lens",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=np.ones(4),
                sample_weight=None,
                expected_exception=ValueError
            ),
            Params(
                description="Pass X as dict but diff lens",
                mm_params=['param1', 'param2'],
                X={k: np.array([-np.arange(5 + int(k == 'param2'))]).T for k in ['param1', 'param2']},
                y=np.arange(5.),
                sample_weight=None,
                expected_exception=ValueError
            ),
            Params(
                description="y is 2d",
                mm_params=['param1', 'param2'],
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=np.ones((5, 2)),
                sample_weight=None,
                expected_xdict={k: torch.stack([torch.arange(5), -torch.arange(5.)], 1)
                                for k in ['param1', 'param2']},
                expected_ydict={'value': torch.ones((5, 2)), 'weight': torch.ones((5, 1))}
            ),
        ]
    )
    @patch('foundry.glm.glm.Glm.expected_model_mat_params_', new_callable=PropertyMock)
    def setup(self, mock_expected_model_mat_params_: PropertyMock, glm: Glm, request: 'FixtureRequest') -> Fixture:
        # this would be set when initializing the module:
        mock_expected_model_mat_params_.return_value = request.param.mm_params

        # call, capturing exceptions if they're expected:
        exception = None
        xdict, ydict = None, None
        try:
            xdict, ydict = glm._build_model_mats(
                X=request.param.X,
                y=request.param.y,
                sample_weight=request.param.sample_weight,
                include_y=request.param.expect_y
            )
        except Exception as e:
            if not request.param.expected_exception:
                raise e
            exception = e

        return self.Fixture(
            params=request.param,
            xdict=xdict,
            ydict=ydict,
            exception=exception
        )

    def test_xdict(self, setup: Fixture):
        if setup.params.expected_xdict is not None:
            assert_dict_of_tensors_equal(setup.params.expected_xdict, setup.xdict)

    def test_ydict(self, setup: Fixture):
        if setup.params.expected_ydict is not None:
            assert_dict_of_tensors_equal(setup.params.expected_ydict, setup.ydict)

    def test_exception(self, setup: Fixture):
        if setup.params.expected_exception:
            assert setup.exception is not None
            with pytest.raises(setup.params.expected_exception):
                raise setup.exception


class TestInit_Module:
    @pytest.fixture()
    def glm(self, family: Family):
        # todo: we're not mocking module_factory, so more of an integration test.
        return Glm(family=family)

    Params = namedtuple('Params', ['description', 'X', 'y', 'expected_result', 'expected_exception'])

    @pytest.mark.parametrize(
        ids=lambda *p: p[0],
        argnames=Params._fields,
        argvalues=[
            Params(
                description="X is dataframe, y is 1d",
                X=pd.DataFrame({'x1': [1, 2, 3], 'x2': [3, 2, 1]}),
                y=[1, 2, 3],
                expected_result=torch.nn.ModuleDict({
                    'param1': torch.nn.Linear(2, 1),
                    'param2': torch.nn.Linear(2, 1)
                }),
                expected_exception=None
            ),
            Params(
                description="X is dict w/only one param; y is 2d",
                X={'param1': pd.DataFrame({'x1': [1, 2, 3], 'x2': [3, 2, 1]})},
                y=np.ones((3, 2)),
                expected_result=torch.nn.ModuleDict({
                    'param1': torch.nn.Linear(2, 2),
                    'param2': NoWeightModule(2)
                }),
                expected_exception=None
            ),
            Params(
                description="X is an empty dict",
                X={},
                y=np.ones((3, 1)),
                expected_result=torch.nn.ModuleDict({
                    'param1': NoWeightModule(),
                    'param2': NoWeightModule()
                }),
                expected_exception=None
            ),
            Params(
                description="X has invalid params",
                X={'param1': pd.DataFrame({'x1': [1, 2, 3], 'x2': [3, 2, 1]}),
                   'parmesian': pd.DataFrame({'x1': [1, 2, 3], 'x2': [3, 2, 1]})},
                y=np.ones(3),
                expected_result=None,
                expected_exception=ValueError
            )
        ]
    )
    def test__init_module(self,
                          glm: Glm,
                          X: ModelMatrix,
                          y: ModelMatrix,
                          expected_result: torch.nn.ModuleDict,
                          expected_exception: Optional[Type[Exception]],
                          description: str):
        """
        Test that _init_module calls the `module_factory`, handling x and y shapes properly.
        """
        with pytest.raises(NotFittedError):
            assert glm.expected_model_mat_params_

        if expected_exception:
            with pytest.raises(expected_exception):
                glm._init_module(X=X, y=y)
        else:
            glm.module_ = glm._init_module(X=X, y=y)
            result_sd_shapes = {k: v.shape for k, v in glm.module_.state_dict().items()}
            expected_result_sd_shapes = {k: v.shape for k, v in expected_result.state_dict().items()}
            assert result_sd_shapes == expected_result_sd_shapes
            if isinstance(X, dict):
                assert glm.expected_model_mat_params_ == list(X.keys())
            else:
                assert glm.expected_model_mat_params_ == glm.family.params


def test_predict():
    """
    Test that predict generates the expected output.
    """
    # TODO
    pass


def test__get_penalty():
    """
    Test that _get_penalty handles (1) no-penalty, (2) single float penalty, (3) penalty-per-param. Also raises on
    incorrect dict-keys.

    Also test that penalty has the expected directionality: bigger weights mean larger penalty.
    """
    pass


def test__get_dist_kwargs():
    pass


# @formatter:off
@pytest.mark.parametrize(
    argnames=['value', 'weight', 'penalty', 'expected_lp'],
    argvalues=[
        (torch.tensor([1., 2.]), None,                   0., torch.tensor([-1.5])),
        (torch.tensor([1., 2.]), torch.tensor([2., 1.]), 0., torch.tensor([-4 / 3.])),
        (torch.tensor([1., 2.]), None,                   9., torch.tensor([-6.])),
        (torch.tensor([1., 2.]), torch.tensor([2., 1.]), 8., torch.tensor([-4.])),
    ]
)
# @formatter:on
def test_log_prob(value: torch.Tensor,
                  weight: Optional[torch.Tensor],
                  penalty: float,
                  expected_lp: torch.Tensor,
                  family: Family):
    """
    Test that family.log_prob and _get_penalty are called, and that increasing the penalty decreases the log prob.
    """
    # init glm:
    glm = Glm(family=family, penalty=penalty)

    # mock family.log_prob
    family.log_prob = Mock(autospec=family.log_prob)
    family.log_prob.side_effect = lambda distribution, value, weight: -value * weight

    # mock penalty:
    glm._get_penalty = Mock(autospec=glm._get_penalty)
    glm._get_penalty.return_value = penalty

    # init module:
    y = {'value': value}
    if weight is not None:
        y['weight'] = weight
    glm._module_ = glm._init_module({}, y=y)

    # call get_log_prob:
    mm_dict, lp_dict = glm._build_model_mats(X={}, y=y, include_y=True)
    lp = glm.get_log_prob(mm_dict, lp_dict)

    # should have called family.log_prob:
    family.log_prob.assert_called()

    # log-prob should match expectation:
    assert_scalars_equal(lp, expected_lp)


def test_fit():
    """
    Integration test (e.g. make_model_mats succeeds b/c expected-params were populated
    """
    pass
