from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Type
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd
import pytest
import torch

from sklearn.preprocessing import LabelEncoder
from torch.distributions import constraints, identity_transform

from foundry.glm.family import Family
from foundry.glm.glm import ModelMatrix, Glm
from foundry.glm.util import NoWeightModule
from foundry.util import to_2d, ToSliceDict
from tests.conftest import assert_dict_of_tensors_equal, assert_scalars_equal, assert_tensors_equal


def test_categorical_integration():
    y = pd.Series([0] * 3 + [1] * 2 + [2] * 5, name='cat').to_frame()
    X = pd.DataFrame(index=y.index)
    glm = Glm(family='categorical')
    glm.fit(X=X, y=y, max_loss=float('inf'), verbose=False)
    assert set(glm.predict(X=X)) == {2}
    np.testing.assert_allclose(glm.predict_proba(X=X).mean(0), np.asarray([.30, .20, .50]), atol=.001)


class _FakeDist:
    arg_constraints = {
        'param1': constraints.real,
        'param2': constraints.real
    }

    def __init__(self, param1: torch.Tensor, param2: torch.Tensor):
        self.param1 = param1
        self.param2 = param2


@pytest.fixture()
def fake_family() -> Family:
    return Family(
        distribution_cls=_FakeDist,
        params_and_links={k: identity_transform for k in _FakeDist.arg_constraints}
    )


class TestClassifierPredict:
    @pytest.mark.parametrize(
        argnames=['family_nm', 'expected'],
        argvalues=[
            ('binomial', True),
            ('gaussian', False)
        ]
    )
    def test_method_added_to_instance_dynamically(self, family_nm: str, expected: bool):
        """
        method should only exist in instances if their distribution has 'probs'
        """
        family = Glm._init_family(Glm, family_nm)
        glm = Glm(family=family)
        glm._fit = Mock(glm._fit, autospec=True)
        glm.fit(X=None, y=None)
        assert hasattr(family.distribution_cls, 'probs') == expected
        assert hasattr(glm, 'predict_proba') == expected

    def test_method(self):
        family = Family(
            distribution_cls=torch.distributions.Binomial,
            params_and_links={'probs': torch.distributions.transforms.identity_transform}
        )
        glm = Glm(family=family)
        # mock label-encoder:
        glm.label_encoder_ = create_autospec(LabelEncoder, instance=True)
        glm.label_encoder_.classes_ = [0, 1]
        # mock _fit, so that fit() just calls init_family etc.:
        glm._fit = Mock(glm._fit, autospec=True)
        # mock build model-mats, we won't be using inputs:
        glm._build_model_mats = Mock(glm._build_model_mats, autospec=True)
        glm._build_model_mats.return_value = {}, None, None
        # mock get_dist_kwargs, we want to control the output:
        glm._get_family_kwargs = Mock(glm._get_family_kwargs, autospec=True)
        glm._get_family_kwargs.return_value = {
            'probs': to_2d(torch.tensor([.1, .9, .1, .9])),
            'total_count': to_2d(torch.tensor([1, 1, 2, 2]))
        }
        #
        glm.fit(X=None, y=None)
        preds = glm.predict_proba(X=None)
        assert len(preds.shape) == 2
        assert preds.shape[1] == 2
        # first col is p(y=0):
        np.testing.assert_array_equal(
            preds[:, [0]],
            1 - glm._get_family_kwargs.return_value['probs']
        )
        # second col is p(y=1):
        np.testing.assert_array_equal(
            preds[:, [1]],
            glm._get_family_kwargs.return_value['probs']
        )

        # make sure default behavior of predict is to choose classes with highest probability
        glm.predict(X=None)
        glm.label_encoder_.inverse_transform.assert_called_once()
        np.testing.assert_array_equal(
            glm.label_encoder_.inverse_transform.call_args_list[0][0],
            np.array([[0, 1, 0, 1]])
        )

        # make sure predict with `mean` works as expected:
        np.testing.assert_array_equal(
            glm.predict(X=None, type='mean'),
            glm._get_family_kwargs.return_value['probs'] * glm._get_family_kwargs.return_value['total_count']
        )


class TestBuildModelMats:
    # TODO: should instead test `_get_xdict` and `_get_ydict` separately

    @pytest.fixture()
    def glm(self, fake_family: Family):
        glm = Glm(family=fake_family)

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
                expected_exception=RuntimeError
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
            Params(
                description="Using 'remainder',",
                mm_params={'param1': ['x1'], 'param2': 'remainder'},
                X=pd.DataFrame({'x1': np.arange(5), 'x2': -np.arange(5)}),
                y=np.ones(5),
                sample_weight=None,
                expected_xdict={
                    'param1': to_2d(torch.arange(5.)),
                    'param2': -to_2d(torch.arange(5.))
                },
                expected_ydict={'value': torch.ones(5, 1), 'weight': torch.ones((5, 1))}
            ),
        ]
    )
    def setup(self, glm: Glm, request: 'FixtureRequest') -> Fixture:

        # call, capturing exceptions if they're expected:
        exception = None
        xdict, ydict = None, None
        try:
            # this would be set in _init_module
            glm.to_slice_dict_ = ToSliceDict(request.param.mm_params).fit(request.param.X)
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
    # TODO: this test needs to be split into multiple tests and partially rewritten:
    #   - half the test is testing that `to_slice_dict_` was initialized properly
    #   - testing this^ by checking if it behaves properly; instead should test that its init method got correct args
    #   - not mocking `module_factory`
    @pytest.fixture()
    def glm(self, fake_family: Family):
        return Glm(family=fake_family)

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
                    'param2': NoWeightModule(1)
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
                expected_result=torch.nn.ModuleDict({
                    'param1': torch.nn.Linear(2, 1),
                    'param2': NoWeightModule(1)
                }),
                # no exception for invalid params in init_module, these come later, since we want to allow pass-thru
                # arguments to the distribution (e.g. total_count) for binomial)
                expected_exception=None
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

        if expected_exception:
            with pytest.raises(expected_exception):
                glm._init_module(X=X, y=y)
        else:
            glm._init_module(X=X, y=y)
            result_sd_shapes = {k: v.shape for k, v in glm.module_.state_dict().items()}
            expected_result_sd_shapes = {k: v.shape for k, v in expected_result.state_dict().items()}
            assert result_sd_shapes == expected_result_sd_shapes
            if isinstance(X, dict):
                assert list(glm.to_slice_dict_.mapping) == list(X.keys())
            else:
                assert list(glm.to_slice_dict_.mapping) == [glm.family.params[0]]


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
                  fake_family: Family):
    """
    Test that family.log_prob and _get_penalty are called, and that increasing the penalty decreases the log prob.
    """
    # init glm:
    glm = Glm(family=fake_family, penalty=penalty, col_mapping=[])

    # mock family.log_prob
    fake_family.log_prob = Mock(autospec=fake_family.log_prob)
    fake_family.log_prob.side_effect = lambda distribution, value, weight: -value * weight

    # mock penalty:
    glm._get_penalty = Mock(autospec=glm._get_penalty)
    glm._get_penalty.return_value = penalty

    # init module:
    y = {'value': value}
    if weight is not None:
        y['weight'] = weight
    glm._init_module({}, y=y)

    # call get_log_prob:
    mm_dict, lp_dict = glm._build_model_mats(X={}, y=y, include_y=True)
    lp = glm.get_log_prob(mm_dict, lp_dict)

    # should have called family.log_prob:
    fake_family.log_prob.assert_called()

    # log-prob should match expectation:
    assert_scalars_equal(lp, expected_lp)


def test_fit():
    """
    Integration test (e.g. make_model_mats succeeds b/c expected-params were populated
    """
    pass
