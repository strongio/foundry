import pdb
from dataclasses import dataclass
from typing import Optional, Union, Dict, Sequence, Type
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions.transforms import identity_transform

from foundry.glm.family import Family
from foundry.glm.glm import ModelMatrix, Glm
from foundry.util import to_2d
from tests.util import assert_dict_of_tensors_equal


class _FakeDist:
    arg_constraints = {
        'param1': identity_transform,
        'param2': identity_transform
    }


@pytest.fixture()
def family() -> Family:
    return Family(
        distribution_cls=_FakeDist(),
        params_and_links={k: identity_transform for k in _FakeDist.arg_constraints}
    )


class TestBuildModelMats:
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
                expected_ydict={}
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
        ]
    )
    def setup(self, family: Family, request: 'FixtureRequest') -> Fixture:
        glm = Glm(family=family)

        # module is used to determine dtype/device:
        glm.module_ = Mock(spec_set=['dtype', 'device'])
        glm.module_.dtype = torch.get_default_dtype()
        glm.module_.device = torch.device('cpu')

        # this would be set when initializing the module:
        glm.expected_model_mat_params_ = request.param.mm_params

        # call, capturing exceptions if they're expected:
        exception = None
        xdict, ydict = None, None
        try:
            xdict, ydict = glm._build_model_mats(
                X=request.param.X,
                y=request.param.y,
                sample_weight=request.param.sample_weight,
                expect_y=request.param.expect_y
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

    # def test__init_module(self, glm: Glm, X: ModelMatrix, y: ModelMatrix) -> torch.nn.ModuleDict:
    #     pass

    # def test__init_optimizer(self, glm: Glm):
    #     pass
    #
    # def test__init_family(self, family: Union[Family, str]):
    #     pass
