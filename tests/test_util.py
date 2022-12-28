import pandas as pd
import pytest
import torch
from pandas.core.arrays import SparseArray

from foundry.util import to_tensor, to_2d
from tests.conftest import assert_tensors_equal


@pytest.mark.parametrize(
    argnames=['x', 'kwargs', 'expected'],
    argvalues=[
        pytest.param(
            pd.DataFrame({'a': SparseArray([0, 0, 0, 1])}),
            {'dtype': torch.get_default_dtype()},
            to_2d(torch.tensor([0., 0., 0., 1.])).to_sparse(),
            id='sparse'
        )
    ]
)
def test_to_tensor(x: object, kwargs: dict, expected: torch.Tensor):
    assert_tensors_equal(to_tensor(x, **kwargs), expected)
