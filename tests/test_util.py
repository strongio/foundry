import pickle

import pandas as pd
import pytest
import torch
from pandas.core.arrays import SparseArray

from foundry.util import to_tensor, to_2d, SliceDict
from tests.conftest import assert_tensors_equal


def test_slice_dict_serialize():
    orig = SliceDict(**{'test': torch.as_tensor([2, 3])})
    unserialized = pickle.loads(pickle.dumps(orig))
    assert set(orig) == set(unserialized)
    assert_tensors_equal(orig['test'], unserialized['test'])


@pytest.mark.parametrize(
    argnames=['x', 'sparse_threshold', 'expected'],
    argvalues=[
        pytest.param(
            pd.DataFrame({'a': SparseArray([0, 0, 0, 1])}),
            0.30,
            to_2d(torch.tensor([0., 0., 0., 1.])).to_sparse(),
            id='sparse enough'
        ),
        pytest.param(
            pd.DataFrame({'a': SparseArray([0, 0, 1, 1])}),
            0.30,
            to_2d(torch.tensor([0., 0., 1., 1.])),
            id='sparse, but not enough'
        ),
        pytest.param(
            pd.DataFrame({'a': SparseArray([0, 0, 0, 1]), 'b': [1.5, 2., 3., 4.]}),
            .90,
            torch.tensor([[0., 0., 0., 1.], [1.5, 2., 3., 4.]]).T.to_sparse(),
            id='mix of sparse and dense, sparse enough'
        ),
        pytest.param(
            pd.DataFrame({'a': SparseArray([0, 0, 0, 1]), 'b': [1.5, 2., 3., 4.]}),
            .10,
            torch.tensor([[0., 0., 0., 1.], [1.5, 2., 3., 4.]]).T,
            id='mix of sparse and dense, not sparse enough'
        ),
        pytest.param(
            pd.DataFrame({'a': [0, 0, 0, 1], 'b': [1.5, 2., 3., 4.]}),
            .99,
            torch.tensor([[0., 0., 0., 1.], [1.5, 2., 3., 4.]]).T,
            id='just dense'
        ),
        pytest.param(
            pd.DataFrame(index=range(4)),
            .99,
            torch.empty((4, 0)),
            id='empty'
        ),
    ]
)
def test_to_tensor(x: pd.DataFrame, sparse_threshold: float, expected: torch.Tensor):
    x_copy = x.copy(deep=True)
    assert_tensors_equal(
        to_tensor(x, dtype=torch.get_default_dtype(), sparse_threshold=sparse_threshold),
        expected
    )
    # make sure not modified in-place:
    pd.testing.assert_frame_equal(x_copy, x)
