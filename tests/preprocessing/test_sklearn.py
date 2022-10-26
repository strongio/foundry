import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.pipeline import make_pipeline

from foundry.preprocessing import make_drop_transformer, make_column_selector


@pytest.fixture()
def small_dataframe():
    return pd.DataFrame({
        "A": [0., 1., 2., 3.],
        "B": [0.5, -0.5, 0.5, -0.5],
        "C": [-1., -2., -3., -4.],
    })


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
                dict(),
                pd.DataFrame({"A": [0., 1., 2., 3., ], "B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4., ]})
        ),
        (
                {"names": "A"},
                pd.DataFrame({"B": [0.5, -0.5, 0.5, -0.5], "C": [-1., -2., -3., -4.]})
        ),
        (
                {"names": ["A", "B"]},
                pd.DataFrame({"C": [-1., -2., -3., -4., ]})
        ),
        (
                {"pattern": "[AB]"},
                pd.DataFrame({"C": [-1., -2., -3., -4., ]})
        ),
        pytest.param(
            {"names": "A", "pattern": ".*"},
            None,
            marks=pytest.mark.xfail(raises=ValueError)
        ),
    ],
    ids=[
        "no_drop",
        "drop_by_name",
        "drop_by_names",
        "drop_by_regex",
        "err_too_many",
    ]
)
def test_make_drop_transformer(small_dataframe, kwargs, expected):
    my_pipeline = make_pipeline(
        make_drop_transformer(**kwargs)
    )

    test = my_pipeline.fit(small_dataframe).transform(small_dataframe)

    assert_frame_equal(expected, test)


def test_make_column_selector():
    small_dataframe = pd.DataFrame({
        "A": [0., 1., 2., 3., ],
        "B": [0.5, -0.5, 0.5, -0.5],
        "C": [-1., -2., -3., -4., ]
    })
    column_selector = make_column_selector("[AB]")

    assert column_selector(small_dataframe) == ["A", "B"]
    assert "[AB]" in repr(column_selector)
