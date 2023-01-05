# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd

import foundry

from data.uci import get_wine_dataset
X, y = get_wine_dataset()


from foundry.glm import Glm
from foundry.util import to_2d

X

y

Glm("categorical", ).fit(X, to_2d(y))

y

type(y)


