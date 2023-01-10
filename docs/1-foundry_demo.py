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

# +
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import foundry
from foundry.glm import Glm
from foundry.util import to_2d

from data.uci import get_wine_dataset, get_online_news_dataset

# -

X, y = get_online_news_dataset()


X_s = StandardScaler().fit(X).transform(X)

X_s.shape

Glm("gaussian", ).fit(X_s, y.values / 10, max_iter=400)

theta = np.array([[-0.5], [0.5], [3]])
X = np.hstack([np.random.randn(100, 2), np.ones((100, 1))])

eps = np.random.randn(100, 1)

y = X @ theta + eps

my_glm = Glm("gaussian", ).fit(X[:, :-1], y)

my_glm._coef_mvnorm_.covariance_matrix


