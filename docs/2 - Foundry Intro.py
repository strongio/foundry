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

# # Introduction
#
# In this notebook, we do a comprehensive introduction into the functionality of `foundry`. Intended as a supplement to `scikit-learn`, this is a demonstration of how the current packages in foundry: (1) `foundry.preprocessing`, (2) `foundry.glm`, and (3) `foundry.evaluation` support work in machine learning, predictive analytics, and explainable models. 

# +
import numpy as np
import pandas as pd
from plotnine import *
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn import svm

from foundry.glm import Glm
from foundry.preprocessing import (
    ColumnDropper,
    DataFrameTransformer, 
    InteractionFeatures, 
    as_transformer, 
    identity, 
    make_column_selector,
)


from data.uci import get_online_news_dataset, get_census_dataset

# +
X, y = get_census_dataset()

# Cleaning
X.loc[X["country"] == "Holand-Netherlands", "country"] = '?' # Only one example
rows_with_missing_values = (X == "?").any(axis=1)
X, y = X.loc[~rows_with_missing_values, :], y.loc[~rows_with_missing_values]
# -

# # Final Model

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# +
final_pipeline = make_pipeline(
    # Newer versions of sklearn, or don't care for pd.Dataframe over np.ndarray?
    # Then you can use sklearn.compose.ColumnTransformer
    DataFrameTransformer(
        transformers=[
            (
                "onehot", 
                OneHotEncoder(sparse=False), 
                ("workclass", "education", "married", "occupation", "relationship", "race", "sex", "country")
            ),
            (
                "log1p",
                as_transformer(np.log1p),
                make_column_selector("capital")
            ),
            (
                "pass",
                as_transformer(identity),
                ("age", "education_years", "capitalgain", "capitalloss", "hoursperweek")
            )
        ]
    ),
    as_transformer(lambda df: df.astype('float64')),
    InteractionFeatures(
        [
            (
                make_column_selector("pass"), 
                make_column_selector("onehot")
            )
        ]
    ),
    StandardScaler(),
    Glm("categorical")
#     DecisionTreeClassifier()
#     LogisticRegression()
)




# +
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.10, random_state=97, shuffle=True,
# )

# final_pipeline.fit(
#     X_train, y_train, 
#     glm__estimate_laplace_coefs=False
# )
# accuracy_score(y_test, final_pipeline.predict(X_test))
# -

results = cross_validate(
    final_pipeline, 
    X, y, cv=StratifiedKFold(shuffle=True, random_state=100, n_splits=10), 
    scoring='accuracy', 
    fit_params={"glm__verbose": False, "glm__estimate_laplace_coefs": False,}
)
print(results)
print(f"average accuracy: {results['test_score'].mean()}")

trained_model = final_pipeline.fit(X, y, glm__estimate_laplace_coefs=False, glm__verbose=False)


