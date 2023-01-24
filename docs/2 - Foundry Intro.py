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
from sklearn.pipeline import make_pipeline, Pipeline
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

categorical_features = ("workclass", "education", "married", "occupation", "relationship", "race", "sex", "country")
# -

# Cleaning
X.loc[X["country"] == "Holand-Netherlands", "country"] = '?' # Only one example'
rows_with_missing_values = (X == "?").any(axis=1)
X, y = X.loc[~rows_with_missing_values, :], y.loc[~rows_with_missing_values]
X["education"] = pd.Categorical(
    X["education"], 
    categories=[
        'Preschool',
        '1st-4th',
        '5th-6th',
        '7th-8th',
        '9th',
        '10th',
        '11th',
        '12th',
        'HS-grad',
        'Prof-school',
        'Assoc-voc',
        'Assoc-acdm',
        'Some-college',
        'Bachelors',
        'Masters',
        'Doctorate',
    ], ordered=True)
X = X.astype({feature: "category" for feature in categorical_features})
y = y.astype("category")

# # Train a model with feature engineering 

classifier: Pipeline = make_pipeline(
    # Newer versions of sklearn, or don't care for pd.Dataframe over np.ndarray?
    # Then you can use sklearn.compose.ColumnTransformer
    DataFrameTransformer(
        transformers=[
            (
                "onehot", 
                OneHotEncoder(sparse=False), 
                categorical_features
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
)

# +
# results = cross_validate(
#     final_pipeline, 
#     X, y, cv=StratifiedKFold(shuffle=True, random_state=100, n_splits=10), 
#     scoring='accuracy', 
#     fit_params={"glm__verbose": False, "glm__estimate_laplace_coefs": False,}
# )
# print(results)
# print(f"average accuracy: {results['test_score'].mean()}")
# -

trained_model = classifier.fit(X, y, glm__estimate_laplace_coefs=False)

# ## Evaluation - Marginal Effects

from foundry.evaluation import MarginalEffects

me = MarginalEffects(pipeline=trained_model, )
me(
    X, 
    y.map(lambda item: 1 if item == ">50K" else 0).astype(float), 
    vary_features = ["sex"], 
    groupby_features=["occupation"], 
    marginalize_aggfun=None, 
    y_aggfun='mean'
)
me.plot() + theme(axis_text_x=element_text(rotation=90, hjust=1))

me = MarginalEffects(pipeline=trained_model, )
me(
    X, 
    y.map(lambda item: 1 if item == ">50K" else 0).astype(float), 
    vary_features = ["education"], 
    groupby_features=["occupation"], 
    marginalize_aggfun=None, 
    y_aggfun='mean'
)
me.plot(include_actuals=False) + theme(axis_text_x=element_text(rotation=90, hjust=1)) + ylab("P(Income > 50k)")

me = MarginalEffects(pipeline=trained_model, )
me(
    X, 
    y.map(lambda item: 1 if item == ">50K" else 0).astype(float), 
    vary_features = ["education_years"], 
    groupby_features=["occupation"], 
    marginalize_aggfun=None, 
    y_aggfun='mean'
)
me.plot(include_actuals=True) + theme(axis_text_x=element_text(rotation=90, hjust=1))


