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
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

import foundry
from foundry.glm import Glm
from foundry.util import to_tensor

from data.uci import get_online_news_dataset, get_wine_dataset

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
# -

# # Introduction
#
# `foundry.glm` is a package that seeks to learn the parameters of various parametric distributions. It uses gradient descent to minimize the negative log-likehood of the observed data, as a linear function (with activation) of some covariates.
#
# minimize the negative log-likehood <-->? maximimum likelihood estimator
#
# In this notebook, we demonstrate the functionality of the `foundry.glm`. We start off with a couple of basic examples with synthetic data, followed by a quick experiment with the online news dataset (https://archive.ics.uci.edu/ml/datasets/online+news+popularity).
#
# The synthetic data demonstrations we show are the following:
# 1. Linear Regression
# 2. Exponential Regression
# 3. Logistic Regression
# 4. Multiclass Classification
#

# # Synthetic Data Demonstrations

# ## Linear Regression
#
# Linear regression follows the model that $y \sim N(\vec{x}\theta, \sigma^2)$, where $N$ is the gaussian distribution. 
#
# In the following, we show how to do that with the `Glm` class.

# +
# Define the model
linear_glm = Glm("gaussian")
# More options (will be explained later):
# linear_glm = Glm("gaussian", penalty=1E-5)
# linear_glm = Glm("gaussian", penalty=1E-5, col_mapping={"loc": ["column1", "column2", "column3"]} 

# Define some data
bias_true = 3.
X, y_true, coefs_true = make_regression(
    n_samples=10**5, # X.shape[0]
    n_features=30, # X.shape[1]
    n_informative=10, # 20 features don't predict the outcome
    bias=bias_true, # E[y | X = 0]
    effective_rank=15, # Increase correlation of features, as if there were approx 15 independent ones. 
    noise=0.1, # defines variance
    coef=True, # give the real model coefficients
    random_state=97,
)

# fit the model
linear_glm.fit(X, y_true)
# More options:
# linear_glm.fit(X, y_true, sample_weight=torch.rand(y_true.shape[0]))
# linear_glm.fit(X, y_true, max_iter = 10**5)

# evaluate the model
print(f"Linear model R2: {r2_score(y_true, linear_glm.predict(X))}")
print(f"Reducible error R2: {r2_score(y_true, X @ coefs_true + bias_true)}")

# what is the actual reducible error?
print(f"Linear model log-prob: {linear_glm.score(X, y_true)}")
print(f"Linear model log-prob: {linear_glm.score(X, X @ coefs_true + bias_true)}")

# -

# ### Modelling heteroscedastic errors
#
# Above we can see how the Glm class can do linear regression. Ordinary least squares modelling typically assumes that the output variance is homoscedastic - i.e. that the variance $\sigma^2$ is constant, and that the error term $\epsilon = y - X\theta \sim N(0, \sigma^2)$. Sometimes that may not be the case, and we want to model the error as well. 
#
# We demonstrate this by adding some more noise to the outputs. This noise has zero mean, but variance that is a function of X. 

# +
# The error depends on the last 5 features of X
error_coefs_true = torch.cat([
    torch.zeros(coefs_true.shape[0] - 5, 1, dtype=torch.float), 
    torch.rand(5, 1, dtype=torch.float) * 1000
])

# Sample these additional errors
additional_error: torch.Tensor = torch.cat([
    torch.distributions
    .Normal(
        loc=torch.zeros(len(X_slice)), 
        scale=torch.exp(to_tensor(X_slice, dtype=torch.float) @ error_coefs_true - 6).flatten()
    )
    .sample()
    for X_slice in np.array_split(to_tensor(X), 1000)
])

# Add back to y
y_add_error = y_true + additional_error.numpy()


# +
# Fit the model
linear_glm_modified = Glm(
    "gaussian", 
    penalty=1E-5, 
    col_mapping = {"loc": slice(None, None, 1), "scale": slice(-10, None, 1)}
)
linear_glm_modified.fit(X, y_add_error, max_iter=10**8, stopping=(.00001, ))

# Evaluate the model

# We expect that R2 is the same between the two models, since we added noise with zero mean 
print(f"Linear model R2: {r2_score(y_add_error, linear_glm.fit(X, y_add_error).predict(X).flatten())}")
print(f"Linear (heteroscedastic) model R2: {r2_score(y_add_error, linear_glm_modified.predict(X).flatten())}")

# But the log likelihood should have increased, since we capture the difference in variance
print(f"Linear model log-prob: {linear_glm.score(X, y_add_error)}")
print(f"Linear (heteroscedastic) model log-prob: {linear_glm_modified.score(X, y_add_error)}")
# -

# ### Model introspection
#
# Above we can see how the Glm class can do linear regression. Ordinary least squares modelling typically assumes that the output variance is homoscedastic - i.e. that the variance $\sigma^2$ is constant, and that the error term $\epsilon = y - X\theta \sim N(0, \sigma^2)$. Sometimes that may not be the case, and we want to model the error as well. 
#
# We demonstrate this by adding some more noise to the outputs. This noise has zero mean, but variance that is a function of X. 

print(linear_glm_modified.module_._modules["loc"]._parameters)
print(linear_glm_modified.module_._modules["scale"]._parameters)

print("loc coefficients: ")
print(to_tensor(coefs_true))
print("scale coefficients: ")
print(error_coefs_true[-10:].flatten())

# ## Exponential Regression
#
# Here's another GLM - modeling the time between independent events. For example, one might be interested in the time between visitors to a website, as a function of \[number of ads, content placement, font_size, ...\]  

# +
# Sample new target data from an exponential distribution
y_exp_true = torch.distributions.Exponential(rate=torch.exp(to_tensor(y_true))).sample()

# Fit a model
exp_model = Glm("exponential").fit(X, y_exp_true)

# Verify that the learned coefficients are similar to the existing ones
print("\nLearned parameters:")
print(exp_model.module_._modules["rate"]._parameters)

print("\nTrue parameters:")
print(coefs_true, bias_true)
# -

# # Binary Classification
#
# This is the standard method for predicting a binary outcome - e.g. \{cancer; no cancer\}. 

# +
X, y_true = make_classification(
    n_samples=10**5,
    n_features=30,
    n_informative=10,
    n_redundant=15,
    random_state=97,
)

binary_model = Glm("bernoulli").fit(X, y_true)

print(f"Accuracy: {accuracy_score(y_true, binary_model.predict(X))}")
# -

# # Multiclass Classification

# +
X, y_true = make_classification(
    n_samples=10**5,
    n_features=30,
    n_informative=10,
    n_redundant=15,
    n_classes=5,
    random_state=97,
)

classification_model = Glm("categorical", penalty=10.).fit(X, y_true)

print(f"Accuracy: {accuracy_score(y_true, classification_model.predict(X))}")
# -

# # Real Data 
#
# ## Online News Dataset
#
# We're going to do a quick regression task - modelling the number of 'shares' an article gets, based on a number of features collected about the article.  

# +
X, y_true = get_online_news_dataset()

pipeline = make_pipeline(
    StandardScaler(),
    Glm("gaussian", penalty=0.01)
)

cross_validate(
    estimator=pipeline,
    X=X,
    y=y_true,
    scoring='r2',
    cv=KFold(n_splits=5, shuffle=True),
    fit_params={"glm__estimate_laplace_coefs": False, "glm__verbose": False},
    verbose=1
)
# -

# As we can see above - the model isn't sophisticated enough to adequately predict the number of share a particular article gets. We will tackle this in the next notebook. 

# ## Wine Dataset
#
# Now for a quick classification task - modeling which of three (3) cultivars that a wine is, based on a number of measures of chemical composition of the wine. Covariates includes \[alcohol, malic acid, ash, alcalinity of ash, ...\]

# +
X, y_true = get_wine_dataset()


pipeline = make_pipeline(
    StandardScaler(),
    Glm("categorical", penalty=0.01)
)

cross_validate(
    estimator=pipeline,
    X=X,
    y=y_true,
    scoring='accuracy',
    cv=KFold(n_splits=5, shuffle=True),
    fit_params={"glm__estimate_laplace_coefs": False, "glm__verbose": False,},
    return_train_score=True,
)
# -

# Wonderful! This dataset is simpmle enough, and the features are sufficiently predictive of the type (cultivar) of wine. 

# # Helpful Tips and Tricks
# 1. I get the error "Second order estimation of parameter distribution failed. Constructing approximate covariance." What can I do? 
#
#     a. Add a penalty. `model = Glm("gaussian", penalty=XYZ)`
#     
#     b. Check for redundant features and remove them.
#     
#     c. Check that the model actually converged (`model.fit(X, y, stopping=FOO, max_iter=BAR, max_loss=BAZ`)
#     
#     d. Similar to c., check that the features have been scaled properly!
#   
# 2. Feature scaling matters! Ensure that the features have been scaled properly. `Glm` relies on gradient descent, and strongly favors proper feature scaling.

# +
X, y_true = make_classification(
    n_samples=10**5,
    n_features=30,
    n_informative=10,
    n_redundant=15,
    random_state=97,
)

scaler = StandardScaler().fit(X)

# Fit a model with poorly scaled features
# This one takes 29 epochs
binary_model_slow = Glm("categorical").fit(scaler.transform(X)/1000, y_true, estimate_laplace_coefs=False)
# This one takes 4 epochs
binary_model_fast = Glm("categorical").fit(scaler.transform(X), y_true, estimate_laplace_coefs=False)

print(f"Accuracy: {accuracy_score(y_true, binary_model_slow.predict(X))}")
print(f"Accuracy: {accuracy_score(y_true, binary_model_fast.predict(X))}")
