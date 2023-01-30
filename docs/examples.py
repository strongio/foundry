# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Foundry
#
# Foundry is a package for forging interpretable predictive modeling pipelines with a sklearn style-API. It includes:
#
# - A `Glm` class with a [Pytorch](https://pytorch.org) backend. This class is highly extensible, supporting (almost) any distribution in pytorch's [distributions](https://pytorch.org/docs/stable/distributions.html) module.
# - A `preprocessing` module that includes helpful classes like `DataFrameTransformer` and `InteractionFeatures`.
# - An `evaluation` module with tools for interpreting any sklearn-API model via `MarginalEffects`.
#
# You should use Foundry to augment your workflows if any of the following are true:
#
# - You are attempting to model a target that is 'weird': for example, highly skewed data, binomial count-data, censored or truncated data, etc.
# - You need some help battling some annoying aspects of feature-engineering: for example, you want an expressive way of specifying interaction-terms in your model; or perhaps you just want consistent support for getting feature-names despite being stuck on python 3.7.
# - You want to interpret your model: for example, perform statistical inference on its parameters, or understand the direction and functional-form of its predictors.
#
# ## Getting Started
#
# `foundry` can be installed with pip:
#
# ```bash
# pip install git+https://github.com/strongio/foundry.git#egg=foundry
# ```
#
# Let's walk through a quick example:

# %%
# data:
from foundry.data import get_click_data
# preprocessing:
from foundry.preprocessing import DataFrameTransformer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline
# glm:
from foundry.glm import Glm
# evaluation:
from foundry.evaluation import MarginalEffects

# %% [markdown]
# Here's a dataset of click user pageviews and clicks for domain with lots of pages:

# %%
df_train, df_val = get_click_data()
df_train

# %% [markdown]
# We'd like to build a model that let's us predict future click-rates for different pages (page_id), page-attributes (e.g. market), and user-attributes (e.g. platform), and also _learn_ about each of these features -- e.g. perform statistical inference on model-coefficients ("are users with missing user-agent data significantly worse than average?")
#
# Unfortunately, these data don't fit nicely into the typical regression/classification divide: each observations captures _counts_ of clicks and _counts_ of pageviews. Our target is the click-rate (clicks/views) and our sample-weight is the pageviews.
#
# One workaround would be to expand our dataset so that each row indicates `is_click` (True/False) -- then we could use a standard classification algorithm:

# %%
df_train_expanded, df_val_expanded = get_click_data(expanded=True)
df_train_expanded

# %% [markdown]
# But this is hugely inefficient: our dataset of ~400K explodes to almost 8MM. 
#     
# Within `foundry`, we have the `Glm`, which supports binomial data directly:

# %%
Glm('binomial', penalty=10_000)

# %% [markdown]
# Let's set up a sklearn model pipeline using this Glm. We'll use `foundry`'s `DataFrameTransformer` to support passing feature-names to the Glm (newer versions of sklearn support this via the `set_output()` API).

# %%
preproc = DataFrameTransformer([
    (
        'one_hot', 
        make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), 
        ['attributed_source', 'user_agent_platform', 'page_id', 'page_market']
    )
    ,
    (
        'power', 
        PowerTransformer(),
        ['page_feat1', 'page_feat2', 'page_feat3']
    )
])

# %%
glm = make_pipeline(
    preproc, 
    Glm('binomial', penalty=1_000)
).fit(
    X=df_train,
    y={
        'value' : df_train['num_clicks'],
        'total_count' : df_train['num_views']
    },
)

# %% [markdown]
# By default, the `Glm` will estimate not just the parameters of our model, but also the uncertainty associated with them. We can access a dataframe of these with the `coef_dataframe_` attribute:

# %%
df_coefs = glm[-1].coef_dataframe_
df_coefs

# %% [markdown]
# Using this, it's easy to plot our model-coefficients:

# %%
df_coefs[['param', 'trans', 'term']] = df_coefs['name'].str.split('__', n=3, expand=True)

df_coefs[df_coefs['name'].str.contains('page_feat')].plot('term', 'estimate', kind='bar', yerr='se')
df_coefs[df_coefs['name'].str.contains('user_agent_platform')].plot('term', 'estimate', kind='bar', yerr='se')

# %% [markdown]
# Model-coefficients are limited because they only give us a single number, and for non-linear models (like our binomial GLM) this doesn't tell the whole story. For example, how could we translate the importance of `page_feat3` into understanable terms? This only gets more difficult if our model includes interaction-terms.
#
# To aid in this, there is `MarginalEffects`, a tool for plotting our model-predictions as a function of each predictor:

# %%
glm_me = MarginalEffects(glm)
glm_me.fit(
    X=df_val_expanded, 
    y=df_val_expanded['is_click'],
    vary_features=['page_feat3']
).plot()

# %% [markdown]
# Here we see that how this predictor's impact on click-rates varies due to floor effects. 
#
# As a bonus, we ploted the actual values alongside the predictions, and we can see potential room for improvement in our model: it looks like very high values of this predictor have especially high click-rates, so an extra step in feature-engineering that captures this discontinuity may be warranted.

# %% [markdown]
# ## Examples

# %% [markdown]
# ### Example 1: Clicks and Pageviews
#
# - compare timing and performance to sklearn and statsmodels
# - fourier-features
# - interactions?

# %%
# TODO

# %% [markdown]
# ### Example 2: Churn
#
# - Survival Analysis

# %%
# TODO

# %% [markdown]
# ### Example 3: Mashable
#
# - heteroskedastic?

# %%
import numpy as np
# data:
from foundry.data import get_online_news_dataset
# preprocessing:
from foundry.preprocessing import DataFrameTransformer, SimpleImputer, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import make_pipeline
# glm:
from foundry.glm import Glm

# %%
df_news = get_online_news_dataset()
df_news

# %%
raw_feats = [
    'n_tokens_title', 
    'num_keywords', 
    'LDA_00',
    'LDA_01', 
    'LDA_02', 
    'LDA_03', 
    'LDA_04',
    'global_subjectivity', 
    'global_sentiment_polarity', 
    'global_rate_positive_words', 
    'global_rate_negative_words',
    'rate_positive_words',
    'rate_negative_words', 
    'avg_positive_polarity', 
    'min_positive_polarity',
    'max_positive_polarity', 
    'avg_negative_polarity',
    'min_negative_polarity', 
    'max_negative_polarity', 
    'title_subjectivity',
    'title_sentiment_polarity', 
    'abs_title_subjectivity',
    'abs_title_sentiment_polarity'
]

power_feats = [
    'num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos', 
]

power_zif_feats = [
    'n_tokens_content', 
    'n_unique_tokens',
    'n_non_stop_unique_tokens',
    'average_token_length',
    'kw_max_min',
    'kw_avg_min',
    'kw_min_max',
    'kw_avg_max',
    'kw_min_avg',
    'kw_max_avg',
    'kw_avg_avg', 
    'self_reference_min_shares',
    'self_reference_max_shares',
    'self_reference_avg_shares',
]

binary_feats = [
 'data_channel_is_bus',
 'data_channel_is_entertainment',
 'data_channel_is_lifestyle',
 'data_channel_is_socmed',
 'data_channel_is_tech',
 'data_channel_is_world',
 'weekday_is_friday',
 'weekday_is_monday',
 'weekday_is_saturday',
 'weekday_is_sunday',
 'weekday_is_thursday',
 'weekday_is_tuesday',
 'weekday_is_wednesday',
 'is_weekend'
]


# %%
def na_if_nonpos(values: np.ndarray):
    values = values.copy()
    values[values<=0] = np.nan
    return values

def na_if_extreme(values: np.ndarray, thresh: float = 6.):
    values = values.copy()
    values[np.abs(values) >= thresh] = np.nan
    return values


# %%
preproc = DataFrameTransformer([
    (
        'asis',
        'passthrough',
        binary_feats
    ),
    (
        'power_zif',
        make_pipeline(
            FunctionTransformer(na_if_nonpos),
            PowerTransformer(method='box-cox'),
            SimpleImputer(add_indicator=True),
            FunctionTransformer(na_if_extreme),
            SimpleImputer()
        ),
        power_zif_feats
    ),
    (
        'power',
        make_pipeline(
            PowerTransformer(method='yeo-johnson'),
            FunctionTransformer(na_if_extreme),
            SimpleImputer()
        ),
        power_feats
    ),
    (
        'scale',
        StandardScaler(),
        raw_feats
    )
])

# %%
df_news['shares2'] = df_news['shares'] / 1000

# %%
glm = make_pipeline(
    preproc, 
    Glm(
        'lognormal',
        penalty=100, 
        col_mapping=fam.params
    )
)
glm.fit(X=df_news, y=df_news['shares2'])
