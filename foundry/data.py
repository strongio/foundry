from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.datasets import make_regression
from foundry.glm.glm import family_from_string

import os
import zipfile
import pandas as pd
from io import BytesIO


def get_online_news_dataset(local_path: str = "~/foundry_cache", download_if_missing: bool = True) -> pd.DataFrame:
    local_path = os.path.expanduser(local_path)
    os.makedirs(local_path, exist_ok=True)

    local_path = os.path.join(local_path, "OnlineNewsPopularity.csv")
    try:
        dataset = pd.read_csv(local_path)
    except FileNotFoundError as e:
        if not download_if_missing:
            raise e
        print(str(e))
        import requests
        dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
        response = requests.get(dataset_url)
        zip_archive = zipfile.ZipFile(BytesIO(response.content))
        with zip_archive.open("OnlineNewsPopularity/OnlineNewsPopularity.csv") as csv_file:
            dataset = pd.read_csv(csv_file, sep=", ", engine="python")
        dataset.to_csv(local_path, index=False)

    dataset.rename(columns={'self_reference_avg_sharess': 'self_reference_avg_shares'}, inplace=True)

    return dataset


@torch.inference_mode()
def simulate_data(family: str,
                  n_samples: int,
                  n_features: int,
                  predict_params: Optional[Sequence] = None,
                  **kwargs) -> pd.DataFrame:
    """
    :param family: A family-alias, see ``Glm.family_names``.
    :param n_samples: Number of samples/rows in the dataset.
    :param n_features: Number of features.
    :param predict_params: The distribution-parameters that the features will be predictive of; all others will be
     constant (with value = inverse_link(0.)). Default is the first of family.params.
    :param kwargs: Other keyword arguments to sklearn.datasets.make_regression, such as ``effective_rank`` and
     ``bias``. Note ``bias`` can be an array that's as long as ``predict_params``.
    :return: A dataframe with predictors in format ``x_{i}`` and target ``target``. The ground-truth predictor-
     coefficients are stored in the ``attrs`` attribute of the dataframe.
    """
    family = family_from_string(family)
    if predict_params is None:
        predict_params = [family.params[0]]
    else:
        assert set(predict_params).issubset(family.params)
    n_targets = len(predict_params)

    # default all informative features:
    kwargs['n_informative'] = kwargs.get('n_informative', n_features)
    # sklearn multiplies by 100, we reverse that, but want bias to be 1-1
    kwargs['bias'] = kwargs.get('bias', 0.0) * 100

    # simulate:
    X, y, ground_truth_mat = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0,
        coef=True,
        **kwargs
    )
    # undo squeezing and 100x:
    y = y.reshape(n_samples, n_targets) / 100
    ground_truth_mat = ground_truth_mat.reshape(n_features, n_targets) / 100

    # convert to distribution:
    family_kwargs = dict(zip(predict_params, torch.as_tensor(y.T)))
    for par in family.params:
        if par not in family_kwargs:
            family_kwargs[par] = torch.as_tensor(0.0)
    distribution = family(**family_kwargs)

    # create output
    feature_names = [f'x{i + 1}' for i in range(n_features)]
    out = pd.DataFrame(X, columns=feature_names)

    # set torch random-state:
    random_state = kwargs.get('random_state')
    with torch.random.fork_rng():
        if random_state is not None:
            torch.manual_seed(random_state if isinstance(random_state, int) else hash(random_state))

        # sample from distribution for target:
        out['target'] = distribution.sample().numpy()

    # ground truth:
    ground_truth = {p: {} for p in predict_params}
    for i in range(n_features):
        for par, coef in zip(predict_params, ground_truth_mat[i, :]):
            ground_truth[par][feature_names[i]] = coef
    out.attrs['ground_truth'] = ground_truth

    return out
