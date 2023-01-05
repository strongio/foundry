from io import BytesIO
from typing import List, Tuple
from tempfile import TemporaryDirectory
import requests
import zipfile

import pandas as pd

def get_wine_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    colnames_raw: List[str] = [
        "Target",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]
    colnames: List[str] = list(map(lambda s: s.lower().replace(" ", ""), colnames_raw))

    dataset: pd.DataFrame = pd.read_csv(dataset_url, names=colnames)
    return dataset.iloc[:, 1:], dataset.loc[:, "target"]

def get_online_news_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """ Returns X, y for predicting the number of "shares" for online news

    See https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
    """

    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"

    response = requests.get(dataset_url)
    zip_archive = zipfile.ZipFile(BytesIO(response.content))
    with zip_archive.open("OnlineNewsPopularity/OnlineNewsPopularity.csv") as csv_file:
        dataset: pd.DataFrame = pd.read_csv(csv_file, sep=", ", engine="python")

    return dataset.loc[:, "n_tokens_title":"abs_title_sentiment_polarity"], dataset.loc[:, "shares"]
