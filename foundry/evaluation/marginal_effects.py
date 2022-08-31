from typing import Collection, Dict, Optional, Iterable, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Callable, Sequence, Union


class Binned:
    def __init__(self, col: str, bins: Union[int, Sequence] = 20, **kwargs):
        self.orig_name = col
        self.bins = bins
        self.kwargs = kwargs

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy(deep=False)
        if pd.api.types.is_categorical_dtype(df[self.orig_name]):
            return df[self.orig_name]

        if hasattr(self.bins, '__iter__'):
            return pd.cut(df[self.orig_name], self.bins, **self.kwargs)
        elif self.bins:
            return pd.qcut(df[self.orig_name], self.bins, duplicates='drop', **self.kwargs)
        else:
            return df[self.orig_name]


def binned(col: str, bins: Union[int, Sequence] = 20, **kwargs) -> Binned:
    """
    A concise way of indicating to ``MarginalEffects`` cls: "bin the result by 'col'".

    :param col: The column to bin
    :param bins: Either (a) number of bins, passed to ``pandas.qcut``, or (b) the cuts, passed to ``pandas.cut``
    :param kwargs: More arguments to the ``{q}cut`` function
    :return: A function that takes a dataframe and returns the binned version of the column.
    """

    return Binned(col=col, bins=bins, **kwargs)


def raw(col: str) -> Callable:
    """
    If you pass a raw column-name to ``MarginalEffects``, it'll by default get wrapped in ``binned``. Sometimes you
    really want the raw value, so wrap it in this.

    :param col: The column-name.
    :return: A function that takes a dataframe and returns the column as-is.
    """

    return Binned(col=col, bins=False)


class MarginalEffects:
    def __init__(self, pipeline: Pipeline, predict_method: Optional[str] = None):
        self.pipeline = pipeline
        self.predict_method = predict_method
        self._dataframe = None
        self.config = None

    @property
    def feature_names_in(self) -> Sequence[str]:
        col_transformer: ColumnTransformer = self.pipeline[0]
        if not hasattr(col_transformer, '_columns'):
            raise TypeError("``self.pipeline[0]`` does not have ``_columns``. Is it a ``ColumnTransformer``?")
        return list(set(sum(col_transformer._columns, [])))

    def __call__(self,
                 X: pd.DataFrame,
                 y: Optional[np.ndarray],
                 vary_features: Sequence[Union[str, Binned]],
                 groupby_features: Collection[Union[str, Binned]] = (),
                 vary_features_aggfun: Union[str, dict, Callable] = 'mean',
                 marginalize_aggfun: Union[str, dict, Callable, None] = 'mean',
                 y_aggfun: str = 'mean',
                 **predict_kwargs) -> 'MarginalEffects':
        """
        Prepare a dataframe/plot showing how predictions vary when one or more features are varied, holding
        all others at a constant value.

        :param X: The data with predictors to be passed to the pipeline.
        :param y: The target. Pass `None` if you don't want to include "actuals", only predictions.
        :param vary_features: Strings indicating the feature(s) to vary, so as to observe their independent effect in
         the model. By default will be binned by passing to :function:`foundry.evaluation.binned`. You can pass to this
         function yourself to manually control/remove binning.
        :param groupby_features: Strings indicating the feature(s) to group/segment on, so as to observe different
         effects per segment. By default will be binned by passing to :function:`foundry.evaluation.binned`. You can
         pass to this function yourself to manually control/remove binning.
        :param vary_features_aggfun: The varying feature(s) will be binned, then within each bin we need to convert
         back to numeric before plugging into the model. This string indicates how to do so. Either an aggregation that
         will be applied, or 'mid' to use the midpoint of the bin. The latter will be used regardless when no actual
         data exists in that bin. This argument can also be a dictionary with keys being feature-names.
        :param marginalize_aggfun: How to marginalize across features that are not in the vary/groupby. This is a
         string/callable that will be applied to each feature. Can also use a dictionary of these, with keys being
         feature-names. If set to False/None, will *not* collapse these, but instead call ``predict`` for each
         row then summarize the *predictions* with `y_aggfun`. WARNING: this will be slow for a large dataset,
         downsampling recommended.
        :param y_aggfun: The function to use when aggregating predictions (and actuals, if supplied). Only has an
         effect if ``marginalize_aggfun` is False/None.
        :param predict_kwargs: Keyword-arguments to pass to the pipeline's ``predict`` method.
        """
        orig_colnames = list(X.columns)
        X = X.copy(deep=False)

        # validate/standardize args:
        vary_features = self._standardize_maybe_binned(X, vary_features)
        check_set_membership(vary_features, self.feature_names_in)
        groupby_features = self._standardize_maybe_binned(X, groupby_features)
        check_set_membership(groupby_features, self.feature_names_in)
        assert set(vary_features).isdisjoint(set(groupby_features))

        # no-vary features ----
        if groupby_features:
            groupby_colnames = []
            for old_fname, new_fname, values in self._get_maybe_binned_features(X, groupby_features):
                groupby_colnames.append(new_fname)
                if old_fname == new_fname:
                    continue
                X[new_fname] = values
        else:
            X['_dummy'] = ''
            groupby_colnames = ['_dummy']

        df_no_vary = self._get_df_novary(
            X=X,
            marginalize_aggfun=marginalize_aggfun,
            groupby_colnames=groupby_colnames,
            colnames_to_agg=list(set(self.feature_names_in) - set(vary_features))
        )

        # vary features ----
        vary_features_aggfuns = self._standardize_maybe_dict(
            maybe_dict=vary_features_aggfun,
            keys=vary_features,
            default=vary_features_aggfun.pop('_default', 'mean') if isinstance(vary_features_aggfun, dict) else 'mean'
        )
        # get mappings of binned(vary_feature) -> agg(vary_feature)
        # the former is needed so we don't drop bins with no actual observations
        vary_features_mappings = {}
        for old_fname, new_fname, values in self._get_maybe_binned_features(X, vary_features):
            if old_fname == new_fname:
                vary_features_mappings[new_fname] = None
            else:
                X[new_fname] = values
                vary_features_mappings[new_fname] = self._get_binned_feature_map(
                    X=X,
                    binned_fname=new_fname,
                    fname=old_fname,
                    aggfun=vary_features_aggfuns[old_fname]
                )

        # get a grid of vary feature-bins, including data on 'actual' if available
        agg_kwargs = {'n': (groupby_colnames[0], 'count')}
        if y is not None:
            X['_actual'] = y
            agg_kwargs['actual'] = ('_actual', y_aggfun)
            agg_kwargs['actual_lower'] = ('_actual', lambda s: s.quantile(.25, interpolation='lower'))
            agg_kwargs['actual_upper'] = ('_actual', lambda s: s.quantile(.75, interpolation='higher'))
        df_vary_grid = (X
                        .groupby(groupby_colnames + list(vary_features_mappings.keys()))
                        .agg(**agg_kwargs)
                        .reset_index())
        if 'actual' in df_vary_grid.columns:  # handle binary targets with low-n
            for col in ('actual', 'actual_lower', 'actual_upper'):
                df_vary_grid[col] = df_vary_grid[col].astype('float')

        # map the binned version of vary features back to their aggregated values:
        for binned_fname, df_mapping in vary_features_mappings.items():
            if df_mapping is None:  # binned_fname==fname, i.e. feature was already categorical
                continue
            df_vary_grid = df_vary_grid.merge(df_mapping, on=binned_fname).drop(columns=[binned_fname])

        # merge
        if marginalize_aggfun:
            df_me = df_vary_grid.merge(df_no_vary, how='left', on=groupby_colnames)
        else:
            raise NotImplementedError("TODO")
        df_me['n'].fillna(0, inplace=True)
        if '_dummy' in df_me.columns:
            del df_me['_dummy']

        self.config = {'pred_colnames': []}
        for col, preds in self.get_predictions(X=X.reindex(columns=orig_colnames), **predict_kwargs).items():
            df_me[col] = preds
            self.config['pred_colnames'].append(col)

        self._dataframe = df_me
        self.config.update({
            'vary_features': vary_features,
            'groupby_features': groupby_features
        })
        return self

    def to_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("This `MarginalEffects()` needs to be called on a dataset first.")
        return self._dataframe

    def plot(self,
             x: Optional[str] = None,
             color: Optional[str] = None,
             facets: Optional[Sequence[str]] = None) -> 'ggplot':

        try:
            from plotnine import ggplot, aes, geom_line, geom_hline, facet_wrap, theme, theme_bw
        except ImportError as e:
            raise RuntimeError("plotting requires `plotnine` package") from e
        facets = facets or []

        data = self.to_dataframe()
        if len(self.config['pred_colnames']) > 1:
            data = data.melt(
                id_vars=[c for c in data if c not in self.config['pred_colnames']],
                var_name='prediction',
                value_name='predicted',
                value_vars=self.config['pred_colnames']
            )
            facets.append('prediction')

        available_default_features = list(self.config['vary_features']) + list(self.config['groupby_features'])
        if x and x in available_default_features:
            available_default_features.remove(x)
        if color and color in available_default_features:
            available_default_features.remove(color)
        for facet in facets:
            if facet in available_default_features:
                available_default_features.remove(facet)

        aes_kwargs = {'y': 'predicted'}
        if not x:
            x = available_default_features.pop(0)
        aes_kwargs['x'] = x
        if not color and available_default_features:
            color = available_default_features.pop(0)
        if color:
            aes_kwargs['group'] = aes_kwargs['color'] = color
        if available_default_features:
            if not facets:
                facets = available_default_features
            else:
                warn(f"Adding {available_default_features} to `facets`")
                facets.extend(available_default_features)

        plot = (
                ggplot(data, aes(**aes_kwargs)) +
                geom_line() +
                geom_hline(yintercept=0) +
                theme_bw() +
                theme(figure_size=(8, 6), subplots_adjust={'wspace': 0.10})
        )
        if 'actual' in self._dataframe.columns:
            raise NotImplementedError("TODO")
        if facets:
            plot += facet_wrap(facets, scales='free_y')
        return plot

    @staticmethod
    def _get_maybe_binned_features(X: pd.DataFrame,
                                   features: Dict[str, Binned]) -> Iterable[Tuple[str, str, Optional[pd.Series]]]:
        for fname, bin_fun in features.items():
            binned_feature = bin_fun(X)
            if binned_feature is X[fname]:
                # `bin_fun` will be a no-op if the user passed ``raw(feature)`` or if they passed
                # the feature and it's categorical.
                # TODO: test this works!
                if not pd.api.types.is_categorical_dtype(binned_feature):
                    warn(f"{fname} is not categorical, values not present in the data will be dropped.")
                binned_fname = fname
            else:
                binned_fname = f'{fname}_binned'
            yield fname, binned_fname, binned_feature

    def get_predictions(self, X: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        # pipeline doesn't let us forward any methods aside from predict{_proba}, so we split:
        prep_steps_, estimator_ = self.pipeline[0:-1], self.pipeline[-1]

        # default behavior: use proba if it's available
        if self.predict_method is None:
            if hasattr(estimator_, 'predict_proba'):
                self.predict_method = 'predict_proba'
            else:
                self.predict_method = 'predict'
        predictions = getattr(estimator_, self.predict_method)(prep_steps_.transform(X), **kwargs)

        # validate output:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            assert len(predictions.shape) == 2
            if self.predict_method == 'predict_proba':
                assert predictions.shape[1] == 2
                predictions = predictions[:, 1]
            elif not isinstance(predictions, pd.DataFrame):
                raise RuntimeError(
                    f"the `{self.predict_method}` method of the pipeline returned predictions with shape "
                    f"{predictions.shape}. Only expect output to be not 1d-like if `predict_proba` was used "
                    f"or if output is a dataframe."
                )
        # handle multi-output:
        if isinstance(predictions, pd.DataFrame):
            predictions = {k: v.values for k, v in predictions.to_dict(orient='series').items()}
        else:
            predictions = {'predicted': predictions}

        return predictions

    @staticmethod
    def _standardize_maybe_binned(data: pd.DataFrame, features: Collection[Union[str, Binned]]) -> Dict[str, Binned]:
        """
        Take a collection of items that may either strings or ``Binned`` instances. Convert any strings to ``Binned``
        instances, return a dictionary keyed by the original feature-names.
        """
        if isinstance(features, str):
            features = [features]
        out = {}
        for maybe_binned in features:
            if hasattr(maybe_binned, 'orig_name'):
                deffy_binned = maybe_binned
            elif not isinstance(maybe_binned, str):
                raise ValueError(f"Expected {maybe_binned} to be a string or wrapped in ``binned``.")
            elif pd.api.types.is_categorical_dtype(data[maybe_binned]):
                deffy_binned = Binned(maybe_binned, bins=False)
            elif pd.api.types.is_numeric_dtype(data[maybe_binned]):
                deffy_binned = Binned(maybe_binned)
            else:
                raise ValueError(f"{maybe_binned} is neither categorical nor numeric; unable bin or keep as-is.")
            out[deffy_binned.orig_name] = deffy_binned
        return out

    @staticmethod
    def _standardize_maybe_dict(maybe_dict: any, keys: Collection[str], default: any) -> dict:
        """
        Sometimes we want to allow either of (a) ``arg='median'`` applies median to each feature,
        (b) ``{'f':'median'}`` applies median to feature 'f'. This function standardizes into (b).

        :param maybe_dict: Maybe a dictionary, or maybe the value for all dictionary items.
        :param keys: The keys that should be in the dictionary. Any not present will be added with value `default`.
        :param default: See `keys`.
        :return: A dictionary.
        """
        if isinstance(maybe_dict, dict):
            deffy_dict = maybe_dict.copy()
            for key in keys:
                if key not in deffy_dict:
                    deffy_dict[key] = default
        else:
            deffy_dict = {key: maybe_dict for key in keys}
        return deffy_dict

    @staticmethod
    def _get_binned_feature_map(X: pd.DataFrame,
                                binned_fname: str,
                                fname: str,
                                aggfun: Union[str, Callable]) -> pd.DataFrame:
        """
        Get a dataframe that maps the binned version of a feature to the aggregates of its original values.
        """
        assert binned_fname != fname

        if aggfun == 'mid':
            # creates a df with unique values of `binned_fname` and `nans` for `fname`.
            # this will then get filled with the midpoint below:
            # todo: less hacky way to do this
            df_mapping = X.groupby(binned_fname)[fname].agg('count').reset_index()
            df_mapping[fname] = float('nan')
        else:
            df_mapping = X.groupby(binned_fname)[fname].agg(aggfun).reset_index()

        # for any bins that aren't actually observed, use the midpoint:
        df_mapping[fname].fillna(pd.Series([x.mid for x in df_mapping[binned_fname]]), inplace=True)
        return df_mapping

    def _get_df_novary(self,
                       X: pd.DataFrame,
                       marginalize_aggfun: Union[str, dict, Callable, None],
                       groupby_colnames: Sequence[str],
                       colnames_to_agg: Sequence[str]) -> pd.DataFrame:
        """
        Get a dataframe with all the features except for `vary_features`. These are collapsed to the groups defined by
        groupby_colnames.
        """
        if marginalize_aggfun:
            marginalize_aggfuns = self._standardize_maybe_dict(
                maybe_dict=marginalize_aggfun,
                keys=colnames_to_agg,
                default=marginalize_aggfun.pop('_default', 'mean') if isinstance(marginalize_aggfun, dict) else 'mean'
            )
            agg_kwargs = {}
            for feature in colnames_to_agg:
                if feature in groupby_colnames:
                    # groupby colnames are usually the binned version of a subset of marginalize_features.
                    # but sometimes (e.g. was already categorical so no binning needed) the groupby is all we need
                    continue
                agg_kwargs[feature] = (feature, marginalize_aggfuns[feature])
            if not agg_kwargs:
                raise NotImplementedError("TODO")

            # collapse:
            df_no_vary = X.groupby(groupby_colnames).agg(**agg_kwargs).reset_index()
            # validate:
            if (df_no_vary.groupby(groupby_colnames).size() > 1).any():
                for col in colnames_to_agg:
                    if (df_no_vary.groupby(groupby_colnames)[col].nunique() > 1).any():
                        raise ValueError(f"The aggfun {agg_kwargs[col]} did not result in 1 value per group for {col}")
                # shouldn't generally get here:
                raise ValueError(f"The ``marginalize_aggfun`` did not result in one row per group:\n{df_no_vary}")
        else:
            df_no_vary = X.loc[:, list(set(list(groupby_colnames) + list(colnames_to_agg)))]
        return df_no_vary


def check_set_membership(maybe_members: Collection, full_set: Collection):
    extras = set(maybe_members) - set(full_set)
    if extras:
        raise ValueError(f"Got unexpected values: {extras}. Expected one/more of: {full_set}")
