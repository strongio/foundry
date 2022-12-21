from typing import Optional, Union, Sequence, Dict, Tuple, Type
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch import distributions
from torch.distributions import transforms

from foundry.glm import Glm
from foundry.glm.family import Family
from foundry.util import transpose_last_dims, ModelMatrix, ToSliceDict


class Covariance:
    def __init__(self, low_rank: Optional[int] = None):
        self.low_rank = low_rank

    def get_num_params(self, rank: int) -> int:
        if self.low_rank:
            raise NotImplementedError('todo')
        else:
            return int(rank + (rank * (rank - 1) / 2))

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        if self.low_rank:
            raise NotImplementedError('todo')
        else:
            nparams = params.shape[-1]
            rank = .5 * (np.sqrt(8 * nparams + 1) - 1)
            log_diag, off_diag = torch.split(params, [rank, nparams - rank], dim=-1)
            L = log_chol_to_chol(log_diag, off_diag)
            return L @ transpose_last_dims(L)


def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
    assert log_diag.shape[:-1] == off_diag.shape[:-1]

    rank = log_diag.shape[-1]
    L1 = torch.diag_embed(torch.exp(log_diag))

    L2 = torch.zeros_like(L1)
    mask = torch.tril_indices(rank, rank, offset=-1)
    L2[mask[0], mask[1]] = off_diag
    return L1 + L2


class Hlm(Glm):
    """
    :param family: A family name; you can see available names with ``Hlm.family_names``.
    :param fixeffs: Sequence of column-names of the fixed-effects in the model-matrix ``X``. Instead of string-
     columns-names, can also supply callables which take the input pandas ``DataFrame`` and return a list of
     column-names.
    :param raneff_design: A dictionary, whose key(s) are the names of grouping factors and whose values are
     column-names for random-effects of that grouping factor. As with ``fixeffs``, either string-column-names or
     callables can be used.
    :param penalty: A multiplier for L2 penalty on coefficients. Can be a single value, or a dictionary of these to
     support different penalties per distribution-parameter. Values can either be floats, or can be functions that take
     a ``torch.nn.Module`` as the first argument and that module's param-names as second argument, and returns a scalar
     penalty that will be applied to the log-prob.
    """
    # these don't yet include re-cov params, those are added in _init_family:
    family_names = {
        'binomial': (
            distributions.Binomial,
            {
                'probs': transforms.SigmoidTransform(),
            },
        ),
        'gaussian': (
            torch.distributions.Normal,
            {
                'loc': transforms.identity_transform,
                'scale': transforms.ExpTransform(),
            }
        )
    }

    def __init__(self,
                 family: Union[str, Family],
                 fixeffs: Sequence[Union[str, callable]],
                 raneff_design: Dict[str, Sequence[Union[str, callable]]],
                 raneff_covariance: Union[dict, str, Covariance] = 'full_rank',
                 penalty: Union[float, Sequence[float], Dict[str, float]] = 0.):

        self.fixeffs = fixeffs
        self.raneff_design = raneff_design
        self.raneff_covariance = raneff_covariance
        super().__init__(
            family=family,
            penalty=penalty,
            col_mapping=None
        )

    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            **kwargs) -> 'Glm':

        # need to do these *before* init_family is called, b/c raneff design/cov determines args to family
        self.fixeffs, self.raneff_design = self._standardize_design(X, self.fixeffs, self.raneff_design)
        self.raneff_covariance = self._standardize_raneff_cov(self.raneff_covariance)

        return super(Hlm, self).fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            groups=groups,
            **kwargs
        )

    def _init_col_mapping(self, X: ModelMatrix):
        if not isinstance(X, pd.DataFrame):
            raise NotImplementedError("Currently only DataFrame X supported")
        # [dist_param, *gfs, group_ids]
        col_mapping = {self.family.params[0]: self.fixeffs}
        col_mapping.update(self.raneff_design)
        assert 'group_ids' not in col_mapping
        col_mapping['group_ids'] = list(self.raneff_design)
        self.to_slice_dict_ = ToSliceDict(mapping=col_mapping)
        self.to_slice_dict_.fit(X)

    @classmethod
    def _standardize_design(cls,
                            X: pd.DataFrame,
                            fixeffs: Sequence[Union[str, callable]],
                            raneff_design: Dict[str, Sequence[Union[str, callable]]],
                            verbose: bool = True
                            ) -> Tuple[Sequence[str], Dict[str, Sequence[str]]]:
        any_callables = False
        standarized_cols = {}
        assert 'fixeffs' not in raneff_design
        for cat, to_standardize in {**raneff_design, **{'fixeffs': fixeffs}}.items():
            if cat not in standarized_cols:
                standarized_cols[cat] = []
            for i, cols in enumerate(to_standardize):
                if callable(cols):
                    any_callables = True
                    cols = cols(X)
                    if not cols:
                        warn(f"Element {i} of {cat} was a callable that, when given X, returned no rows.")
                    # (1) callables might have overlapping matches, drop dupes, (2) don't include grouping factor
                    cols = [c for c in cols if c not in standarized_cols[cat] and c not in raneff_design]
                if isinstance(cols, str):
                    if cols not in X.columns:
                        raise RuntimeError(f"No column named {cols}")
                    cols = [cols]
                standarized_cols[cat].extend(cols)
        if any_callables and verbose:
            print(f"Model-features: {standarized_cols}")
        fixeffs = standarized_cols.pop('fixeffs')
        return fixeffs, standarized_cols

    def _standardize_raneff_cov(self, raneff_covariance: Union[dict, str, Covariance]) -> dict:
        if isinstance(raneff_covariance, dict):
            raneff_covariance = raneff_covariance.copy()
        else:
            raneff_covariance = {gf: raneff_covariance for gf in self.raneff_design}

        for gf, gf_preds in self.raneff_design.items():
            gf_cov = raneff_covariance[gf]
            if isinstance(gf_cov, str):
                low_rank = None
                if gf_cov.startswith('low_rank'):
                    low_rank = gf_cov.replace('low_rank', '').lstrip('_')
                    low_rank = int(low_rank) if low_rank.isdigit() else int(np.sqrt(len(gf_preds)))
                elif gf_cov != 'full_rank':
                    raise ValueError("If covariance is a string, expect 'full_rank' or 'low_rank{rank}'")
                gf_cov = Covariance(low_rank=low_rank)
            raneff_covariance[gf] = gf_cov

        return raneff_covariance

    def _get_module_num_outputs(self, y: np.ndarray, dist_param_name: str) -> int:
        if dist_param_name in self.raneff_covariance:
            return self.raneff_covariance.get_num_params(len(self.raneff_design[dist_param_name]))
        return super()._get_module_num_outputs(y=y, dist_param_name=dist_param_name)

    def _init_family(self, family: Union[Family, str]) -> Family:
        if isinstance(family, str):
            distribution_cls, params_and_links, *args = self.family_names[family]
            if args:
                raise NotImplementedError

            # add random-effects parameters:
            for gf, cov in self.raneff_covariance.items():
                assert gf not in params_and_links
                params_and_links[f're_cov_{gf}'] = cov

            family = HlmFamily(
                distribution_cls=distribution_cls,
                params_and_links=params_and_links
            )
        return family


class HlmFamily(Family):

    def __init__(self,
                 distribution_cls: Type[torch.distributions.Distribution],
                 params_and_links: Dict[str, callable],
                 is_classifier: Optional[bool] = None,
                 mcmc_likelihood: Optional[bool] = None):
        # TODO: what about composed multivariate?

        if issubclass(distribution_cls, distributions.Normal):
            if mcmc_likelihood is None:
                mcmc_likelihood = True
        else:
            if mcmc_likelihood is not None:
                raise NotImplementedError(f"Must use mcmc_likelihood for {distribution_cls.__name__}")

        self.mcmc_likelihood = mcmc_likelihood

        super().__init__(
            distribution_cls=distribution_cls,
            params_and_links=params_and_links,
            is_classifier=is_classifier
        )

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        # the usual:
        dist_kwargs = {}
        for p, ilink in self.params_and_links.items():
            dist_kwargs[p] = ilink(kwargs.pop(p))

        # ok so now dist_kwargs should have [loc, *re_cov_{gfs}, optional[scale]]
        # and kwargs should have [{gfs}, group_ids].
        # 1. for mcmc:
        #    a. for one grouping factor: need to sim betas for each gf from each re_cov_gf, return mvnorm of sims
        #    b. for >1 grouping factor: one giant mvnorm
        # 2. for closed_form:
        #    a. for one grouping factor: one mvnorm with num_groups batch-dim
        #    b. for >1 grouping factor: one giant mvnorm
        raise NotImplementedError
        # dist_kwargs.update(kwargs)
        # return self.distribution_cls(**dist_kwargs)

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        value, weight = self._validate_values(value, weight, distribution)
        # CHALLENGE: do we need to get group_ids back?

        # log_probs = distribution.log_prob(value)
        # log_probs = weight * log_probs
        # return log_probs

    @staticmethod
    def log_cdf(distribution: distributions.Distribution, value: torch.Tensor, lower_tail: bool) -> torch.Tensor:
        raise NotImplementedError  # TODO
