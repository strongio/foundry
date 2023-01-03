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
        col_mapping = {self.family.loc_param: self.fixeffs}
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
        full_design = raneff_design.copy()
        assert 'fixeffs' not in raneff_design
        full_design['fixeffs'] = fixeffs

        _any_callables = False
        standarized_cols = {cat: [] for cat in full_design}
        for cat, to_standardize in full_design.items():
            for i, cols in enumerate(to_standardize):
                if callable(cols):
                    _any_callables = True
                    cols = cols(X)
                    if not cols:
                        warn(f"Element {i} of {cat} was a callable that, when given X, returned no rows.")
                    # (1) callables might have overlapping matches, drop dupes, (2) don't include grouping factor
                    cols = [c for c in cols if c not in standarized_cols[cat] and c not in raneff_design]
                elif isinstance(cols, str):
                    raise RuntimeError("`cols` must be sequence of strings, not a single string.")
                standarized_cols[cat].extend(cols)
        if _any_callables and verbose:
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
                    raise ValueError("If covariance is a string, expected 'full_rank' or 'low_rank{rank}'")
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
            params_and_links = params_and_links.copy()
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
                 supports_predict_proba: Optional[bool] = None,
                 mc_likelihood: Union[int, bool, None] = None):
        # TODO: what about composed multivariate?

        if issubclass(distribution_cls, distributions.Normal):
            if mc_likelihood is None:
                mc_likelihood = True
        else:
            if mc_likelihood is not None:
                raise NotImplementedError(f"Must use ``mc_likelihood`` for {distribution_cls.__name__}")

        self.mc_likelihood = mc_likelihood
        self._mc_white_noise = {}

        super().__init__(
            distribution_cls=distribution_cls,
            params_and_links=params_and_links,
            supports_predict_proba=supports_predict_proba
        )

        self.loc_param = self.params[0]

    def get_mc_white_noise(self, grouping_factor: str, num_sims: int, num_res: int, **kwargs) -> torch.Tensor:
        shape = (num_sims, num_res)
        if grouping_factor in self._mc_white_noise:
            assert tuple(self._mc_white_noise[grouping_factor].shape) == shape
        else:
            self._mc_white_noise[grouping_factor] = torch.randn(shape, **kwargs)
        return self._mc_white_noise[grouping_factor]

    def __call__(self, **kwargs) -> torch.distributions.Distribution:
        cov_params = {}
        raw_dist_params = {}
        for nm, ilink in self.params_and_links.items():
            if nm.startswith('re_cov_'):
                cov_params[nm.replace('re_cov_', '')] = ilink(kwargs.pop(nm))
            else:
                raw_dist_params[nm] = kwargs.pop(nm)

        # XXX
        re_model_mats = {k:mm for k, mm in kwargs.items() if k!='group_ids'}
        gf_names = list(re_model_mats)
        re_distributions = {
            nm: torch.distributions.MultivariateNormal(loc=0, covariance_matrix=cov_params[nm])
            for nm in gf_names
        }

        if self.mc_likelihood:
            num_sims = self.mc_likelihood if isinstance(self.mc_likelihood, int) else 500
            #    a. for one grouping factor: need to sim betas for each gf from each re_cov_gf, return mvnorm of sims
            if len(re_model_mats) == 1:
                Xgf = re_model_mats[gf_names[0]]
                Xgf = torch.cat([torch.ones_like(Xgf[:, 0]), Xgf], 1)  # TODO: intercept-only
                white_noise = self.get_mc_white_noise(
                    grouping_factor=gf_names[0],
                    num_sims=num_sims,
                    num_res=Xgf.shape[-1]
                )
                re_samples = (re_distributions[gf_names[0]].scale_tril @ white_noise.unsqueeze(-1)).squeeze(-1)
                yhat_r_samples = (Xgf.unsqueeze(-1) * re_samples.T.unsqueeze(0)).sum(1)
                # y_dist = torch.distributions.Normal(loc=yhat_samples, scale=self.residual_var ** .5)
                # integral_wrt_b_random[ p(y|b_fixed, b_random)] * p(b_random|re_cov) ]
                #   ~=
                # mean[ p(y|b_fixed, b_random_i)) ], b_random_i ~ mvnorm(re_cov)
                # log_integrand = y_dist.log_prob(y.unsqueeze(-1))
                dist_kwargs = {}
                for nm, raw_param in raw_dist_params.items():
                    raw_param = raw_param.unsqueeze(-1)
                    if nm == self.loc_param:
                        raw_param = raw_param + yhat_r_samples
                    ilink = self.params_and_links[nm]
                    dist_kwargs[nm] = ilink(raw_param)
                return self.distribution_cls(**dist_kwargs)
            else:
                #    b. for >1 grouping factor: one giant mvnorm
                raise NotImplementedError
        else:
            #    a. for one grouping factor: one mvnorm with num_groups batch-dim
            #    b. for >1 grouping factor: one giant mvnorm
            pass

    def log_prob(self,
                 distribution: torch.distributions.Distribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None,
                 group_ids: pd.DataFrame = None) -> torch.Tensor:
        if weight is not None:
            # TODO
            raise NotImplementedError
        if self.mc_likelihood:
            log_probs = distribution.log_prob(value)
            ngroups = cache['group_ids_seq'].max() + 1
            group_ids_broad = cache['group_ids_seq'].expand(-1, nsim)
            lps_per_group = torch.zeros((ngroups, nsim)).scatter_add(0, group_ids_broad, log_integrand)
            log_probs_unnorm = lps_per_group.logsumexp(1)
            log_probs = log_probs_unnorm - math.log(nsim)
        # log_probs = weight * log_probs
        # return log_probs
