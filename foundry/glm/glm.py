import os

from time import sleep
from typing import Union, Sequence, Optional, Callable, Tuple, Dict
from warnings import warn

import numpy as np
import pandas as pd
import torch
import torch.nn
from sklearn.base import BaseEstimator

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm.auto import tqdm

from foundry.glm.family import Family
from foundry.glm.family.survival import SurvivalFamily
from foundry.glm.util import NoWeightModule, Stopping
from foundry.hessian import hessian
from foundry.penalty import L2
from foundry.util import FitFailedException, is_invalid, get_to_kwargs, to_tensor, to_2d

ModelMatrix = Union[np.ndarray, pd.DataFrame, dict]

N_FIT_RETRIES = int(os.getenv('GLM_N_FIT_RETRIES', 10))

from sklearn.exceptions import NotFittedError


class Glm(BaseEstimator):
    """
    :param family: Either a :class:`foundry.glm.family.Family`, or a string alias. You can see available aliases with
     ``Glm.family_aliases()``.
    :param penalty: A multiplier for L2 penalty on coefficients. Can be a single float, or a dictionary of these to
     support different penalties per distribution-parameter. Instead of floats, can also pass functions that take a
     ``torch.nn.Module`` as the first argument and that module's param-names as second argument, and returns a scalar
     penalty that will be applied to the log-prob.
    :param predict_params: Many distributions have multiple parameters: for example the normal distribution has a
     location and scale parameter. If a single dataframe/matrix is passed to  ``fit()``, the default behavior is to
     use these to separately predict each of loc/scale. Sometimes this is not desired: for example, we only want to use
     predictors to predict the location, and the scale should be 'intercept only'. This can be accomplished with
     `predict_params=['loc']` (replacing 'loc' with the relevant name(s) for your distribution of interest. For finer-
     grained control (e.g. using some predictors for some params and others for others), see options about passing a
     dictionary to ``fit()``.
    """

    def __init__(self,
                 family: Union[str, Family],
                 penalty: Union[float, Sequence[float], Dict[str, float]] = 0.,
                 predict_params: Optional[Sequence[str]] = None):

        self.family = family
        self.penalty = penalty
        self.predict_params = predict_params

        # set in _init_module:
        self._module_ = None
        self._module_param_names_ = None

        # laplace params:
        self._coef_mvnorm_ = None
        self.converged_ = None
        self._laplace_coefs_names_ = None

        # updated on each fit():
        self._fit_failed = 0

    @property
    def module_(self) -> torch.nn.ModuleDict:
        if self._module_ is None:
            raise NotFittedError("Tried to access `module_` prior to fitting.")
        return self._module_

    @module_.setter
    def module_(self, value: torch.nn.ModuleDict):
        self._module_ = value

    @property
    def expected_model_mat_params_(self) -> Sequence[str]:
        return [p for p, m in self.module_.items() if not isinstance(m, NoWeightModule)]

    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            **kwargs) -> 'Glm':
        """
        :param X: A dataframe/array of predictors, or a dictionary of these. If a dict, then keys should correspond to
         ``self.family.params``.
        :param y:
        :param sample_weight:
        :param groups:
        :param reset:
        :param callbacks:
        :param stopping: args/kwargs to pass to :class:`foundry.glm.util.Stopping` (e.g. ``(.01,)`` would
         use abstol of .01).
        :param max_iter:
        :param max_loss:
        :param verbose:
        :param estimate_laplace_coefs:
        :return:
        """
        self.family = self._init_family(self.family)

        if isinstance(self.penalty, (tuple, list)):
            raise NotImplementedError  # TODO
        else:
            if groups is not None:
                warn("`groups` argument will be ignored because self.penalty is a single value not a sequence.")
            return self._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

    @staticmethod
    def family_aliases() -> dict:
        out = Family.aliases.copy()
        out.update({f'survival_{nm}': f for f, nm in SurvivalFamily.aliases.items()})
        return out

    @staticmethod
    def _init_family(family: Union[Family, str]) -> Family:
        if isinstance(family, str):
            if family.startswith('survival'):
                family = SurvivalFamily.from_name(family.replace('survival', '').lstrip('_'))
            else:
                family = Family.from_name(family)
        return family

    @retry(retry=retry_if_exception_type(FitFailedException), reraise=True, stop=stop_after_attempt(N_FIT_RETRIES + 1))
    def _fit(self,
             X: ModelMatrix,
             y: ModelMatrix,
             sample_weight: Optional[np.ndarray] = None,
             reset: bool = True,
             callbacks: Sequence[Callable] = (),
             stopping: Union['Stopping', tuple, dict] = (.001,),
             max_iter: int = 200,
             max_loss: float = 10.,
             verbose: bool = True,
             estimate_laplace_coefs: bool = True) -> 'Glm':

        # tallying fit-failurs:
        if self._fit_failed:
            if verbose:
                print(f"Retry attempt {self._fit_failed}/{N_FIT_RETRIES}")
            reset = True

        # build model:
        if self._module_ is None or reset:
            if self._module_ is not None and verbose:
                warn("Resetting module with reset=True")
            self.module_ = self._init_module(X, y)

        # optimizer:
        self.optimizer_ = self._init_optimizer()

        # stopping:
        if isinstance(stopping, dict):
            stopping = Stopping(**stopping, optimizer=self.optimizer_)
        elif isinstance(stopping, (list, tuple)):
            stopping = Stopping(*stopping, optimizer=self.optimizer_)
        else:
            assert isinstance(stopping, Stopping)
            stopping.optimizer = self.optimizer_

        # build model-mats:
        mm_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)

        # progress bar:
        prog = None
        if verbose:
            prog = self._init_pb()

        # 'closure' for torch.optimizer.Optimizer.step method:
        epoch = 0

        def closure():
            self.optimizer_.zero_grad()
            loss = -self.get_log_prob(mm_dict, lp_dict)
            if is_invalid(loss):
                self._fit_failed += 1
                raise FitFailedException("`nan`/`inf` loss")
            loss.backward()
            if prog:
                prog.update()
                prog.set_description(
                    f"Epoch {epoch:,}; Loss {loss.item():.4}; Convergence {stopping.get_info()}"
                )
            return loss

        # fit-loop:
        assert max_iter > 0
        for epoch in range(max_iter):
            try:
                if prog:
                    prog.reset()
                    prog.set_description(f"Epoch {epoch:,}; Loss -; Convergence {stopping.get_info()}")
                train_loss = self.optimizer_.step(closure).item()
                for callback in callbacks:
                    callback(self, train_loss)

                if stopping(train_loss):
                    break

            except KeyboardInterrupt:
                sleep(1)
                break
            finally:
                self.optimizer_.zero_grad(set_to_none=True)

        if train_loss > max_loss:
            self._fit_failed += 1
            raise FitFailedException(f"train_loss ({train_loss}) too high")

        self._fit_failed = 0

        if estimate_laplace_coefs:
            if verbose:
                print("Estimating laplace coefs... (you can safely keyboard-interrupt to cancel)")
            try:
                self.estimate_laplace_coefs(X=X, y=y, sample_weight=sample_weight)
            except KeyboardInterrupt:
                pass

        return self

    def get_log_prob(self, mm_dict: dict, lp_dict: dict, mean: bool = True) -> torch.Tensor:
        """
        Get the penalized log prob, applying weights as needed.
        """
        log_probs = self.family.log_prob(self.family(**self._get_dist_kwargs(**mm_dict)), **lp_dict)
        penalty = self._get_penalty()
        log_prob = log_probs.sum() - penalty
        if mean:
            log_prob = log_prob / lp_dict['weight'].sum()
        return log_prob

    def _get_dist_kwargs(self, **kwargs) -> dict:
        """
        Call each entry in the module_ dict to get predicted family params.
        """
        out = {}
        for dp in self.module_:
            if dp in self.expected_model_mat_params_:
                out[dp] = self.module_[dp](kwargs.pop(dp))
            else:
                out[dp] = self.module_[dp](kwargs.get(dp, None))
        if kwargs:
            warn(f"Unrecognized kwargs to `_get_dist_kwargs()`: {set(kwargs)}")
        return out

    def _get_xdict(self, X: ModelMatrix) -> Dict[str, torch.Tensor]:
        _to_kwargs = get_to_kwargs(self.module_)

        # convert to dict:
        if isinstance(X, dict):
            Xdict = X.copy()
        else:
            # TODO: if originally passed a dict but are now passing a dataframe, this will lead to cryptic errors later
            Xdict = {p: X for p in self.expected_model_mat_params_}

        # validate:
        if set(Xdict) != set(self.expected_model_mat_params_):
            raise ValueError(
                f"X has keys {set(Xdict)} but (from prev. call to fit()), was "
                f"expecting {self.expected_model_mat_params_}"
            )

        # convert to tensors:
        x_len = None
        for nm in self.expected_model_mat_params_:
            Xdict[nm] = to_tensor(Xdict[nm], **_to_kwargs)
            assert len(Xdict[nm].shape) == 2, f"len(X['{nm}'].shape)!=2"
            if x_len is None:
                x_len = Xdict[nm].shape[0]
            elif x_len != Xdict[nm].shape[0]:
                raise ValueError("Not all X entries have same shape[0]")
            if is_invalid(Xdict[nm]):
                raise ValueError(f"nans/infs in X ({nm})")
        return Xdict

    def _get_ydict(self, y: ModelMatrix, sample_weight: Optional[np.ndarray]) -> Dict[str, torch.Tensor]:
        if isinstance(y, dict):
            assert 'value' in y
            ydict = y.copy()
        else:
            ydict = {'value': y}

        y_len = ydict['value'].shape[0]

        if 'weight' in ydict:
            if sample_weight is not None:
                raise ValueError("Please pass either `sample_weight` or y=dict(weight=), but not both.")
        else:
            ydict['weight'] = sample_weight

        if ydict.get('weight', None) is None:
            ydict['weight'] = torch.ones(y_len)
        else:
            assert (ydict['weight'] >= 0).all()

        _to_kwargs = get_to_kwargs(self.module_)

        for k in list(ydict):
            if not hasattr(ydict[k], '__iter__'):
                # wait until we have a use-case
                raise NotImplementedError(
                    f"Unclear how to handle {k}, please report this error to the package maintainer"
                )
            ydict[k] = to_2d(to_tensor(ydict[k], **_to_kwargs))
            # note: no `is_invalid` check, some families might support missing values or infs
            if ydict[k].shape[0] != y_len:
                raise ValueError(f"{k}.shape[0] does not match y.shape[0]")
        return ydict

    def _build_model_mats(self,
                          X: ModelMatrix,
                          y: Optional[ModelMatrix],
                          sample_weight: Optional[np.ndarray] = None,
                          include_y: bool = False) -> Tuple[dict, Optional[dict]]:
        """
        :param X: A dataframe/ndarray/tensor, or dictionary of these.
        :param y: An optional dataframe/ndarray/tensor, or dictionary of these.
        :param sample_weight: Optional sample-weights
        :param include_y: If True, then will raise if y is not present; if False, will return None in place of y.
        :return: Two dictionaries: one of model-mat kwargs (for prediction) and one of target-related kwargs
         (for evaluation i.e. log-prob).
        """
        _to_kwargs = get_to_kwargs(self.module_)

        Xdict = self._get_xdict(X)

        if not include_y:
            return Xdict, None

        if y is None:
            raise ValueError("Must pass `y` when fitting.")

        ydict = self._get_ydict(y=y, sample_weight=sample_weight)

        if Xdict and list(Xdict.values())[0].shape[0] != ydict['value'].shape[0]:
            raise ValueError("Expected X.shape[0] to match y.shape[0]")

        return Xdict, ydict

    def _init_module(self, X: ModelMatrix, y: ModelMatrix) -> torch.nn.ModuleDict:

        if isinstance(X, dict):
            if not set(X.keys()).issubset(set(self.family.params)):
                raise ValueError(f"X should be a subset of {self.family.params}")
        else:
            X = {dp: X for dp in self.predict_params or self.family.params}

        if isinstance(y, dict):
            y = y['value']
        y = to_2d(np.asanyarray(y))
        output_dim = y.shape[1]

        self._module_param_names_ = {}
        self.module_ = torch.nn.ModuleDict()
        for dp in self.family.params:
            Xp = X.get(dp, None)
            if Xp is None or not Xp.shape[1]:
                # always use the base approach when no input features:
                module, nms = Glm.module_factory(X=None, output_dim=output_dim)
            else:
                module, nms = self.module_factory(X=Xp, output_dim=output_dim)
            self.module_[dp] = module
            self._module_param_names_[dp] = {k: np.asarray(v) for k, v in nms.items()}

        return self.module_

    @classmethod
    def module_factory(cls,
                       X: Union[None, pd.DataFrame, np.ndarray],
                       output_dim: int) -> Tuple[torch.nn.Module, dict]:
        """
        Given a model-matrix and output-dim, produce a ``torch.nn.Module`` that predicts a distribution-parameter. The
        default produces a ``torch.nn.Linear`` layer. Additionally, this function returns a dictionary whose keys are
        param-names and whose values are the names of the individual param-elements. For the default case, each weight
        is named according to the column-names in X (or "x{i}" if X is not a dataframe).

        :param X: A dataframe or ndarray.
        :param output_dim: The number of output dimensions.
        :return: A tuple of (module, dict). The dictionary should correspond to ``module.named_parameters()``: the dict
         keys should be module param-names, and the dict values should be arrays (or coercible to arrays) of strings
         whose shape matches the parameter shapes.
        """
        if X is None or not X.shape[1]:
            module = NoWeightModule(dim=output_dim)
            columns = []
        else:
            module = torch.nn.Linear(in_features=X.shape[1], out_features=output_dim, bias=True)
            with torch.no_grad():
                module.weight.normal_(std=.1)
            columns = list(X.columns) if hasattr(X, 'columns') else [f'x{i}' for i in range(X.shape[1])]

        module_param_names = {'bias': [], 'weight': []}
        if output_dim == 1:
            # for most common case of 1d output, param names are just feature-names
            module_param_names['bias'].append('bias')
            module_param_names['weight'].append(columns)
        else:
            # if multi-output, we prefix with output idx:
            for i in range(output_dim):
                module_param_names['bias'].append(f'y{i}__bias')
                module_param_names['weight'].append([f'y{i}__{c}' for c in columns])

        return module, module_param_names

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(
            self.module_.parameters(), max_iter=10, line_search_fn='strong_wolfe', lr=.25
        )

    def _init_pb(self) -> Optional[tqdm]:
        max_eval = self.optimizer_.param_groups[0].get('max_eval', False)
        if max_eval:
            return tqdm(total=self.optimizer_.param_groups[0]['max_eval'])
        return None

    @torch.no_grad()
    def predict(self,
                X: ModelMatrix,
                type: str = 'mean',
                kwargs_as_is: Union[bool, dict] = False,
                **kwargs) -> np.ndarray:
        """
        Get the predicted distribution, then extract an attribute from that distribution and return as an ndarray.

        :param X: An array or dictionary of arrays.
        :param type: The type of the prediction -- i.e. the attribute to be extracted from the resulting
         ``torch.Distribution``. The default is 'mean'. If that attribute is callable, will be called with ``kwargs``.
        :param kwargs_as_is: If the ``type`` is callable, then kwargs that are lists/arrays will be converted to tensors, and
         if 1D then will be unsqueezed to 2D (unsqueezing is to avoid accidental broadcasting, wherein (e.g.) a
         distribution with batch_shape of N*1 receives a ``value`` w/shape (N,1), resulting in a (N,N) tensor, and
         usually an OOM error). Can be ``True``, or can pass a dict to set true/false on per-kwarg basis.
        :param kwargs: Keyword arguments to pass if ``type`` is callable. See ``kwargs_as_is``.
        :return: A ndarray of predictions.
        """
        Xdict, *_ = self._build_model_mats(X=X, y=None)
        dist_kwargs = self._get_dist_kwargs(**Xdict)
        dist = self.family(**dist_kwargs)
        result = getattr(dist, type)
        if callable(result):
            if not isinstance(kwargs_as_is, dict):
                kwargs_as_is = {k: kwargs_as_is for k in kwargs}
            for k in list(kwargs):
                if not kwargs_as_is.get(k, False) and isinstance(kwargs[k], (torch.Tensor, np.ndarray, pd.Series)):
                    kwargs[k] = to_2d(to_tensor(kwargs[k], **get_to_kwargs(self.module_)))
            result = result(**kwargs)
        elif kwargs:
            warn(f"Ignoring {set(kwargs)}, `{dist.__class__.__name__}.{type}` not callable.")
        return result.numpy()

    def _get_penalty(self) -> torch.Tensor:
        """
        Get penalty on sum(log_prob) based on the module-weights and self.penalty.
        """
        if not self.penalty:
            return torch.zeros(1, **get_to_kwargs(self.module_))

        # standardize to dictionary with param-names:
        if isinstance(self.penalty, dict):
            if set(self.penalty) != set(self.module_.keys()):
                raise ValueError(
                    f"``self.penalty.keys()`` is {set(self.penalty)}, but expected {set(self.module_.keys())}"
                )
        else:
            self.penalty = {k: self.penalty for k in self.module_.keys()}

        # standardize to values = callables:
        for param_name in list(self.penalty):
            maybe_callable = self.penalty[param_name]
            if not callable(maybe_callable):
                # if not callable, it's a multiplier --i.e. L2's 'precision'
                self.penalty[param_name] = L2(precision=maybe_callable)

        # call each:
        to_sum = []
        for param_name, penalty_fun in self.penalty.items():
            to_sum.append(
                penalty_fun(self.module_[param_name], self._module_param_names_[param_name])
            )
        return torch.stack(to_sum).sum()

    @property
    def coef_dataframe_(self) -> pd.DataFrame:
        if self._coef_mvnorm_ is None:
            raise RuntimeError("Must call ``estimate_laplace_coefs()`` first.")
        out = []
        with torch.no_grad():
            ses = self._coef_mvnorm_.covariance_matrix.diag().sqrt()
            for name, estimate, se in zip(self._laplace_coefs_names_, self._coef_mvnorm_.mean, ses):
                estimate = estimate.item()
                se = se.item()
                if not self.converged_:
                    se = float('nan')
                out.append({'name': name, 'estimate': estimate, 'se': se})
        return pd.DataFrame(out)

    def estimate_laplace_coefs(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None):
        self._laplace_coefs_names_, means, hess = self._estimate_laplace_coefs(X=X, y=y, sample_weight=sample_weight)

        # create mvnorm for laplace approx:
        with torch.no_grad():
            cov = torch.inverse(hess)
            try:
                self._coef_mvnorm_ = torch.distributions.MultivariateNormal(
                    means, covariance_matrix=cov, validate_args=True
                )
                self.converged_ = True
            except ValueError as e:
                if 'constraint PositiveDefinite' in str(e):
                    warn(f"Model failed to converge; `laplace_params` cannot be estimated. cov.diag():\n{cov.diag()}")
                    fake_cov = torch.eye(hess.shape[0]) * 1e-10
                    self._coef_mvnorm_ = torch.distributions.MultivariateNormal(means, covariance_matrix=fake_cov)
                    self.converged_ = False
                else:
                    raise e

    def _estimate_laplace_coefs(self,
                                X: ModelMatrix,
                                y: ModelMatrix,
                                sample_weight: Optional[np.ndarray] = None
                                ) -> Tuple[Sequence[str], torch.Tensor, torch.Tensor]:
        all_param_names = []
        all_params = []
        for dp in self.family.params:
            for nm, param_values in self.module_[dp].named_parameters():
                param_names = self._module_param_names_[dp][nm]
                assert param_names.shape == param_values.shape, f"param_names.shape!=param_values.shape for {dp}.{nm}"
                all_param_names.extend(f"{dp}__{pnm}" for pnm in param_names.reshape(-1))
                all_params.append(param_values)  # TODO: any way to assert reshape(-1) matches internals of hessian?
        means = torch.cat([p.reshape(-1) for p in all_params])

        mm_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        log_prob = self.get_log_prob(mm_dict=mm_dict, lp_dict=lp_dict, mean=False)
        hess = hessian(output=-log_prob, inputs=all_params, allow_unused=True, progress=False)

        return all_param_names, means, hess
