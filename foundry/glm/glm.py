import os

from time import sleep
from typing import Union, Sequence, Optional, Callable, Tuple, Dict
from warnings import warn

import numpy as np
import pandas as pd

import torch
import torch.nn
from torch import distributions
from torch.distributions import transforms

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm

from foundry.covariance import Covariance
from foundry.hessian import hessian
from foundry.penalty import L2

from .family import Family, CeilingWeibull
from .family.survival import SurvivalFamily
from .util import NoWeightModule, Stopping, SigmoidTransformForClassification, Multinomial, SoftmaxKp1

from foundry.util import (
    FitFailedException, is_invalid, get_to_kwargs, to_tensor, to_2d, is_array, ModelMatrix, ToSliceDict, to_1d
)
from sklearn.exceptions import NotFittedError

N_FIT_RETRIES = int(os.getenv('GLM_N_FIT_RETRIES', 10))


class Glm(BaseEstimator):
    """
    :param family: A family name; you can see available names with ``Glm.family_names``. (Advanced: you can also pass
     the :class:`foundry.glm.family.Family` instead of a name). Some names can be prefixed with ``survival_`` for
     support for censored data.
    :param penalty: A multiplier for L2 penalty on coefficients. Can be a single value, or a dictionary of these to
     support different penalties per distribution-parameter. Values can either be floats, or can be functions that take
     a ``torch.nn.Module`` as the first argument and that module's param-names as second argument, and returns a scalar
     penalty that will be applied to the log-prob.
    :param col_mapping: Many distributions have multiple parameters: for example the normal distribution has a
     location and scale parameter. If a single dataframe/matrix is passed to  :class:`foundry.glm.Glm.fit`, the default
     behavior is to use this to predict the *first* parameter (e.g. loc) while other parameters (e.g. scale) have no
     predictors (i.e. are "intercept-only"). Sometimes this is not desired: for example, we want to use predictors to
     predict both. This can be either a list or a dictionary. For example, ``col_mapping=['loc','scale']`` will create a
      dict with the full model-matrix assigned to both params. A dictionary allows finer-grained control, e.g.:
     ``col_mapping={'loc':[col1,col2,col3],'scale':[col1]}``. Finally, instead of the dict-values being lists of
     columns, these can be functions that takes the data and return the relevant columns: e.g.
     ``col_mapping={'loc':sklearn.compose.make_column_selector('^col+.'), 'scale':[col1]}``.
    :param sparse_mm_threshold: Density threshold for creating a sparse model-matrix. If X has density less than this,
     the model-matrix will be sparse; otherwise it will be dense. Default .05.
     ``col_mapping={'loc':sklearn.compose.make_column_selector('^col.+'), 'scale':[col1]}``.
    """
    family_names = {
        'bernoulli': (
            distributions.Bernoulli,
            {'probs': SigmoidTransformForClassification()},
        ),
        'binomial': (
            distributions.Binomial,
            {'probs': transforms.SigmoidTransform()},
        ),
        'categorical': (
            distributions.Categorical,
            {'probs': SoftmaxKp1()}
        ),
        'multinomial': (
            Multinomial,
            {'probs': SoftmaxKp1()}
        ),
        'poisson': (
            distributions.Poisson,
            {'rate': transforms.ExpTransform()}
        ),
        'negative_binomial': (
            distributions.NegativeBinomial,
            {'probs': transforms.SigmoidTransform(), 'total_count': transforms.ExpTransform()},
            False
        ),
        'exponential': (
            torch.distributions.Exponential,
            {
                'rate': transforms.ExpTransform(),
            }
        ),
        'weibull': (
            torch.distributions.Weibull,
            {
                'scale': transforms.ExpTransform(),
                'concentration': transforms.ExpTransform()
            }
        ),
        'normal': (
            torch.distributions.Normal,
            {
                'loc': transforms.identity_transform,
                'scale': transforms.ExpTransform()
            }
        ),
        'multivariate_normal': (
            torch.distributions.MultivariateNormal,
            {
                'loc': transforms.identity_transform,
                'covariance_matrix': Covariance()
            }
        ),
        'ceiling_weibull': (
            CeilingWeibull,
            {
                'scale': transforms.ExpTransform(),
                'concentration': transforms.ExpTransform(),
                'ceiling': transforms.SigmoidTransform()
            }
        )
    }
    family_names['gaussian'] = family_names['normal']
    family_names['mvnorm'] = family_names['multivariate_normal']

    def __init__(self,
                 family: Union[str, Family],
                 penalty: Union[float, Sequence[float], Dict[str, float]] = 0.,
                 col_mapping: Union[list, dict, None] = None,
                 sparse_mm_threshold: float = .05):

        self.family = family
        self.penalty = penalty
        self.col_mapping = col_mapping
        self.sparse_mm_threshold = sparse_mm_threshold

        # set in _init_module:
        self._module_ = None
        self._module_param_names_ = None
        self.label_encoder_ = None
        self.to_slice_dict_ = None

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

    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            **kwargs) -> 'Glm':
        """
        :param X: A array/dataframe of predictors, or a dictionary of these. If a dict, then keys should correspond to
         ``self.family.params``.
        :param y: An array of targets. This can instead be a dictionary, with the target-value in the "value" entry,
         and additional auxiliary information (e.g. sample-weights, upper/lower censoring for survival modeling) in
         other entries.
        :param sample_weight: The weight for each row. If performing cross-validation, this argument should not be used,
         as sklearn does not support it; instead, pass a :class:`foundry.util.SliceDict` for the ``y`` argument, with a
         'sample_weight' entry.
        :param groups: todo
        :param reset: If calling ``fit()`` more than once, should the module/weights be reinitialized. Default True.
        :param callbacks: A list of callbacks: functions that take the ``Glm`` instance as a first argument and the
         train-loss as a second argument.
        :param stopping: Controls stopping based on converging loss/parameters. This argument is passed to
         :class:`foundry.glm.util.Stopping` (e.g. ``(.01,)`` would use abstol of .01).
        :param max_iter: The max. number of iterations before stopping training regardless of convergence. Default 200.
        :param max_loss: If training stops and loss is higher than this, a class:`foundry.util.FitFailedException` will
         be raised and fitting will be retried with a different set of inits.
        :param verbose: Whether to allow print statements and a progress bar during training. Default True.
        :param estimate_laplace_coefs: If true, then after fitting, the hessian of the optimized parameters will be
         estimated; this can then be used for confidence-intervals and statistical inference (see ``coef_dataframe_``).
         Can set to False if you want to save time and skip this step.
        :return: This ``Glm`` instance.
        """
        self.family = self._init_family(self.family)

        if self.family.supports_predict_proba:
            # sklearn uses `hasattr` in some cases to check for `predict_proba`, so can't just
            # have it error out for some distributions -- need to add it dynamically.
            # noinspection PyAttributeOutsideInit
            self.predict_proba = self._predict_proba

        if isinstance(self.penalty, (tuple, list)):
            raise NotImplementedError  # TODO
        else:
            if groups is not None:
                warn("`groups` argument will be ignored because self.penalty is a single value not a sequence.")
            return self._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

    def _predict_proba(self,
                       X: ModelMatrix,
                       kwargs_as_is: Union[bool, dict] = False,
                       **kwargs) -> np.ndarray:
        """
        Return the probability of each class.
        """
        assert self.family.supports_predict_proba
        probs = self.predict(X=X, type='probs', kwargs_as_is=kwargs_as_is, **kwargs)
        assert len(probs.shape) == 2

        # pytorch distributions can vary in behavior: e.g. bernoulli outputs a single prediction for 2-classes,
        # while categorical/multinomial output a final dim whose extent is equal to the number of classes
        # so bernoulli/binomial breaks sklearn convention and we need to add a 2nd col
        if probs.shape[-1] == 1:
            if self.label_encoder_ is not None:
                expected_num_classes = len(self.label_encoder_.classes_)
                if expected_num_classes != 2:
                    raise RuntimeError(
                        f"There are {expected_num_classes} ``self.label_encoder_.classes_``, but distribution "
                        f"``probs.shape[-1]`` is {probs.shape[-1]}"
                    )
            # currently no way to get `expected_num_classes` if label_encoder_ is not present (afaik, only binomial)
            assert np.all((probs >= 0) & (probs <= 1))
            probs = np.concatenate([1 - probs, probs], axis=1)
            probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def _init_family(self, family: Union[Family, str]) -> Family:
        """ if family is a string, turns family to a Family. else return unchanged. """
        if isinstance(family, str):
            if family.startswith('survival'):
                family = family.replace('survival', '').lstrip('_')
                return SurvivalFamily(*self.family_names[family])
            else:
                return Family(*self.family_names[family])
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
             max_loss: float = np.inf,
             verbose: bool = True,
             estimate_laplace_coefs: bool = True) -> 'Glm':

        if verbose:
            print("Estimating model parameters")

        # tallying fit-failurs:
        if self._fit_failed:
            if verbose:
                print(f"Retry attempt {self._fit_failed}/{N_FIT_RETRIES}")
            reset = True

        # build model:
        if self._module_ is None or reset:
            if self._module_ is not None and verbose:
                warn("Resetting module with reset=True")
            # initialize module:
            self._init_module(X, y)

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
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)

        # progress bar:
        prog = None
        if verbose:
            prog = self._init_pb()

        epoch = 0

        # 'closure' for torch.optimizer.Optimizer.step method:
        def closure():
            self.optimizer_.zero_grad()
            loss = -self.get_log_prob(x_dict, lp_dict)
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
            except ValueError as e:
                raise FitFailedException() from e
            finally:
                self.optimizer_.zero_grad(set_to_none=True)

        if train_loss > max_loss:
            self._fit_failed += 1
            raise FitFailedException(f"train_loss ({train_loss}) too high, consider increasing max_loss ({max_loss})")

        self._fit_failed = 0

        if verbose:
            print("Estimating model parameters success!")

        if estimate_laplace_coefs:
            if verbose:
                print("Estimating laplace coefs... (you can safely keyboard-interrupt to cancel)")
            try:
                self.estimate_laplace_coefs(X=X, y=y, sample_weight=sample_weight)
            except KeyboardInterrupt:
                if verbose:
                    print("Estimation laplace coefs interrupted")

        if estimate_laplace_coefs and verbose:
            if self.converged_:
                print("Estimating laplace coefs success!")
            else:
                print("Estimating laplace coefs unsuccessful! See warnings / errors.")

        return self

    def get_log_prob(self,
                     x_dict: dict,
                     lp_dict: dict,
                     mean: bool = True,
                     include_penalty: bool = True) -> torch.Tensor:
        """
        Get the penalized log prob, applying weights as needed.
        """
        log_probs = self.family.log_prob(
            self.family(**self._get_family_kwargs(**x_dict)),  # distribution(ilink(X \theta))
            **lp_dict
        )
        penalty = self._get_penalty() if include_penalty else 0.
        log_prob = log_probs.sum() - penalty
        if mean:
            log_prob = log_prob / lp_dict['weight'].sum()
        return log_prob

    def _get_family_kwargs(self, **kwargs) -> dict:
        """
        Call each entry in the module_ dict to get predicted family params. Any leftover keywords are passed as-is.
        """
        out = {}
        for dp, dp_module in self.module_.items():
            mm = kwargs.pop(dp, None)
            if isinstance(dp_module, NoWeightModule):
                if mm is not None and mm.numel():
                    raise RuntimeError(f"When fitted, no predictors were passed for {dp}, so can't pass them now.")
                out[dp] = dp_module(mm)
            else:
                out[dp] = dp_module(mm)
        out.update(kwargs)  # remaining are passthru
        return out

    def _get_xdict(self, X: ModelMatrix, sparse_threshold: float) -> Dict[str, torch.Tensor]:
        _to_kwargs = get_to_kwargs(self.module_)

        Xdict = self.to_slice_dict_.transform(X)

        # convert to tensors:
        for nm in list(Xdict):
            if is_array(Xdict[nm]):
                Xdict[nm] = to_tensor(Xdict[nm], sparse_threshold=sparse_threshold, **_to_kwargs)

            if nm in self.family.params:
                # model-mat params
                assert len(Xdict[nm].shape) == 2, f"len(X['{nm}'].shape)!=2"
                if is_invalid(Xdict[nm]):
                    raise ValueError(f"nans/infs in X ({nm})")

        return Xdict

    def _get_ydict(self, y: ModelMatrix, sample_weight: Optional[np.ndarray]) -> Dict[str, torch.Tensor]:
        # Checks of arguments
        if isinstance(y, dict) and 'weight' in y.keys() and sample_weight is not None:
            raise ValueError("Please pass either `sample_weight` or y=dict(weight=), but not both.")
        if isinstance(y, dict) and 'value' not in y.keys():
            raise ValueError("y is dict but does not contain 'value' matrix")

        # Define the ydict here
        if isinstance(y, dict):
            ydict = y.copy()
        else:
            ydict = {'value': y}

        y_len = ydict['value'].shape[0]

        ydict['weight'] = ydict.get(
            'weight',
            sample_weight if sample_weight is not None else torch.ones(y_len)
        )
        assert (ydict['weight'] >= 0).all()

        _to_kwargs = get_to_kwargs(self.module_)

        # Checks on ydict before return
        for k in list(ydict):
            if not hasattr(ydict[k], '__iter__'):
                # wait until we have a use-case
                raise NotImplementedError(
                    f"Unclear how to handle {k}, please report this error to the package maintainer"
                )

            # special handling for classification:
            if k == 'value' and self.label_encoder_ is not None:
                ydict['value'] = self.label_encoder_.transform(to_1d(ydict['value']))
                ydict['value'] = to_tensor(ydict['value'], **_to_kwargs)
                continue

            # standard handling:
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

        Xdict = self._get_xdict(X, sparse_threshold=self.sparse_mm_threshold)

        if not include_y:
            return Xdict, None

        if y is None:
            raise ValueError("Must pass `y` when fitting.")

        ydict = self._get_ydict(y=y, sample_weight=sample_weight)

        if Xdict and list(Xdict.values())[0].shape[0] != ydict['value'].shape[0]:
            raise ValueError("Expected X.shape[0] to match y.shape[0]")

        return Xdict, ydict

    def _init_module(self, X: ModelMatrix, y: ModelMatrix) -> torch.nn.ModuleDict:
        # memorize the col -> dist-param mapping
        self._init_col_mapping(X)

        # standardize X and y:
        X = self.to_slice_dict_.transform(X)
        if isinstance(y, dict):
            y = y['value']
        y = to_2d(np.asanyarray(y))

        # if classification, handle label encoding:
        self.label_encoder_ = None
        if self.family.supports_predict_proba and not self.family.has_total_count:
            if y.shape[-1] != 1:
                raise ValueError(
                    f"GLM w/family {self.family} expects a 1D ``y`` whose values are the class-labels. However,"
                    f"y.shape is {y.shape}."
                )
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(to_1d(y))
            assert len(self.label_encoder_.classes_) > 1

        # create modules that predict params:
        self._module_param_names_ = {}
        self.module_ = torch.nn.ModuleDict()
        for dp in self.family.params:
            Xp = X.get(dp, None)
            module_num_outputs = self._get_module_num_outputs(y, dp)
            if Xp is None or not Xp.shape[1]:
                # always use the base approach when no input features:
                module, nms = Glm.module_factory(X=None, output_dim=module_num_outputs)
            else:
                module, nms = self.module_factory(X=Xp, output_dim=module_num_outputs)
            self.module_[dp] = module
            self._module_param_names_[dp] = {k: np.asarray(v) for k, v in nms.items()}

    def _init_col_mapping(self, X: ModelMatrix):
        col_mapping = self.col_mapping
        if col_mapping is None and not isinstance(X, dict):
            # if col_mapping is None and they didn't pass a dict for X, default is use first family param
            # otherwise use col_mapping as-is (if X is a dict, will be used by ToSliceDict)
            col_mapping = [self.family.params[0]]
        self.to_slice_dict_ = ToSliceDict(mapping=col_mapping)
        self.to_slice_dict_.fit(X)

    def _get_module_num_outputs(self, y: np.ndarray, dist_param_name: str) -> int:
        assert len(y.shape) == 2
        if self.label_encoder_ is not None:
            # classification
            y_dim = len(self.label_encoder_.classes_)
        else:
            # regression:
            y_dim = y.shape[1]

        # for most distribution-params, the number of outputs we need to predict is dictated by the structure of `y`:
        # we need to predict one value if `y` is (n, 1), two values if `y` is (n, 2), etc.
        # however, there are exceptions:
        # - parameters of a covariance-matrix
        # - parameters of a categorical distribution (k-1 classes)
        # the way this is implemented is by having that inverse-link-function also have a `get_param_dim` method
        # which takes the shape of y and returns the number of parameters needed
        ilink = self.family.params_and_links[dist_param_name]
        get_param_dim = getattr(ilink, 'get_param_dim', None)
        if get_param_dim:
            return get_param_dim(y_dim)
        else:
            return y_dim

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
            return tqdm(total=max_eval)
        return None

    @torch.no_grad()
    def score(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Uses log_prob (without penalty) for scoring.
        """
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        return self.get_log_prob(x_dict, lp_dict, mean=True, include_penalty=False).item()

    @torch.no_grad()
    def predict(self,
                X: ModelMatrix,
                type: Optional[str] = None,
                kwargs_as_is: Union[bool, dict] = False,
                **kwargs) -> np.ndarray:
        """
        Get the predicted distribution, then extract an attribute from that distribution and return as an ndarray.

        :param X: An array or dictionary of arrays.
        :param type: The type of the prediction -- i.e. the attribute to be extracted from the resulting
         ``torch.Distribution``. The default depends on the family. If ``self.family.supports_predict_proba``, and the
         distribution doesn't have a ``total_count`` parametere (e.g. multinomial), then this method will predict the
         class. Otherwise, this distribution will predict the mean of the distribution.
        :param kwargs_as_is: If the ``type`` is callable, then kwargs that are arrays will be converted to tensors,
         and if 1D then will be unsqueezed to 2D (unsqueezing is to avoid accidental broadcasting, wherein (e.g.) a
         distribution with batch_shape of N*1 receives a ``value`` w/shape (N,1), resulting in a (N,N) tensor,
         and usually an OOM error). Can be ``True``, or can pass a dict to set true/false on per-kwarg basis.
        :param kwargs: Keyword arguments to pass if ``type`` is callable. See ``kwargs_as_is``.
        :return: A ndarray of predictions.
        """
        if type is None:
            if self.label_encoder_ is not None:
                probs = self.predict_proba(X, kwargs_as_is=kwargs_as_is, **kwargs)
                class_idxs = probs.argmax(axis=1)
                return self.label_encoder_.inverse_transform(to_1d(class_idxs))
            else:
                type = 'mean'

        x_dict, *_ = self._build_model_mats(X=X, y=None)
        if 'validate_args' in kwargs:
            x_dict['validate_args'] = kwargs.pop('validate_args')
        dist_kwargs = self._get_family_kwargs(**x_dict)
        dist = self.family(**dist_kwargs)
        result = getattr(dist, type)
        if callable(result):
            if not isinstance(kwargs_as_is, dict):
                kwargs_as_is = {k: kwargs_as_is for k in kwargs}
            for k in list(kwargs):
                if not kwargs_as_is.get(k, False) and is_array(kwargs[k]):
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

            self.converged_ = False

            try:
                # cov = torch.inverse(hess)  # Hello floating point errors my old friend.
                cholesky_hess = torch.linalg.cholesky(hess)  # This is way more numerically stable
                inv_cholesky_hess = torch.linalg.inv(cholesky_hess)
                cov = inv_cholesky_hess.T @ inv_cholesky_hess

                # if `hess` has one eigenvalue of zero, then the matrix will be non-invertible. This usually means one
                # of the features is constant, or is completely colinear with another feature.

                # if `hess` is poorly conditioned, but is positive definite (PD), it may not remain PD after inversion.
                # Using the Cholesky decomposition helps, but can still cause issues

                # if `hess` is negative definite (ND), the model hasn't converged. Consider looking at the stopping
                # criterion.

                self._coef_mvnorm_ = torch.distributions.MultivariateNormal(
                    means, covariance_matrix=cov, validate_args=True
                )

                self.converged_ = True

            except RuntimeError as e:
                warn(f"RunTimeError(\"{(str(e))}\")")
                warn("Second order estimation of parameter distribution failed. Constructing approximate covariance.")
                fake_cov = torch.diag(torch.diag(hess).pow(-1).clip(min=1E-5))
                self._coef_mvnorm_ = torch.distributions.MultivariateNormal(means, covariance_matrix=fake_cov)

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

        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        log_prob = self.get_log_prob(x_dict=x_dict, lp_dict=lp_dict, mean=False)
        hess = hessian(output=-log_prob.squeeze(), inputs=all_params, allow_unused=True, progress=False)

        return all_param_names, means, hess
