import os
from time import sleep
from typing import Union, Sequence, Optional, Callable, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import torch

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm

from foundry.glm.family import Family
from foundry.glm.family.survival import SurvivalFamily
from foundry.util import FitFailedException, is_invalid, get_to_kwargs, to_tensor, to_2d

ModelMatrix = Union[np.ndarray, pd.DataFrame, dict]

N_FIT_RETRIES = int(os.getenv('GLM_N_FIT_RETRIES', 10))

from sklearn.exceptions import NotFittedError


class Glm:

    def __init__(self,
                 family: Union[str, Family],
                 penalty: Union[float, Sequence[float]] = 0.):
        self.family = family
        self.penalty = penalty

        # set in _init_module:
        self._module_ = None
        self.expected_model_mat_params_ = None

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
        if self._expected_model_mat_params_ is None:
            raise NotFittedError("Tried to access `expected_model_mat_params_` prior to fitting.")
        return self._expected_model_mat_params_

    @expected_model_mat_params_.setter
    def expected_model_mat_params_(self, value: Sequence[str]):
        self._expected_model_mat_params_ = value

    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            **kwargs) -> 'Glm':
        """
        :param X:
        :param y:
        :param sample_weight:
        :param groups:
        :param reset:
        :param callbacks:
        :param tol:
        :param patience:
        :param max_iter:
        :param max_loss:
        :param verbose:
        :return:
        """
        self.family = self._init_family(self.family)

        if isinstance(self.penalty, (tuple, list, np.ndarray)):
            raise NotImplementedError  # TODO
        else:
            if groups is not None:
                warn("`groups` argument will be ignored because self.penalty is a single value not a sequence.")
            return self._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

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
             reset: bool = False,
             callbacks: Sequence[Callable] = (),
             tol: float = .0001,
             patience: int = 2,
             max_iter: int = 200,
             max_loss: float = 10.,
             verbose: bool = True) -> 'Glm':

        # tallying fit-failurs:
        if self._fit_failed:
            if verbose:
                print(f"Retry attempt {self._fit_failed}/{N_FIT_RETRIES}")
            reset = True

        # build model:
        if self._module_ is None or reset:
            self.module_ = self._init_module(X, y)

        # optimizer:
        self.optimizer_ = self._init_optimizer()

        # build model-mats:
        mm_dict, lp_dict = self._build_model_mats(X, y, sample_weight, expect_y=True)

        # progress bar:
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

            prog.update()
            prog.set_description(f'Epoch: {epoch}; Loss: {loss:.4f}')
            return loss

        # fit-loop:
        assert max_iter > 0
        prev_train_loss = float('inf')
        num_lower = 0
        for epoch in range(max_iter):
            try:
                prog.reset()
                train_loss = self.optimizer_.step(closure).item()
                for callback in callbacks:
                    callback(self, train_loss)
                if abs(train_loss - prev_train_loss) < tol:
                    num_lower += 1
                else:
                    num_lower = 0
                if num_lower == patience:
                    break
                prev_train_loss = train_loss
            except KeyboardInterrupt:
                sleep(1)
                break
            finally:
                self.optimizer_.zero_grad(set_to_none=True)

        if train_loss > max_loss:
            self._fit_failed += 1
            raise FitFailedException(f"train_loss ({train_loss}) too high")

        self._fit_failed = 0
        return self

    def get_log_prob(self, mm_dict: dict, lp_dict: dict, reduce: bool = True) -> torch.Tensor:
        log_probs = self.family.log_prob(self.family(**self._get_dist_kwargs(**mm_dict)), **lp_dict)
        penalty = self._get_penalty()
        if reduce:
            log_probs = (log_probs * lp_dict['weight']).sum() + penalty
            log_probs = log_probs / lp_dict['weight']
        return log_probs

    def _get_penalty(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_dist_kwargs(self, **kwargs) -> dict:
        """
        Call each entry in the module_ dict to get predicted family params.
        """
        raise NotImplementedError

    def _build_model_mats(self,
                          X: ModelMatrix,
                          y: Optional[ModelMatrix],
                          sample_weight: Optional[np.ndarray],
                          expect_y: bool = False) -> Tuple[dict, dict]:
        """
        :param X: A dataframe/ndarray/tensor, or dictionary of these.
        :param y: An optional dataframe/ndarray/tensor, or dictionary of these.
        :param sample_weight: Optional sample-weights
        :param expect_y: If True, then will raise if y is not present.
        :return: Two dictionaries: one of model-mat kwargs (for prediction) and one of target-related kwargs
         (for evaluation i.e. log-prob).
        """
        _to_kwargs = get_to_kwargs(self.module_)

        # convert to dict:
        if isinstance(X, dict):
            Xdict = X.copy()
        else:
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

        if not expect_y:
            return Xdict, {}

        # handle target:
        if y is None:
            raise ValueError("Must pass `y` when fitting.")
        else:
            if isinstance(y, dict):
                assert 'value' in y
                ydict = y.copy()
            else:
                ydict = {'value': y}

        y_len = ydict['value'].shape[0]

        if x_len != y_len:
            raise ValueError("Expected X.shape[0] to match y.shape[0]")

        if 'weight' in ydict:
            if sample_weight is not None:
                raise ValueError("Please pass either `sample_weight` or y=dict(weight=), but not both.")
        else:
            ydict['weight'] = sample_weight

        if ydict.get('weight', None) is None:
            ydict['weight'] = torch.ones(y_len)
        else:
            assert (ydict['weight'] >= 0).all()

        for k in list(ydict):
            if not hasattr(ydict[k], '__iter__'):
                # wait until we have a use-case
                raise NotImplementedError(
                    f"Unclear how to handle {k}, please report this error to the package maintainer"
                )
            ydict[k] = to_2d(to_tensor(ydict[k], **_to_kwargs))
            if is_invalid(ydict[k]):
                raise ValueError(f"nans/infs in {k}")
            if ydict[k].shape[0] != y_len:
                raise ValueError(f"{k}.shape[0] does not match y.shape[0]")

        return Xdict, ydict

    def _init_module(self, X: ModelMatrix, y: ModelMatrix) -> torch.nn.ModuleDict:
        raise NotImplementedError

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
                avoid_broadcast_1d: bool = True,
                **kwargs) -> np.ndarray:
        raise NotImplementedError
