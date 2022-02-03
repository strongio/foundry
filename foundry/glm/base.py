import os
from time import sleep
from typing import Union, Sequence, Optional, Callable, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from foundry.glm.distribution import GlmDistribution
from foundry.glm.distribution.survival import SurvivalGlmDistribution
from foundry.util import SliceDict, FitFailedException, is_invalid, get_to_kwargs, to_tensor

ModelMatrix = Union[np.ndarray, pd.DataFrame, SliceDict]

N_FIT_RETRIES = int(os.getenv('GLM_N_FIT_RETRIES', 10))


class Glm(BaseEstimator):

    def __init__(self,
                 distribution: Union[str, GlmDistribution],
                 penalty: Union[float, Sequence[float]] = 0.):
        self.distribution = distribution
        self.penalty = penalty
        self.model_ = None
        self._fit_failed = 0

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
        :return: A dictionary of model-mat kwargs (for prediction) and of target-related kwargs
         (for evaluation i.e. log-prob).
        """
        _to_kwargs = get_to_kwargs(self.model_)

        # convert to dict:
        if isinstance(X, dict):
            Xdict = X
        else:
            Xdict = {p: X for p in self.expected_model_mat_params_}

        # convert to tensors:
        for nm in list(Xdict):
            if nm not in self.expected_model_mat_params_:
                raise RuntimeError(
                    f"In original call to `fit()`, did not get predictors for param `{nm}`, but now that's included."
                )
            Xdict[nm] = to_tensor(Xdict[nm], **_to_kwargs)
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

        if sample_weight is None:
            sample_weight = torch.ones(y_len)

        assert (sample_weight >= 0).all()
        ydict['weight'] = sample_weight

        for k in list(ydict):
            if hasattr(ydict[k], '__iter__'):  # TODO: less hacky way to do this?
                ydict[k] = to_tensor(ydict[k])
                if is_invalid(ydict[k]):
                    raise ValueError(f"nans/infs in {k}")
                if ydict[k].shape[0] != y_len:
                    raise ValueError(f"{k}.shape[0] does not match y.shape[0]")

        return Xdict, ydict

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
        self.distribution = self._init_distribution(self.distribution)

        if isinstance(self.penalty, (tuple, list, np.ndarray)):
            raise NotImplementedError  # TODO
        else:
            if groups is not None:
                warn("`groups` argument will be ignored because self.penalty is a single value not a sequence.")
            return self._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

    @staticmethod
    def _init_distribution(distribution: Union[GlmDistribution, str]) -> GlmDistribution:
        if isinstance(distribution, str):
            if distribution.startswith('survival'):
                distribution = SurvivalGlmDistribution.from_name(distribution.replace('survival', '').lstrip('_'))
            else:
                distribution = GlmDistribution.from_name(distribution)
        return distribution

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
        if self.model_ is None or reset:
            self.model_ = self._build_model(X, y, sample_weight=sample_weight)

        # build model-mats:
        mm_dict, lp_dict = self._build_model_mats(X, y, sample_weight, expect_y=True)

        # optimizer:
        self.optimizer_ = self._initialize_optimizer()

        # progress bar:
        prog = self._initialize_pb()

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
        log_probs = self.distribution.log_prob(self.distribution(**self._get_dist_kwargs(**mm_dict)), **lp_dict)
        penalty = self._get_penalty()
        if reduce:
            log_probs = (log_probs * lp_dict['weight']).sum() + penalty
            log_probs = log_probs / lp_dict['weight']
        return log_probs

    def _get_dist_kwargs(self, **kwargs) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self,
                X: ModelMatrix,
                type: str = 'mean',
                avoid_broadcast_1d: bool = True,
                **kwargs) -> np.ndarray:
        raise NotImplementedError
