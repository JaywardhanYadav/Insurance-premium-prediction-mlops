"""Ensemble regressor combining XGBRegressor and LGBMRegressor."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Ensemble estimator that averages predictions from an XGBoost and a LightGBM regressor.

    Parameters
    ----------
    xgb_model : xgb.XGBRegressor, default=None
        Fitted or un-fitted XGBoost regressor model.
    lgb_model : lgb.LGBMRegressor, default=None
        Fitted or un-fitted LightGBM regressor model.
    xgb_weight : float, default=0.5
        Weight assigned to the XGBoost predictions. The LightGBM model will receive
        a weight of ``1.0 - xgb_weight``.
    """

    def __init__(
        self,
        xgb_model: Optional[xgb.XGBRegressor] = None,
        lgb_model: Optional[lgb.LGBMRegressor] = None,
        xgb_weight: float = 0.5,
    ) -> None:
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.xgb_weight = xgb_weight

    def fit(self, X: Any, y: Any) -> EnsembleRegressor:
        """Fit both underlying models.

        Note: In our pipeline, we fit the models externally (with early stopping) and
        pass the fitted instances to the constructor. This fit method is implemented
        for scikit-learn compliance.
        """
        if self.xgb_model is not None:
            self.xgb_model.fit(X, y)
        if self.lgb_model is not None:
            self.lgb_model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression target by averaging model predictions.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions.
        """
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Both xgb_model and lgb_model must be initialized before predicting.")

        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)

        lgb_weight = 1.0 - self.xgb_weight
        return self.xgb_weight * xgb_pred + lgb_weight * lgb_pred

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "xgb_weight": self.xgb_weight,
        }

    def set_params(self, **params: Any) -> EnsembleRegressor:
        """Set parameters for this estimator."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
