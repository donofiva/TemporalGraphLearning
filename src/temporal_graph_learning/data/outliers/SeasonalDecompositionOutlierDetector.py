import numpy as np
import pandas as pd

from typing import Union
from sklearn.base import BaseEstimator, OutlierMixin
from statsmodels.tsa.seasonal import seasonal_decompose


class SeasonalDecompositionOutlierDetector(BaseEstimator, OutlierMixin):

    def __init__(self, period: int, distance: int):

        # Store seasonal decomposer configuration
        self._period = period
        self._distance = distance

        # Sklearn interface
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True

    def _detect_outliers(self, X: Union[np.ndarray, pd.Series]):

        # Store seasonal decomposition
        seasonal_decomposition = seasonal_decompose(
            X.interpolate(method='linear').ffill().bfill().values,
            model='additive',
            period=self._period
        )

        # Extract residuals
        residuals = seasonal_decomposition.resid

        # Extract threshold
        outlier_threshold = self._distance * np.nanstd(residuals)
        outliers = np.abs(residuals) > outlier_threshold

        # Here format outliers
        return np.where(outliers, 1, -1)

    def predict(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:

        # Parse input
        if isinstance(X, pd.DataFrame):
            channels = [X[column] for column in X.columns]

        elif isinstance(X, np.ndarray) or isinstance(X, pd.Series):
            channels = [pd.Series(X)]

        else:
            raise Exception("Unsupported input type.")

        # Perform seasonal decomposition
        return np.array([
            self._detect_outliers(channel)
            for channel in channels
        ]).T

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X)
        return self.predict(X)
