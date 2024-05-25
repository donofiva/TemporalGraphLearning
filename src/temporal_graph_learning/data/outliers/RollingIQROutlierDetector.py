import numpy as np
import pandas as pd

from typing import Union
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted


class RollingIQROutlierDetector(BaseEstimator, OutlierMixin):

    def __init__(
            self,
            lower_quantile=0.25,
            upper_quantile=0.75,
            window_size=10,
            width=1.5,
            center=False
    ):

        # Store IQR configuration
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.width = width

        # Store rolling statistics configuration
        self.window_size = window_size
        self.center = center

        # Parametrization
        self._inter_quantile_range = None
        self._lower_bound = None
        self._upper_bound = None

        # Sklearn interface
        self.is_fitted_ = False

    def fit(self, X: Union[pd.Series, np.ndarray], y=None):

        # Ensure X is a DataFrame, otherwise raise an error
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.Series):
            raise ValueError("Input must be a unidimensional NumPy array or a Pandas series.")

        # Covert series to DataFrame for simplicity
        X = pd.Series(X)

        # Compute rolling quantiles
        rolling_lower_quantile = X.rolling(window=self.window_size, center=self.center).quantile(self.lower_quantile)
        rolling_upper_quantile = X.rolling(window=self.window_size, center=self.center).quantile(self.upper_quantile)

        # Compute IQR and bounds
        self._inter_quantile_range = rolling_upper_quantile - rolling_lower_quantile
        self._lower_bound = rolling_lower_quantile - self.width * self._inter_quantile_range
        self._upper_bound = rolling_upper_quantile + self.width * self._inter_quantile_range

        # Flag estimator as fitted
        self.is_fitted_ = True

        return self

    def predict(self, X: Union[pd.Series, np.ndarray]):

        # Ensure X is a DataFrame, otherwise raise an error
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.Series):
            raise ValueError("Input must be a unidimensional NumPy array or a Pandas series.")

        # Check if the estimator is fitted
        check_is_fitted(self, 'is_fitted_')

        # Covert series to DataFrame for simplicity
        X = pd.Series(X)

        # Estimate outliers
        outliers = (X < self._lower_bound) | (X > self._upper_bound)

        # Return classification
        return np.where(outliers, -1, 1)

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X)
        return self.predict(X)