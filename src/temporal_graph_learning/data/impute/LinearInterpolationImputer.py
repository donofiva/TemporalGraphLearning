import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin


class LinearInterpolationImputer(BaseEstimator, TransformerMixin):

    @staticmethod
    def _linear_interpolation(series: pd.Series):

        # Store actual values
        actual_values = ~series.isnull()

        # Define interpolation method
        linear_interpolation = interp1d(
            series[actual_values].index,
            series[actual_values],
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False
        )

        # Perform interpolation on buffer series
        return pd.Series(linear_interpolation(series.index), index=series.index)

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Initial input type detection
        X_type = type(X)

        # Apply interpolation
        if isinstance(X, list):
            X = self._linear_interpolation(pd.Series(X))

        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X = self._linear_interpolation(pd.Series(X))

        elif isinstance(X, pd.Series):
            X = self._linear_interpolation(pd.Series(X))

        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = pd.DataFrame(X).apply(self._linear_interpolation)

        elif isinstance(X, pd.DataFrame):
            X = X.apply(self._linear_interpolation)

        # Convert back to original input type
        if X_type in {list, np.ndarray}:
            return X.to_numpy()

        return X

