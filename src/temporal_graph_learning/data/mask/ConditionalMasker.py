import numpy as np
import pandas as pd

from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class MaskingComparisonType(Enum):

    GREATER = 0
    LESS = 1
    EQUAL = 2
    NOT_EQUAL = 3
    IN = 4
    NOT_IN = 5

    def compare(self, target, value):
        return {
            MaskingComparisonType.GREATER: lambda x, y: x > y,
            MaskingComparisonType.LESS: lambda x, y: x < y,
            MaskingComparisonType.EQUAL: lambda x, y: x == y,
            MaskingComparisonType.NOT_EQUAL: lambda x, y: x != y,
            MaskingComparisonType.IN: lambda x, y: x.isin(y),
            MaskingComparisonType.NOT_IN: lambda x, y: ~x.isin(y),
        }[self](target, value)


@dataclass
class MaskingComparison:

    dimension: str
    type: MaskingComparisonType
    value: Any


@dataclass
class MaskingCondition:

    dimensions: Union[str, List[str]]
    comparisons: List[MaskingComparison]


class ConditionalMasker(BaseEstimator, TransformerMixin):

    def __init__(self, masking_conditions: List[MaskingCondition] = None):
        self._masking_conditions = masking_conditions or set()

    @staticmethod
    def _parse_masking_condition_dimensions(X, dimensions: Union[str, List[str]]):

        # Select all the dimensions on __ALL__ wildcard
        if dimensions == "__ALL__":
            return X.columns

        # Otherwise return dimensions
        return dimensions

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Make sure that the input is a Pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Create a copy of the DataFrame to avoid modifying the original data
        X = X.copy()

        # Apply masking conditions iteratively
        for condition in self._masking_conditions:

            # Create a mask for all comparisons related to the current condition
            conditional_mask = pd.Series([True] * X.shape[0], index=X.index)

            # Evaluate all comparisons related to the current condition and accumulate masking conditions
            for comparison in condition.comparisons:
                conditional_mask &= comparison.type.compare(X[comparison.dimension], comparison.value)

            # Apply the combined mask to the target column
            X.loc[conditional_mask, self._parse_masking_condition_dimensions(X, condition.dimensions)] = np.nan

        return X
