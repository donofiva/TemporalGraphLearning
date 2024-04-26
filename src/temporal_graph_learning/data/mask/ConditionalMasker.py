import numpy as np
import pandas as pd

from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class MaskingComparison(Enum):

    GREATER = 0
    LESS = 1
    EQUAL = 2
    NOT_EQUAL = 3
    IN = 4
    NOT_IN = 5

    def compare(self, target, value):
        return {
            MaskingComparison.GREATER: lambda x, y: x > y,
            MaskingComparison.LESS: lambda x, y: x < y,
            MaskingComparison.EQUAL: lambda x, y: x == y,
            MaskingComparison.NOT_EQUAL: lambda x, y: x != y,
            MaskingComparison.IN: lambda x, y: x.isin(y),
            MaskingComparison.NOT_IN: lambda x, y: ~x.isin(y),
        }[self](target, value)


@dataclass
class MaskingCondition:

    dimension: str
    comparison: MaskingComparison
    value: Any


@dataclass
class MaskingRule:

    dimensions: Union[str, List[str]]
    conditions: List[MaskingCondition]


class ConditionalMasker(BaseEstimator, TransformerMixin):

    def __init__(self, masking_rules: List[MaskingRule] = None):
        self._masking_rules = masking_rules or set()

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
        for masking_rule in self._masking_rules:

            # Create a mask for all comparisons related to the current condition
            mask = pd.Series([True] * X.shape[0], index=X.index)

            # Evaluate all comparisons related to the current condition and accumulate masking conditions
            for masking_condition in masking_rule.conditions:
                mask &= masking_condition.comparison.compare(
                    X[masking_condition.dimension],
                    masking_condition.value
                )

            # Apply the combined mask to the target column
            X.loc[mask, self._parse_masking_condition_dimensions(X, masking_rule.dimensions)] = np.nan

        return X
