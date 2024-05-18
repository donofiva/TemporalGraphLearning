import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Any


class Dataset:

    # Transformations
    @staticmethod
    def get_missing_values_by_dimensions(
            dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame(
            dataframe.isna().sum(),
            columns=['MISSING_VALUES']
        )

    @staticmethod
    def slice_on_dimensions(
            dataframe: pd.DataFrame,
            dimensions: List[str],
            include_all: bool = False,
            drop_dimensions: bool = False
    ) -> List[Tuple[Any, pd.DataFrame]]:

        # Define dataframe slices
        slice_entities_to_dataframe = list(dataframe.groupby(dimensions, as_index=False))
        slice_entities_to_dataframe += int(include_all) * [(tuple(['ALL'] * len(dimensions)), dataframe)]

        # Convert dataframe slices to datasets
        return [
            (
                slice_entities,
                dataframe.drop(columns=dimensions if drop_dimensions else []).reset_index(drop=True)
            )
            for slice_entities, dataframe in slice_entities_to_dataframe
        ]

    @staticmethod
    def extend_with_lagged_dimensions_values(
            dataframe: pd.DataFrame,
            dimensions: List[str],
            lags: int,
            dimension_to_fill_value: Dict = None
    ) -> Tuple[List[str], pd.DataFrame]:

        # Check if the dimension exists in the dataframe
        if set(dimensions).difference(dataframe.columns):
            raise ValueError(f"Some dimensions not found in the DataFrame.")

        # Validate window_size is a positive integer
        if not isinstance(lags, int) or lags < 1:
            raise ValueError("Lags must be a positive integer.")

        # Copy dataframe to preserve initial structure
        dataframe = dataframe.copy()

        # Create new columns buffer
        lagged_columns = []

        # Parse fill values map
        if dimension_to_fill_value is None:
            dimension_to_fill_value = {}

        for dimension in dimensions:

            # Retrieve fill value for the current dimension
            fill_value = dimension_to_fill_value.get(dimension, np.nan)

            # Store lagged dimensions
            for lag in range(1, lags + 1):

                # Define and store new column
                lagged_column = f"{dimension}_LAG_{lag}"
                lagged_columns.append(lagged_column)

                # Create new column
                dataframe[lagged_column] = dataframe[dimension].shift(lag)
                dataframe.loc[:lag - 1, lagged_column] = fill_value

        return lagged_columns, dataframe

    @staticmethod
    def extend_with_leading_dimension_values(
            dataframe: pd.DataFrame,
            dimensions: List[str],
            leads: int
    ):

        # Check if the dimension exists in the dataframe
        if set(dimensions).difference(dataframe.columns):
            raise ValueError(f"Some dimensions not found in the DataFrame.")

        # Validate window_size is a positive integer
        if not isinstance(leads, int) or leads < 0:
            raise ValueError("Leads must be a non-negative integer.")

        # Copy dataframe to preserve initial structure
        dataframe = dataframe.copy()

        # Create new columns buffer
        leading_columns = []

        # Store leading dimensions
        for dimension in dimensions:

            # Create future target columns
            for lead in range(0, leads + 1):

                # Define new column name
                leading_column = f'{dimension}_LEAD_{lead}'
                leading_columns.append(leading_column)

                # Store new column
                dataframe[leading_column] = dataframe[dimension].shift(-lead)

            # Remove target dimension
            dataframe = dataframe.drop(columns=dimension)

        # Preserve only not-nan values
        dataframe = dataframe[~dataframe[leading_columns].isna().any(axis=1)]

        return leading_columns, dataframe
