import pandas as pd

from typing import List, Tuple, Any


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
            include_all: bool = False
    ) -> List[Tuple[Any, pd.DataFrame]]:

        # Define dataframe slices
        slice_entities_to_dataframe = list(dataframe.groupby(dimensions, as_index=False))
        slice_entities_to_dataframe += int(include_all) * [(tuple(['ALL'] * len(dimensions)), dataframe)]

        # Convert dataframe slices to datasets
        return [
            (slice_entities, dataframe.reset_index(drop=True))
            for slice_entities, dataframe in slice_entities_to_dataframe
        ]

    @staticmethod
    def extend_with_shifted_dimension(
            dataframe: pd.DataFrame,
            dimension: str,
            window_size: int
    ) -> Tuple[List[str], pd.DataFrame]:

        # Check if the dimension exists in the dataframe
        if dimension not in dataframe.columns:
            raise ValueError(f"Column '{dimension}' not found in the DataFrame.")

        # Validate window_size is a positive integer
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        # Copy dataframe to preserve initial structure
        dataframe = dataframe.copy()

        # Create new columns buffer
        columns = []

        # Create future target columns
        for i in range(1, window_size + 1):

            # Define new column name
            column = f'{dimension}_SHIFT_{i}'

            # Store new column
            columns.append(column)
            dataframe[column] = dataframe[dimension].shift(-i)

        # Enforce final format
        dataframe = dataframe.drop(columns=dimension)
        dataframe = dataframe.dropna(ignore_index=True)

        return columns, dataframe
