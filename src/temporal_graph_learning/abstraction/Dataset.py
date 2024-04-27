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
