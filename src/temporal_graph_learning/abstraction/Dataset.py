import pandas as pd

from typing import List, Tuple, Any


class Dataset:

    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    # Transformations
    def slice_on_dimensions(self,  dimensions: List[str], include_all: bool = False) -> List[Tuple[Any, "Dataset"]]:

        # Define dataframe slices
        slice_entities_to_dataframe = list(self._dataframe.groupby(dimensions, as_index=False))
        slice_entities_to_dataframe += int(include_all) * [('ALL', ), self._dataframe]

        # Convert dataframe slices to datasets
        return [
            (slice_entities, Dataset(dataframe))
            for slice_entities, dataframe in slice_entities_to_dataframe
        ]

    # Getters
    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe
