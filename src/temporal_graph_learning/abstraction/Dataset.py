import pandas as pd

from typing import List, Tuple, Any


class Dataset:

    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    # Transformations
    def get_missing_values_by_dimension(self, label: str = None) -> "Dataset":
        return Dataset(
            pd.DataFrame(
                self._dataframe.isna().sum(),
                columns=[label or 'MISSING_VALUES']
            )
        )

    def pivot_on_dimensions(
            self,
            column_dimensions: List[str],
            index_dimensions: List[str],
            values_dimension: str
    ) -> "Dataset":
        return Dataset(
            pd.pivot_table(
                self._dataframe,
                columns=column_dimensions,
                index=index_dimensions,
                values=values_dimension
            ).reset_index(drop=True)
        )

    def slice_on_dimensions(self,  dimensions: List[str], include_all: bool = False) -> List[Tuple[Any, "Dataset"]]:

        # Define dataframe slices
        slice_entities_to_dataframe = list(self._dataframe.groupby(dimensions, as_index=False))
        slice_entities_to_dataframe += int(include_all) * [(('ALL', ), self._dataframe)]

        # Convert dataframe slices to datasets
        return [
            (slice_entities, Dataset(dataframe.reset_index(drop=True)))
            for slice_entities, dataframe in slice_entities_to_dataframe
        ]

    # Getters
    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe
