import pandas as pd
from typing import List, Dict, Hashable, Union


class TabularDatasetParser:

    def __init__(self, dataset: pd.DataFrame):

        # Store dataframe and replicate dataframe to store parsed dataset
        self._dataset = dataset.copy()
        self._dataset_backup = dataset.copy()

        # Store scaler references
        self._dimension_to_scaler = {}

    # Dataset methods
    def get_dataset(self) -> pd.DataFrame:
        return self._dataset

    def get_dataset_dimensions(self) -> List[str]:
        return list(self._dataset.columns)

    def retrieve_dimension_from_dataset(self, dimension: str) -> pd.Series:
        return self._dataset[dimension]

    def retrieve_dimensions_from_dataset(self, dimensions: List[str]) -> pd.DataFrame:
        return self._dataset[dimensions]

    def drop_dimensions(self, dimensions: List[str]):
        self._dataset.drop(columns=dimensions, inplace=True)

    def convert_dimension(self, dimension: str, dtype) -> pd.Series:
        return self.retrieve_dimension_from_dataset(dimension).astype(dtype=dtype)

    def store_dimension(self, dimension: str, data: Union[pd.Series, pd.DataFrame]):
        self._dataset[dimension] = data

    def split_on_dimensions(self, dimensions: List[str]) -> Dict[Hashable, "TabularDatasetParser"]:
        return {
            dimensions: TabularDatasetParser(dataset_slice.reset_index(drop=True))
            for dimensions, dataset_slice in self._dataset.groupby(dimensions, as_index=False)
        }
