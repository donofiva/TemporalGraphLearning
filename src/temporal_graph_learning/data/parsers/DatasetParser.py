import torch
import pandas as pd

from typing import List, Tuple, Dict, Any, Hashable, Union
from sklearn.model_selection import train_test_split
from temporal_graph_learning.data.scalers.Scaler import Scaler


class DatasetParser:

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

    def split_on_dimensions(self, dimensions: List[str]) -> Dict[Hashable, "DatasetParser"]:
        return {
            dimensions: DatasetParser(dataset_slice.reset_index(drop=True))
            for dimensions, dataset_slice in self._dataset.groupby(dimensions, as_index=False)
        }

    # Scaler methods
    def scale_dimension(self, dimension: str, scaler: Scaler) -> Tuple[pd.DataFrame, Any]:
        return scaler.initialize_scaler_and_scale_data(self.retrieve_dimensions_from_dataset([dimension]))

    def get_scaler_by_dimension(self, dimension: str):
        return self._dimension_to_scaler.get(dimension)

    def store_scaler_by_dimension(self, dimension: str, scaler):
        self._dimension_to_scaler[dimension] = scaler

    # Split methods
    def train_test_split(
            self,
            *dimensions_set: List[str],
            test_size=0.2,
            shuffle=False,
            stratify=None
    ) -> Tuple["DatasetParser", ...]:
        return tuple(
            DatasetParser(split)
            for split in train_test_split(
                *[self.retrieve_dimensions_from_dataset(dimensions) for dimensions in dimensions_set],
                test_size=test_size,
                shuffle=shuffle,
                stratify=stratify
            )
        )

    # Tensor methods
    def to_tensor(self):
        return torch.tensor(
            self._dataset.values,
            dtype=torch.float32,
        )

    def get_dimensions_from_dataset_parsed_as_tensor(self, dimensions: List[str]):
        return torch.tensor(
            self.retrieve_dimensions_from_dataset(dimensions).values,
            dtype=torch.float32
        )
