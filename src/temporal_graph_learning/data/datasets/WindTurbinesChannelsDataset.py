from typing import List
from torch.utils.data import Dataset
from temporal_graph_learning.data.datasets.WindTurbineChannelsDataset import WindTurbineChannelsDataset


class WindTurbinesChannelsDataset(Dataset):

    def __init__(self, wind_turbine_channels_datasets: List[WindTurbineChannelsDataset]):

        # Store datasets and lengths
        self._datasets = wind_turbine_channels_datasets
        self._lengths = [
            len(wind_turbine_channels_dataset)
            for wind_turbine_channels_dataset in wind_turbine_channels_datasets
        ]

        # Store total length
        self._total_length = sum(self._lengths)

        # Store index map
        self._global_index_to_dataset_index_and_local_index = self._build_index_map(wind_turbine_channels_datasets)

    @staticmethod
    def _build_index_map(wind_turbine_channels_datasets: List[WindTurbineChannelsDataset]):
        return [
            (dataset_index, local_index)
            for dataset_index, dataset in enumerate(wind_turbine_channels_datasets)
            for local_index in range(len(dataset))
        ]

    def __len__(self):
        return self._total_length

    def __getitem__(self, global_index):

        # Map global index to dataset index and local index
        dataset_index, local_index = self._global_index_to_dataset_index_and_local_index[global_index]

        # Retrieve data from corresponding dataset
        return self._datasets[dataset_index][local_index]