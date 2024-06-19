import numpy as np
import pandas as pd
import torch

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from temporal_graph_learning.data.parsers.WindTurbinesChannelsParser import WindTurbinesChannelsParser
from temporal_graph_learning.data.parsers.WindTurbinesPositionParser import WindTurbinesPositionParser, ConnectivityType
from temporal_graph_learning.data.datasets.WindTurbinesConnectivityChannelsDataset import WindTurbinesConnectivityChannelsDataset


class WindTurbinesPositionChannelsParser:

    def __init__(
            self,
            dataset_channels: pd.DataFrame,
            dataset_position: pd.DataFrame,
            enable_mask: bool = False,
            self_loops: bool = False
    ):

        # Initialize parsers
        self._parser_channels = WindTurbinesChannelsParser(
            dataset=dataset_channels
        )

        self._parser_position = WindTurbinesPositionParser(
            dataset=dataset_position,
            enable_mask=enable_mask,
            enable_self_loops=self_loops
        )

    def get_parser_channels(self) -> WindTurbinesChannelsParser:
        return self._parser_channels

    def get_parser_position(self) -> WindTurbinesPositionParser:
        return self._parser_position

    def store_parser_channels(self, parser_channels: WindTurbinesChannelsParser):
        self._parser_channels = parser_channels

    # Dataset methods
    def build_wind_turbines_channels_series(self, channels_excluded: List[str]) -> np.ndarray:

        # Build channels series
        wind_turbines_channels_series = np.array([
            parser_channels_split.get_dataset().values
            for _, parser_channels_split in self._parser_channels.split_on_dimensions(['TURBINE']).items()
            for _ in [parser_channels_split.drop_dimensions(channels_excluded)]
        ])

        # Reshape and convert channels series
        wind_turbines_channels_series = wind_turbines_channels_series.reshape(
            wind_turbines_channels_series.shape[0],
            -1
        ).astype(float)

        return wind_turbines_channels_series.astype(float)

    def build_and_get_wind_turbines_connectivity_matrices(
            self,
            connectivity_type: ConnectivityType,
            threshold: float,
            channels_series: np.ndarray
    ) -> np.ndarray:

        # Initialize adjacency matrices buffer
        connectivity_matrices = []

        # Build distance matrix
        distance_matrix = self._parser_position.build_distance_matrix(
            connectivity_type=connectivity_type,
            channels_series=channels_series
        )

        # Split dataset by day and timeslot to retrieve adjacency matrices
        for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items():

            # Define mask
            mask = parser_channels_split.retrieve_dimensions_from_dataset(['DATA_AVAILABLE'])

            # Retrieve adjacency matrix
            connectivity_matrix = self._parser_position.build_connectivity_matrix(
                distance_matrix=distance_matrix,
                mask=mask,
                threshold=threshold
            )

            # Store adjacency matrix
            connectivity_matrices.append(connectivity_matrix)

        return np.array(connectivity_matrices)

    def build_and_get_wind_turbines_channels_grids(self) -> np.ndarray:

        # Initialize channels grid buffer
        channels_grids = []

        # Split dataset by day and timeslot to retrieve channels grid
        for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items():

            # Convert timeslot and enforce schema
            parser_channels_split.transform_timeslot()
            parser_channels_split.drop_dimensions(['DAY', 'TURBINE', 'DATA_AVAILABLE'])

            # Store channel grid
            channels_grid = parser_channels_split.get_dataset().values
            channels_grids.append(channels_grid)

        return np.array(channels_grids)

    def build_and_get_wind_turbines_masks_grids(self) -> np.ndarray:
        return np.array([
            parser_channels_split.retrieve_dimensions_from_dataset(['DATA_AVAILABLE'])
            for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items()
        ])

    def build_and_get_wind_turbines_targets_grids(self, channel: str) -> np.ndarray:
        return np.array([
            parser_channels_split.retrieve_dimensions_from_dataset([channel])
            for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items()
        ])

    def build_wind_turbines_connectivity_channels_dataset(
            self,
            connectivity_type: ConnectivityType,
            threshold: float,
            channels_series: np.ndarray,
            window: int = 1,
            lag: int = 1,
            horizon: int = 1
    ) -> WindTurbinesConnectivityChannelsDataset:

        # Build connectivity matrices, channels, masks and targets grids
        wind_turbines_channels_grids = self.build_and_get_wind_turbines_channels_grids()
        wind_turbines_masks_grids = self.build_and_get_wind_turbines_masks_grids()
        wind_turbines_targets_grids = self.build_and_get_wind_turbines_targets_grids(channel='ACTIVE_POWER')
        wind_turbines_connectivity_matrices = self.build_and_get_wind_turbines_connectivity_matrices(
            connectivity_type=connectivity_type,
            threshold=threshold,
            channels_series=channels_series
        )

        # Convert arrays to tensors
        wind_turbines_connectivity_matrices_tensor = torch.tensor(
            wind_turbines_connectivity_matrices,
            dtype=torch.float32
        )

        wind_turbines_channels_grids_tensor = torch.tensor(
            wind_turbines_channels_grids,
            dtype=torch.float32
        )

        wind_turbines_masks_grids_tensor = torch.tensor(
            wind_turbines_masks_grids,
            dtype=torch.float32
        )

        wind_turbines_targets_grids_tensor = torch.tensor(
            wind_turbines_targets_grids,
            dtype=torch.float32
        )

        # Build dataset
        return WindTurbinesConnectivityChannelsDataset(
            connectivity_matrices=wind_turbines_connectivity_matrices_tensor,
            channels_grids=wind_turbines_channels_grids_tensor,
            masks_grids=wind_turbines_masks_grids_tensor,
            targets_grids=wind_turbines_targets_grids_tensor,
            window=window,
            lag=lag,
            horizon=horizon
        )

    def build_wind_turbines_connectivity_channels_train_and_test_datasets(
            self,
            connectivity_type: ConnectivityType,
            threshold: float,
            channels_series: np.ndarray,
            test_size: float = 0.2,
            window: int = 1,
            lag: int = 1,
            horizon: int = 1
    ) -> Tuple[WindTurbinesConnectivityChannelsDataset, WindTurbinesConnectivityChannelsDataset]:

        # Build connectivity matrices, channels, masks and targets grids
        wind_turbines_channels_grids = self.build_and_get_wind_turbines_channels_grids()
        wind_turbines_masks_grids = self.build_and_get_wind_turbines_masks_grids()
        wind_turbines_targets_grids = self.build_and_get_wind_turbines_targets_grids(channel='ACTIVE_POWER')
        wind_turbines_connectivity_matrices = self.build_and_get_wind_turbines_connectivity_matrices(
            connectivity_type=connectivity_type,
            threshold=threshold,
            channels_series=channels_series
        )

        # Convert arrays to tensors
        wind_turbines_connectivity_matrices_tensor = torch.tensor(
            wind_turbines_connectivity_matrices,
            dtype=torch.float32
        )

        wind_turbines_channels_grids_tensor = torch.tensor(
            wind_turbines_channels_grids,
            dtype=torch.float32
        )

        wind_turbines_masks_grids_tensor = torch.tensor(
            wind_turbines_masks_grids,
            dtype=torch.float32
        )

        wind_turbines_targets_grids_tensor = torch.tensor(
            wind_turbines_targets_grids,
            dtype=torch.float32
        )

        # Dataset split
        (
            wind_turbines_connectivity_matrices_tensor_train,
            wind_turbines_connectivity_matrices_tensor_test,
            wind_turbines_channels_grids_tensor_train,
            wind_turbines_channels_grids_tensor_test,
            wind_turbines_masks_grids_tensor_train,
            wind_turbines_masks_grids_tensor_test,
            wind_turbines_targets_grids_tensor_train,
            wind_turbines_targets_grids_tensor_test
        ) = train_test_split(
            wind_turbines_connectivity_matrices_tensor,
            wind_turbines_channels_grids_tensor,
            wind_turbines_masks_grids_tensor,
            wind_turbines_targets_grids_tensor,
            test_size=test_size,
            shuffle=False,
            stratify=None
        )

        # Build datasets
        wind_turbines_connectivity_channels_dataset_train = WindTurbinesConnectivityChannelsDataset(
            connectivity_matrices=wind_turbines_connectivity_matrices_tensor_train,
            channels_grids=wind_turbines_channels_grids_tensor_train,
            masks_grids=wind_turbines_masks_grids_tensor_train,
            targets_grids=wind_turbines_targets_grids_tensor_train,
            window=window,
            lag=lag,
            horizon=horizon
        )

        wind_turbines_connectivity_channels_dataset_test = WindTurbinesConnectivityChannelsDataset(
            connectivity_matrices=wind_turbines_connectivity_matrices_tensor_test,
            channels_grids=wind_turbines_channels_grids_tensor_test,
            masks_grids=wind_turbines_masks_grids_tensor_test,
            targets_grids=wind_turbines_targets_grids_tensor_test,
            window=window,
            lag=lag,
            horizon=horizon
        )

        return wind_turbines_connectivity_channels_dataset_train, wind_turbines_connectivity_channels_dataset_test
