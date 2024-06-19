import numpy as np
import pandas as pd
import torch

from typing import Tuple
from sklearn.model_selection import train_test_split
from temporal_graph_learning.data.scalers.Scaler import Scaler
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

# if __name__ == '__main__':
#
#     # Read datasets
#     folder = '/Users/ivandonofrio/Workplace/Thesis/TemporalGraphLearning/assets'
#     dataset_channels = pd.read_csv(f'{folder}/wind_turbines_channels.csv')
#     dataset_position = pd.read_csv(f'{folder}/wind_turbines_position.csv')
#
#     # Initialize parser
#     parser = WindTurbinesPositionChannelsParser(
#         dataset_channels,
#         dataset_position,
#     )
#
#     # Retrieve channels parsers
#     parser_channels = parser.get_parser_channels()
#
#     # Define scaling dimensions
#     dimensions_to_scale = {
#         'WIND_SPEED',
#         'EXTERNAL_TEMPERATURE',
#         'INTERNAL_TEMPERATURE',
#         'REACTIVE_POWER',
#         'ACTIVE_POWER'
#     }
#
#     # Store scalers externally
#     wind_turbine_to_parser_channels = parser_channels.split_on_dimensions(['TURBINE'])
#     wind_turbine_to_channel_to_scaler = {}
#
#     # Scale all turbines
#     for (wind_turbine, ), parser_channels_split in wind_turbine_to_parser_channels.items():
#
#         # Apply data transformation
#         parser_channels_split.aggregate_and_transform_blades_pitch_angle()
#         parser_channels_split.transform_wind_direction()
#         parser_channels_split.transform_nacelle_direction()
#         parser_channels_split.transform_masks()
#
#         # Scale data
#         for dimension_to_scale in dimensions_to_scale:
#
#             # Apply scaling
#             dimension_scaled, scaler = parser_channels_split.scale_dimension(
#                 dimension_to_scale,
#                 Scaler.MIN_MAX
#             )
#
#             # Store scaled data
#             parser_channels_split.store_dimension(dimension_to_scale, dimension_scaled)
#
#             # Store scaler
#             wind_turbine_to_channel_to_scaler.setdefault(wind_turbine, {}).setdefault(dimension_to_scale, scaler)
#
#     # Generate new channels parser from scaled features
#     parser_channels = WindTurbinesChannelsParser.from_chunks(*wind_turbine_to_parser_channels.values())
#
#     # Generate wind turbine channels array
#     channels = np.array([
#         parser_channels_split.get_dataset().values
#         for _, parser_channels_split in wind_turbine_to_parser_channels.items()
#         for _ in [parser_channels_split.drop_dimensions(['TURBINE', 'DAY', 'TIMESLOT', 'ACTIVE_POWER'])]
#     ])
#
#     # Store
#     parser.store_parser_channels(parser_channels)
#
#     parser.build_wind_turbines_connectivity_channels_dataset(
#         ConnectivityType.COSINE_DISTANCE,
#         0.15,
#         channels
#     )
#
#     # parser.prova()