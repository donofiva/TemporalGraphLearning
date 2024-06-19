import numpy as np
import pandas as pd
import torch

from temporal_graph_learning.data.scalers.Scaler import Scaler
from temporal_graph_learning.data.parsers.WindTurbinesChannelsParser import WindTurbinesChannelsParser
from temporal_graph_learning.data.parsers.WindTurbinesPositionParser import WindTurbinesPositionParser, ConnectivityType
from temporal_graph_learning.data.datasets.WindTurbinesChannelsConnectivityDataset import WindTurbineChannelsPositionDataset


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
    def build_and_get_connectivity_matrices(
            self,
            connectivity_type: ConnectivityType,
            threshold: float,
            channels: np.ndarray
    ) -> np.ndarray:

        # Initialize adjacency matrices buffer
        connectivity_matrices = []

        # Build distance matrix
        distance_matrix = self._parser_position.build_distance_matrix(
            connectivity_type=connectivity_type,
            channels=channels
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

    def build_and_get_wind_turbines_channels_grid(self) -> np.ndarray:

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

    def prova(self):

        # Retrieve adjacency matrices
        adjacency_matrices = self.get_adjacency_matrices()
        wind_turbines_channels_grid = self.get_wind_turbines_channels_grid()
        targets = np.array([
            parser_channels_split.retrieve_dimensions_from_dataset(['ACTIVE_POWER'])
            for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items()
        ])

        # Convert to tensors
        adjacency_matrices_tensor = torch.tensor(adjacency_matrices, dtype=torch.float32)
        wind_turbines_channels_grid_tensor = torch.tensor(wind_turbines_channels_grid, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        print(adjacency_matrices_tensor.shape)
        print(wind_turbines_channels_grid_tensor.shape)
        print(targets_tensor.shape)

        dataset = WindTurbineChannelsPositionDataset(
            adjacency_matrices_tensor,
            wind_turbines_channels_grid_tensor,
            targets_tensor,
            1,
            1,
            1
        )

        print(dataset[2])

if __name__ == '__main__':

    # Read datasets
    folder = '/Users/ivandonofrio/Workplace/Thesis/TemporalGraphLearning/assets'
    dataset_channels = pd.read_csv(f'{folder}/wind_turbines_channels.csv')
    dataset_position = pd.read_csv(f'{folder}/wind_turbines_position.csv')

    # Initialize parser
    parser = WindTurbinesPositionChannelsParser(
        dataset_channels,
        dataset_position,
    )

    # Retrieve channels parsers
    parser_channels = parser.get_parser_channels()

    # Define scaling dimensions
    dimensions_to_scale = {
        'WIND_SPEED',
        'EXTERNAL_TEMPERATURE',
        'INTERNAL_TEMPERATURE',
        'REACTIVE_POWER',
        'ACTIVE_POWER'
    }

    # Store scalers externally
    wind_turbine_to_parser_channels = parser_channels.split_on_dimensions(['TURBINE'])
    wind_turbine_to_channel_to_scaler = {}

    # Scale all turbines
    for (wind_turbine, ), parser_channels_split in wind_turbine_to_parser_channels.items():

        # Apply data transformation
        parser_channels_split.aggregate_and_transform_blades_pitch_angle()
        parser_channels_split.transform_wind_direction()
        parser_channels_split.transform_nacelle_direction()
        parser_channels_split.transform_masks()

        # Scale data
        for dimension_to_scale in dimensions_to_scale:

            # Apply scaling
            dimension_scaled, scaler = parser_channels_split.scale_dimension(
                dimension_to_scale,
                Scaler.MIN_MAX
            )

            # Store scaled data
            parser_channels_split.store_dimension(dimension_to_scale, dimension_scaled)

            # Store scaler
            wind_turbine_to_channel_to_scaler.setdefault(wind_turbine, {}).setdefault(dimension_to_scale, scaler)

    # Generate new channels parser from scaled features
    parser_channels = WindTurbinesChannelsParser.from_chunks(*wind_turbine_to_parser_channels.values())

    # Generate wind turbine channels array
    channels = np.array([
        parser_channels_split.get_dataset().values
        for _, parser_channels_split in wind_turbine_to_parser_channels.items()
        for _ in [parser_channels_split.drop_dimensions(['TURBINE', 'DAY', 'TIMESLOT', 'ACTIVE_POWER'])]
    ])

    # Store
    parser.store_parser_channels(parser_channels)

    connectivity_matrices = parser.build_and_get_connectivity_matrices(
        ConnectivityType.PHYSICAL_DISTANCE,
        threshold=2000,
        channels=channels
    )

    parser.prova()

    # parser.prova()