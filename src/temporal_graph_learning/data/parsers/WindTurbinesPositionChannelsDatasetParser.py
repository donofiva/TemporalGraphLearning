import numpy as np
import pandas as pd
import torch

from temporal_graph_learning.data.scalers.Scaler import Scaler
from temporal_graph_learning.data.parsers.WindTurbinesChannelsDatasetParser import WindTurbinesChannelsDatasetParser
from temporal_graph_learning.data.parsers.WindTurbinesPositionDatasetParser import WindTurbinesPositionDatasetParser
from temporal_graph_learning.data.datasets.WindTurbinesChannelsConnectivityDataset import WindTurbineChannelsPositionDataset


class WindTurbinesPositionChannelsDatasetParser:

    def __init__(
            self,
            dataset_channels: pd.DataFrame,
            dataset_position: pd.DataFrame,
            mask_channels: bool = True,
            self_loops: bool = False
    ):

        # Initialize parsers
        self._parser_channels = WindTurbinesChannelsDatasetParser(dataset_channels)
        self._parser_position = WindTurbinesPositionDatasetParser(dataset_position, self_loops=self_loops)

        # Store configuration
        self._mask_channels = mask_channels

    def get_parser_channels(self) -> WindTurbinesChannelsDatasetParser:
        return self._parser_channels

    def get_parser_position(self) -> WindTurbinesPositionDatasetParser:
        return self._parser_position

    def store_parser_channels(self, parser_channels: WindTurbinesChannelsDatasetParser):
        self._parser_channels = parser_channels

    # Dataset methods
    def get_adjacency_matrices(self) -> np.ndarray:

        # Initialize adjacency matrices buffer
        adjacency_matrices = []

        # Split dataset by day and timeslot to retrieve adjacency matrices
        for _, parser_channels_split in self._parser_channels.split_on_dimensions(['DAY', 'TIMESLOT']).items():

            # Define mask
            mask = parser_channels_split.retrieve_dimensions_from_dataset(['DATA_AVAILABLE'])
            mask = mask if self._mask_channels else np.ones_like(mask.values)

            # Retrieve adjacency matrix
            adjacency_matrix = self._parser_position.get_masked_adjacency_matrix_from_euclidean_distance_matrix_by_threshold(
                mask=mask,
                distance_threshold=2000
            )

            # Store adjacency matrix
            adjacency_matrices.append(adjacency_matrix)

        return np.array(adjacency_matrices)

    def get_wind_turbines_channels_grid(self) -> np.ndarray:

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
    parser = WindTurbinesPositionChannelsDatasetParser(
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
        parser_channels_split

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
    parser_channels = WindTurbinesChannelsDatasetParser.from_chunks(*wind_turbine_to_parser_channels.values())

    # Store
    parser.store_parser_channels(parser_channels)
    parser.prova()

    # parser.prova()