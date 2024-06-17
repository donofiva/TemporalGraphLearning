import numpy as np
import pandas as pd

from temporal_graph_learning.data.parsers.WindTurbinesChannelsDatasetParser import WindTurbinesChannelsDatasetParser
from temporal_graph_learning.data.parsers.WindTurbinesPositionDatasetParser import WindTurbinesPositionDatasetParser


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

    def prova(self):

        # Retrieve masked adjacency matrices
        adjacency_matrices_masked = np.array([
            self._parser_position.get_masked_adjacency_matrix_from_euclidean_distance_matrix_by_threshold(
                mask=parser_channels_slice.retrieve_dimensions_from_dataset_parsed(['DATA_AVAILABLE']),
                distance_threshold=2000
            )
            for _, parser_channels_slice in self._parser_channels.split_on_dimension(['DAY', 'TIMESLOT']).items()
        ])

        # TODO: Find a way to perform meaningful scaling and return stuff for the dataset


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

    parser.prova()