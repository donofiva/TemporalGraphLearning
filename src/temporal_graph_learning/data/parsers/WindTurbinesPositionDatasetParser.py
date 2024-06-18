import numpy as np
import pandas as pd

from typing import Dict, Union
from scipy.spatial.distance import pdist, squareform
from temporal_graph_learning.data.parsers.DatasetParser import DatasetParser


class WindTurbinesPositionDatasetParser(DatasetParser):

    def __init__(self, dataset: pd.DataFrame, self_loops: bool = False):

        # Initialize superclass
        super().__init__(dataset)

        # Store graph configuration
        self._self_loops = self_loops

        # Build graph
        self._wind_turbine_index_to_wind_turbine = self._build_wind_turbine_index_to_wind_turbine_map()
        self._euclidean_distance_matrix = self._build_euclidean_distance_matrix()

    # Internal methods
    def _build_wind_turbine_index_to_wind_turbine_map(self) -> Dict[int, int]:
        return dict(enumerate(self._dataset.index))

    def _build_euclidean_distance_matrix(self) -> np.ndarray:

        # Parse dataset
        dataset = self._dataset.sort_values('TURBINE', ascending=True).set_index('TURBINE')

        # Compute square euclidean distance matrix
        positions = dataset[['X_AXIS', 'Y_AXIS']].values
        distance_matrix = squareform(pdist(positions, 'euclidean'))

        return distance_matrix

    # Dataset methods
    def get_euclidean_distance_matrix(self) -> np.ndarray:
        return self._euclidean_distance_matrix

    def get_masked_adjacency_matrix_from_euclidean_distance_matrix_by_threshold(
            self,
            mask: Union[np.ndarray, pd.Series, pd.DataFrame],
            distance_threshold: float
    ) -> np.ndarray:

        # Convert series to array
        if isinstance(mask, pd.DataFrame) and mask.shape[1] == 1:
            mask = mask.values.squeeze()

        elif isinstance(mask, pd.Series):
            mask = np.array(mask)

        # Remove additional empty dimensions
        mask = mask.squeeze()

        # Map unavailable nodes to high distance values
        mask = np.where(mask == 0, np.inf, 1)

        # Retrieve masked adjacency matrix
        euclidean_distance_matrix = self._euclidean_distance_matrix.copy()

        # Avoid self loops if required
        if not self._self_loops:
            np.fill_diagonal(
                euclidean_distance_matrix,
                np.inf
            )

        # Apply mask
        euclidean_distance_matrix_masked = euclidean_distance_matrix * np.outer(mask, mask)

        # Retrieve adjacency matrix by threshold
        return (euclidean_distance_matrix_masked <= distance_threshold).astype(int)







