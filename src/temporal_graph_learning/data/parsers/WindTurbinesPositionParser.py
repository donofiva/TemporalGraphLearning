import numpy as np
import pandas as pd

from enum import Enum, unique
from typing import Dict, Union, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from temporal_graph_learning.data.parsers.TabularDatasetParser import TabularDatasetParser


@unique
class ConnectivityType(Enum):

    PHYSICAL_DISTANCE = 0
    COSINE_DISTANCE = 1


class WindTurbinesPositionParser(TabularDatasetParser):

    def __init__(self, dataset: pd.DataFrame, enable_mask: bool = False, enable_self_loops: bool = False):

        # Initialize superclass
        super().__init__(dataset)

        # Store graph configuration
        self._enable_mask = enable_mask
        self._enable_self_loops = enable_self_loops

        # Mappings
        self._wind_turbine_index_to_wind_turbine = self._build_wind_turbine_index_to_wind_turbine_map()

    def _build_wind_turbine_index_to_wind_turbine_map(self) -> Dict[int, int]:
        return dict(enumerate(self._dataset.index))

    # Connectivity methods
    def _build_connectivity_mask(self, mask: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:

        if isinstance(mask, pd.DataFrame) and mask.shape[1] == 1:
            mask = mask.values.squeeze()

        elif isinstance(mask, pd.Series):
            mask = np.array(mask)

        # Remove additional empty dimensions
        mask = mask.squeeze()

        # If masking is required, keep mask, otherwise replace it with neutral mask
        if not self._enable_mask:
            mask = np.ones_like(mask)

        # Map unavailable nodes to high distance values
        mask = np.where(mask == 0, np.inf, 1)

        return np.outer(mask, mask)

    def _build_physical_distance_matrix(self) -> np.ndarray:

        # Parse dataset
        dataset = self._dataset.sort_values('TURBINE', ascending=True).set_index('TURBINE')

        # Build distance matrix
        positions = dataset[['X_AXIS', 'Y_AXIS']].values
        physical_distance_matrix = squareform(pdist(positions, 'euclidean'))

        # Disable self-loops if required
        if not self._enable_self_loops:
            np.fill_diagonal(physical_distance_matrix, np.inf)

        return physical_distance_matrix

    def _build_cosine_distance_matrix(self, channels_series: np.ndarray) -> np.ndarray:

        # Enforce channels dimensions
        if channels_series.ndim < 2 or channels_series.ndim > 3:
            raise Exception("Incompatible channel dimensions")

        # Flatten channels
        channels_series_flattened = channels_series.reshape(channels_series.shape[0], -1).astype(float)

        # Remove missing values
        missing_values = np.isnan(channels_series_flattened).any(axis=0)
        channels_series_flattened = channels_series_flattened[:, ~missing_values]

        # Compute cosine distance matrix
        cosine_distance_matrix = 1 - cosine_similarity(channels_series_flattened)

        # Disable self-loops if required
        if not self._enable_self_loops:
            np.fill_diagonal(cosine_distance_matrix, np.inf)

        return cosine_distance_matrix

    # Dataset methods
    def build_distance_matrix(
            self,
            connectivity_type: ConnectivityType,
            channels_series: np.ndarray
    ) -> np.ndarray:

        if connectivity_type == ConnectivityType.PHYSICAL_DISTANCE:
            distance_matrix = self._build_physical_distance_matrix()

        elif connectivity_type == ConnectivityType.COSINE_DISTANCE:
            distance_matrix = self._build_cosine_distance_matrix(channels_series=channels_series)

        else:
            raise Exception("Unknown connectivity type")

        return distance_matrix

    def build_connectivity_matrix(
            self,
            distance_matrix: np.ndarray,
            mask: Union[np.ndarray, pd.Series, pd.DataFrame],
            threshold: float
    ) -> np.ndarray:

        # Build connectivity mask
        connectivity_mask = self._build_connectivity_mask(mask)

        # Build connectivity matrix
        connectivity_matrix = distance_matrix * connectivity_mask
        connectivity_matrix = (connectivity_matrix <= threshold).astype(int)

        return connectivity_matrix
