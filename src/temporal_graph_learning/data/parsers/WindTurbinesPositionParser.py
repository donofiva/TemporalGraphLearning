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
    def get_position(self) -> pd.DataFrame:

        # Set wind turbine index
        position = self._dataset.sort_values('TURBINE')
        position = position.set_index('TURBINE')

        return position
