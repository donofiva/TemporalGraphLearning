import numpy as np
import pandas as pd

from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class KNNImputer:

    def __init__(
            self,
            neighbours: int = 1,
            min_items: int = 1,
            enable_self_similarity: bool = False
    ):

        # Store imputer configuration
        self._neighbours = neighbours
        self._min_items = min_items
        self._enable_self_similarity = enable_self_similarity

        # Store channels and masks
        self._wind_turbines_count = 0
        self._index_to_wind_turbine = {}
        self._masks = None
        self._channels = None
        self._similarity_matrix = None

        # Store
        self._wind_turbine_to_similar_wind_turbines = {}
        self._wind_turbine_to_similarity_values = {}
        self._timestamp_to_inactive_wind_turbines = {}
        self._timestamp_to_wind_turbine_to_similar_wind_turbines = {}

    @staticmethod
    def _build_index_to_wind_turbine_map(index) -> Dict[int, int]:
        return dict(enumerate(sorted(set(index))))

    @staticmethod
    def _swap_dataframe_index_level(df: pd.DataFrame) -> pd.DataFrame:
        return df.swaplevel(axis=1).sort_index(axis=1)

    def _compute_similarity_matrix(self, matrix: np.ndarray) -> np.ndarray:

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(matrix)

        # Handle self-similarity
        if not self._enable_self_similarity:
            np.fill_diagonal(similarity_matrix, 0.0)

        return similarity_matrix

    @staticmethod
    def _stack_dataframe_on_index_level(df: pd.DataFrame) -> pd.DataFrame:
        return df.stack(0, future_stack=True)

    def _build_wind_turbine_to_similar_wind_turbines_map(self, similarity_matrix: np.array) -> Dict[int, np.ndarray]:
        return {
            self._index_to_wind_turbine[index]: np.array([
                self._index_to_wind_turbine[similar_index]
                for similar_index in similar_indices[:-1]
            ])
            for index_to_similar_indices in [np.argsort(similarity_matrix, axis=1)[:, ::-1]]
            for index, similar_indices in enumerate(index_to_similar_indices)
        }

    def _build_wind_turbine_to_similarity_values_map(self, similarity_matrix: np.ndarray) -> Dict[int, np.ndarray]:
        return {
            self._index_to_wind_turbine[index]: np.array(similarity_values[:-1])
            for index_to_similarity_values in [np.sort(similarity_matrix, axis=1)[:, ::-1]]
            for index, similarity_values in enumerate(index_to_similarity_values)
        }

    def _build_timestamp_to_inactive_wind_turbines(self) -> Dict[pd.Timestamp, np.ndarray]:
        return {
            timeslot: mask[mask == 0].index.get_level_values(0).values
            for timeslot, mask in self._masks.iterrows()
        }

    def _build_timestamp_to_wind_turbine_to_similar_wind_turbines(self) -> Dict[pd.Timestamp, Dict[int, np.ndarray]]:
        return {
            timestamp: {
                wind_turbine: similar_wind_turbines[
                    ~np.isin(
                        similar_wind_turbines,
                        self._timestamp_to_inactive_wind_turbines[timestamp]
                    )
                ][:self._neighbours]
                for wind_turbine, similar_wind_turbines in self._wind_turbine_to_similar_wind_turbines.items()
            }
            for timestamp in self._channels.index
        }

    def fit(self, channels: pd.DataFrame, masks: pd.DataFrame):

        # Retrieve mask and channels
        self._masks = masks
        self._channels = channels
        self._index_to_wind_turbine = self._build_index_to_wind_turbine_map(channels.columns.get_level_values(0))
        self._wind_turbines_count = len(self._index_to_wind_turbine)

        # Process data
        channels = self._channels.copy()
        channels = self._swap_dataframe_index_level(channels)
        channels = self._stack_dataframe_on_index_level(channels)

        # Compute similarity matrix
        self._similarity_matrix = self._compute_similarity_matrix(channels.dropna().values.T)

        # Store similarity indices and values
        self._wind_turbine_to_similar_wind_turbines = self._build_wind_turbine_to_similar_wind_turbines_map(
            similarity_matrix=self._similarity_matrix
        )

        self._wind_turbine_to_similarity_values = self._build_wind_turbine_to_similarity_values_map(
            similarity_matrix=self._similarity_matrix
        )

        # Extract inactive turbines by timeslot
        self._timestamp_to_inactive_wind_turbines = self._build_timestamp_to_inactive_wind_turbines()
        self._timestamp_to_wind_turbine_to_similar_wind_turbines = (
            self._build_timestamp_to_wind_turbine_to_similar_wind_turbines()
        )

    def impute_index(self, timestamp: pd.Timestamp, wind_turbine: int, channel: str):

        # Make sure that there are enough active wind turbine
        if self._wind_turbines_count - self._timestamp_to_inactive_wind_turbines[timestamp].shape[0] < self._min_items:
            return np.nan

        # Retrieve channel from similar active wind turbines
        similar_wind_turbines_channel = self._channels.loc[
            timestamp,
            pd.IndexSlice[
                self._timestamp_to_wind_turbine_to_similar_wind_turbines[timestamp][wind_turbine],
                channel
            ]
        ]

        # Perform imputation
        return np.nanmean(similar_wind_turbines_channel.values[:self._neighbours])

    def impute_indices(self, indices: List[Tuple[pd.Timestamp, int, str]]) -> Dict[Tuple[pd.Timestamp, int, str], float]:
        return dict(
            zip(
                indices,
                list(map(lambda index: self.impute_index(*index), indices))
            )
        )

    def impute(self, channels: pd.DataFrame):

        # Preserve original dataframe
        channels = channels.copy()

        # Retrieve missing values indices
        missing_values_indices = channels.isna().stack(level=[0, 1], future_stack=True)[lambda x: x].index.tolist()

        # Impute indices missing values
        index_to_imputed_value = self.impute_indices(missing_values_indices)

        # Set values
        for (timestamp, wind_turbine, channel), imputed_value in index_to_imputed_value.items():
            channels.loc[timestamp, (wind_turbine, channel)] = imputed_value

        return channels