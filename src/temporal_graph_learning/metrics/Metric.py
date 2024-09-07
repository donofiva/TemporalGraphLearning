import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional


class Metric(ABC):

    @abstractmethod
    def __init__(self, time_index: Optional[int] = None):
        self.time_index = time_index

    def select_time_index(
            self,
            targets: pd.DataFrame,
            predictions: pd.DataFrame,
            masks: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.time_index is not None:

            # Slice index
            index_slice = targets.columns.get_level_values(1) == targets.columns.levels[1][self.time_index]

            # Slice dataframes
            return (
                targets.iloc[:, index_slice],
                predictions.iloc[:, index_slice],
                masks.iloc[:, index_slice] if masks is not None else None
            )

        return targets, predictions, masks

    @staticmethod
    def mask(targets: pd.DataFrame, masks: Optional[pd.DataFrame]):
        return targets if masks is None else targets.where(masks == 1, 0)

    @abstractmethod
    def aggregate(
            self,
            targets: pd.DataFrame,
            predictions: pd.DataFrame,
            masks: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @staticmethod
    def slice(
            targets: pd.DataFrame,
            predictions: pd.DataFrame,
            masks: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

        if targets.columns.nlevels == 1:
            return [(targets, predictions, masks)]

        elif targets.columns.nlevels == 2:
            return [
                (
                    targets.loc[:, pd.IndexSlice[entity, :]],
                    predictions.loc[:, pd.IndexSlice[entity, :]],
                    None if masks is None else masks.loc[:, pd.IndexSlice[entity, :]]
                )
                for entity in targets.columns.get_level_values(0).unique()
            ]

    @staticmethod
    def parse_values(dataframe: pd.DataFrame) -> np.array:
        return dataframe.values

    def parse_masks(self, masks: Optional[pd.DataFrame], shape) -> np.array:
        return np.ones(shape) if masks is None else self.parse_values(masks)

    @abstractmethod
    def compute(self, targets: np.ndarray, predictions: np.ndarray, masks: np.array):
        raise NotImplementedError

    @staticmethod
    def average(values: Union[np.array, List[float]], count: int) -> float:
        return np.sum(values) / count

    def evaluate(
            self,
            targets: pd.DataFrame,
            predictions: pd.DataFrame,
            masks: pd.DataFrame = None
    ):

        # Preserve dataframes
        targets = targets.copy()
        predictions = predictions.copy()
        masks = masks.copy() if masks is not None else None

        # Enforce masks schema
        masks.columns = targets.columns

        # Select time index
        targets, predictions, masks = self.select_time_index(targets, predictions, masks)

        # Mask values
        targets = self.mask(targets, masks)
        predictions = self.mask(predictions, masks)

        # Aggregate values
        targets, predictions, masks = self.aggregate(targets, predictions, masks)

        # Slice buffer
        metrics_slice = []

        # Split data
        for targets_slice, predictions_slice, masks_slice in self.slice(targets, predictions, masks):

            # Convert data to matrix
            shape = targets_slice.shape

            # Parse targets, predictions and masks
            targets_slice = self.parse_values(targets_slice)
            predictions_slice = self.parse_values(predictions_slice)
            masks_slice = self.parse_masks(masks_slice, shape)

            # Horizon buffer
            metrics_slice_horizon = []

            # Iterate over horizons
            for horizon in range(shape[1]):

                # Retrieve horizon slice
                targets_slice_horizon = targets_slice[:, horizon]
                predictions_slice_horizon = predictions_slice[:, horizon]
                masks_slice_horizon = masks_slice[:, horizon]

                # Compute metric slice
                metric_slice_horizon = self.compute(
                    targets_slice_horizon,
                    predictions_slice_horizon,
                    masks_slice_horizon
                )

                metrics_slice_horizon.append(metric_slice_horizon)

            metric_slice = self.average(metrics_slice_horizon, shape[1])
            metrics_slice.append(metric_slice)

        return self.average(metrics_slice, len(metrics_slice))
