import numpy as np
import pandas as pd

from typing import Tuple, Optional
from temporal_graph_learning.metrics.Metric import Metric


class RMSEWindFarm(Metric):

    def __init__(self, time_index: Optional[int] = None):
        super().__init__(time_index)

    def aggregate(
            self,
            targets: pd.DataFrame,
            predictions: pd.DataFrame,
            masks: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # Aggregate targets and predictions
        targets = targets.groupby(level=1, axis=1).sum()
        predictions = predictions.groupby(level=1, axis=1).sum()

        # Aggregate masks
        if masks is None:

            # Initialize masks as copy of target and then assign identity mask
            masks = targets.copy()
            masks.loc[:, :] = 1

        else:

            # Else preserve timeslot if at least wind farm is available
            masks = masks.groupby(level=1, axis=1).max()

        return targets, predictions, masks

    def compute(self, targets: np.ndarray, predictions: np.ndarray, masks: np.array):
        return np.sqrt(((targets - predictions) ** 2).sum() / masks.sum())
