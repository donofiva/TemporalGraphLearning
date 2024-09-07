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
        pass

    def compute(self, targets: np.ndarray, predictions: np.ndarray, masks: np.array):
        pass
