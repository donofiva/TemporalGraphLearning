import pandas as pd
from sklearn.metrics import mean_absolute_error


class MAEWindFarmAtTimeIndex:

    def __init__(self, time_index: int):
        self._time_index = time_index

    def compute(self, targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> float:
        return mean_absolute_error(
            targets.groupby(level=1, axis=1).sum().iloc[:, self._time_index],
            targets_predicted.groupby(level=1, axis=1).sum().iloc[:, self._time_index]
        )
