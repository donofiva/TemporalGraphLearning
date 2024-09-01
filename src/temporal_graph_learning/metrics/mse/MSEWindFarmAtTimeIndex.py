import pandas as pd
from sklearn.metrics import mean_squared_error


class MSEWindFarmAtTimeIndex:

    def __init__(self, time_index: int):
        self._time_index = time_index

    def compute(self, targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> float:
        return mean_squared_error(
            targets.groupby(level=1, axis=1).sum().iloc[:, self._time_index],
            targets_predicted.groupby(level=1, axis=1).sum().iloc[:, self._time_index]
        )
