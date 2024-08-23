import pandas as pd
from sklearn.metrics import root_mean_squared_error


class RMSEWindFarm:

    def __init__(self):
        pass

    @staticmethod
    def compute(targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> float:
        return root_mean_squared_error(
            targets.groupby(level=1, axis=1).sum(),
            targets_predicted.groupby(level=1, axis=1).sum()
        )
