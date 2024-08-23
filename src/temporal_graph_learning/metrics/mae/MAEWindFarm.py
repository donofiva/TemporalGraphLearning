import pandas as pd
from sklearn.metrics import mean_absolute_error


class MAEWindFarm:

    def __init__(self):
        pass

    @staticmethod
    def compute(targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> float:
        return mean_absolute_error(
            targets.groupby(level=1, axis=1).sum(),
            targets_predicted.groupby(level=1, axis=1).sum()
        )
