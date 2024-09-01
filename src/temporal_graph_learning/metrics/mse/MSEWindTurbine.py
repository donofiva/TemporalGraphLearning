import pandas as pd
from sklearn.metrics import mean_squared_error


class MSEWindTurbine:

    def __init__(self):
        pass

    @staticmethod
    def compute(targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> pd.DataFrame:

        # Metrics buffer
        metrics = {}

        # Populate metrics buffer
        for wind_turbine in targets.columns.get_level_values(0).unique():
            metrics[wind_turbine] = mean_squared_error(
                targets.loc[:, pd.IndexSlice[wind_turbine, :]],
                targets_predicted.loc[:, pd.IndexSlice[wind_turbine, :]]
            )

        # Build metrics dataframe
        return pd.DataFrame(
            list(metrics.values()),
            index=list(metrics.keys()),
            columns=['MSE']
        )
