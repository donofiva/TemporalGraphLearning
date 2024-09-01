import pandas as pd
from sklearn.metrics import mean_absolute_error


class MAEWindTurbineAtTimeIndex:

    def __init__(self, time_index: int):
        self._time_index = time_index

    def compute(self, targets: pd.DataFrame, targets_predicted: pd.DataFrame) -> pd.DataFrame:

        # Metrics buffer
        metrics = {}

        # Populate metrics buffer
        for wind_turbine in targets.columns.get_level_values(0).unique():
            metrics[wind_turbine] = mean_absolute_error(
                targets.loc[:, pd.IndexSlice[wind_turbine, :]].iloc[:, self._time_index],
                targets_predicted.loc[:, pd.IndexSlice[wind_turbine, :]].iloc[:, self._time_index]
            )

        # Build metrics dataframe
        return pd.DataFrame(
            list(metrics.values()),
            index=list(metrics.keys()),
            columns=[f'MAE_AT_TIME_INDEX_{self._time_index}']
        )
