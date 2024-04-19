import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, turbines: pd.DataFrame, timeseries: pd.DataFrame):

        # Store datasets
        self._turbines = turbines.set_index(['TURBINE'])
        self._timeseries = timeseries.set_index(['TURBINE', 'DAY', 'TIMESLOT'])

        # Store entities
        self._days = self._timeseries.index.unique('DAY')
        self._timeslots = self._timeseries.index.unique('TIMESLOT')

    def get_all_turbines(self) -> np.array:
        return self._turbines.index.values

    def get_all_days(self) -> np.array:
        return self._days.values

    def get_all_timeslots(self) -> np.array:
        return self._timeslots.values

    def get_turbine_dataset(self) -> pd.DataFrame:
        return self._turbines

    def get_timeseries_dataset(self) -> pd.DataFrame:
        return self._timeseries
