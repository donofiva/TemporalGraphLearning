import pandas as pd
from typing import Tuple, List


class Timeseries:

    SAMPLES_PER_HOUR = 6
    HOURS_PER_DAY = 24

    @staticmethod
    def get_daily_index_and_labels_from_timeseries_index(timeseries_index: pd.Series) -> Tuple[List[int], List[str]]:

        # Define samples per day and parse index
        samples_per_day = Timeseries.SAMPLES_PER_HOUR * Timeseries.HOURS_PER_DAY
        timeseries_index = timeseries_index.tolist()

        # Define daily index and labels
        daily_index = timeseries_index[samples_per_day::samples_per_day] + timeseries_index[-1:]
        daily_labels = [f'Day {index + 1}' for index, _ in enumerate(daily_index[:-1])]

        return daily_index, daily_labels
