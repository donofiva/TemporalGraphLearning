import numpy as np
import pandas as pd

from typing import Union, List, Tuple
from temporal_graph_learning.charts.Plot import Plot


class Lineplot:

    def __init__(self, highlight_missing_values: bool = False):

        # Store lineplot configuration
        self._highlight_missing_values = highlight_missing_values

    @staticmethod
    def get_daily_index_and_labels_from_timeseries_index(timeseries_index: List[int]) -> Tuple[List[int], List[str]]:

        # Define temporal anchors to produce ticks
        samples_per_hour = 6
        hours_per_day = 24

        # Generate daily ticks and labels
        daily_index = timeseries_index[::(samples_per_hour * hours_per_day)] + [timeseries_index[-1]]
        daily_labels = [''] + [f'Day {index + 1}' for index, _ in enumerate(daily_index[:-1])]

        return daily_index, daily_labels

    def draw_on_plot(self, datapoints: Union[pd.Series, List], plot: Plot):

        # Convert datapoints to NumPy array
        datapoints = np.array(datapoints)
        indices = np.arange(0, len(datapoints), 1)

        # Draw lineplot
        plot.get_pointer().plot(indices, datapoints)

        # Highlight missing values if required
        if self._highlight_missing_values:
            plot.get_pointer().fill_between(
                indices,
                0,
                1,
                where=np.isnan(datapoints),
                alpha=.3,
                color='red',
                transform=plot.get_pointer().get_xaxis_transform(),
                linewidth=0
            )
