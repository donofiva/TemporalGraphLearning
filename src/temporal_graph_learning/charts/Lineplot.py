import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot


class Lineplot:

    def __init__(self, highlight_missing_values: bool = False):

        # Store lineplot configuration
        self._highlight_missing_values = highlight_missing_values

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
