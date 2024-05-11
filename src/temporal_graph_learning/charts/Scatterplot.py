import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot


class Scatterplot:

    def __init__(self, opacity: float = 1):

        # Store heatmap configuration
        self._opacity = min(1.0, opacity)

    def draw_on_plot(self, datapoints_x: Union[pd.Series, List], datapoints_y: Union[pd.Series, List], plot: Plot):

        # Convert datapoints to NumPy array
        datapoints_x = np.array(datapoints_x)
        datapoints_y = np.array(datapoints_y)

        # Draw scatterplot
        plot.get_pointer().scatter(datapoints_x, datapoints_y, alpha=self._opacity)
