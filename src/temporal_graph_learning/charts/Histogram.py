import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot


class Histogram:

    def __init__(self, bins: int, left_boundary: float = None, right_boundary: float = None):

        # Histogram bins
        self.bins = bins

        # Store histogram interval bounds
        self._left_boundary = left_boundary
        self._right_boundary = right_boundary

    def draw_on_plot(self, datapoints: Union[pd.Series, List], plot: Plot):

        # Convert datapoints to NumPy array
        datapoints = np.array(datapoints)

        # Parse histogram interval bounds
        left_bound = self._left_boundary or datapoints.min()
        right_bound = self._right_boundary or datapoints.max()

        # Bin datapoints
        counts, edges = np.histogram(
            datapoints,
            np.linspace(left_bound, right_bound, self.bins)
        )

        # Draw histogram
        plot.get_pointer().stairs(counts, edges, fill=True)
        plot.get_pointer().vlines(edges, 0, counts.max(), colors='w')
