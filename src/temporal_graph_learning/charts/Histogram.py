import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot


class Histogram:

    def __init__(self, bins: int, left_bound: float = None, right_bound: float = None):

        # Histogram bins
        self.bins = bins

        # Store histogram interval bounds
        self._left_boundary = left_bound
        self._right_boundary = right_bound

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


# if __name__ == '__main__':
#
#     plots = Plots(1, 1, row_width=12, column_width=9)
#     plot = plots.get_plot_from_list_by_index(0)
#     plot.set_axis_labels('Ivan', 'Nadine')
#     plot.toggle_top_and_right_border(False)
#     plot.set_x_axis_boundaries(-10, 10)
#
#     hist = Histogram(100, -10, 10)
#     hist.draw_on_plot(np.random.normal(size=10000), plot)
#
#     plots.show()
