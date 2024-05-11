import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot


class VerticalBarplot:

    def __init__(self, width: float = 0.8):

        # Store bar plot configuration
        self._width = width

    def draw_on_plot(self, entities: Union[pd.Series, List], height: Union[pd.Series, List], plot: Plot):

        # Convert datapoints to NumPy array
        entities = np.array(entities)
        height = np.array(height)

        # Draw scatterplot
        plot.get_pointer().bar(entities, height, width=self._width)
