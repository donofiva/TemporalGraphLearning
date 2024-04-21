import numpy as np
import pandas as pd
import seaborn as sns

from typing import Union
from temporal_graph_learning.charts.Plot import Plot


class Heatmap:

    def __init__(
            self,
            maximum_value: float = None,
            minimum_value: float = None,
            show_values: bool = False,
            color_map=None
    ):

        self.maximum_value = maximum_value
        self.minimum_value = minimum_value
        self.show_values = show_values
        self.color_map = color_map

    def draw_on_plot(self, datapoints: Union[pd.DataFrame, np.array], plot: Plot):

        # Covert Pandas dataframe to NumPy matrix
        if isinstance(datapoints, pd.DataFrame):
            datapoints = datapoints.values

        # Draw heatmap
        sns.heatmap(
            datapoints,
            vmin=self.minimum_value,
            vmax=self.maximum_value,
            annot=self.show_values,
            square=True,
            cmap=self.color_map,
            ax=plot.get_pointer()
        )
