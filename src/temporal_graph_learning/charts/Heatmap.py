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
            color_map=None,
            symmetric: bool = False
    ):

        # Store heatmap configuration
        self._maximum_value = maximum_value
        self._minimum_value = minimum_value
        self._show_values = show_values
        self._color_map = color_map
        self._symmetric = symmetric

    def draw_on_plot(self, datapoints: Union[pd.DataFrame, np.array], plot: Plot):

        # Covert Pandas dataframe to NumPy matrix
        if isinstance(datapoints, pd.DataFrame):
            datapoints = datapoints.values

        # Heatmaps usually do not require any mask
        triangular_mask = None

        # Symmetric heatmaps require triangular mask to cover redundant data
        if self._symmetric:
            triangular_mask = np.triu(np.ones_like(datapoints, dtype=bool))

        # Draw heatmap
        sns.heatmap(
            datapoints,
            vmin=self._minimum_value,
            vmax=self._maximum_value,
            annot=self._show_values,
            square=True,
            cmap=self._color_map,
            mask=triangular_mask,
            ax=plot.get_pointer()
        )
