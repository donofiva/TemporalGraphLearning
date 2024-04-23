import numpy as np
import pandas as pd

from typing import Union, List
from temporal_graph_learning.charts.Plot import Plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class AutoCorrelation:

    def __init__(self, lags: int, partial: bool = False):

        # Store auto correlation plot configuration
        self._lags = lags
        self._partial = partial

    def draw_on_plot(self, datapoints: Union[List, np.array, pd.Series], plot: Plot):

        if self._partial:
            plot_pacf(
                x=datapoints,
                lags=self._lags,
                zero=False,
                ax=plot.get_pointer()
            )

        else:
            plot_acf(
                x=datapoints,
                lags=self._lags,
                zero=False,
                ax=plot.get_pointer()
            )
