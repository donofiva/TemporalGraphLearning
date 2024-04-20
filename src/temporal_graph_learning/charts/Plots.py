import itertools
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from temporal_graph_learning.charts.Plot import Plot


class Plots:

    def __init__(
            self,
            rows: int,
            columns: int,
            row_width: int = 1,
            column_width: int = 1,
            share_x_axis: bool = False,
            share_y_axis: bool = False
    ):

        # Store rows and columns configuration
        self.rows = rows
        self.columns = columns

        # Generate subplots references
        self._figure, plots_grid = plt.subplots(
            rows,
            columns,
            figsize=(row_width * rows, column_width * columns),
            sharex=share_x_axis,
            sharey=share_y_axis
        )

        # Build plots abstraction from plots grid
        self._plots_grid = self._build_plots_abstraction_from_plots_grid(plots_grid)

        # Squeeze plots grid into a single list
        self._plots_list = self._squeeze_plots_grid(self._plots_grid)

    def __del__(self):
        plt.close(self._figure)

    # Setup
    def _build_plots_abstraction_from_plots_grid(self, plots_grid) -> List[List[Plot]]:

        # Make sure to always have a plots matrix
        plots_grid = np.reshape(plots_grid, (self.rows, self.columns))

        # Build Plot abstraction
        return [[Plot(plot_pointer) for plot_pointer in plots_row] for plots_row in plots_grid]

    @staticmethod
    def _squeeze_plots_grid(plots_grid: List[List[Plot]]) -> List[Plot]:
        return list(itertools.chain.from_iterable(plots_grid))

    # Getters
    def get_plots_grid(self) -> List[List[Plot]]:
        return self._plots_grid

    def get_plots_list(self) -> List[Plot]:
        return self._plots_list

    def get_plots_from_grid_by_row_index(self, row_index: int) -> List[Plot]:
        return self._plots_grid[row_index]

    def get_plot_from_grid_by_indexes(self, row_index: int, column_index: int) -> Plot:
        return self._plots_grid[row_index][column_index]

    def get_plot_from_list_by_index(self, index: int) -> Plot:
        return self._plots_list[index]

    # Configuration
    @staticmethod
    def set_tight_layout():
        plt.tight_layout()

    # Output
    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def save(destination: str):
        plt.savefig(destination)
