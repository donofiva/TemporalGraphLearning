from enum import Enum, unique
from temporal_graph_learning.charts.Plot import Plot


@unique
class FontSize(Enum):

    SMALL = 'small'


class Text:

    def __init__(self, x_position_padding: float, y_position_padding: float):

        # Store heatmap configuration
        self._x_position_padding = x_position_padding
        self._y_position_padding = y_position_padding

    def draw_on_plot(self, label: str, x_position: float, y_position: float, plot: Plot):

        # Draw text
        plot.get_pointer().text(
            x_position + self._x_position_padding,
            y_position + self._y_position_padding,
            label,
            fontsize=FontSize.SMALL.value
        )
