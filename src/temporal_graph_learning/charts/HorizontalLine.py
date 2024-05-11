from temporal_graph_learning.charts.Plot import Plot, Colors, LineStyle


class HorizontalLine:

    def __init__(self):

        # Store bar plot configuration
        self._color = Colors.RED
        self._line_style = LineStyle.DASHED

    def draw_on_plot(self, y: float, plot: Plot):

        # Draw scatterplot
        plot.get_pointer().axhline(y, color=self._color.value, linestyle=self._line_style.value)
