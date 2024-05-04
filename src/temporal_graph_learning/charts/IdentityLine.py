from temporal_graph_learning.charts.Plot import Plot


class Lineplot:

    def __init__(self):
        pass

    @staticmethod
    def draw_on_plot(plot: Plot):
        plot.get_pointer().axline(
            (0, 0),
            slope=1,
            color='red'
        )
