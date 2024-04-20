import matplotlib.pyplot as plt

from enum import Enum, unique


@unique
class Position(Enum):

    TOP = 'top'
    RIGHT = 'right'
    BOTTOM = 'bottom'
    LEFT = 'left'


class Plot:

    def __init__(self, pointer: plt.Axes):
        self._pointer = pointer

    # Borders
    def _toggle_border(self, border_position: Position, is_visible: bool):
        self._pointer.spines[border_position.value].set_visible(is_visible)

    def toggle_top_border(self, is_active: bool):
        self._toggle_border(Position.TOP, is_active)

    def toggle_right_border(self, is_active: bool):
        self._toggle_border(Position.RIGHT, is_active)

    def toggle_bottom_border(self, is_active: bool):
        self._toggle_border(Position.BOTTOM, is_active)

    def toggle_left_border(self, is_active: bool):
        self._toggle_border(Position.LEFT, is_active)

    def toggle_top_and_right_border(self, is_active: bool):
        self.toggle_top_border(is_active)
        self.toggle_right_border(is_active)

    # Labels
    def set_x_axis_label(self, label: str):
        self._pointer.set_xlabel(label)

    def set_y_axis_label(self, label: str):
        self._pointer.set_ylabel(label)

    def set_axis_labels(self, x_axis_label: str, y_axis_label: str):
        self.set_x_axis_label(x_axis_label)
        self.set_y_axis_label(y_axis_label)

    # Boundaries
    def set_x_axis_boundaries(self, left_boundary: float, right_boundary: float):
        self._pointer.set_xlim(left_boundary, right_boundary)

    def set_y_axis_boundaries(self, left_boundary: float, right_boundary: float):
        self._pointer.set_ylim(left_boundary, right_boundary)

    # Blank axes
    def empty(self):
        self._pointer.axis('off')

    # Title
    def set_title(self, title: str):
        self._pointer.set_title(title)

    # Drawing
    def get_pointer(self):
        return self._pointer