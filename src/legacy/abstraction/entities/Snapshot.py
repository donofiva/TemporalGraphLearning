import pandas as pd

from legacy.abstraction.entities.Day import Day
from legacy.abstraction.entities.Turbine import Turbine
from legacy.abstraction.entities.Dimension import Dimension
from legacy.abstraction.entities import Timeslot


class Snapshot:

    def __init__(self, day: Day, timeslot: Timeslot, turbine_to_dimension_to_datapoints: pd.DataFrame):

        # Store entity identifiers
        self.day = day
        self.timeslot = timeslot

        # Store snapshot datapoints
        self._turbine_to_dimension_to_grid = turbine_to_dimension_to_datapoints

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: "Turbine"):
        return type(self) is type(other) and hash(self) == hash(other)

    @property
    def key(self):
        return self.day, self.timeslot

    def get_grid_by_dimension(self, dimension: Dimension):
        return self._turbine_to_dimension_to_grid.loc[:, dimension.name]

    def get_datapoint_by_turbine_and_dimension(self, turbine: Turbine, dimension: Dimension):
        return self._turbine_to_dimension_to_grid.loc[turbine, dimension.name]
