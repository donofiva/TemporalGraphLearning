import numpy as np

from typing import Dict, Set
from src.abstraction.entities.Day import Day
from src.abstraction.entities.Turbine import Turbine
from src.abstraction.entities.Dimension import Dimension
from src.abstraction.entities.Timeslot import Timeslot


class GridSnapshot:

    @classmethod
    def build_from_day_timeslot_dimension_and_turbines(
            cls,
            day: Day,
            timeslot: Timeslot,
            dimension: Dimension,
            turbines: Set[Turbine]
    ):
        return cls(
            day=day,
            timeslot=timeslot,
            dimension=dimension,
            turbine_to_datapoint={
                turbine: turbine.get_timeseries_by_dimension(dimension).get_datapoint_by_day_and_timeslot(
                    day=day,
                    timeslot=timeslot
                )
                for turbine in turbines
            }
        )

    def __init__(self, day: Day, timeslot: Timeslot, dimension: Dimension, turbine_to_datapoint: Dict[Turbine, float]):

        # Store entity identifiers
        self.day = day
        self.timeslot = timeslot
        self.dimension = dimension

        # Store entity attributes
        self._turbine_to_datapoint = turbine_to_datapoint

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: "Turbine"):
        return type(self) is type(other) and hash(self) == hash(other)

    @property
    def key(self):
        return self.day, self.timeslot, self.dimension

    def to_array(self) -> np.array:
        return np.array([*self._turbine_to_datapoint.values()])