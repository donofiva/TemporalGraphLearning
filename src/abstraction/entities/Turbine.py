from typing import List, Dict, Set, Optional
from src.abstraction.entities.Dimension import Dimension
from src.abstraction.entities.Timeseries import Timeseries


class Turbine:

    _entities: Dict[int, "Turbine"] = dict()

    @classmethod
    def build_from_index_and_coordinates(cls, index: int, x_axis: float, y_axis: float) -> "Turbine":

        # Initialize and try to store temporary entity
        index = int(index)
        turbine = cls(index, x_axis, y_axis)
        cls._entities.setdefault(turbine.key, turbine)

        # Return stored entity
        return cls._entities[turbine.key]

    @classmethod
    def get_from_index(cls, index: int) -> "Turbine":
        return cls._entities.get(index)

    @classmethod
    def get_all(cls) -> Set["Turbine"]:
        return set(cls._entities.values())

    @classmethod
    def get_all_sorted(cls) -> List["Turbine"]:
        return sorted(cls._entities.values())

    def __init__(self, index: int, x_axis: float, y_axis: float):

        # Store entity identifiers
        self.index = index

        # Store entity attributes
        self.x_axis = x_axis
        self.y_axis = y_axis

        # Map dimension to timeseries
        self._dimension_to_timeseries: Dict[Dimension, Timeseries] = dict()

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: "Turbine"):
        return type(self) is type(other) and hash(self) == hash(other)

    def __lt__(self, other: "Turbine"):
        return self.index < other.index

    def __le__(self, other: "Turbine"):
        return self < other or self == other

    @property
    def key(self):
        return self.index

    def store_timeseries_by_dimension(self, dimension: Dimension, timeseries: Timeseries):
        self._dimension_to_timeseries[dimension] = timeseries

    def get_timeseries_by_dimension(self, dimension: Dimension) -> Optional[Timeseries]:
        return self._dimension_to_timeseries.get(dimension)
