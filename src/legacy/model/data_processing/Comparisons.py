from typing import Any, Set
from dataclasses import dataclass
from legacy.abstraction.entities.Dimension import Dimension
from legacy.data import Comparator


@dataclass(frozen=True)
class Comparison:

    dimension: Dimension
    comparator: Comparator
    value: Any


@dataclass
class Comparisons:

    comparisons: Set[Comparison]
