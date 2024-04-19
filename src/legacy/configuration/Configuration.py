from typing import Dict, List, Any
from dataclasses import dataclass
from legacy.data.Field import Field
from legacy.model import Comparisons


@dataclass
class Configuration:

    abnormal_values: List[Comparisons]
    field_to_filter_values: Dict[Field, List[Any]]
