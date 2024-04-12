from typing import Dict, List, Any
from dataclasses import dataclass
from tgl.data.Field import Field


@dataclass
class Configuration:

    field_to_filter_values: Dict[Field, List[Any]]
