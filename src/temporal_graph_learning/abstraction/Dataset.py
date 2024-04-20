import pandas as pd

from typing import List, Tuple, Any


class Dataset:

    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    def slice_on_dimensions(self,  dimensions: List[str], include_all: bool = False) -> List[Tuple[Any, pd.DataFrame]]:
        return (
            list(self._dataframe.groupby(dimensions, as_index=False)) +
            ([] if not include_all else [('ALL', self._dataframe)])
        )
