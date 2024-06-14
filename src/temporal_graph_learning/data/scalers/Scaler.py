import pandas as pd

from typing import Tuple, Any
from enum import Enum, unique
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@unique
class Scaler(Enum):

    MIN_MAX = 0
    STANDARD = 1

    def get_scaler(self):

        if self == Scaler.MIN_MAX:
            return MinMaxScaler()

        elif self == Scaler.STANDARD:
            return StandardScaler()

    def scale_and_return_scaled_data_and_scaler(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:

        # Initialize scaler
        scaler = self.get_scaler()

        # Scale data
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        # Return scaled data and scaler
        return data_scaled, scaler
