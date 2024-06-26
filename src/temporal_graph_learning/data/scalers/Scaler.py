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

    @staticmethod
    def scale_data(data: pd.DataFrame, scaler) -> pd.DataFrame:

        data_scaled = scaler.transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        return data_scaled

    def initialize_scaler_and_scale_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:

        # Initialize scaler
        scaler = self.get_scaler()
        scaler.fit(data)

        # Scale data
        data_scaled = self.scale_data(data, scaler)

        # Return scaled data and scaler
        return data_scaled, scaler
