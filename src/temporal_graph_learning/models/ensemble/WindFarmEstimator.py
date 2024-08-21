import pandas as pd

from typing import Tuple, Dict


class WindFarmEstimator:

    def __init__(self, wind_turbine_to_estimator: Dict, apply_mask: bool = True):
        self._wind_turbine_to_estimator = wind_turbine_to_estimator
        self._apply_mask = apply_mask

    def predict(self, wind_turbine_to_X_mask: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]):

        # Predictions buffer
        y_wind_farm = []

        # Iterate over wind turbine in wind farm
        for wind_turbine, estimator in self._wind_turbine_to_estimator.items():

            # Retrieve estimator
            X_wind_turbine, mask_wind_turbine = wind_turbine_to_X_mask[wind_turbine]

            # Predict y and store it as a dataframe
            y_wind_turbine = estimator.predict(X_wind_turbine)
            y_wind_turbine = pd.DataFrame(
                y_wind_turbine,
                columns=mask_wind_turbine.columns,
                index=mask_wind_turbine.index
            )

            # Apply mask, if required
            if self._apply_mask:

                # Preserve mask
                mask_wind_turbine = mask_wind_turbine.copy()
                mask_wind_turbine.columns = y_wind_turbine.columns

                # Apply mask
                y_wind_turbine = y_wind_turbine.where(mask_wind_turbine, 0.0)

            # Store prediction
            y_wind_farm.append(y_wind_turbine)

        # Produce resulting dataset
        y_wind_farm = pd.concat(y_wind_farm, axis=0)
        y_wind_farm = y_wind_farm.groupby(y_wind_farm.index).sum()

        return y_wind_farm
