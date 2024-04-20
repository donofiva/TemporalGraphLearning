import pandas as pd

from pathlib import Path
from legacy.configuration import Configuration
from legacy.abstraction.entities.Dimension import Dimension


class DataManager:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def _filter_input_file(self, input_file: pd.DataFrame):

        # Apply viable filters
        for field, filter_values in self.configuration.field_to_filter_values.items():
            if field.name in input_file.columns:
                input_file = input_file[input_file[field.name].isin(filter_values)]

        # Return filtered input file
        return input_file

    def read_turbines_input_file(self, turbines_input_file_path: Path) -> pd.DataFrame:

        # Read input file
        turbines = pd.read_csv(turbines_input_file_path)
        turbines.columns = ['TURBINE', 'X_AXIS', 'Y_AXIS']

        # Filter input file
        return self._filter_input_file(turbines)

    def read_timeseries_input_file(self, timeseries_input_file_path: Path) -> pd.DataFrame:

        # Load input file
        timeseries = pd.read_csv(timeseries_input_file_path)

        # Convert column names using dimension names
        timeseries.columns = [
            'TURBINE',
            'DAY',
            'TIMESLOT',
            *[dimension.name for dimension in Dimension]
        ]

        # Make sure that input file is properly sorted
        timeseries = timeseries.sort_values(['TURBINE', 'DAY', 'TIMESLOT'])

        # Filter input file
        return self._filter_input_file(timeseries)
