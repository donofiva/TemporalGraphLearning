import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Hashable
from temporal_graph_learning.data.parsers.TabularDatasetParser import TabularDatasetParser
from temporal_graph_learning.data.datasets.WindTurbineChannelsDataset import WindTurbineChannelsDataset


class WindTurbinesChannelsParser(TabularDatasetParser):

    @classmethod
    def from_chunks(cls, *parsers: "WindTurbinesChannelsParser"):
        return cls(
            pd.concat(
                list(map(lambda parser: parser.get_dataset(), parsers)),
                ignore_index=True
            )
        )

    def __init__(self, dataset: pd.DataFrame):
        super().__init__(dataset)

    # Feature engineering methods
    def aggregate_and_transform_blades_pitch_angle(self, drop_dimensions: bool = True):

        # Retrieve blade pitch angles
        blade_pitch_angles = self.retrieve_dimensions_from_dataset([
            'PITCH_ANGLE_FIRST_BLADE',
            'PITCH_ANGLE_SECOND_BLADE',
            'PITCH_ANGLE_THIRD_BLADE'
        ])

        # Aggregate blade pitch angles
        blade_pitch_angle_aggregated = blade_pitch_angles.mean(axis=1)

        # Convert angle to radians and transform
        blade_pitch_angle_aggregated = np.radians(blade_pitch_angle_aggregated)
        blade_pitch_angle_aggregated_cos = np.cos(blade_pitch_angle_aggregated)

        # Store aggregated blade pitch angle
        self.store_dimension('PITCH_ANGLE_COS', blade_pitch_angle_aggregated_cos)

        # Remove blade pitch angles
        if drop_dimensions:
            self.drop_dimensions([
                'PITCH_ANGLE_FIRST_BLADE',
                'PITCH_ANGLE_SECOND_BLADE',
                'PITCH_ANGLE_THIRD_BLADE'
            ])

    def transform_wind_direction(self, drop_dimensions: bool = True):

        # Retrieve wind direction
        wind_direction = self.retrieve_dimension_from_dataset('WIND_DIRECTION')

        # Convert angle to radians and transform
        wind_direction = np.radians(wind_direction)
        wind_direction_cos = np.cos(wind_direction)

        # Store transformed wind direction
        self.store_dimension('WIND_DIRECTION_COS', wind_direction_cos)

        # Drop wind direction
        if drop_dimensions:
            self.drop_dimensions(['WIND_DIRECTION'])

    def transform_nacelle_direction(self, drop_dimensions: bool = True):

        # Retrieve wind direction
        nacelle_direction = self.retrieve_dimension_from_dataset('NACELLE_DIRECTION')

        # Convert angle to radians and transform
        nacelle_direction = np.radians(nacelle_direction)
        nacelle_direction_cos = np.cos(nacelle_direction)
        nacelle_direction_sin = np.sin(nacelle_direction)

        # Store transformed nacelle direction
        self.store_dimension('NACELLE_DIRECTION_COS', nacelle_direction_cos)
        self.store_dimension('NACELLE_DIRECTION_SIN', nacelle_direction_sin)

        # Drop nacelle direction
        if drop_dimensions:
            self.drop_dimensions(['NACELLE_DIRECTION'])

    def transform_masks(self):

        # Transform masks
        masks = self.convert_dimension('DATA_AVAILABLE', int)

        # Store transformed masks
        self.store_dimension('DATA_AVAILABLE', masks)

    # Dataset methods
    @staticmethod
    def _set_datetime_index(dataset: pd.DataFrame) -> pd.DataFrame:

        # Define dummy start date
        start_date = '2024-01-01'

        # Convert day and timeslot to datetime
        dataset['DATETIME'] = (
            pd.to_datetime(dataset['DAY'], unit='D', origin=pd.Timestamp(start_date)) +
            pd.to_timedelta(dataset['TIMESLOT'].map(lambda t: f'{t}:00'))
        )

        # Remove day and timeslot columns
        dataset = dataset.drop(columns=['DAY', 'TIMESLOT'])

        # Set datetime index
        dataset = dataset.set_index('DATETIME')

        return dataset

    @staticmethod
    def _set_wind_turbine_multi_index(dataset: pd.DataFrame) -> pd.DataFrame:

        # Add wind turbine to index
        dataset = dataset.set_index('TURBINE', append=True)

        # Move wind turbine index as column index
        dataset = dataset.unstack('TURBINE')

        # Enforce multi-index format
        dataset.columns = dataset.columns.swaplevel(0, 1)
        dataset = dataset.sort_index(axis=1, level=0)

        return dataset

    def get_target_mask_and_channels(
            self,
            target_labels: List[str] = ['ACTIVE_POWER'],
            mask_label: str = 'DATA_AVAILABLE',
            preserve_target_as_channel: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # Preserve dataset
        channels = self._dataset.copy()

        # Set datetime index
        channels = self._set_datetime_index(channels)
        channels = self._set_wind_turbine_multi_index(channels)

        # Retrieve channels labels
        channels_drop = [mask_label] + ([] if preserve_target_as_channel else target_labels)

        return (
            channels.loc[:, pd.IndexSlice[:, target_labels]],
            channels.loc[:, pd.IndexSlice[:, mask_label]],
            channels.drop(columns=channels.loc[:, pd.IndexSlice[:, channels_drop]].columns)
        )



if __name__ == '__main__':

    folder = '/Users/ivandonofrio/Workplace/Thesis/TemporalGraphLearning/assets'
    dataset_channels = pd.read_csv(f'{folder}/wind_turbines_channels.csv')

    parser = WindTurbinesChannelsParser(dataset_channels)
    print(parser.get_target_mask_and_channels())