import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Hashable
from temporal_graph_learning.data.parsers.DatasetParser import DatasetParser
from temporal_graph_learning.data.datasets.WindTurbineChannelsDataset import WindTurbineChannelsDataset


class WindTurbinesChannelsDatasetParser(DatasetParser):

    def __init__(self, dataset: pd.DataFrame, device: str = 'cpu'):
        super().__init__(dataset, device)

    # Dataset methods
    def split_on_dimension(self, dimensions: List[str]) -> Dict[Hashable, "WindTurbinesChannelsDatasetParser"]:
        return {
            dimensions: WindTurbinesChannelsDatasetParser(dataset_slice.reset_index(drop=True), self._device)
            for dimensions, dataset_slice in self._dataset.groupby(dimensions, as_index=False)
        }

    # Split methods
    def train_test_split(
            self,
            *dimensions_set: List[str],
            test_size=0.2
    ) -> Tuple["DatasetParser", ...]:
        return super().train_test_split(
            *dimensions_set,
            test_size=test_size,
            shuffle=False,
            stratify=None
        )

    # Feature engineering methods
    def apply_cyclical_time_encoding(self):

        # Retrieve timeslots
        timeslots = self.retrieve_dimension_from_dataset('TIMESLOT')

        # Apply cyclical encoding
        minutes = timeslots.map(lambda ts: int(ts.split(':')[0]) * 60 + int(ts.split(':')[1]))
        radians = minutes / 1440 * 2 * np.pi
        time_sin = np.sin(radians)
        time_cos = np.cos(radians)

        # Store encoded dimensions
        self.store_dimension('TIME_SIN', time_sin)
        self.store_dimension('TIME_COS', time_cos)

        # Remove timeslot dimension
        self.drop_dimensions(['TIMESLOT'])

    def aggregate_and_transform_blades_pitch_angle(self):

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
        blade_pitch_angle_aggregated = np.cos(blade_pitch_angle_aggregated)

        # Store aggregated blade pitch angle
        self.store_dimension('PITCH_ANGLE', blade_pitch_angle_aggregated)

        # Remove blade pitch angles
        self.drop_dimensions([
            'PITCH_ANGLE_FIRST_BLADE',
            'PITCH_ANGLE_SECOND_BLADE',
            'PITCH_ANGLE_THIRD_BLADE'
        ])

    def transform_nacelle_direction(self):
        pass

    def convert_masks_to_int(self):

        # Convert and store masks
        masks = self.convert_dimension('DATA_AVAILABLE', int)
        self.store_dimension('DATA_AVAILABLE', masks)

    # PyTorch dataset methods
    def build_wind_turbine_channels_dataset(
            self,
            window: int = 1,
            horizon: int = 1
    ) -> WindTurbineChannelsDataset:

        # Retrieve channels, mask and target
        channels = DatasetParser(
            self.retrieve_dimensions_from_dataset_parsed([
                    'WIND_SPEED',
                    'WIND_DIRECTION',
                    'EXTERNAL_TEMPERATURE',
                    'INTERNAL_TEMPERATURE',
                    'NACELLE_DIRECTION',
                    'PITCH_ANGLE',
                    'REACTIVE_POWER',
                    'ACTIVE_POWER',
                    'TIME_COS',
                    'TIME_SIN',
            ])
        )

        mask = DatasetParser(
            self.retrieve_dimensions_from_dataset_parsed([
                'DATA_AVAILABLE'
            ])
        )

        target = DatasetParser(
            self.retrieve_dimensions_from_dataset_parsed([
                'ACTIVE_POWER'
            ])
        )

        # Build dataset
        return WindTurbineChannelsDataset(
            channels=channels.to_tensor(),
            masks=mask.to_tensor(),
            targets=target.to_tensor(),
            window=window,
            horizon=horizon
        )


    def build_wind_turbine_channels_train_and_test_datasets(
            self,
            test_size: float = 0.2,
            window: int = 1,
            horizon: int = 1
    ) -> Tuple[WindTurbineChannelsDataset, WindTurbineChannelsDataset]:

        # Define dimension sets
        channels_dimensions = [
            'WIND_SPEED',
            'WIND_DIRECTION',
            'EXTERNAL_TEMPERATURE',
            'INTERNAL_TEMPERATURE',
            'NACELLE_DIRECTION',
            'PITCH_ANGLE',
            'REACTIVE_POWER',
            'ACTIVE_POWER',
            'TIME_COS',
            'TIME_SIN',
        ]

        mask_dimensions = [
            'DATA_AVAILABLE'
        ]

        target_dimensions = [
            'ACTIVE_POWER'
        ]

        # Perform train test split
        (
            channels_train,
            channels_test,
            mask_train,
            mask_test,
            target_train,
            target_test
        ) = self.train_test_split(
            channels_dimensions, mask_dimensions, target_dimensions,
            test_size=test_size
        )

        # Build datasets
        train_dataset = WindTurbineChannelsDataset(
            channels=channels_train.to_tensor(),
            masks=mask_train.to_tensor(),
            targets=target_train.to_tensor(),
            window=window,
            horizon=horizon
        )

        test_dataset = WindTurbineChannelsDataset(
            channels=channels_test.to_tensor(),
            masks=mask_test.to_tensor(),
            targets=target_test.to_tensor(),
            window=window,
            horizon=horizon
        )

        return train_dataset, test_dataset
