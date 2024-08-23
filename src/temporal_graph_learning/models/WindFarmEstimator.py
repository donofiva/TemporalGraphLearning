import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Dict
from sklearn.model_selection import train_test_split


class WindFarmEstimator:

    def __init__(
            self,
            channels: pd.DataFrame,
            masks: pd.DataFrame,
            targets: pd.DataFrame, estimator,
            test_size: float = 0.2,
            shuffle_dataset: bool = False,
            mask_predictions: bool = True
    ):

        # Store channels, masks and targets
        self._channels = channels
        self._masks = masks
        self._targets = targets

        # Store estimator template
        self._estimator = estimator

        # Store dataset split configuration
        self._test_size = test_size
        self._shuffle_dataset = shuffle_dataset

        # Store masking logic configuration
        self._mask_predictions = mask_predictions

        # Train and test split
        self._channels_train = None
        self._channels_test = None
        self._masks_train = None
        self._masks_test = None
        self._targets_train = None
        self._targets_test = None

        # Wind turbines split
        self._wind_turbines: List[int] = []
        self._wind_turbine_to_estimator: Dict = {}

    def _get_estimator(self):
        return copy.deepcopy(self._estimator)

    def _load_train_test_split(self):
        (
            self._channels_train,
            self._channels_test,
            self._masks_train,
            self._masks_test,
            self._targets_train,
            self._targets_test
        ) = train_test_split(
            self._channels,
            self._masks,
            self._targets,
            test_size=self._test_size,
            shuffle=self._shuffle_dataset
        )

    def _load_wind_turbines(self):
        self._wind_turbines = self._channels.columns.get_level_values(0).unique().tolist()

    def _load_estimators(self):
        self._wind_turbine_to_estimator = {
            wind_turbine: self._get_estimator()
            for wind_turbine in self._wind_turbines
        }

    def initialize(self):

        # Load wind turbines and estimators
        self._load_wind_turbines()
        self._load_estimators()

        # Split dataset
        self._load_train_test_split()

    def fit(self):

        # Fit model for each wind turbine
        for wind_turbine in tqdm(self._wind_turbines):

            # Retrieve estimator and dataset slices on wind turbine
            estimator = self.get_estimator_by_wind_turbine(wind_turbine)
            channels, _, targets = self.get_train_channels_masks_and_targets_by_wind_turbine(wind_turbine)

            # Train estimator
            estimator.fit(channels, targets)

    def predict(self) -> pd.DataFrame:

        # Prediction buffer
        predictions = []

        # Fit model for each wind turbine
        for wind_turbine in tqdm(self._wind_turbines):

            # Retrieve estimator and dataset slices on wind turbine
            estimator = self.get_estimator_by_wind_turbine(wind_turbine)
            channels, _, _ = self.get_test_channels_masks_and_targets_by_wind_turbine(wind_turbine)

            # Retrieve predictions from estimator
            prediction = estimator.predict(channels)
            predictions.append(prediction)

        # Store predictions
        predictions = np.hstack(predictions)

        # Store predictions dataframe
        return pd.DataFrame(predictions, columns=self._targets_test.columns)

    def get_train_channels_masks_and_targets(self):
        return (
            self._channels_train,
            self._masks_train,
            self._targets_train
        )

    def get_test_channels_masks_and_targets(self):
        return (
            self._channels_test,
            self._masks_test,
            self._targets_test
        )

    def get_train_channels_masks_and_targets_by_wind_turbine(self, wind_turbine: int):
        return (
            self._channels_train.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._masks_train.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._targets_train.loc[:, pd.IndexSlice[wind_turbine, :]],
        )

    def get_test_channels_masks_and_targets_by_wind_turbine(self, wind_turbine: int):
        return (
            self._channels_test.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._masks_test.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._targets_test.loc[:, pd.IndexSlice[wind_turbine, :]],
        )

    def get_estimator_by_wind_turbine(self, wind_turbine: int):
        return self._wind_turbine_to_estimator.get(wind_turbine)
