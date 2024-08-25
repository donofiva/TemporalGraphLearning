import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Dict
from sklearn.model_selection import train_test_split

from temporal_graph_learning.data.preprocessing.LagLeadDatasetPreprocessor import LagLeadDatasetPreprocessor


class WindFarmEstimator:

    def __init__(
            self,
            channels: pd.DataFrame,
            masks: pd.DataFrame,
            targets: pd.DataFrame,
            lag_lead_preprocessor: LagLeadDatasetPreprocessor,
            scaler,
            pca,
            estimator,
            test_size: float = 0.2,
            mask_predictions: bool = True
    ):

        # Store channels, masks and targets
        self._channels = channels
        self._masks = masks
        self._targets = targets

        # Store estimator template
        self._lag_lead_preprocessor = lag_lead_preprocessor
        self._scaler = scaler
        self._pca = pca
        self._estimator = estimator

        # Store dataset split configuration
        self._test_size = test_size

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
        self._wind_turbine_to_channels_scaler: Dict = {}
        self._wind_turbine_to_targets_scaler: Dict = {}
        self._wind_turbine_to_pca = {}
        self._wind_turbine_to_estimator: Dict = {}
        self._wind_turbine_to_train: Dict = {}
        self._wind_turbine_to_test: Dict = {}

    def _get_estimator(self):
        return copy.deepcopy(self._estimator)

    def _get_scaler(self):
        return copy.deepcopy(self._scaler)

    def _get_pca(self):
        return copy.deepcopy(self._pca)

    @staticmethod
    def _initialize_scaling(df: pd.DataFrame, scaler):
        scaler.fit_transform(df)

    @staticmethod
    def _apply_scaling(df: pd.DataFrame, scaler) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(df),
            index=df.index,
            columns=df.columns
        )

    @staticmethod
    def _inverse_scaling(df: pd.DataFrame, scaler) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.inverse_transform(df),
            index=df.index,
            columns=df.columns
        )

    @staticmethod
    def _initialize_pca(df: pd.DataFrame, pca):
        pca.fit(df)

    @staticmethod
    def _apply_pca(df: pd.DataFrame, pca) -> pd.DataFrame:
        return pd.DataFrame(
            pca.transform(df),
            index=df.index,
            columns=[f'PC_{component}' for component in range(pca.n_components_)]
        )

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
            test_size=self._test_size
        )

    def _load_wind_turbines(self):
        self._wind_turbines = self._channels.columns.get_level_values(0).unique().tolist()

    def _load_estimators(self):
        self._wind_turbine_to_estimator = {
            wind_turbine: self._get_estimator()
            for wind_turbine in self._wind_turbines
        }

    def _load_scaler(self):

        self._wind_turbine_to_channels_scaler = {
            wind_turbine: self._get_scaler()
            for wind_turbine in self._wind_turbines
        }

        self._wind_turbine_to_targets_scaler = {
            wind_turbine: self._get_scaler()
            for wind_turbine in self._wind_turbines
        }

    def _load_pca(self):
        self._wind_turbine_to_pca = {
            wind_turbine: self._get_pca()
            for wind_turbine in self._wind_turbines
        }

    def initialize(self):
        self._load_wind_turbines()
        self._load_scaler()
        self._load_pca()
        self._load_estimators()

    def apply_temporal_window_and_decomposition(self):

        # For each wind turbine, extract temporal representation and apply decomposition
        for wind_turbine in tqdm(self._wind_turbines):

            # Retrieve channels, masks and targets for each wind turbine
            channels = self._channels.loc[:, pd.IndexSlice[wind_turbine, :]]
            masks = self._masks.loc[:, pd.IndexSlice[wind_turbine, :]]
            targets = self._targets.loc[:, pd.IndexSlice[wind_turbine, :]]

            # Apply lag lead transformation
            channels, masks, targets = self._lag_lead_preprocessor.transform(channels, masks, targets)

            # Split dataset
            (
                channels_train,
                channels_test,
                masks_train,
                masks_test,
                targets_train,
                targets_test
            ) = train_test_split(
                channels,
                masks,
                targets,
                test_size=self._test_size,
                shuffle=False
            )

            # For each turbine, fit scaler and apply to train and test targets
            channels_scaler = self._wind_turbine_to_channels_scaler[wind_turbine]
            self._initialize_scaling(channels_train, channels_scaler)
            channels_train = self._apply_scaling(channels_train, channels_scaler)
            channels_test = self._apply_scaling(channels_test, channels_scaler)

            # For each turbine, fit scaler to train targets
            targets_scaler = self._wind_turbine_to_targets_scaler[wind_turbine]
            self._initialize_scaling(targets_train, targets_scaler)
            targets_train = self._apply_scaling(targets_train, targets_scaler)

            # If required, apply PCA
            pca = self._wind_turbine_to_pca[wind_turbine]

            if pca is not None:

                # For each turbine, fit PCA to train targets
                self._initialize_pca(channels_train, self._wind_turbine_to_pca[wind_turbine])
                channels_train = self._apply_pca(channels_train, self._wind_turbine_to_pca[wind_turbine])
                channels_test = self._apply_pca(channels_test, self._wind_turbine_to_pca[wind_turbine])

            self._wind_turbine_to_train[wind_turbine] = (channels_train, masks_train, targets_train)
            self._wind_turbine_to_test[wind_turbine] = (channels_test, masks_test, targets_test)

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
            channels, _, targets = self.get_test_channels_masks_and_targets_by_wind_turbine(wind_turbine)

            # Retrieve predictions from estimator
            prediction = estimator.predict(channels)
            prediction = pd.DataFrame(prediction, index=targets.index, columns=targets.columns)

            # Inverse scaling
            scaler = self._wind_turbine_to_targets_scaler[wind_turbine]
            prediction = self._inverse_scaling(prediction, scaler)

            # Store prediction
            predictions.append(prediction)

        # Store predictions
        return pd.concat(predictions, axis=1)

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
        return self._wind_turbine_to_train[wind_turbine]

    def get_test_channels_masks_and_targets_by_wind_turbine(self, wind_turbine: int):
        return self._wind_turbine_to_test[wind_turbine]

    def get_estimator_by_wind_turbine(self, wind_turbine: int):
        return self._wind_turbine_to_estimator.get(wind_turbine)
