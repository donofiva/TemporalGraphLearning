import os
import copy
import pickle
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
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
            decomposer,
            estimator,
            test_size: float = 0.2,
            mask_predictions: bool = True,
            label: str = ''
    ):

        # Store estimator template
        self._lag_lead_preprocessor = lag_lead_preprocessor
        self._scaler = scaler
        self._decomposer = decomposer
        self._estimator = estimator

        # Store label
        self._label = label

        # Store dataset split configuration
        self._test_size = test_size

        # Store masking logic configuration
        self._mask_predictions = mask_predictions

        # Wind turbines split
        self._wind_turbines: List[int] = []
        self._wind_turbine_to_scaler_channels: Dict = {}
        self._wind_turbine_to_scaler_targets: Dict = {}
        self._wind_turbine_to_decomposer = {}
        self._wind_turbine_to_estimator: Dict = {}

        # Store dataset
        self._channels: pd.DataFrame = channels
        self._masks: pd.DataFrame = masks
        self._targets: pd.DataFrame = targets

        # Store train test split
        self._channels_train: Optional[pd.DataFrame] = None
        self._channels_test: Optional[pd.DataFrame] = None
        self._masks_train: Optional[pd.DataFrame] = None
        self._masks_test: Optional[pd.DataFrame] = None
        self._targets_train: Optional[pd.DataFrame] = None
        self._targets_test: Optional[pd.DataFrame] = None

    def __del__(self):
        del self._channels_train, self._channels_test, self._masks_train, self._masks_test, self._targets_train, self._targets_test

    # Replicating entities by wind turbine
    def _get_estimator(self):
        return copy.deepcopy(self._estimator)

    def _get_scaler(self):
        return copy.deepcopy(self._scaler)

    def _get_decomposer(self):
        return copy.deepcopy(self._decomposer)

    # Scaler
    @staticmethod
    def _initialize_scaler(df: pd.DataFrame, scaler):
        scaler.fit_transform(df)

    @staticmethod
    def _scale(df: pd.DataFrame, scaler) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.transform(df),
            index=df.index,
            columns=df.columns
        )

    @staticmethod
    def _invert_scaling(df: pd.DataFrame, scaler) -> pd.DataFrame:
        return pd.DataFrame(
            scaler.inverse_transform(df),
            index=df.index,
            columns=df.columns
        )

    # Decomposer
    @staticmethod
    def _initialize_decomposer(df: pd.DataFrame, decomposer):
        decomposer.fit(df)

    @staticmethod
    def _decompose(df: pd.DataFrame, decomposer) -> pd.DataFrame:
        return pd.DataFrame(
            decomposer.transform(df),
            index=df.index,
            columns=pd.MultiIndex.from_product([
                df.columns.get_level_values(0).unique(),
                [f'PC_{component}' for component in range(decomposer.n_components_)]
            ])
        )

    # Initialize wind farm estimator
    def _store_wind_turbines(self):
        self._wind_turbines = self._channels.columns.get_level_values(0).unique().tolist()

    def _store_scaler_by_wind_turbine(self):

        self._wind_turbine_to_channels_scaler = {
            wind_turbine: self._get_scaler()
            for wind_turbine in self._wind_turbines
        }

        self._wind_turbine_to_targets_scaler = {
            wind_turbine: self._get_scaler()
            for wind_turbine in self._wind_turbines
        }

    def _store_decomposer_by_wind_turbine(self):
        self._wind_turbine_to_decomposer = {
            wind_turbine: self._get_decomposer()
            for wind_turbine in self._wind_turbines
        }

    def _store_estimators_by_wind_turbine(self):
        self._wind_turbine_to_estimator = {
            wind_turbine: self._get_estimator()
            for wind_turbine in self._wind_turbines
        }

    def initialize(self):
        self._store_wind_turbines()
        self._store_scaler_by_wind_turbine()
        self._store_decomposer_by_wind_turbine()
        self._store_estimators_by_wind_turbine()

    # Apply scaling and decomposition
    def _train_test_split(self, channels: pd.DataFrame, masks: pd.DataFrame, targets: pd.DataFrame):
        return train_test_split(
            channels,
            masks,
            targets,
            test_size=self._test_size,
            shuffle=False
        )

    def apply_temporal_window_and_decomposition(self):

        # Dataset chunks
        channels_train_chunks = []
        channels_test_chunks = []
        masks_train_chunks = []
        masks_test_chunks = []
        targets_train_chunks = []
        targets_test_chunks = []

        # For each wind turbine, extract temporal representation and apply decomposition
        for wind_turbine in tqdm(self._wind_turbines):

            # Retrieve channels, masks and targets for each wind turbine
            channels = self._channels.loc[:, pd.IndexSlice[wind_turbine, :]]
            masks = self._masks.loc[:, pd.IndexSlice[wind_turbine, :]]
            targets = self._targets.loc[:, pd.IndexSlice[wind_turbine, :]]

            # Apply lag lead transformation
            channels, masks, targets = self._lag_lead_preprocessor.transform(channels, masks, targets)

            # Perform train-test split
            (
                channels_train,
                channels_test,
                masks_train,
                masks_test,
                targets_train,
                targets_test
            ) = self._train_test_split(channels, masks, targets)

            # Retrieve scaler and decomposer
            scaler_channels = self._wind_turbine_to_channels_scaler[wind_turbine]
            scaler_targets = self._wind_turbine_to_targets_scaler[wind_turbine]
            decomposer = self._wind_turbine_to_decomposer[wind_turbine]

            # For each turbine, initialize channels scaler
            self._initialize_scaler(channels_train, scaler_channels)

            # And scale train and test channels
            channels_train = self._scale(channels_train, scaler_channels)
            channels_test = self._scale(channels_test, scaler_channels)

            # For each turbine, initialize targets scaler
            self._initialize_scaler(targets_train, scaler_targets)

            # And scale train targets
            targets_train = self._scale(targets_train, scaler_targets)

            # Apply decomposition if required
            if decomposer is not None:

                # For each turbine, initialize decomposer
                self._initialize_decomposer(channels_train, decomposer)

                # And scale train and test channels
                channels_train = self._decompose(channels_train, decomposer)
                channels_test = self._decompose(channels_test, decomposer)

            # Store chunks
            channels_train_chunks.append(channels_train)
            channels_test_chunks.append(channels_test)
            masks_train_chunks.append(masks_train)
            masks_test_chunks.append(masks_test)
            targets_train_chunks.append(targets_train)
            targets_test_chunks.append(targets_test)

            # Free memory
            del channels_train, channels_test, masks_train, masks_test, targets_train, targets_test

        # Store split
        self._channels_train = pd.concat(channels_train_chunks, axis=1)
        self._channels_test = pd.concat(channels_test_chunks, axis=1)
        self._masks_train = pd.concat(masks_train_chunks, axis=1)
        self._masks_test = pd.concat(masks_test_chunks, axis=1)
        self._targets_train = pd.concat(targets_train_chunks, axis=1)
        self._targets_test = pd.concat(targets_test_chunks, axis=1)

        # Free memory
        del channels_train_chunks, channels_test_chunks, masks_train_chunks, masks_test_chunks, targets_train_chunks, targets_test_chunks

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
            prediction = self._invert_scaling(prediction, scaler)

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
        return (
            self._channels_train.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._masks_train.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._targets_train.loc[:, pd.IndexSlice[wind_turbine, :]]
        )

    def get_test_channels_masks_and_targets_by_wind_turbine(self, wind_turbine: int):
        return (
            self._channels_test.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._masks_test.loc[:, pd.IndexSlice[wind_turbine, :]],
            self._targets_test.loc[:, pd.IndexSlice[wind_turbine, :]]
        )

    def get_estimator_by_wind_turbine(self, wind_turbine: int):
        return self._wind_turbine_to_estimator.get(wind_turbine)

    # Pickle
    def store(self, directory: str):

        # Make directories
        os.makedirs(directory, exist_ok=True)

        # Store model
        with open(f'{directory}/{self._label}.pickle', 'wb') as file:
            pickle.dump(self, file)
