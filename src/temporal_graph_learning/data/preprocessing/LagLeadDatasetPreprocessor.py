import pandas as pd


class LagLeadDatasetPreprocessor:

    def __init__(self, window, delay, horizon):

        self.window = window
        self.delay = delay
        self.horizon = horizon

    def _lag_features(self, X: pd.DataFrame):

        # Preserve X
        X = X.copy()

        # Lagged X buffer
        lagged_X_slices = []

        # Apply backward looking window
        for lag in range(self.window - 1, 0, -1):

            lagged_X_slice = X.shift(lag)
            lagged_X_slice.columns = [f"{col}_W{lag}" for col in X.columns]
            lagged_X_slices.append(lagged_X_slice)

        # Store X_w = 0
        X.columns = [f"{col}_W0" for col in X.columns]
        lagged_X_slices.append(X)

        # Concatenate all lagged dataframes
        lagged_X = pd.concat(lagged_X_slices, axis=1)

        return lagged_X

    def _lead_targets(self, y: pd.DataFrame):

        # Preserve y
        y = y.copy()

        # Lead y buffer
        lead_y_slices = []

        # Apply forward-looking horizon
        for lead in range(self.horizon):

            lead_y_slice = y.shift(-(self.delay + lead))
            lead_y_slice.columns = [f"{col}_D{self.delay}_H{lead}" for col in y.columns]
            lead_y_slices.append(lead_y_slice)

        # Concatenate all lead dataframes
        lead_y = pd.concat(lead_y_slices, axis=1)

        return lead_y

    def transform(self, X: pd.DataFrame, y: pd.DataFrame):

        # Create lagged features and lead targets
        lagged_X = self._lag_features(X)
        lead_y = self._lead_targets(y)

        # Align the indices by dropping rows with NaNs from both dataframes
        valid_indices = lagged_X.dropna().index.intersection(lead_y.dropna().index)
        lagged_X.dropna().index.intersection(lead_y.dropna().index)

        # Filter datasets
        lagged_X = lagged_X.loc[valid_indices]
        lead_y = lead_y.loc[valid_indices]

        return lagged_X, lead_y