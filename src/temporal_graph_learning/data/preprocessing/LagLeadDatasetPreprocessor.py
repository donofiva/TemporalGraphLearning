import pandas as pd
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class LagLeadDatasetPreprocessor:

    def __init__(self, window: int, delay: int, horizon: int):

        self.window = window
        self.delay = delay
        self.horizon = horizon

    @staticmethod
    def _shift_and_rename(df: pd.DataFrame, shifts: list, suffixes: list):

        # Initialize buffer for shifted dataframes
        shifted_dfs = []

        # Apply shifts and rename columns
        for shift, suffix in zip(shifts, suffixes):
            shifted_df = df.shift(shift)

            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                shifted_df.columns = pd.MultiIndex.from_tuples(
                    [(col[0], f"{col[1]}_{suffix}") for col in df.columns]
                )

            else:
                shifted_df.columns = [f"{col}_{suffix}" for col in df.columns]

            # Append to buffer
            shifted_dfs.append(shifted_df)

        # Concatenate all shifted dataframes
        return pd.concat(shifted_dfs, axis=1)

    def _lag_features(self, X: pd.DataFrame):

        # Define shifts and suffixes for lagging
        shifts = list(range(self.window - 1, -1, -1))
        suffixes = [f"W{lag}" for lag in shifts]

        # Handle multi-index columns by grouping at the first level
        if isinstance(X.columns, pd.MultiIndex):

            lagged_X_slices = [
                self._shift_and_rename(group, shifts, suffixes)
                for _, group in X.groupby(level=0, axis=1)
            ]

            return pd.concat(lagged_X_slices, axis=1)

        # Apply lagging directly for single-level columns
        return self._shift_and_rename(X, shifts, suffixes)

    def _lead_targets(self, y: pd.DataFrame):

        # Define shifts and suffixes for leading
        shifts = [-(self.delay + lead) for lead in range(self.horizon)]
        suffixes = [f"D{self.delay}_H{lead}" for lead in range(self.horizon)]

        # Handle multi-index columns by grouping at the first level
        if isinstance(y.columns, pd.MultiIndex):

            lead_y_slices = [
                self._shift_and_rename(group, shifts,  suffixes)
                for _, group in y.groupby(level=0, axis=1)
            ]

            return pd.concat(lead_y_slices, axis=1)

        # Apply leading directly for single-level columns
        return self._shift_and_rename(y, shifts,  suffixes)

    def transform(self, X: pd.DataFrame, y: pd.DataFrame):

        # Create lagged features and lead targets
        lagged_X = self._lag_features(X)
        lead_y = self._lead_targets(y)

        # Align the indices by dropping rows with NaNs from both dataframes
        valid_indices = lagged_X.dropna().index.intersection(lead_y.dropna().index)

        # Filter datasets by valid indices
        lagged_X = lagged_X.loc[valid_indices]
        lead_y = lead_y.loc[valid_indices]

        return lagged_X, lead_y
