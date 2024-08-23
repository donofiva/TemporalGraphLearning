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

    def _lag_features(self, channels: pd.DataFrame):

        # Define shifts and suffixes for lagging
        shifts = list(range(self.window - 1, -1, -1))
        suffixes = [f"W{lag}" for lag in shifts]

        # Handle multi-index columns by grouping at the first level
        if isinstance(channels.columns, pd.MultiIndex):

            lagged_channels_slices = [
                self._shift_and_rename(group, shifts, suffixes)
                for _, group in channels.groupby(level=0, axis=1)
            ]

            return pd.concat(lagged_channels_slices, axis=1)

        # Apply lagging directly for single-level columns
        return self._shift_and_rename(channels, shifts, suffixes)

    def _lead_targets(self, targets: pd.DataFrame):

        # Define shifts and suffixes for leading
        shifts = [-(self.delay + lead) for lead in range(self.horizon)]
        suffixes = [f"D{self.delay}_H{lead}" for lead in range(self.horizon)]

        # Handle multi-index columns by grouping at the first level
        if isinstance(targets.columns, pd.MultiIndex):

            lead_targets_slices = [
                self._shift_and_rename(group, shifts,  suffixes)
                for _, group in targets.groupby(level=0, axis=1)
            ]

            return pd.concat(lead_targets_slices, axis=1)

        # Apply leading directly for single-level columns
        return self._shift_and_rename(targets, shifts,  suffixes)

    def transform(self, channels: pd.DataFrame, masks: pd.DataFrame, targets: pd.DataFrame):

        # Create lagged features and lead targets
        lagged_channels = self._lag_features(channels)
        lead_masks = self._lead_targets(masks)
        lead_targets = self._lead_targets(targets)

        # Align the indices by dropping rows with NaNs from both dataframes
        valid_indices = lagged_channels.dropna().index.intersection(lead_targets.dropna().index)

        # Filter datasets by valid indices
        lagged_channels = lagged_channels.loc[valid_indices]
        lead_masks = lead_masks.loc[valid_indices]
        lead_targets = lead_targets.loc[valid_indices]

        return lagged_channels, lead_masks, lead_targets
