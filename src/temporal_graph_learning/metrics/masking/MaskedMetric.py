import pandas as pd


class MaskedMetric:

    def __init__(self, metric, masks: pd.DataFrame):

        self._metric = metric
        self._masks = masks

    def _mask_targets(self, targets: pd.DataFrame):

        # Apply mask
        targets_masked = targets.values * self._masks.values

        # Rebuild targets
        return pd.DataFrame(
            targets_masked,
            index=targets.index,
            columns=targets.columns
        )

    def compute(self, targets: pd.DataFrame, targets_predicted: pd.DataFrame):
        return self._metric.compute(
            self._mask_targets(targets),
            self._mask_targets(targets_predicted)
        )