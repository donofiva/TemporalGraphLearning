import pandas as pd


class MaskedMetric:

    def __init__(self, metric):
        self._metric = metric

    @staticmethod
    def _mask_targets(targets: pd.DataFrame, masks: pd.DataFrame):

        # Apply mask
        targets_masked = targets.values * masks.values

        # Rebuild targets
        return pd.DataFrame(
            targets_masked,
            index=targets.index,
            columns=targets.columns
        )

    def compute(self, targets: pd.DataFrame, targets_predicted: pd.DataFrame, masks: pd.DataFrame):
        return self._metric.compute(
            self._mask_targets(targets, masks),
            self._mask_targets(targets_predicted, masks)
        )