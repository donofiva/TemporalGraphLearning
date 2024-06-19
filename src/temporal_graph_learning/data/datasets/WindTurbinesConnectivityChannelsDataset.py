from torch import Tensor
from torch.utils.data import Dataset


class WindTurbinesConnectivityChannelsDataset(Dataset):

    def __init__(
            self,
            connectivity_matrices: Tensor,
            channels_grids: Tensor,
            masks_grids: Tensor,
            targets_grids: Tensor,
            window: int,
            lag: int,
            horizon: int
    ):

        # Store tensor data
        self._connectivity_matrices = connectivity_matrices
        self._channels_grids = channels_grids
        self._masks_grids = masks_grids
        self._targets_grids = targets_grids

        # Store input window and prediction horizon
        self._window = window
        self._lag = lag
        self._horizon = horizon

    def __len__(self):
        return (
            self._channels_grids.shape[0] -
            self._window -
            self._lag -
            self._horizon +
            1
        )

    def __getitem__(self, index):
        return (
            self._connectivity_matrices[index:(index + self._window)],
            self._channels_grids[index:(index + self._window)],
            self._masks_grids[index:(index + self._window)],
            self._masks_grids[(index + self._window + self._lag - 1):(index + self._window + self._lag - 1 + self._horizon)],
            self._targets_grids[(index + self._window + self._lag - 1):(index + self._window + self._lag - 1 + self._horizon)]
        )
