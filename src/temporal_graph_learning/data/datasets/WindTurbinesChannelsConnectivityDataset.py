from torch import Tensor
from torch.utils.data import Dataset


class WindTurbineChannelsPositionDataset(Dataset):

    def __init__(
            self,
            adjacency_matrices: Tensor,
            channels: Tensor,
            targets: Tensor,
            window: int,
            lag: int,
            horizon: int
    ):

        # Store tensor data
        self._adjacency_matrices = adjacency_matrices
        self._channels = channels
        self._targets = targets

        # Store input window and prediction horizon
        self._window = window
        self._lag = lag
        self._horizon = horizon

    def __len__(self):
        return (
            self._channels.shape[0] -
            self._window -
            self._lag -
            self._horizon +
            1
        )

    def __getitem__(self, index):
        return (
            self._adjacency_matrices[index:(index + self._window)],
            self._channels[index:(index + self._window)],
            self._targets[(index + self._window + self._lag - 1):(index + self._window + self._lag - 1 + self._horizon)]
        )
