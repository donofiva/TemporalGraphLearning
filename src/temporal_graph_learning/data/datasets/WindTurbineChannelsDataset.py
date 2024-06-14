from torch.utils.data import Dataset


class WindTurbineChannelsDataset(Dataset):

    def __init__(
            self,
            channels,
            masks,
            targets,
            window: int,
            horizon: int
    ):

        # Store tensor data
        self._channels = channels
        self._masks = masks
        self._targets = targets

        # Store input window and prediction horizon
        self._window = window
        self._horizon = horizon

    def __len__(self):
        return (
            self._channels.shape[0] -
            self._window -
            self._horizon +
            1
        )

    def __getitem__(self, index):
        return (
            self._channels[index:(index + self._window)],
            self._masks[index:(index + self._window)],
            self._masks[(index + self._window):(index + self._window + self._horizon)],
            self._targets[(index + self._window):(index + self._window + self._horizon)]
        )
