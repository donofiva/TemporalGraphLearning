import numpy as np
import pandas as pd

from typing import Union
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

from tsl.ops.similarities import gaussian_kernel
from tsl.datasets.prototypes import DatetimeDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin


class SDWPFDataset(DatetimeDataset, MissingValuesMixin):

    similarity_options = {'physical_proximity', 'cosine_similarity'}

    def __init__(
            self,
            target: pd.DataFrame,
            mask: pd.DataFrame,
            channels: pd.DataFrame,
            position: pd.DataFrame,
            similarity_score: str = 'physical_proximity'
    ):

        # Initialize superclass
        super().__init__(
            target=target,
            mask=mask,
            covariates={'channels_': channels},
            similarity_score=similarity_score
        )

        # Add physical position as covariate
        self.add_covariate('position', position)

    # Connectivity
    def _compute_physical_proximity(self) -> np.ndarray:

        # Parse position
        position = self.position[['X_AXIS', 'Y_AXIS']].values

        # Compute square physical distance matrix
        physical_distance_matrix = squareform(pdist(position, 'euclidean'))

        # Apply gaussian filter to extract physical proximity
        return gaussian_kernel(
            physical_distance_matrix,
            theta=np.std(physical_distance_matrix)
        )

    def _compute_cosine_similarity(self) -> np.ndarray:

        # Stack vertically exogenous variables for pairwise cosine similarity
        channels_stack = [
            channel.droplevel(1, axis=1).reset_index(drop=True)
            for channel in self.exogenous.values()
        ]

        channels_stack = pd.concat(
            channels_stack,
            axis=0,
            ignore_index=True
        )

        # Enforce shape and remove missing values
        channels_stack = channels_stack.values.T
        channels_stack = channels_stack[:, ~np.isnan(channels_stack).any(axis=0)]

        # Compute pairwise cosine similarity matrix
        return cosine_similarity(channels_stack)

    def compute_similarity(self, method: str, **kwargs) -> np.ndarray:

        if method == 'physical_proximity':
            return self._compute_physical_proximity()

        if method == 'cosine_similarity':
            return self._compute_cosine_similarity()

    def get_target(self):
        return self.target

    def get_mask(
            self,
            dtype: Union[type, str, np.dtype] = None,
            as_dataframe: bool = False
    ):
        return super().get_mask()

    def get_channels(self):
        return self.channels_



# if __name__ == '__main__':
#
#     # Read datasets
#     folder = '/Users/ivandonofrio/Workplace/Thesis/TemporalGraphLearning/assets'
#     dataset_channels = pd.read_csv(f'{folder}/wind_turbines_channels.csv')
#     dataset_position = pd.read_csv(f'{folder}/wind_turbines_position.csv')
#
#     # Parse position dataset
#     dataset_position.set_index('TURBINE', inplace=True)
#
#     # Parse temporal dataset
#     # Parse date to datetime and get rid of day and timeslot
#     start_date = '2024-01-01'
#
#     dataset_channels['DATETIME'] = (
#         pd.to_datetime(dataset_channels['DAY'], unit='D', origin=pd.Timestamp(start_date)) +
#         pd.to_timedelta(dataset_channels['TIMESLOT'].map(lambda t: f'{t}:00'))
#     )
#
#     dataset_channels.set_index('DATETIME', inplace=True)
#     dataset_channels.drop(columns=['DAY', 'TIMESLOT'], inplace=True)
#
#     # Enforce index format
#     dataset_channels = dataset_channels.set_index('TURBINE', append=True)
#     dataset_channels = dataset_channels.unstack('TURBINE')
#     dataset_channels.columns = dataset_channels.columns.swaplevel(0, 1)
#     dataset_channels.sort_index(axis=1, level=0, inplace=True)
#
#     # Extract data
#     target = dataset_channels.loc[:, pd.IndexSlice[:, 'ACTIVE_POWER']]
#     mask = dataset_channels.loc[:, pd.IndexSlice[:, 'DATA_AVAILABLE']]
#     channels = dataset_channels.drop(columns=dataset_channels.loc[:, pd.IndexSlice[:, ['ACTIVE_POWER', 'DATA_AVAILABLE']]].columns)
#
#     # Build covariates
#     channels_labels = {column[1] for column in channels.columns}
#
#     channels = {
#         channel_label: channels.loc[:, pd.IndexSlice[:, channel_label]]
#         for channel_label in channels_labels
#     }
#
#     sdwpf = SDWPFDataset(target, mask, channels, dataset_position)
#
#     dataset_covariates = {
#         'timestamp': sdwpf.datetime_encoded('hour').values,
#         **sdwpf.covariates
#     }
#
#     # Try to build SpatioTemporalDataset
#     stdataset = SpatioTemporalDataset(
#         target=sdwpf.target,
#         mask=sdwpf.mask,
#         covariates=dataset_covariates,
#         connectivity=sdwpf.get_connectivity(
#             method='physical_proximity',
#             threshold=0.7
#         )
#     )
#
#     dm = SpatioTemporalDataModule(
#         dataset=stdataset,
#         scalers=None,
#         splitter=TemporalSplitter(val_len=0.1, test_len=0.1),
#         batch_size=8,
#         workers=1,
#     )
#
#     predictor = Predictor(
#         model_class=GraphWaveNetModel,
#         model_kwargs={
#             'n_nodes': sdwpf.n_nodes,
#             'input_size': sdwpf.n_channels,
#             'output_size': sdwpf.n_channels,
#             'horizon': sdwpf.horizon,
#             'exog_size': sdwpf.input_map.u.shape[-1]
#         },
#         optim_class=torch.optim.Adam,
#         optim_kwargs={},
#         loss_fn=loss_fn,
#         metrics=log_metrics,
#         scheduler_class=scheduler_class,
#         scheduler_kwargs=scheduler_kwargs,
#         scale_target=cfg.scale_target,
#     )
#
#     print(stdataset[1])