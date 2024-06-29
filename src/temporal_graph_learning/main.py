import numpy as np
import pandas as pd

from typing import List
from sklearn.preprocessing import MinMaxScaler

from temporal_graph_learning.data.parsers.WindTurbinesChannelsParser import WindTurbinesChannelsParser
from temporal_graph_learning.data.impute.KNNImputer import KNNImputer


if __name__ == '__main__':

    # Read datasets
    folder = '/Users/ivandonofrio/Workplace/Thesis/TemporalGraphLearning/assets'
    dataset_channels = pd.read_csv(f'{folder}/wind_turbines_channels.csv')

    # Initialize parsers
    parser = WindTurbinesChannelsParser(dataset_channels)
    index_to_wind_turbine = dict(enumerate(sorted(parser.get_dataset().TURBINE.unique())))

    # Perform relevant transformations
    parser.transform_masks()
    parser.transform_wind_direction()
    parser.transform_nacelle_direction()
    parser.aggregate_and_transform_blades_pitch_angle()

    # Parse dataset
    _, masks, channels = parser.get_target_mask_and_channels(preserve_target_as_channel=False)

    # Initialize dataset
    scaler = MinMaxScaler()

    # Scale dataset
    channels_scaled = pd.DataFrame(
        scaler.fit_transform(channels),
        columns=channels.columns,
        index=channels.index
    )

    # Initialize and fit KNN imputer
    knn_imputer = KNNImputer(min_items=50, neighbours=10)
    knn_imputer.fit(channels_scaled, masks)

    # Impute missing values
    channels_scaled_imputed = knn_imputer.impute(channels_scaled)
    print(channels_scaled.isna().sum().sum())
    print(channels_scaled_imputed.isna().sum().sum())
    exit()

    #
    print(
        list(
            map(
                lambda timestamp_wind_turbine_channel: knn_imputer.impute(*timestamp_wind_turbine_channel),
                missing_values
            )
        )
    )

    # Time it
    print(
        timeit.timeit(
            lambda: knn_imputer.impute(
                pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=20),
                1,
                'EXTERNAL_TEMPERATURE'
            ),
            number=100000
        )
    )
    exit()

    # Retrieve missing values
    channels_scaled_nan = channels_scaled[channels_scaled.isna().any(axis=1)]

    for timeslot in channels_scaled_nan.index:

        for (wind_turbine, channel) in channels_scaled_nan.columns:
            if pd.isna(channels_scaled_nan.loc[timeslot, (wind_turbine, channel)]):
                knn_imputer.impute(timeslot, wind_turbine, channel)

        print(timeslot)

    exit()

    print(channels_scaled[channels_scaled.isna().any(axis=1)])

    exit()

    # # Swap channels with wind turbines index
    # channels_scaled_swapped = channels_scaled.swaplevel(axis=1).sort_index(axis=1)
    #
    # # Stack dataset on channels
    # channels_scaled_swapped_stack = channels_scaled_swapped.stack(0, future_stack=True)
    # channels_scaled_swapped_stack_values = channels_scaled_swapped_stack.dropna().values.T
    #
    # # Compute cosine similarity and remove self-loops
    # similarity_matrix = cosine_similarity(channels_scaled_swapped_stack_values)
    # np.fill_diagonal(similarity_matrix, 0)
    #
    # # Store closest available turbines by timestamp and wind turbine
    # timestamp_to_wind_turbine_to_closest_wind_turbines = {}
    #
    # # Timestamp breakdown
    # for index, mask in masks.iterrows():
    #
    #     # Store mask as array
    #     mask = mask.values
    #
    #     # Store
    #     wind_turbine_to_closest_wind_turbines = timestamp_to_wind_turbine_to_closest_wind_turbines.setdefault(index, {})
    #
    #     # Make sure that you have enough active turbines
    #     if mask.sum() > 110:
    #
    #         # Extract closest turbine
    #         closest_wind_turbines_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    #         closest_wind_turbines_values = np.take_along_axis(similarity_matrix, closest_wind_turbines_indices, axis=1)
    #
    #         # Extract valid closest turbines
    #         for wind_turbine_index in range(closest_wind_turbines_indices.shape[0]):
    #
    #             # Retrieve close wind turbines and associated mask
    #             closest_wind_turbines = closest_wind_turbines_indices[wind_turbine_index, :]
    #             closest_wind_turbines_mask = mask[closest_wind_turbines]
    #
    #             # Store available close wind turbines
    #             wind_turbine_to_closest_wind_turbines[wind_turbine_index] = list(
    #                 closest_wind_turbines[closest_wind_turbines_mask == 1][:5]
    #             )
    #
    # # Test function
    # wind_turbine_index = 0
    # wind_turbine = index_to_wind_turbine[wind_turbine_index]
    # timestamp = list(timestamp_to_wind_turbine_to_closest_wind_turbines.keys())[1]
    # mask = masks.loc[timestamp]
    # close_turbine = [
    #     index_to_wind_turbine[i]
    #     for i in timestamp_to_wind_turbine_to_closest_wind_turbines[timestamp][wind_turbine_index]
    # ]
    #
    # retrieve_and_aggregate_neighbour_channels(
    #     timestamp,
    #     wind_turbine,
    #     close_turbine,
    #     channels,
    #     'EXTERNAL_TEMPERATURE'
    # )

    #
    # print(timestamp_to_wind_turbine_to_closest_wind_turbines)
    # exit()
    #
    # mask = np.ones_like(cosine_similarity_matrix)
    #
    # masked_similarity = np.where(mask, cosine_similarity_matrix, -np.inf)
    # top_n_indices = np.argsort(masked_similarity, axis=1)[:, -5:][:, ::-1]
    # top_n_values = np.take_along_axis(masked_similarity, top_n_indices, axis=1)
    #
    # print(top_n_indices, top_n_values)

    # fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    # sns.heatmap(cosine_similarity_matrix, vmin=0, vmax=1, ax=ax)
    # plt.show()

    # # Stack channels
    # channels_stack = channels.stack(0, dropna=False, future_stack=True)
    #
    # # Extract principal components
    # pca = PCA(n_components=2)
    # pca.fit(channels_stack.dropna())
    #
    # print(pca.explained_variance_)
    # print(pca.explained_variance_ratio_)

    # print(channels)