import pandas as pd

from typing import Set, Tuple
from src.abstraction.entities.Turbine import Turbine
from src.abstraction.entities.Day import Day
from src.abstraction.entities.Timeslot import Timeslot
from src.abstraction.entities.Dimension import Dimension
from src.abstraction.entities.Timeseries import Timeseries
from src.abstraction.entities.GridSnapshot import GridSnapshot
from src.threads.MultiThreadingManager import MultiThreadingManager


class AbstractionManager:

    @staticmethod
    def build_turbines_abstraction(locations_table: pd.DataFrame) -> Set[Turbine]:
        return set(
            Turbine.build_from_index_and_coordinates(row.TURBINE, row.X_AXIS, row.Y_AXIS)
            for _, row in locations_table.iterrows()
        )

    @staticmethod
    def _store_timeseries(turbine_to_turbine_slice: Tuple[Turbine, pd.DataFrame]):

        # Explode arguments
        turbine, turbine_slice = turbine_to_turbine_slice

        # Generate and store timeseries
        for dimension in Dimension:

            # Generate timeseries dictionary
            day_to_timeslot_to_datapoint = {
                day: dict(zip(
                    turbine_day_slice['TIMESLOT'],
                    turbine_day_slice[dimension.name]
                ))
                for day, turbine_day_slice in turbine_slice.groupby('DAY', as_index=False)
            }

            # Initialize
            timeseries = Timeseries(day_to_timeslot_to_datapoint)

            # Store timeseries
            turbine: Turbine = turbine
            turbine.store_timeseries_by_dimension(dimension, timeseries)

    @staticmethod
    def _slice_and_store_timeseries(timeseries_table: pd.DataFrame):

        # Initialize multi-threading manager
        multi_threading_manager = MultiThreadingManager(number_of_workers=25)

        # Slice dataset turbine-wise
        turbine_to_turbine_slice = [
            (turbine, turbine_slice)
            for turbine, turbine_slice
            in timeseries_table.groupby('TURBINE', as_index=False)
        ]

        # Populate timeseries representation
        multi_threading_manager.parallelize_pool(
            turbine_to_turbine_slice,
            AbstractionManager._store_timeseries
        )

    @staticmethod
    def build_timeseries_abstraction(timeseries_table: pd.DataFrame):

        # Backup timeseries table
        timeseries_table = timeseries_table.copy()

        # Map timeseries entities
        timeseries_table['TURBINE'] = timeseries_table['TURBINE'].map(Turbine.get_from_index)
        timeseries_table['DAY'] = timeseries_table['DAY'].map(Day.build_from_index)
        timeseries_table['TIMESLOT'] = timeseries_table['TIMESLOT'].map(Timeslot.build_from_hh_mm_string)

        # Store timeseries
        AbstractionManager._slice_and_store_timeseries(timeseries_table)

    @staticmethod
    def _build_grid_snapshot(day_timeslot_dimension_turbines: Tuple[Day, Timeslot, Dimension, Turbine]):
        return GridSnapshot.build_from_day_timeslot_dimension_and_turbines(*day_timeslot_dimension_turbines)

    @staticmethod
    def build_grid_snapshots_abstraction(turbines: Set[Turbine]) -> Set[GridSnapshot]:

        # Initialize multi-threading manager
        multi_threading_manager = MultiThreadingManager(number_of_workers=25)

        # Initialize grid snapshot primitives
        day_timeslot_dimension_turbines = [
            (day, timeslot, dimension, turbines)
            for day in Day.get_all()
            for timeslot in Timeslot.get_all()
            for dimension in Dimension
        ]

        # Generate grid snapshot
        grid_snapshots = multi_threading_manager.parallelize_pool(
            day_timeslot_dimension_turbines,
            AbstractionManager._build_grid_snapshot
        )

        return set(grid_snapshots)
