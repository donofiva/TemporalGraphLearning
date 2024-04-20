import pandas as pd

from typing import Set
from legacy.abstraction.entities.Turbine import Turbine
from temporal_graph_learning.legacy.abstraction.entities.Day import Day
from legacy.abstraction.entities import Timeslot
from legacy.abstraction.entities import Snapshot


class AbstractionManager:

    @staticmethod
    def build_turbines_abstraction(turbines_table: pd.DataFrame) -> Set[Turbine]:
        return {
            Turbine.build_from_index_and_coordinates(row.TURBINE, row.X_AXIS, row.Y_AXIS)
            for _, row in turbines_table.iterrows()
        }

    @staticmethod
    def replace_keys_with_entities(timeseries_table: pd.DataFrame) -> pd.DataFrame:

        timeseries_table['TURBINE'] = timeseries_table['TURBINE'].map(Turbine.get_from_index)
        timeseries_table['DAY'] = timeseries_table['DAY'].map(Day.build_from_index)
        timeseries_table['TIMESLOT'] = timeseries_table['TIMESLOT'].map(Timeslot.build_from_hh_mm_string)

        return timeseries_table

    @staticmethod
    def store_timeseries_abstraction(timeseries_table: pd.DataFrame):

        # Store timeseries
        for turbine, timeseries_slice in timeseries_table.groupby('TURBINE', as_index=False):

            # Define timeseries slice on turbine
            timeseries_slice = timeseries_slice.drop(columns='TURBINE').set_index(['DAY', 'TIMESLOT'])

            # Store timeseries
            turbine: Turbine = turbine
            turbine.store_timeseries(timeseries_slice)

        return timeseries_table

    @staticmethod
    def build_snapshots_abstraction(timeseries_table: pd.DataFrame) -> Set[Snapshot]:
        return {
            Snapshot(day, timeslot, snapshot.set_index('TURBINE'))
            for (day, timeslot), snapshot in timeseries_table.groupby(['DAY', 'TIMESLOT'], as_index=False)
        }
