from typing import List, Set

from legacy.abstraction.entities.Turbine import Turbine
from legacy.abstraction.entities import Snapshot
from legacy.model import Comparisons


class AbnormalValuesManager:

    def __init__(self, abnormal_values: List[Comparisons]):
        self.abnormal_values = abnormal_values

    def evaluate_abnormal_values(self, snapshots: Set[Snapshot]):
        for snapshot in snapshots:
            for turbine in Turbine.get_all():

                for an in self.abnormal_values:
                    if all([
                        comparison.comparator.compare(
                            snapshot.get_datapoint_by_turbine_and_dimension(
                                turbine,
                                comparison.dimension
                            ),
                            comparison.value
                        )
                        for comparison in an.comparisons
                    ]):
                        # print(f"Unfeasible turbine {turbine.index} at TS: {snapshot.timeslot} at day {snapshot.day}")
                        break