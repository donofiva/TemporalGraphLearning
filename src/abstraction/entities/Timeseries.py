from typing import Dict
from src.abstraction.entities.Day import Day
from src.abstraction.entities.Timeslot import Timeslot


class Timeseries:

    def __init__(self, day_to_timeslot_to_datapoint: Dict[Day, Dict[Timeslot, float]]):
        self._day_to_timeslot_to_datapoint: Dict[Day, Dict[Timeslot, float]] = day_to_timeslot_to_datapoint

    def get_timeslot_to_datapoint_by_day(self, day: Day) -> Dict[Timeslot, float]:
        return self._day_to_timeslot_to_datapoint.get(day, {})

    def get_datapoint_by_day_and_timeslot(self, day: Day, timeslot: Timeslot) -> float:
        return self.get_timeslot_to_datapoint_by_day(day).get(timeslot)