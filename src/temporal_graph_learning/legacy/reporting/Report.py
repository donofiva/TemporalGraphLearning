from enum import IntEnum, unique


@unique
class Report(IntEnum):

    TURBINES = 0
    TIMESERIES = 1
