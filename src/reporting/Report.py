from enum import IntEnum, unique


@unique
class Report(IntEnum):

    LOCATIONS = 0
    TIMESERIES = 1
