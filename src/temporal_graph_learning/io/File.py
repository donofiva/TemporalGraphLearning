from enum import Enum, unique


@unique
class File(Enum):

    TURBINES = 'turbines.csv'
    TIMESERIES = 'timeseries.csv'
