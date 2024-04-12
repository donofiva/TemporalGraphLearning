from enum import Enum, unique


@unique
class File(Enum):

    LOCATIONS = 'locations.csv'
    TIMESERIES = 'timeseries.csv'
