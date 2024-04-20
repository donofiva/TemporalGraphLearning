from enum import Enum, unique


@unique
class Field(Enum):

    TURBINE = 0
    DAY = 1
    TIMESLOT = 2
