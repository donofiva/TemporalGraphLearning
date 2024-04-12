from enum import Enum, unique


@unique
class Directory(Enum):

    INPUTS = 'inputs'
    OUTPUTS = 'outputs'
