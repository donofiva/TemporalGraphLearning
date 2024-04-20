from enum import Enum, unique


@unique
class Comparator(Enum):

    EQUALS = 0
    GREATER = 1
    LESS = 2

    def _get_lambda(self):
        return {
            Comparator.EQUALS: lambda x, y: x == y,
            Comparator.GREATER: lambda x, y: x > y,
            Comparator.LESS: lambda x, y: x < y
        }[self]

    def compare(self, left_hand_value, right_hand_value):
        return self._get_lambda()(left_hand_value, right_hand_value)
