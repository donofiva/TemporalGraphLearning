from typing import List, Dict, Set


class Day:

    _entities: Dict[int, "Day"] = dict()

    @classmethod
    def build_from_index(cls, index: int):

        # Initialize and try to store temporary entity
        day = cls(index)
        cls._entities.setdefault(day.key, day)

        return cls._entities[day.key]

    @classmethod
    def get_from_index(cls, index: int) -> "Day":
        return cls._entities.get(index)

    @classmethod
    def get_all(cls) -> Set["Day"]:
        return set(cls._entities.values())

    @classmethod
    def get_all_sorted(cls) -> List["Day"]:
        return sorted(cls._entities.values())

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: "Day"):
        return type(self) is type(other) and hash(self) == hash(other)

    def __lt__(self, other: "Day"):
        return self.index < other.index

    def __le__(self, other: "Day"):
        return self < other or self == other

    def __str__(self):
        return f'DAY_{self.index}'

    @property
    def key(self):
        return self.index