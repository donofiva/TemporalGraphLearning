from typing import Dict, Tuple, Set


class Timeslot:

    _entities: Dict[Tuple[int, int], "Timeslot"] = dict()

    @classmethod
    def build_from_hh_mm_string(cls, token: str):

        # Parse hours and minutes
        hours, minutes = cls.parse_hh_mm_string(token)

        # Initialize and try to store temporary entity
        timeslot = Timeslot(hours, minutes)
        cls._entities[timeslot.key] = cls(hours, minutes)

        return cls._entities[timeslot.key]

    @classmethod
    def get_from_hh_mm_string(cls, token: str) -> "Timeslot":
        return cls._entities.get(cls.parse_hh_mm_string(token))

    @classmethod
    def get_all(cls) -> Set["Timeslot"]:
        return set(cls._entities.values())

    def __init__(self, hours: int, minutes: int):

        self.hours = hours
        self.minutes = minutes

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: "Timeslot"):
        return type(self) is type(other) and hash(self) == hash(other)

    def __lt__(self, other: "Timeslot"):
        return self.hours < other.hours or (self.hours == other.hours and self.minutes == other.minutes)

    def __le__(self, other: "Timeslot"):
        return self < other or self == other

    def __str__(self):
        return f"{str(self.hours).zfill(2)}:{str(self.minutes).zfill(2)}"

    @property
    def key(self):
        return self.hours, self.minutes

    @staticmethod
    def parse_hh_mm_string(token: str) -> Tuple[int, int]:

        # Parse tokens to int
        tokens = list(map(int, token.split(':')))

        return tokens[0], tokens[1]
