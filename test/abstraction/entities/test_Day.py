import unittest

from legacy.abstraction.entities import Day


class TestDay(unittest.TestCase):

    def test_entities(self):

        Day.build_from_index(1)
        Day.build_from_index(1)

        self.assertEqual(len(Day.get_all()), 1)

    def test_equality(self):

        day_a = Day.build_from_index(1)
        day_b = Day.build_from_index(1)

        self.assertEqual(day_a, day_b)

    def test_inequality(self):

        day_a = Day.build_from_index(1)
        day_b = Day.build_from_index(2)

        self.assertNotEqual(day_a, day_b)

    def test_less_than(self):

        day_a = Day.build_from_index(1)
        day_b = Day.build_from_index(2)

        self.assertLess(day_a, day_b)

    def test_less_or_equal_than(self):

        day_a = Day.build_from_index(1)
        day_b = Day.build_from_index(2)
        day_c = Day.build_from_index(2)

        self.assertLessEqual(day_a, day_b)
        self.assertLessEqual(day_b, day_c)

