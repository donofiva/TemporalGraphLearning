import unittest

from legacy.abstraction.entities import Timeslot


class TestDay(unittest.TestCase):

    def test_entities(self):

        Timeslot.build_from_hh_mm_string("10:00")
        Timeslot.build_from_hh_mm_string("10:00")

        self.assertEqual(len(Timeslot.get_all()), 1)

    def test_equality(self):

        day_a = Timeslot.build_from_hh_mm_string("10:00")
        day_b = Timeslot.build_from_hh_mm_string("10:00")

        self.assertEqual(day_a, day_b)

    def test_inequality(self):

        day_a = Timeslot.build_from_hh_mm_string("10:00")
        day_b = Timeslot.build_from_hh_mm_string("11:00")

        self.assertNotEqual(day_a, day_b)

    def test_less_than(self):

        day_a = Timeslot.build_from_hh_mm_string("10:00")
        day_b = Timeslot.build_from_hh_mm_string("11:00")

        self.assertLess(day_a, day_b)

    def test_less_or_equal_than(self):

        day_a = Timeslot.build_from_hh_mm_string("10:00")
        day_b = Timeslot.build_from_hh_mm_string("11:00")
        day_c = Timeslot.build_from_hh_mm_string("11:00")

        self.assertLessEqual(day_a, day_b)
        self.assertLessEqual(day_b, day_c)