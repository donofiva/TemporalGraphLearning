import unittest
import pandas as pd

from temporal_graph_learning.abstraction.Dataset import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'DAY': [1, 1, 2, 2],
            'TIMESLOT': [1, 2, 1, 2],
            'VALUE': [100.0, 200.0, 300.0, 400.0]
        })

    def test_extend_with_shifted_targets_basic(self):

        columns, result_df = Dataset.extend_with_shifted_dimension(self.df, 'VALUE', 2)

        # Check if the correct columns were added
        expected_columns = ['VALUE_SHIFT_1', 'VALUE_SHIFT_2']
        self.assertListEqual(columns, expected_columns)

        # Check if the new columns have correct values
        pd.testing.assert_series_equal(
            result_df['VALUE_SHIFT_1'],
            pd.Series([200.0, 300.0], name='VALUE_SHIFT_1')
        )

        pd.testing.assert_series_equal(
            result_df['VALUE_SHIFT_2'],
            pd.Series([300.0, 400.0], name='VALUE_SHIFT_2')
        )

    def test_column_not_found(self):
        with self.assertRaises(ValueError):
            _, _ = Dataset.extend_with_shifted_dimension(self.df, 'Nonexistent', 1)

    def test_negative_window_size(self):
        with self.assertRaises(ValueError):
            _, _ = Dataset.extend_with_shifted_dimension(self.df, 'VALUE', -1)

    def test_zero_window_size(self):
        with self.assertRaises(ValueError):
            _, _ = Dataset.extend_with_shifted_dimension(self.df, 'VALUE', 0)

    def test_window_size_too_large(self):

        columns, result_df = Dataset.extend_with_shifted_dimension(self.df, 'VALUE', 10)

        # Check if all the extended columns are filled with None from a certain point
        self.assertTrue(result_df['VALUE_SHIFT_10'].isna().all())
