import unittest

import numpy as np

from temporal_graph_learning.data.mask.ConditionalMasker import *


class TestConditionalMasker(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [10, 20, 30, 40, 50]
        })

    def test_mask_greater_than(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='A',
                comparisons=[
                    MaskingComparison('C', MaskingComparisonType.GREATER, 25)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([1, 2, np.nan, np.nan, np.nan], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_mask_less_than(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='B',
                comparisons=[
                    MaskingComparison('C', MaskingComparisonType.LESS, 25)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, np.nan, 3, 2, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_multiple_conditions(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='B',
                comparisons=[
                    MaskingComparison('A', MaskingComparisonType.GREATER, 1),
                    MaskingComparison('C', MaskingComparisonType.LESS, 50)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([5, np.nan, np.nan, np.nan, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_equal_to(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='B',
                comparisons=[
                    MaskingComparison('A', MaskingComparisonType.EQUAL, 2)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        print(result)
        expected = pd.Series([5, np.nan, 3, 2, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_not_equal_to(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='B',
                comparisons=[
                    MaskingComparison('A', MaskingComparisonType.NOT_EQUAL, 3)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, np.nan, 3, np.nan, np.nan], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_in(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='A',
                comparisons=[
                    MaskingComparison('A', MaskingComparisonType.IN, {2, 4})
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([1, np.nan, 3, np.nan, 5], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_mask_not_in(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='A',
                comparisons=[
                    MaskingComparison('A', MaskingComparisonType.NOT_IN, {2, 4})
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, 2, np.nan, 4, np.nan], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_input_validation(self):
        masker = ConditionalMasker([
            MaskingCondition(
                dimensions='A',
                comparisons=[
                    MaskingComparison('C', MaskingComparisonType.GREATER, 25)
                ]
            )
        ])
        with self.assertRaises(TypeError):
            masker.transform(np.array([1, 2, 3]))

    def test_no_conditions(self):
        masker = ConditionalMasker([])
        result = masker.transform(self.data.copy())
        pd.testing.assert_frame_equal(result, self.data)