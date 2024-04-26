import unittest
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
            MaskingRule(
                dimensions='A',
                conditions=[
                    MaskingCondition('C', MaskingComparison.GREATER, 25)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([1, 2, np.nan, np.nan, np.nan], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_mask_less_than(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='B',
                conditions=[
                    MaskingCondition('C', MaskingComparison.LESS, 25)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, np.nan, 3, 2, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_multiple_conditions(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='B',
                conditions=[
                    MaskingCondition('A', MaskingComparison.GREATER, 1),
                    MaskingCondition('C', MaskingComparison.LESS, 50)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([5, np.nan, np.nan, np.nan, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_equal_to(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='B',
                conditions=[
                    MaskingCondition('A', MaskingComparison.EQUAL, 2)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        print(result)
        expected = pd.Series([5, np.nan, 3, 2, 1], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_not_equal_to(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='B',
                conditions=[
                    MaskingCondition('A', MaskingComparison.NOT_EQUAL, 3)
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, np.nan, 3, np.nan, np.nan], name='B')
        pd.testing.assert_series_equal(result['B'], expected)

    def test_mask_in(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='A',
                conditions=[
                    MaskingCondition('A', MaskingComparison.IN, {2, 4})
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([1, np.nan, 3, np.nan, 5], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_mask_not_in(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='A',
                conditions=[
                    MaskingCondition('A', MaskingComparison.NOT_IN, {2, 4})
                ]
            )
        ])
        result = masker.transform(self.data.copy())
        expected = pd.Series([np.nan, 2, np.nan, 4, np.nan], name='A')
        pd.testing.assert_series_equal(result['A'], expected)

    def test_input_validation(self):
        masker = ConditionalMasker([
            MaskingRule(
                dimensions='A',
                conditions=[
                    MaskingCondition('C', MaskingComparison.GREATER, 25)
                ]
            )
        ])
        with self.assertRaises(TypeError):
            masker.transform(np.array([1, 2, 3]))

    def test_no_conditions(self):
        masker = ConditionalMasker([])
        result = masker.transform(self.data.copy())
        pd.testing.assert_frame_equal(result, self.data)