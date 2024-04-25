import unittest
import numpy as np
import pandas as pd

from temporal_graph_learning.data.impute.LinearInterpolationImputer import LinearInterpolationImputer


class TestLinearInterpolationImputer(unittest.TestCase):

    def setUp(self):
        self.imputer = LinearInterpolationImputer()

    def test_with_list_input(self):
        data = [1, 2, np.nan, 4, 5, np.nan, 7]
        expected = np.array([1, 2, 3, 4, 5, 6, 7])
        result = self.imputer.transform(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_numpy_array(self):
        data = np.array([1, 2, np.nan, 4, 5, np.nan, 7])
        expected = np.array([1, 2, 3, 4, 5, 6, 7])
        result = self.imputer.transform(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_pandas_series(self):
        data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7])
        expected = pd.Series([1, 2, 3, 4, 5, 6, 7])
        result = self.imputer.transform(data)
        pd.testing.assert_series_equal(result, expected, check_dtype=False, check_names=False)

    def test_with_pandas_dataframe(self):
        data = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [np.nan, 2, 3, 4]})
        expected = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4]})
        result = self.imputer.transform(data)
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_with_2d_numpy_array(self):
        data = np.array([[1, np.nan], [np.nan, 4], [3, 5]])
        expected = np.array([[1, 3.0], [2.0, 4], [3, 5]])
        result = self.imputer.transform(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_edge_nan_values(self):
        data = np.array([np.nan, np.nan, 3, 4, 5, np.nan, np.nan])
        expected = np.array([1, 2, 3, 4, 5, 6, 7])
        result = self.imputer.transform(data)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fit_method(self):
        # Testing fit to ensure it doesn't do anything and returns self
        result = self.imputer.fit([1, 2, np.nan])
        self.assertEqual(result, self.imputer, "Fit method should return self")