import unittest
import numpy as np
from pipeline import *

class TestPredep(unittest.TestCase):

    def test_predep_basic(self):
        # Arrange
        x = np.random.rand(100)

        # Act
        result = predep(x)

        # Assert
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)  # Non-negative due to KDE

    def test_predep_empty_array(self):
        # Arrange
        x = np.array([])

        # Act
        with self.assertRaises(ValueError):
            predep(x)

    def test_predep_invalid_num_boots(self):
        # Arrange
        x = np.random.rand(100)
        num_boots = 0

        # Act
        with self.assertRaises(ValueError):
            predep(x, num_boots)

class TestClusterAssign(unittest.TestCase):

    def test_cluster_assign_basic(self):
        # Arrange
        x = np.array([1, 2, 3, 4])
        clusters = [[1, 2], [3, 4]]

        # Act
        labels = cluster_assign(x, clusters)

        # Assert
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def test_cluster_assign_empty_clusters(self):
        # Arrange
        x = np.array([1, 2])
        clusters = []

        # Act
        with self.assertRaises(ValueError):
            cluster_assign(x, clusters)

    def test_cluster_assign_out_of_range(self):
        # Arrange
        x = np.array([10])
        clusters = [[1, 2]]

        # Act
        labels = cluster_assign(x, clusters)

        # Assert
        self.assertEqual(labels.tolist(), [-1])

class TestCalcPredepXy(unittest.TestCase):
    def test_calc_predep_xy_empty_data(self):
        # Arrange
        data = np.array([])
        clusters = [[1, 2]]

        # Act
        with self.assertRaises(IndexError):
            calc_predep_xy(data, clusters)

    def test_calc_predep_xy_single_data_point(self):
        # Arrange
        data = np.array([[1, 2]])
        clusters = [[1, 2]]

        # Act
        predep_i = calc_predep_xy(data, clusters)

        # Assert
        self.assertEqual(predep_i, 0.0)  # No calculation for single point

class TestBootstrap(unittest.TestCase):

    def test_bootstrap_basic(self):
        # Arrange
        num_boots = 50
        data = np.array([[i,i] for i in range(100)])
        segmentation_method = lambda x: [[0,50],[50,100]]  # Mock segmentation

        # Act
        bootstrap_stats = bootstrap(data, segmentation_method,num_boots = num_boots)

        # Assert
        self.assertEqual(len(bootstrap_stats), num_boots)  # Number of replicates
        # Check for some non-zero values (replace with more specific assertions)
        self.assertGreater(sum(bootstrap_stats), 0.0)
if __name__ == '__main__':
    unittest.main()
