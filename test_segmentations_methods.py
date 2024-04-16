import unittest
import numpy as np
from segmentations_methods import *


class TestSegmentation(unittest.TestCase):

  def test_segmentation_basic(self):
    # Arrange
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    n_clusters = 2

    # Act
    clusters = segmentation_by_predep(data, n_clusters)

    # Assert
    self.assertEqual(len(clusters), n_clusters + 1)
    # Check that each cluster is a list of indices
    for cluster in clusters:
      self.assertIsInstance(cluster, list)

  def test_segmentation_single_cluster(self):
    # Arrange
    data = np.array([[1, 2]])
    n_clusters = 1

    # Act
    clusters = segmentation_by_predep(data, n_clusters)

    # Assert
    self.assertEqual(len(clusters), n_clusters)
    self.assertEqual(clusters[0], [-np.inf, np.inf])

  def test_get_cluster_boundaries_empty(self):
    # Arrange
    clusters = []

    # Act
    cluster_boundaries = get_cluster_boundaries(clusters)

    # Assert
    self.assertEqual(cluster_boundaries, [])

class TestSegmentationProportional(unittest.TestCase):

  def test_segmentation_basic(self):
    # Arrange
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    n_clusters = 2  # Target 2 clusters (0.1 proportion not guaranteed)

    # Act
    clusters = segmentation_proportional(data, n_clusters)

    # Assert
    self.assertEqual(len(clusters), n_clusters + 1)
    for cluster in clusters:
      self.assertIsInstance(cluster, list)

  def test_segmentation_single_cluster(self):
    # Arrange
    data = np.array([[1, 2]])
    n_clusters = 1

    # Act
    clusters = segmentation_proportional(data, n_clusters)

    print(clusters)

    # Assert
    self.assertEqual(len(clusters), n_clusters)
    self.assertEqual(clusters[0], [-np.inf,np.inf])  # All data in single cluster

  def test_segmentation_proportional(self):
    # Arrange
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[1, 2], [3, 4], [5, 6], [7, 8]])
    n_clusters = 0.2  # Target 20% of data per cluster (may not be exact)

    # Act
    clusters = segmentation_proportional(data, n_clusters)

    # Assert (may need adjustment based on get_cluster_boundaries implementation)
    self.assertEqual(len(clusters), int(len(data) * n_clusters))

  def test_get_cluster_boundaries_empty(self):
    # Arrange
    clusters = []

    # Act
    cluster_boundaries = get_cluster_boundaries(clusters)

    # Assert
    self.assertEqual(cluster_boundaries, [])


if __name__ == '__main__':
  unittest.main()
