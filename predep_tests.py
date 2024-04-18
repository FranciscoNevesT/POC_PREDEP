from sklearn.model_selection import train_test_split
from pipeline import pipeline
import numpy as np
from multiprocessing import Pool


class MetricTester:
  """
  A class for testing metrics using multiprocessing.
  """

  def __init__(self):
    self.segmentation_args = None
    self.segmentation_method = None

  def compute_metric(self, args):
    """
    Compute metric using the provided arguments.

    Args:
        args (any): Arguments required for computing the metric.

    Returns:
        any: The computed metric value.
    """

    if self.segmentation_method is None:
      raise ValueError("Segmentation_method not defined")


    data,theoretic_result = self.initialize_data(args=args)
    data_train, data_test = train_test_split(data, test_size=self.segmentation_args.get('test_size', 0.5))
    predep_alpha = pipeline(data_train=data_train, data_test=data_test, data_validation=None,
                            segmentation_method=self.segmentation_method)
    return predep_alpha,theoretic_result

  def multi_test(self, num_tests, num_processes, args):
    """
    Perform multiple tests using multiprocessing.

    Args:
        num_tests (int): Number of tests to perform.
        num_processes (int): Number of processes to use for multiprocessing.
        args (dict): Arguments required for each test.

    Returns:
        list: List of results from each test.
    """

    if self.segmentation_method is None:
      raise ValueError("Segmentation_method not defined")


    with Pool(processes=num_processes) as pool:
      data_list = [args for _ in range(num_tests)]
      results = pool.map(self.compute_metric, data_list)
    return results

  def set_segmentation_method(self, segmentation_method, segmentation_args):
    """
    Set the segmentation method and its arguments.

    Args:
        segmentation_method (function): The segmentation method to use.
        segmentation_args (dict): Arguments required for the segmentation method.
    """
    self.segmentation_method = segmentation_method
    self.segmentation_args = segmentation_args

  def initialize_data(self, args):
    """
    Initialize data for computing the metric.

    Args:
        args (any): Arguments required for initializing the data.

    Returns:
        any: Initialized data.
    """
    raise NotImplementedError("Subclasses must implement initialize_data")


class CaseGaussian(MetricTester):
  """
  A subclass of MetricTester for testing metrics with Gaussian distribution.
  """

  def __init__(self):
    super().__init__()

  def initialize_data(self, args):
    """
    Initialize Gaussian distributed data for computing the metric.

    Args:
        args (dict): Arguments for initializing the data.
            - sample_size (int): Size of the sample.
            - alpha (float): Covariance parameter.

    Returns:
        numpy.ndarray: Initialized data with Gaussian distribution.
    """
    if 'sample_size' not in args or 'alpha' not in args:
      raise ValueError("Both 'sample_size' and 'alpha' must be provided in args.")

    sample_size = args['sample_size']
    alpha = args['alpha']
    mean = [0, 0]
    cov = [[1, alpha],
           [alpha, 1]]

    data = np.random.multivariate_normal(mean, cov, sample_size)

    theoretic_result = 1 - np.sqrt(1 - alpha ** 2)

    return data,theoretic_result

