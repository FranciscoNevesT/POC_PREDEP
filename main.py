import numpy as np
from pipeline import *
from segmentations_methods import *
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

mean = [0, 0]
cov = [[1, 0.8],
      [0.8, 1]]
n_samples = 100000

def calculate_and_append(data):
  data_train, data_test = train_test_split(data, test_size=0.1)
  predep_alpha = pipeline(data_train=data_train, data_test=data_test, data_validation=None,
                             segmentation_method=segmentation_proportional)
  return predep_alpha
if __name__ == '__main__':

  bootstrap_stats = []
  num_processes = 4  # Adjust based on your hardware

  # Use Pool for parallel execution
  with Pool(processes=num_processes) as pool:
    data_list = [np.random.multivariate_normal(mean, cov, n_samples) for _ in range(1000)]
    bootstrap_stats = pool.map(calculate_and_append, data_list)

  sns.boxplot(data=bootstrap_stats)
  plt.show()

  # Calculate the percentile for the 99% confidence interval (0.5% on each tail)
  percentiles = [0.5, 99.5]

  print(bootstrap_stats)

  # Use np.percentile to get the confidence interval bounds
  confidence_interval = np.percentile(bootstrap_stats, percentiles, axis=0)

  # Print the confidence interval
  print("Confidence Interval (99%):", confidence_interval)