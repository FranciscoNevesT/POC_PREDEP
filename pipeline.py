import numpy as np
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def predep(x,num_boots = 10000):
  if len(np.unique(x)) == 1:
    return 0

  num_boots = min(max(50,len(x) * len(x)),num_boots)

  x1 = np.random.choice(x, size=num_boots, replace=True)
  x2 = np.random.choice(x, size=num_boots, replace=True)

  x_d = x1 - x2

  kde = gaussian_kde(x_d)

  return kde.evaluate(0)[0]

def cluster_assign(x,clusters):

    if len(clusters) == 0 or len(x) == 0:
        raise ValueError

    labels = []

    for i in x:
        assigned_label = False
        for c in range(len(clusters)):
            min_v = min(clusters[c])
            max_v = max(clusters[c])

            if min_v <= i and i <= max_v:
                labels.append(c)
                assigned_label = True
                break

        if assigned_label == False:
            labels.append(-1)

    return np.array(labels)

def calc_predep_xy(data,clusters):
  labels = cluster_assign(data[:,1],clusters)
  predep_i = 0
  for j in np.unique(labels):
      data_index = labels == j

      if np.sum(data_index)  <= 1:
          continue
      data_i = data[data_index]

      predep_i += predep(data_i[:, 0]) * len(data_i) / len(data)
  return predep_i


def pipeline(data_train, data_test, data_validation, segmentation_method):
    predep_x = predep(data_test[:, 0])

    clusters = segmentation_method(data_train)

    print(clusters)

    predep_xy = calc_predep_xy(data_train,clusters)

    predep_alpha = (predep_xy - predep_x)/predep_xy

    return predep_alpha

def bootstrap(data, segmentation_method, num_boots=100):
  # Create empty list to store bootstrap replicates
  bootstrap_stats = []

  # Loop for the desired number of replicates
  for _ in range(num_boots):
    # Sample with replacement to create a bootstrap replicate
    data_replicate = resample(data, replace=True)

    data_train,data_test = train_test_split(data_replicate,test_size=0.9)

    # Calculate the statistic (replace with your specific statistic calculation)
    predep_alpha = pipeline(data_train = data_train, data_test = data_test, data_validation=None, segmentation_method = segmentation_method)
    # Append the statistic to the list
    bootstrap_stats.append(predep_alpha)

  bootstrap_stats = np.array(bootstrap_stats)

  return bootstrap_stats

