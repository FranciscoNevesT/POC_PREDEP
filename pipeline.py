import numpy as np
from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy

def predep(x,num_boots = 5000):
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

  n_clusters = len(clusters)
  predep_i = 0
  for j in range(n_clusters):
      data_index = labels == j

      if np.sum(data_index)  <= 1:
          continue
      data_i = data[data_index]

      predep_i += predep(data_i[:, 0]) * len(data_i) / len(data)
  return predep_i


def pipeline(data_train,data_test,data_validation,segmetantion_method):
    predep_x = predep(data_test[:, 0])

    clusters = segmetantion_method(data_train)

    predep_xy = calc_predep_xy(data_train,clusters)

    predep_alpha = (predep_xy - predep_x)/predep_xy

    return predep_alpha


