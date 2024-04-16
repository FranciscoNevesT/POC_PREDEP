import numpy as np
from pipeline import predep


# Todo: Consertar o codigo
def segmentation_by_predep(data, max_clusters = 5):
  data_index = np.argsort(data[:, 1])
  data = data[data_index]

  y = data[:,1]
  x = data[:,0]

  clusters_y = [[i] for i in y]
  clusters_x = [[i] for i in x]
  clusters_v = [0 for _ in range(len(y))]

  while len(clusters_y) > max_clusters:
    index_max = None
    predep_dif_max = -np.inf
    predep_max = None
    for i in range(len(clusters_y) - 1):
      cluster_a = clusters_x[i]
      cluster_b = clusters_x[i + 1]

      cluster_c = np.array(cluster_a + cluster_b)

      predep_i = predep(np.array(cluster_c)) * (len(cluster_c) / len(y))

      predep_dif = predep_i - clusters_v[i] - clusters_v[i + 1]

      if predep_dif > predep_dif_max:
        predep_dif_max = predep_dif
        predep_max = predep_i
        index_max = i

    clusters_y[index_max] = clusters_y[index_max] + clusters_y[index_max + 1]
    clusters_x[index_max] = clusters_x[index_max] + clusters_x[index_max + 1]
    clusters_v[index_max] = predep_max

    clusters_y.pop(index_max + 1)
    clusters_x.pop(index_max + 1)
    clusters_v.pop(index_max + 1)

  cluster_boundaries = get_cluster_boundaries(clusters_y)

  return cluster_boundaries


def segmentation_proportional(data,n_clusters = 10):
  if n_clusters < 1 and n_clusters > 0:
    n_clusters = int(len(data) * n_clusters)

  data_index = np.argsort(data[:, 1])
  data = data[data_index]

  y = data[:, 1]
  x = data[:, 0]

  i = 0
  clusters_y = []
  while len(clusters_y) < n_clusters:
    plus = (len(data) - i) // (n_clusters -len(clusters_y))

    y_i = y[i:i+plus]
    clusters_y.append(y_i)

    i =  i + plus

  cluster_boundaries = get_cluster_boundaries(clusters_y)

  return cluster_boundaries


def get_cluster_boundaries(clusters):
  if len(clusters) == 0:
    return []
  elif len(clusters) == 1:
    return [[-np.inf,np.inf]]

  clusters_div = [[-np.inf, min(clusters[0])]]
  for i in range(len(clusters) - 1):
    min_v = min(clusters[i])
    max_v = min(clusters[i + 1])

    clusters_div.append([min_v, max_v])
  clusters_div.append([min(clusters[-1]), np.inf])

  return clusters_div