from predep_tests import CaseGaussian
from segmentations_methods import *
import pandas as pd

if __name__ == '__main__':

  case = CaseGaussian()
  segmentation_method = segmentation_histogram
  segmentation_args = {}

  args = {'sample_size': 1000,'alpha':0.8}

  case.set_segmentation_method(segmentation_method=segmentation_method,segmentation_args = segmentation_args)

  results = case.multi_test(num_tests=100,num_processes=5,args=args)
  results = pd.DataFrame(results)
  print(results)

