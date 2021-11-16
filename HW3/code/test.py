from hw3 import Stump,bagging,adaboost
import matplotlib.pyplot as plt
import numpy as np
from hw3_utils import visualization, get_dataset_fixed
# import hw3_utils



data,labels= get_dataset_fixed()
weights = [1.0] * labels.size
#print(data)

#stump = Stump(data, labels, weights)
# print(stump.dimension,stump.threshold)
#print(data)
n_classifiers = 20
n_samples = 15
classifiers,weight=adaboost(data, labels, n_classifiers)
#classifiers = bagging(data, labels, n_classifiers, n_samples, seed=0)
visualization(data, labels,classifiers , weights)

