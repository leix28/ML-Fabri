import dataloader
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

if __name__ == "__main__":
    dtrain, dval, dtest = dataloader.load('material_dataset.txt', 'vgg_fea.npz')

    out = open('vgg_nn.txt', 'w')

    for i in [5, 10]:
        model = KNeighborsClassifier(n_neighbors=i, n_jobs=4)

        model.fit(dtrain[0], dtrain[1])
        pre_val = model.predict(dval[0])
        pre_test = model.predict(dtest[0])

        acc_val = (np.sum(pre_val == dval[1]), len(dval[1]))
        acc_test = (np.sum(pre_test == dtest[1]), len(dtest[1]))
        print(i, acc_val, acc_test, sep='\t', file=out)
