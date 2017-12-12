import dataloader
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

if __name__ == "__main__":
    dtrain, dval, dtest = dataloader.load('material_dataset.txt', 'vgg_fea.npz')

    out = open('svm-vgg.txt', 'w')

    for i in range(10):
        cc = 10 ** np.random.uniform(-2, 2)
        inter = 10 ** np.random.uniform(0, 2)
        pen = 'l1' if np.random.randint(2) == 0 else 'l2'
        loss = 'squared_hinge'

        model = LinearSVC(penalty=pen, loss=loss, dual=False, C=cc, intercept_scaling=inter, verbose=1, max_iter=1000)
        # model = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
        # model = DecisionTreeClassifier()

        # dtrain = list(zip(dtrain[0], dtrain[1]))
        # random.shuffle(dtrain)
        # dtrain =list(zip(*dtrain))

        model.fit(dtrain[0], dtrain[1])
        pre_val = model.predict(dval[0])
        pre_test = model.predict(dtest[0])

        acc_val = (np.sum(pre_val == dval[1]), len(dval[1]))
        acc_test = (np.sum(pre_test == dtest[1]), len(dtest[1]))
        print(cc, inter, pen, loss, acc_val, acc_test, sep='\t', file=out)
