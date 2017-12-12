import dataloader
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

def compute_acc(x, y):
    z = np.asarray([v in u for u, v in zip(x, y)], dtype='float32')
    return np.sum(z) / np.size(z)


if __name__ == "__main__":
    dtrain, dval, dtest = dataloader.load('material_dataset.txt', 'alex_fea.npz')

    cc = 0.02349261481109219
    inter = 34.310644314644186
    pen = 'l1'
    loss = 'squared_hinge'

    model = LinearSVC(penalty=pen, loss=loss, dual=False, C=cc, intercept_scaling=inter, verbose=1, max_iter=1000)

    model.fit(dtrain[0], dtrain[1])
    pre_val = model.decision_function(dval[0])
    pre_test = model.decision_function(dtest[0])

    pre_val = np.argsort(pre_val)[:, -3:]
    pre_test = np.argsort(pre_test)[:, -3:]

    acc_val = compute_acc(pre_val, dval[1])
    acc_test = compute_acc(pre_test, dtest[1])

    print(model.classes_)
    print(np.sum(model.predict(dtest[0]) == dtest[1]), len(dtest[1]))
    print(cc, inter, pen, loss, acc_val, acc_test, sep='\t')
