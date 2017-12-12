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

    out = open('dt-alex.txt', 'w')

    for i in range(10):

        criterion = 'gini' if np.random.randint(2) == 0 else 'entropy'
        min_samples_split = int(10 ** np.random.uniform(0, 2)) + 1

        model = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_features='log2')

        model.fit(dtrain[0], dtrain[1])
        pre_val = model.predict(dval[0])
        pre_test = model.predict(dtest[0])

        acc_val = np.sum(pre_val == dval[1]) / len(dval[1])
        acc_test = np.sum(pre_test == dtest[1]) / len(dtest[1])
        print(criterion, min_samples_split, acc_val, acc_test, sep='\t', file=out)
        print(criterion, min_samples_split, acc_val, acc_test, sep='\t')

        pre_val = model.predict_proba(dval[0])
        pre_test = model.predict_proba(dtest[0])
        pre_val = np.argsort(pre_val)[:, -3:]
        pre_test = np.argsort(pre_test)[:, -3:]

        acc_val = compute_acc(pre_val, dval[1])
        acc_test = compute_acc(pre_test, dtest[1])
        print('--', acc_val, acc_test, sep='\t', file=out)
        print('--', acc_val, acc_test, sep='\t')
