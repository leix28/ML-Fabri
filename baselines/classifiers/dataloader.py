import numpy as np

def load(datasplit, feature):
    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    X = np.load(feature)['arr_0']

    assert len(labels) == len(X)

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for label, fea in zip(labels, X):
        if label[2] == '0':
            X_train.append(fea)
            Y_train.append(int(label[1]))
        elif label[2] == '1':
            X_val.append(fea)
            Y_val.append(int(label[1]))
        elif label[2] == '2':
            X_test.append(fea)
            Y_test.append(int(label[1]))

    print(len(X_train), len(X_val), len(X_test))
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
