#!/usr/bin/env python
# encoding: utf-8
# File Name: dataloader.py

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA, RotatingMarker
import cv2
import gc
import random

def load(datasplit, feature):
    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    X = np.load(feature)['arr_0']

    print len(labels), len(X)
    assert len(labels) == len(X)

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    ids = range(len(labels))
    random.seed(0)
    random.shuffle(ids)
    #for label, fea in zip(labels, X):
    for i in ids:
        label = labels[i]
        fea = X[i]
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
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    fout = open('y_test_label.txt', 'w')
    for i in range(len(Y_test)):
        print >>fout, Y_test[i]
        fout.flush()
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def center():
    def f(img):
        #return img[360-112:360+112, 480-112:480+112]
        return img[360-114:360+113, 480-114:480+113]
    return 'center_227x227', f

def rescaled(d):
    def f(img):
        #return cv2.resize(img[360-d:360+d, 480-d:480+d], (224, 224))
        return cv2.resize(img[360-d:360+d, 480-d:480+d], (227, 227))
    return 'rescaled_{}_227x227'.format(d), f

def save_images(datasplit, params):
    gc.collect()

    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    name, f = params

    widgets = [name, Percentage(), ' ', Bar(marker=RotatingMarker()),
            ' ', ETA()]
    pb = ProgressBar(widgets=widgets, maxval=len(labels)).start()

    X = []
    for c, label in enumerate(labels):
        fn = '../../../data/dataset/' + label[0]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.copy(f(img))
        X.append(x)
        del img
        pb.update(c+1)
    pb.finish()

    imgs_fn = '../../../storage/{}.npz'.format(name)
    np.savez(imgs_fn, X)


if __name__ == '__main__':
    #save_images('../../data_preprocessing/material_dataset.txt', center())
    #save_images('../../data_preprocessing/material_dataset.txt', rescaled(360))
    #save_images('../../data_preprocessing/material_dataset.txt', rescaled(180))
    load('../../../data_preprocessing/material_dataset.txt', '../../../../storage/center_227x227.npz')

