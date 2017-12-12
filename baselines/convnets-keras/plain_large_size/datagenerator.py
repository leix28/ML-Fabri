#!/usr/bin/env python
# encoding: utf-8
# File Name: dataloader.py

import numpy as np
from progressbar import ProgressBar, Percentage, Bar, ETA, RotatingMarker
from keras.utils import np_utils
import cv2
import gc
import random

def test_generator(datasplit, feature, batch_size=32, val=True, nb_classes=14, sz=227):
    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    X, Y, Z = [], [], []

    ids = range(len(labels))
    random.seed(0)
    random.shuffle(ids)

    def f(im):
        hf = int(sz/2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        points = [
                (360-hf, 480-hf)
                #(360-hf-sz, 480-hf-sz)
                #(360-hf-sz, 480-hf),
                #(360-hf, 480-hf-sz),
                #(360-hf, 480+hf),
                #(360+hf, 480-hf)
                #(360-hf-2*sz, 480-hf),
                #(360-hf, 480-hf-2*sz),
                #(360-hf, 480+hf+sz),
                #(360+hf+sz, 480-hf)
                ]
        '''
        points = [(360-114, 480-114)]
        '''
        ret = []
        for p in points:
            x, y = p
            ret.append(im[x:x+sz,y:y+sz])
        return ret

    cnt = 0
    for i in ids:
        label = labels[i]
        flag = False
        if label[2] == '0':
            pass
            #X_train.append(fea)
            #Y_train.append(int(label[1]))
        elif label[2] == '1' and val:
            flag = True
        elif label[2] == '2' and not(val):
            flag = True

        if flag:
            im = cv2.imread('../../../../data/dataset/' + label[0])
            fea = f(im)
            for j in range(len(fea)):
                X.append(fea[j])
                Y.append(int(label[1]))
                Z.append([label[0], int(label[1]), j])
                cnt += 1

                if len(X) == batch_size:
                    X = np.array(X).astype('float32')
                    X = (X - 128) / 128
                    Y = np_utils.to_categorical(Y, nb_classes)
                    yield X, Y, Z
                    X, Y, Z = [], [], []
            if cnt > 100:
                pass
                #break

def load(datasplit, feature):
    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    #X = np.load(feature)['arr_0']
    #print len(labels), len(X)
    #assert len(labels) == len(X)

    X_train, Y_train, Z_train = [], [], []
    X_val, Y_val, Z_val = [], [], []
    X_test, Y_test, Z_test = [], [], []

    ids = range(len(labels))
    random.seed(0)
    random.shuffle(ids)
    #for label, fea in zip(labels, X):

    def f(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        points = [(360-114-227, 480-114-227),
                (360-114-227, 480+113),
                (360-114, 480-114),
                (360+113, 480-114-227),
                (360+113, 480+113)]
        '''
        points = [(360-114, 480-114)]
        '''
        ret = []
        for p in points:
            x, y = p
            ret.append(im[x:x+227,y:y+227])
        return ret

    for i in ids:
        label = labels[i]
        if label[2] == '0':
            pass
            #X_train.append(fea)
            #Y_train.append(int(label[1]))
        elif label[2] == '1':
            im = cv2.imread('../../../../data/dataset/' + label[0])
            fea = f(im)
            for j in range(len(fea)):
                X_val.append(fea[j])
                Y_val.append(int(label[1]))
                Z_val.append([label[0], int(label[1]), j])
        elif label[2] == '2':
            im = cv2.imread('../../../../data/dataset/' + label[0])
            fea = f(im)
            for j in range(len(fea)):
                X_test.append(fea[j])
                Y_test.append(int(label[1]))
                Z_test.append([label[0], int(label[1]), j])
            if len(X_test) % 1000 == 0:
                print len(X_test)
                #break

    print(len(X_train), len(X_val), len(X_test))
    #X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    fout = open('y_test_label.txt', 'w')
    for i in range(len(Y_test)):
        print >>fout, Y_test[i]
        fout.flush()
    return (X_train, Y_train, Z_train), (X_val, Y_val, Z_val), (X_test, Y_test, Z_test)

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


def data_gen(datasplit, d=300, sz=227, batch_size=32, datagen=None, nb_classes=14):
    gc.collect()

    def f(img):
        #return cv2.resize(img[360-d:360+d, 480-d:480+d], (224, 224))
        x = random.randint(360-d, 360+d-sz)
        y = random.randint(360-d, 360+d-sz)
        #x = 360-114
        #y = 480-114
        return img[x:x+sz, y:y+sz]

    labels = open(datasplit).readlines()
    labels = [item.rstrip().split() for item in labels]

    random.seed(0)
    random.shuffle(labels)

    X = []
    y = []
    while True:
        for c, label in enumerate(labels):
            if label[2] == '0':
                fn = '../../../../data/dataset/' + label[0]
                img = cv2.imread(fn)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = f(img)
                X.append(x)
                y.append(int(label[1]))
                if len(X) == batch_size:
                    y = np_utils.to_categorical(y, nb_classes)
                    X = np.array(X).astype('float32')
                    X = (X-128)/128.0
                    if datagen:
                        X, y = next(datagen.flow(X, y, batch_size=batch_size))
                    yield X, y
                    del X, y
                    X = []
                    y = []



if __name__ == '__main__':
    #save_images('../../data_preprocessing/material_dataset.txt', center())
    #save_images('../../data_preprocessing/material_dataset.txt', rescaled(360))
    #save_images('../../data_preprocessing/material_dataset.txt', rescaled(180))
    #load('../../../data_preprocessing/material_dataset.txt', '../../../../storage/center_227x227.npz')
    print next(data_gen('../../../data_preprocessing/material_dataset.txt'))

