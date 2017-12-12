#!/usr/bin/env python
# encoding: utf-8
# File Name: dataloader.py

import numpy as np
import numpy.matlib
from progressbar import ProgressBar, Percentage, Bar, ETA, RotatingMarker
from keras.utils import np_utils
import cv2
import gc
import random

def test_generator(datasplit, feature, batch_size=32, val=True, nb_classes=14, sz=600):
    labels = open(datasplit).readlines()
    labels = [item.split() for item in labels]

    X, Y, Z = [], [], []

    ids = range(len(labels))
    random.seed(0)
    random.shuffle(ids)

    def f(im):
        hf = int(sz/2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ret = []
        ret.append(im[360-hf:360+hf,480-hf:480+hf])
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
            if cnt > 1000:
                pass
                #break


def data_gen(datasplit, d=360, sz=600, batch_size=32, datagen=None, nb_classes=14, outs=31):
    gc.collect()

    def f(img):
        #return cv2.resize(img[360-d:360+d, 480-d:480+d], (224, 224))
        x = random.randint(360-d, 360+d-sz)
        y = random.randint(480-d, 480+d-sz)
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
                    y = np.reshape(numpy.matlib.repmat(y,1,outs*outs), [batch_size,outs,outs,nb_classes])
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

