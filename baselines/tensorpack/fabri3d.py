#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: fabri.py
# Author: Yue Wang <valianter.wang@gmail.com

import os
import pickle
import numpy as np
import six
import cv2
from six.moves import range

from tensorpack import *
from collections import defaultdict
from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.dataflow.base import RNGDataFlow


__all__ = ['Fabri']


class Fabri3D(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, dir='../../data'):
        assert train_or_test in ['train', 'val', 'test']
        assert dir is not None
        self.data = defaultdict(list)
        self.label = defaultdict(str)
        if train_or_test == 'train':
            fnames = 'train.txt'
        elif train_or_test == 'val':
            fnames = 'val.txt'
        else:
            fnames = 'test.txt'
        with open(os.path.join(dir, fnames)) as f:
            for idx, line in enumerate(f.readlines()):
                fn, label = line.strip().split()
                fn = os.path.join(dir, 'dataset',fn)
                if not os.path.isfile(fn):
                    raise ValueError('Failed to find file: ' + fn)
                img = cv2.imread(fn)
                height, width = img.shape
                img = img[height/2-112:height/2+112, width/2-112:width/2+112]
                name = '_'.join(fn.split('_')[0:3])
                self.data[name].append(img)
                self.label[name].append(label)
        for key in self.data.keys():
            if len(self.data[key]) < 15:
                del self.data[key]
                del self.label[key]
            else:
                self.data[key] = np.array(self.data[key][:15]).reshape((224, 244, 15*3))
        self.train_or_test = train_or_test
        self.shuffle = shuffle

    def size(self):
        return len(self.data)

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.data[k], self.label[k]
    
    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 224x224x3
        """
        all_imgs = [x[0] for x in self.data]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean

if __name__ == '__main__':
    ds = Places('val')
    print(ds.size())
    print(ds.get_per_pixel_mean())
