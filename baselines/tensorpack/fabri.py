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

from tensorpack.utils import logger
from tensorpack.utils.fs import download, get_dataset_path
from tensorpack.dataflow.base import RNGDataFlow


__all__ = ['Fabri']


class Fabri(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, dir='../../data'):
        assert train_or_test in ['train', 'val', 'test']
        assert dir is not None
        self.data = []
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
		img = cv2.resize(img, (224, 224))
                self.data.append([img, int(label)])
        self.train_or_test = train_or_test
        self.shuffle = shuffle

    def size(self):
        return len(self.data)

    def get_data(self):
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.data[k]
    
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
