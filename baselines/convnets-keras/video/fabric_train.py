"""
adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import tensorflow as tf
import sys
import datetime
import os
import shutil
from keras.optimizers import Adam, Adadelta
from convnets import AlexNet
from datagenerator import data_gen



import numpy as np
import dataloader
import datagenerator

from keras.backend.tensorflow_backend import set_session
from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

set_session(sess)

t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
print(t)

batch_size = 32
nb_classes = 14
nb_epoch = 200
data_augmentation = True

# The data, shuffled and split between train and test sets:
dataset_fn = '../../../data_preprocessing/material_dataset.txt'
imgs_fn = '../../../../storage/center_227x227.npz'
weights_fn = '../../../../storage/alexnet_weights.h5'

#sz = 227
sz = 128
img_rows = sz
img_cols = sz
img_channels = 3

with tf.device('/gpu:0'):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('alexnet.csv')
    #model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    #model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
    model = AlexNet(nb_classes=nb_classes, sz=sz)
    #model = AlexNet(weights_fn, nb_classes=nb_classes, sz=sz)
    #model = AlexNet(weights_fn, nb_classes=nb_classes)

#opt = Adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0)
#opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy', top_3_accuracy])

if data_augmentation:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    r = 0.2
    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=r*100,
        width_shift_range=r,
        height_shift_range=r,
        shear_range=r,
        zoom_range=r,
        channel_shift_range=r,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    #datagen.fit(X_train)

    def print_log(y_pred, Z, log_fn, k=5):
        fout = open(log_fn, 'w')
        acc1 = 0
        acc3 = 0
        cnt = 0

        j = 0
        while j < len(y_pred):
            ys = []
            for i in range(j, len(y_pred), k):
                img_fn = Z[i][0]
                id = img_fn[:-7]
                if id != Z[j][0][:-7]:
                    break
                label = Z[i][1]
                loc = Z[i][2]
                y_sum = np.sum(y_pred[i:i+k], axis=0)
                ys.append(y_sum)
                print(Z[i][0], Z[j][1], end=' ', file=fout)
                for t in y_sum:
                    print(t, end=' ', file=fout)
                print("", file=fout)

            label = Z[j][1]
            ys = np.mean(ys, axis=0)
            y = [(_, ys[_]) for _ in range(nb_classes)]
            y_sorted = sorted(y, key=lambda d:d[1], reverse=True)
            if y_sorted[0][0] == label:
                acc1 += 1
            if y_sorted[0][0] == label or y_sorted[1][0] == label or y_sorted[2][0] == label:
                acc3 += 1
            cnt += 1

            if i + k >= len(y_pred):
                break
            j = i
        fout.close()
        return acc1 * 1.0 / cnt, acc3 * 1.0 / cnt

    def predict(model, val=True):
        y_preds = []
        Z = []
        for (x, y, z) in datagenerator.test_generator(dataset_fn, imgs_fn, val=val, sz=img_rows):
            y_pred = model.predict(x, batch_size=batch_size)
            y_preds.append(y_pred)
            Z = Z + z
        y_preds = np.vstack(y_preds)
        return y_preds, Z

    log_dir = '../../../../result/alexnet/{}/'.format(t)
    os.mkdir(log_dir)
    shutil.copy('./fabric_train.py', log_dir+'fabric_train.py')
    shutil.copy('./convnets.py', log_dir+'convnets.py')
    G = data_gen('../../../data_preprocessing/material_dataset.txt', batch_size=batch_size, datagen=datagen, sz=sz)

    # Fit the model on the batches generated by datagen.flow().
    for epochs in range(nb_epoch):
        model.fit_generator(#datagen.flow(X_train, Y_train, batch_size=batch_size),
                            #steps_per_epoch=X_train.shape[0] // batch_size,
                            G,
                            steps_per_epoch=2000,
                            epochs=1, verbose=1, max_q_size=100)

        #y_pred_valid = model.predict(X_valid, batch_size=batch_size)
        #y_pred_test = model.predict(X_test, batch_size=batch_size)
        y_pred_valid, Z_valid = predict(model, val=True)
        y_pred_test, Z_test = predict(model, val=False)

        k = 5

        log_fn = log_dir + '.tmp.txt'
        val_acc = print_log(y_pred_valid, Z_valid, log_fn, k=k)
        test_acc = print_log(y_pred_test, Z_test, log_fn, k=k)

        log_fn = log_dir + 'val_{:02d}'.format(epochs) + '_{:.4f}_{:.4f}'.format(val_acc[1], test_acc[1]) + '.txt'
        print_log(y_pred_valid, Z_valid, log_fn, k=k)
        log_fn = log_dir + '{:02d}'.format(epochs) + '_{:.4f}_{:.4f}'.format(val_acc[1], test_acc[1]) + '.txt'
        print_log(y_pred_test, Z_test, log_fn, k=k)

        print(epochs, val_acc, test_acc)



