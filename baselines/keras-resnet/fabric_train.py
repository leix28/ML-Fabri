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

from keras.backend.tensorflow_backend import set_session
set_session(sess)

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import sys
import datetime
import os
import shutil
from keras.optimizers import Adam, Adadelta

import numpy as np
import resnet
import dataloader

from keras.metrics import top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)



t = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
print(t)

batch_size = 32
nb_classes = 14
nb_epoch = 200
data_augmentation = True
#data_augmentation = False


# The data, shuffled and split between train and test sets:
dataset_fn = '../../data_preprocessing/material_dataset.txt'
imgs_fn = '../../../storage/center.npz'
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = dataloader.load(dataset_fn, imgs_fn)

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



# input image dimensions
img_rows, img_cols = X_train.shape[1], X_train.shape[2]
# The CIFAR10 images are RGB.
img_chanels = 3

#new_sz = 4
#X_train = X_train[:,img_rows/2-new_sz:img_rows/2+new_sz]
#X_valid = X_valid[:,img_rows/2-new_sz:img_rows/2+new_sz]
#X_test = X_test[:,img_rows/2-new_sz:img_rows/2+new_sz]

img_rows, img_cols = X_train.shape[1], X_train.shape[2]
print(img_rows, img_cols)


X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_valid -= mean_image
X_test -= mean_image
X_train /= 128.
X_valid /= 128.
X_test /= 128.

with tf.device('/gpu:0'):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('resnet.csv')
#model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)

#opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy', top_3_accuracy])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
              #callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    r = 0.0
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
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    def print_log(y_pred, log_fn):
        fout = open(log_fn, 'w')
        for i in range(1, len(X_test)+1):
            img_fn = 'test/{:08d}.jpg'.format(i)
            print(img_fn, end=' ', file=fout)
            y = y_pred[i-1]
            y = [(j, y[j]) for j in range(nb_classes)]
            y_sorted = sorted(y, key=lambda d:d[1], reverse=True)
            for j in y_sorted[:5]:
                print(j[0], end=' ', file=fout)
            print("", file=fout)
        fout.close()

    log_dir = '../../../result/resnet/{}/'.format(t)
    os.mkdir(log_dir)
    shutil.copy('./fabric_train.py', log_dir+'fabric_train.py')

    # Fit the model on the batches generated by datagen.flow().
    for epochs in range(nb_epoch):
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_data=(X_valid, Y_valid),
                            epochs=1, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger])
        val_acc = model.evaluate(X_valid, Y_valid, verbose=0,  batch_size=batch_size)
        test_acc = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
        print(val_acc, test_acc)

        y_pred = model.predict(X_test, batch_size=batch_size)
        log_fn = log_dir + '{:02d}'.format(epochs) + '_{:.4f}_{:.4f}'.format(val_acc[1],test_acc[1]) + '.txt'
        print_log(y_pred, log_fn)

        y_pred = model.predict(X_valid, batch_size=batch_size)
        log_fn = log_dir + 'val_{:02d}'.format(epochs) + '_{:.4f}_{:.4f}'.format(val_acc[1], test_acc[1]) + '.txt'
        print_log(y_pred, log_fn)



