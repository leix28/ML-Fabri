import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D#, MaxPooling2D, ZeroPadding2D
from keras.layers import MaxPooling2D, ZeroPadding2D, Concatenate
from keras.optimizers import SGD
from keras import backend as K
from keras.layers import Conv2D
import numpy as np
from scipy.misc import imread, imresize, imsave

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D


def AlexNet_FCN(weights_path=None, heatmap=False, nb_classes=14, sz=227):
    data_format = 'channels_last'
    K.set_image_data_format(data_format)
    inputs = Input(shape=(sz, sz, 3))

    strides = 4

    div = 1

    conv_1 = Conv2D(96/div, (11, 11), strides=(strides, strides), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2), data_format=data_format)(conv_2)
    K.set_image_data_format(data_format)
    conv_2 = Concatenate(axis=-1, name='conv_2')([
                       Conv2D(128/div, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(axis=3, ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)])

    K.set_image_data_format(data_format)
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1), data_format=data_format)(conv_3)
    K.set_image_data_format(data_format)
    conv_3 = Conv2D(384/div, (3, 3), activation='relu', name='conv_3')(conv_3)

    K.set_image_data_format(data_format)
    conv_4 = ZeroPadding2D((1, 1), data_format=data_format)(conv_3)
    conv_4 = Concatenate(axis=-1, name='conv_4')([
                       Conv2D(192/div, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(axis=3, ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)])

    K.set_image_data_format(data_format)
    conv_5 = ZeroPadding2D((1, 1), data_format=data_format)(conv_4)
    K.set_image_data_format(data_format)
    conv_5 = Concatenate(axis=-1, name='conv_5')([
                       Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(axis=3, ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)])
    print conv_5.shape

    K.set_image_data_format(data_format)

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten()(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    #dense_1 = Dense(128, activation='relu', name='dense_1')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    #dense_2 = dense_1
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_1)
    prediction = Dense(1000, activation='softmax', name='dense_3')(dense_2)

    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path)

    dense_1 = MaxPooling2D((4, 4), strides=(2, 2), name='convpool')(conv_3)
    dense_1 = Conv2D(4096, (1, 1), activation='relu', name='dense_1_conv')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    dense_2 = Conv2D(4096, (1, 1), activation='relu', name='dense_2_conv')(dense_1)
    dense_2 = Dropout(0.5)(dense_2)
    dense_3 = Conv2D(nb_classes, (1, 1), name='dense_3_conv')(dense_2)
    prediction_new = Softmax4D(axis=-1, name='softmax_')(dense_3)


    model = Model(inputs=inputs, outputs=prediction_new)

    outs = prediction_new.shape[1]

    return model, int(outs)


if __name__ == "__main__":
    #sz = 224
    sz = 600
    im = np.random.random((10, sz, sz, 3))

    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = AlexNet_FCN(sz=sz)
    model.compile(optimizer=sgd, loss='mse')

    out = model.predict(im)
    print out.shape
    while(True):
        pass
