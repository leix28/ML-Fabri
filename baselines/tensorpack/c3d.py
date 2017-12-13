#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import glob
import cv2


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from fabri import Fabri
from tensorflow.contrib.layers import variance_scaling_initializer

BATCH_SIZE = 48

def top_k_smooth_entropy(logits, labels, k):
    # https://arxiv.org/pdf/1512.00486.pdf
    k = k - 1
    inf = 1e4
    p_y = tf.reduce_sum(logits * tf.one_hot(labels, 100), 1)

    values, indices = tf.nn.top_k(logits - inf * tf.one_hot(labels, 100), k)

    mask_top_k = tf.reduce_sum(tf.one_hot(indices, 100), 1)

    exp_p_i_minus_p_y = tf.exp(logits - tf.expand_dims(p_y, 1) - inf * mask_top_k)

    exp_p_i_minus_p_y_smooth = exp_p_i_minus_p_y * (1 - mask_top_k)

    return tf.log(tf.reduce_sum(exp_p_i_minus_p_y_smooth, 1))


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 10, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0 - tf.constant([0.45834960097,0.44674252445,0.41352266842], dtype=tf.float32)
        assert tf.test.is_gpu_available()
        # image = tf.transpose(image, [0, 3, 1, 2])


        net = tf.layers.Conv3D(64, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv1')
        net = tf.layers.max_pooling3d(net, (1, 2, 2), (1, 2, 2))

        net = tf.layers.Conv3D(128, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv2')
        net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2))

        net = tf.layers.Conv3D(256, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv3')
        net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2))

        net = tf.layers.Conv3D(512, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv4')
        net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2))

        net = tf.layers.Conv3D(512, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv5a')
        net = tf.layers.Conv3D(512, (3, 3, 3), (1, 1, 1), 'same', activation=tf.nn.relu, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                kernel_regularizer=0.0001, name='conv5b')
        net = tf.layers.max_pooling3d(net, (2, 2, 2), (2, 2, 2))

        net = FullyConnected('fc6', net, out_dim=4096, nl=tf.nn.relu)
        net = FullyConnected('fc7', net, out_dim=4096, nl=tf.nn.relu)

        logits = FullyConnected('linear', net, out_dim=14, nl=tf.identity)
        
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # cost = top_k_smooth_entropy(logits=logits, labels=label, k=5) 
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = Fabri(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((256, 256)),
            imgaug.RandomCrop((224, 224)),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True),
	        imgaug.Brightness(20),
            imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
	    imgaug.CenterPaste((256, 256)),
            imgaug.RandomCrop((224, 224)),
           # imgaug.Flip(horiz=True),
           # imgaug.Flip(vert=True),
           # imgaug.Brightness(20),
           # imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/places-single-fisrt%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 1e-2), (20, 1e-3), (40, 1e-4), (60, 1e-5), (95, 1e-6), (105, 1e-7)]),
        ],
        model=Model(n=args.n),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
    )


def augment(img):

    def crop(img):
	i = np.random.randint(0, 16)
	j = np.random.randint(0, 16)
	img = img[i:i+160, j:j+160]
	return img

    def brightness(img, delta):
        old_dtype = img.dtype
        img = img.astype('float32')
        img += np.random.uniform(-delta, delta)
        img = np.clip(img, 0, 255)
        return img.astype(old_dtype)

    def contrast(img, factor_range):
        old_dtype = img.dtype
        img = img.astype('float32')
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        r = np.random.uniform(factor_range[0], factor_range[1])
        img = (img - mean) * r + mean
        img = np.clip(img, 0, 255)
        return img.astype(old_dtype)

    def flip(img):
        if np.random.uniform(0, 1.0) > 0.5:
            img = cv2.flip(img, 0)
        if np.random.uniform(0, 1.0) > 0.5:
            img = cv2.flip(img, 1)
        return img

    img = crop(img)
    img = brightness(img, 20)
    img = contrast(img, (0.6, 1.4))
    img = flip(img)
    return img

def run(model_path, test_file, output_file=None):
    pred_config = PredictConfig(
        model=Model(n=args.n),
        session_init=tfutils.get_model_loader(model_path),
        input_names=['input'],
        output_names=['output'])
    predictor = OfflinePredictor(pred_config)
    output_f = open(output_file, 'w')
    for fname in glob.glob(os.path.join(test_file, '*')):
        im = cv2.imread(fname)
	im = cv2.resize(im, (176, 176))
	ims = [augment(im.copy().astype('float32')) for i in xrange(30)]
        output = np.squeeze(predictor([ims]))
	output = output.reshape((30, 100))
	output = np.squeeze(np.sum(output, axis=0))
        output = np.argsort(output)[-5:]
        fname = fname.split('data/')[-1]
        output_f.write('%s %d %d %d %d %d\n'%(fname, output[-1], output[-2], output[-3], output[-4], output[-5]))
        print('%s %d %d %d %d %d'%(fname, output[-1], output[-2], output[-3], output[-4], output[-5]))
    output_f.close()
    # im = cv2.imread(image_path)
    # assert im is not None
    # im = cv2.resize(im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16))
    # outputs = predictor([[im.astype('float32')]])
    # if output is None:
    #     for k in range(6):
    #         pred = outputs[k][0]
    #         cv2.imwrite("out{}.png".format(
    #             '-fused' if k == 5 else str(k + 1)), pred * 255)
    # else:
    #     pred = outputs[5][0]
    #     cv2.imwrite(output, pred * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=225,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--n',default=18, help='Number of units')
    parser.add_argument('--max_epoch',default=300,help='max epoch')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--test_file', default='./data/test')
    parser.add_argument('--output_file', default='./output.txt')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.run:
        run(args.load, args.test_file, args.output_file)
    else:
       SyncMultiGPUTrainer(config).train()
