import os
import cv2
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from caffe_classes import class_names

BATCH_SIZE = 100

def mkbatch():
    files = open('material_dataset.txt').readlines()
    files = [s.split()[0] for s in files]
    cnt = len(files) // BATCH_SIZE
    if len(files) % BATCH_SIZE != 0:
        cnt += 1

    batchlist = []
    for i in range(cnt):
        batchlist.append(files[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    return batchlist

def getbatchdata(filenames):
    imgs = []
    for filename in filenames:
        try:
            im = cv2.imread('data/'+filename)
            im = im[246:473, 366:593]
        except:
            print("fileerror", 'data/'+filename)
            im = np.zeros((227, 227, 3))
        imgs.append(im)
    return np.stack(imgs, 0)


def main():
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])

    #define activation of last layer as score
    feature = [tf.reduce_max(model.conv5, axis=[1, 2]), model.fc7]

    batchlst = mkbatch()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        out_fea2 = []
        for idx, batch in enumerate(batchlst):
            inp = getbatchdata(batch)
            fea1, fea2 = sess.run(feature, feed_dict={x: inp - imagenet_mean, keep_prob: 1})
            out_fea2.append(fea2)
            # print(fea[0])
            if idx % 10 == 0:
                print(idx, '/', len(batchlst))


        out_fea2 = np.concatenate(out_fea2, 0)
        print(out_fea2.shape)
        np.savez('alex_fea', out_fea2)


if __name__ == "__main__":
    main()
