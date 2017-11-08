import numpy as np
import tensorflow as tf

import vgg16
import utils
import cv2

BATCH_SIZE = 100

def mkbatch():
    files = open('material_dataset.txt').readlines()
    cnt = len(files) // BATCH_SIZE
    if len(files) % BATCH_SIZE != 0:
        cnt += 1

    files = [item.split()[0] for item in files]
    batchlist = []
    for i in range(cnt):
        batchlist.append(files[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    return batchlist

def getbatchdata(filenames):
    imgs = []
    for filename in filenames:
        try:
            im = cv2.imread('data/'+filename)
            im = np.asarray(im[248:472, 368:592], dtype='float32') / 255.
        except:
            print("fileerror", 'data/'+filename)
            im = np.zeros((224, 224, 3))
        imgs.append(im)
    return np.stack(imgs, 0)

if __name__ == "__main__":
    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.Session() as sess:
        images = tf.placeholder("float", [None, 224, 224, 3])

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        batchlst = mkbatch()

        feature = []

        out_fea = []

        feature = vgg.relu7

        for idx, batch in enumerate(batchlst):
            inp = getbatchdata(batch)
            fea = sess.run(feature, feed_dict={images: inp})
            out_fea.append(fea)
            # print(fea[0])
            if idx % 10 == 0:
                print(idx, '/', len(batchlst))

        out_fea = np.concatenate(out_fea, 0)
        print(out_fea.shape)
        np.savez('vgg_fea', out_fea)

        # prob = sess.run(vgg.prob, feed_dict={images: batch})
