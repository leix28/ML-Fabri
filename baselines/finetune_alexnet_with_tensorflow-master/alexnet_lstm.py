import os
import cv2
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from caffe_classes import class_names
import time

BATCH_SIZE = 2
SEQ_LEN = 10
LSTM_SIZE = 100
checkpoint_path = "checkpoints"

FLAGS = tf.app.flags.FLAGS


class Dataset(object):
    def __init__(self, sets=[0, 1, 2]):
        self.rng = np.random
        with open('material_dataset.txt') as f:
            imglist = f.readlines()
        imglist = [item.split() for item in imglist]

        def load(lst, shortcut):
            if lst == []:
                return None
            try:
                out = np.load(shortcut)['arr_0']
                print("use ", shortcut)
                return out
            except:
                pass

            out = np.zeros((len(lst), 227, 227, 3), dtype='uint8')
            for idx, item in enumerate(lst):
                im = cv2.imread('data/'+item[0])
                im = im[246:473, 366:593]
                out[idx] = im
                if idx % 1000 == 0:
                    print("load {}/{}".format(idx, len(lst)))
            np.savez(shortcut, out)
            return out

        self.train = list(filter(lambda x: x[2] == '0', imglist)) if 0 in sets else []
        self.val = list(filter(lambda x: x[2] == '1', imglist)) if 1 in sets else []
        self.test = list(filter(lambda x: x[2] == '2', imglist)) if 2 in sets else []

        self.train.sort(key=lambda x: x[0])
        self.val.sort(key=lambda x: x[0])
        self.test.sort(key=lambda x: x[0])

        self.train_img = load(self.train, shortcut='preload/train.npz')
        self.val_img = load(self.val, shortcut='preload/val.npz')
        self.test_img = load(self.test, shortcut='preload/test.npz')

        def genlist(data):
            n = len(data)
            lst = []
            for i in range(n - SEQ_LEN + 1):
                if data[i][0][:-6] == data[i+SEQ_LEN-1][0][:-6]:
                    lst.append(i)
            return lst

        self.train_examples = genlist(self.train)
        self.val_examples = genlist(self.val)
        self.test_examples = genlist(self.test)
        print(len(self.train_examples), len(self.val_examples), len(self.test_examples))

    def get_next_train(self):
        data = self.rng.choice(self.train_examples, BATCH_SIZE)

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.train[data[i]][1])
            imgs[i] = self.train_img[data[i]:data[i]+SEQ_LEN]
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb

    def get_val_batch_num(self):
        return len(self.val_examples) // BATCH_SIZE

    def get_val(self, idx):
        data = self.val_examples[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.val[data[i]][1])
            imgs[i] = self.val_img[data[i]:data[i]+SEQ_LEN]
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb

    def get_test_batch_num(self):
        return len(self.test_examples) // BATCH_SIZE

    def get_test(self, idx):
        data = self.test_examples[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.test[data[i]][1])
            imgs[i] = self.test_img[data[i]:data[i]+SEQ_LEN]
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb

def main(_):
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])

    log = open('log.txt', 'w')

    with tf.Session() as sess:
        with tf.variable_scope('top_lstm'):
            feature = tf.contrib.layers.fully_connected(model.fc7, LSTM_SIZE)
            lstm_input = tf.reshape(feature, (BATCH_SIZE, SEQ_LEN, LSTM_SIZE))
            cell = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
            outputs, state = tf.nn.dynamic_rnn(cell, lstm_input, dtype='float32')

        with tf.variable_scope('softmax'):
            logits = tf.contrib.layers.fully_connected(outputs[:, -1, :], 14, activation_fn=None)
        entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

        varss = tf.trainable_variables()
        l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in varss]) * 0.0001

        loss = entropy + l2_loss
        opt = tf.train.MomentumOptimizer(0.0001, 0.9)
        opt_op = opt.minimize(loss)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        if FLAGS.load != '':
            print("load model from", FLAGS.load)
            saver.restore(sess, FLAGS.load)
        print("Model set up.")

        if not FLAGS.run:
            data = Dataset([0, 1])
            gLoss = []
            for epoch in range(10):
                print("Epoch number: {}".format(epoch+1))
                for step in range(20000):
                    img_batch, label_batch = data.get_next_train()
                    lossv, _ = sess.run((entropy, opt_op), feed_dict={x: img_batch - imagenet_mean,
                                                                        y: label_batch,
                                                                        keep_prob: 0.5})
                    gLoss = gLoss + [lossv]
                    gLoss = gLoss[-50:]
                    if step % 100 == 0:
                        print(epoch, step, np.mean(gLoss))

                # Validate the model on the entire validation set
                print("Start validation")
                test_acc = 0.
                test_count = 0

                for i in range(data.get_val_batch_num()):
                    img_batch, label_batch = data.get_val(i)
                    predict = sess.run(logits, feed_dict={x: img_batch - imagenet_mean,
                                                        y: label_batch,
                                                        keep_prob: 1.})
                    correct = np.sum(np.argmax(predict, axis=1) == label_batch)
                    test_acc += correct
                    test_count += len(label_batch)
                    if i % 100 == 0:
                        print(i, test_acc / test_count)

                test_acc /= test_count
                print("Validation Accuracy = {:.4f}".format(test_acc))
                print("Saving checkpoint of model...")

                print("Epoch {}, Val {}".format(epoch+1, test_acc * 100), file=log)
                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)

        else:
            data = Dataset([2])
            test_acc = 0
            test_count = 0
            test_acc_3 = 0

            def compute_acc(x, y):
                z = np.asarray([v in u for u, v in zip(x, y)], dtype='float32')
                return np.sum(z)

            for i in range(data.get_test_batch_num()):
                img_batch, label_batch = data.get_test(i)
                predict = sess.run(logits, feed_dict={x: img_batch - imagenet_mean,
                                                    y: label_batch,
                                                    keep_prob: 1.})

                pretop3 = np.argsort(predict)[:, -3:]

                correct = np.sum(np.argmax(predict, axis=1) == label_batch)

                test_acc += correct
                test_acc_3 += compute_acc(pretop3, label_batch)
                test_count += len(label_batch)
                if i % 100 == 0:
                    print(i, test_acc / test_count, test_acc_3 / test_count)

            test_acc /= test_count
            test_acc_3 /= test_count
            print(test_acc, test_acc_3)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('load', '', 'load model')
    tf.app.flags.DEFINE_boolean('run', False, 'run test')

    tf.app.run()
