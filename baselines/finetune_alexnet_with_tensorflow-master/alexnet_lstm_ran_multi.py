from __future__ import print_function
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
checkpoint_path = "checkpoints_ran_multi"
TEST_RAN = 10

FLAGS = tf.app.flags.FLAGS


class Dataset(object):
    def __init__(self, sets=[0, 1, 2]):
        self.rng = np.random
        with open('material_dataset_multi.txt') as f:
            imglist = f.readlines()
        imglist = [item.split() for item in imglist]

        self.train = list(filter(lambda x: x[2] == '0', imglist)) if 0 in sets else []
        self.val = list(filter(lambda x: x[2] == '1', imglist)) if 1 in sets else []
        self.test = list(filter(lambda x: x[2] == '2', imglist)) if 2 in sets else []

        self.train.sort(key=lambda x: x[0])
        self.val.sort(key=lambda x: x[0])
        self.test.sort(key=lambda x: x[0])


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
        self.test_examples = self.test_examples * TEST_RAN
        print(len(self.train_examples), len(self.val_examples), len(self.test_examples))

    def random_crop(self, lst, id):
        out = np.zeros((SEQ_LEN, 227, 227, 3), dtype='uint8')
        cx = np.random.randint(600-227)
        cy = np.random.randint(600-227)
        
        for id, item in enumerate(lst[id:id+SEQ_LEN]):
            im = cv2.imread('data/'+item[0])
            im = im[60:660, 180:780]
            out[id] = im[cx:cx+227, cy:cy+227]
        return out
    
    def center_crop(self, lst, id):
        out = np.zeros((SEQ_LEN, 227, 227, 3), dtype='uint8')
        
        for id, item in enumerate(lst[id:id+SEQ_LEN]):
            im = cv2.imread('data/'+item[0])
            im = im[246:473, 366:593]
            out[id] = im
        return out
        
    def get_next_train(self):
        data = self.rng.choice(self.train_examples, BATCH_SIZE)

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')
        lb_wash = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.train[data[i]][1])
            lb_wash[i] = int(self.train[data[i]][3])
            imgs[i] = self.random_crop(self.train, data[i])
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb, lb_wash

    def get_val_batch_num(self):
        return len(self.val_examples) // BATCH_SIZE

    def get_val(self, idx):
        data = self.val_examples[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')
        lb_wash = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.val[data[i]][1])
            lb_wash[i] = int(self.train[data[i]][3])
            imgs[i] = self.center_crop(self.val, data[i])
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb, lb_wash

    def get_test_max_id(self):
        return np.max(self.test_examples)
    
    def get_test_batch_num(self):
        return len(self.test_examples) // BATCH_SIZE

    def get_test(self, idx):
        data = self.test_examples[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

        imgs = np.zeros((BATCH_SIZE, SEQ_LEN, 227, 227, 3))
        lb = np.zeros(BATCH_SIZE, dtype='int32')
        lb_wash = np.zeros(BATCH_SIZE, dtype='int32')

        for i in range(BATCH_SIZE):
            lb[i] = int(self.test[data[i]][1])
            lb_wash[i] = int(self.train[data[i]][3])
            imgs[i] = self.random_crop(self.test, data[i])
        return imgs.reshape((BATCH_SIZE * SEQ_LEN, 227, 227, 3)), lb, lb_wash, data

def main(_):
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.int32, [None])
    y_wash = tf.placeholder(tf.int32, [None])
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
        with tf.variable_scope('softmax_wash'):
            logits_wash = tf.contrib.layers.fully_connected(outputs[:, -1, :], 5, activation_fn=None)
        softmax_prob = tf.nn.softmax(logits)
        softmax_wash_prob = tf.nn.softmax(logits_wash)

        entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
        entropy_wash = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_wash, logits=logits_wash))

        varss = tf.trainable_variables()
        l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in varss]) * 0.0001

        loss = 0.7*entropy + 0.3*entropy_wash + l2_loss
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
            gLoss_wash = []
            for epoch in range(10):
                print("Epoch number: {}".format(epoch+1))
                for step in range(20000):
                    img_batch, label_batch, label_wash_batch = data.get_next_train()
                    lossv, lossv_wash, _ = sess.run((entropy, entropy_wash, opt_op), feed_dict={x: img_batch - imagenet_mean,
                                                                        y: label_batch,
                                                                        y_wash: label_wash_batch,
                                                                        keep_prob: 0.5})
                    gLoss = gLoss + [lossv]
                    gLoss = gLoss[-50:]
                    gLoss_wash = gLoss_wash + [lossv_wash]
                    gLoss_wash = gLoss_wash[-50:]
                    if step % 100 == 0:
                        print(epoch, step, np.mean(gLoss))
                        print(epoch, step, np.mean(gLoss_wash))
                # Validate the model on the entire validation set
                print("Start validation")
                test_acc = 0.
                test_count = 0

                for i in range(data.get_val_batch_num()):
                    img_batch, label_batch, label_wash_batch = data.get_val(i)
                    predict, predict_wash = sess.run((logits, logits_wash), feed_dict={x: img_batch - imagenet_mean,
                                                     y: label_batch,
                                                     y_wash: label_wash_batch,
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

        elif FLAGS.run == 'val':
            data = Dataset([1])
            test_acc = 0
            test_count = 0
            test_acc_3 = 0

            def compute_acc(x, y):
                z = np.asarray([v in u for u, v in zip(x, y)], dtype='float32')
                return np.sum(z)
            
            for i in range(data.get_val_batch_num()):
                img_batch, label_batch, label_wash = data.get_val(i)
                predict, predict_wash = sess.run((logits, logits_wash), feed_dict={x: img_batch - imagenet_mean,
                                                    y: label_batch,
                                                    y_wash: label_wash_batch,
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
        else:
            data = Dataset([2])
            test_acc = 0
            test_count = 0
            test_acc_3 = 0

            def compute_acc(x, y):
                z = np.asarray([v in u for u, v in zip(x, y)], dtype='float32')
                return z

            def compute_topk(x, y, k):
                pretopk = np.argsort(x)[:, -k:]
                acc = compute_acc(pretopk[y != -1], y[y != -1])
                return np.mean(acc)
                
                
            score = np.zeros((data.get_test_max_id() + 1, 14))
            label = -np.ones(data.get_test_max_id() + 1, dtype='int32')
            
            for i in range(data.get_test_batch_num()):
                img_batch, label_batch, label_wash_batch, idx = data.get_test(i)
                predict = sess.run(softmax_prob, feed_dict={x: img_batch - imagenet_mean,
                                                        y: label_batch,
                                                        y_wash: label_wash_batch,
                                                        keep_prob: 1.})
                score[idx] += predict
                label[idx] = label_batch
                if i % 100 == 0:
                    print(i, compute_topk(score, label, 1), compute_topk(score, label, 3))
                
            print("final", compute_topk(score, label, 1), compute_topk(score, label, 3))

if __name__ == "__main__":
    tf.app.flags.DEFINE_string('load', '', 'load model')
    tf.app.flags.DEFINE_string('run', '', 'run test')

    tf.app.run()
