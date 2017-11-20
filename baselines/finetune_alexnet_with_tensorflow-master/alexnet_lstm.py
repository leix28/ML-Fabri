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

class Dataset(object):
    def __init__(self):
        self.rng = np.random
        with open('material_dataset.txt') as f:
            imglist = f.readlines()
        imglist = [item.split() for item in imglist]

        def load(lst):
            out = np.zeros((len(lst), 227, 227, 3), dtype='uint8')
            for idx, item in enumerate(lst):
                im = cv2.imread('data/'+item[0])
                im = im[246:473, 366:593]
                out[idx] = im
                if idx % 1000 == 0:
                    print("load {}/{}".format(idx, len(lst)))
            return out

        self.train = list(filter(lambda x: x[2] == '0', imglist))
        self.train_img = load(self.train)
        self.val = list(filter(lambda x: x[2] == '1', imglist))
        self.val_img = load(self.val)
        self.test = list(filter(lambda x: x[2] == '2', imglist))
        self.test_img = load(self.test)

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

def main():
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)

    #create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])

    data = Dataset()

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
        opt = tf.train.GradientDescentOptimizer(0.01)
        opt_op = opt.minimize(loss)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        print("Model set up.")

        gLoss = []
        for epoch in range(100):
            print("Epoch number: {}".format(epoch+1))
            for step in range(100):
                img_batch, label_batch = data.get_next_train()
                lossv, _ = sess.run((entropy, opt_op), feed_dict={x: img_batch - imagenet_mean,
                                                                    y: label_batch,
                                                                    keep_prob: 0.5})
                gLoss = gLoss + [lossv]
                gLoss = gLoss[-10:]
                if step % 10 == 0:
                    print(epoch, step, np.mean(gLoss))

            # Validate the model on the entire validation set
            print("Start validation")
            test_acc = 0.
            test_count = 0

            for i in range(data.get_val_batch_num()):
                img_batch, label_batch = data.get_val(i)
                predict = sess.run(logits, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.})
                correct = np.sum(np.argmax(predict) == label_batch)
                test_acc += correct
                test_count += len(label_batch)

            test_acc /= test_count
            print("Validation Accuracy = {:.4f}".format(test_acc))
            print("Saving checkpoint of model...")

            print("Epoch {}, Val {}".format(epoch+1, test_acc * 100), file=log)
            # save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            # print("{} Model checkpoint saved at {}".format(datetime.now(),
            #                                                checkpoint_name))


if __name__ == "__main__":
    main()
