import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('imglist', 'imgs.txt', 'List of data images')
flags.DEFINE_string('label', 'label.csv', 'labels')
flags.DEFINE_string('output', 'out.txt', "output")
flags.DEFINE_string('labeldic', 'dict.txt', 'output number to label dict')
flags.DEFINE_string('rate', '7:1:2', 'train, val, test')

FLAGS = flags.FLAGS

def load_data(filename):
    imgs = open(filename).readlines()
    imgs = [(img[:-1], int(img.split('_')[0][2:])) for img in imgs]
    return imgs

def load_label(filename):
    print(filename)
    labels = np.loadtxt(filename, dtype='str', delimiter=',')

    classdic = {}
    count = {}
    N = 0
    for item in labels:
        if not item[1] in classdic:
            classdic[item[1]] = N
            count[N] = 0
            N += 1
        count[classdic[item[1]]] += 1

    labels = [(int(item[0]), item[1]) for item in labels]
    return labels, classdic, count



def main(_):
    thre = np.asarray(FLAGS.rate.split(':'), dtype='float32')
    test_thre = thre[2] / np.sum(thre)
    val_thre = (thre[1] + thre[2]) / np.sum(thre)
    print(test_thre, val_thre)

    imgs = load_data(FLAGS.imglist)
    labels, classdic, count = load_label(FLAGS.label)


    with open(FLAGS.labeldic, 'w')  as f:
        print('id\ttype\tcount', file=f)
        for k, v in classdic.items():
            print(v, k, count[v], file=f, sep='\t')


    cc = np.zeros(len(classdic))
    tp = {}

    for item in labels:
        lb = classdic[item[1]]
        if cc[lb] == 0 or (cc[lb] + 1) < count[lb] * test_thre:
            tp[item[0]] = 2
        elif (cc[lb] + 1) < count[lb] * val_thre:
            tp[item[0]] = 1
        else:
            tp[item[0]] = 0
        cc[lb] += 1


    labels = dict(labels)
    with open(FLAGS.output, 'w') as f:
        for item in imgs:
            try:
                print(item[0], classdic[labels[item[1]]], tp[item[1]], sep='\t', file=f)
            except:
                print(item[0], -1, -1, sep='\t', file=f)


if __name__ == "__main__":
    tf.app.run()
