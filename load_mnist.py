import numpy as np
import os
import gzip

# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
# 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

def load_data(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

def load_mnist(load_class='TRAIN',load_type='ALL',index=0):
    (train_images, train_labels), (test_images, test_labels) = load_data('./MNIST_data/')
    if load_type=='ALL':
        if load_class=='TRAIN':
            return train_images, train_labels
        elif load_class=='TEST':
            return test_images, test_labels
    elif load_type=='SINGLE':
        if load_class=='TRAIN':
            return train_images[index], train_labels[index]
        elif load_class=='TEST':
            return test_images[index], test_labels[index]
    return
