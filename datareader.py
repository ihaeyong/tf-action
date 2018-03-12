import numpy as np
import os
import glob
import utils
import scipy.io as sio
import random


class DataReader(object):

    def __init__(self, input_dir, output_dir, max_len=100, is_shuffle=True):
        # print(name.title() + " data reader initialization...")
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._input_file_list = sorted(glob.glob(input_dir+'/*.npy'))
        self._output_file_list = sorted(glob.glob(output_dir+'/*.npy'))

        self._file_len = len(self._input_file_list)
        self._is_shuffle = is_shuffle

        if self._is_shuffle:
            self._file_perm()  # file permutation
        self._max_len = max_len

        self._start_idx = 0
        self.epoch = 0

    def _zero_padding(self, data):
        pad_len = self._max_len - data.shape[0]
        pad = np.zeros((pad_len, data.shape[1]), dtype=np.float32)
        return np.concatenate((data, pad), axis=0)

    def _file_perm(self):
        r = random.random()
        random.shuffle(self._input_file_list, lambda: r)
        random.shuffle(self._output_file_list, lambda: r)

    def next_batch(self, batch_size):

        if self._start_idx + batch_size > self._file_len:
            print('epoch = %d' % self.epoch)

            x_list = [self._zero_padding(np.load(i)) for i
                      in self._input_file_list[self._start_idx:]]
            y_list = [np.load(i) for i
                      in self._output_file_list[self._start_idx:]]
            self._start_idx = 0
            self.epoch += 1
            if self._is_shuffle:
                self._file_perm()

            return np.asarray(x_list), np.asarray(y_list)
        else:
            x_name = self._input_file_list[self._start_idx:self._start_idx + batch_size]
            y_name = self._output_file_list[self._start_idx:self._start_idx + batch_size]
            x_list = [self._zero_padding(np.load(i)) for i
                      in self._input_file_list[self._start_idx:self._start_idx + batch_size]]
            y_list = [np.load(i) for i
                      in self._output_file_list[self._start_idx:self._start_idx + batch_size]]

            self._start_idx += batch_size

            return np.asarray(x_list), np.asarray(y_list)

    def next_batch_last(self, batch_size):

        if self._start_idx + batch_size > self._file_len:
            print('epoch = %d' % self.epoch)

            x_list = [self._zero_padding(np.load(i)) for i
                      in self._input_file_list[self._start_idx:]]
            y_list = [np.load(i) for i
                      in self._output_file_list[self._start_idx:]]
            self._start_idx = 0
            self.epoch += 1
            if self._is_shuffle:
                self._file_perm()

            return np.asarray(x_list), np.asarray(y_list)
        else:
            x_name = self._input_file_list[self._start_idx:self._start_idx + batch_size]
            y_name = self._output_file_list[self._start_idx:self._start_idx + batch_size]
            x_list = [self._zero_padding(np.load(i)) for i
                      in self._input_file_list[self._start_idx:self._start_idx + batch_size]]
            y_list = [np.load(i) for i
                      in self._output_file_list[self._start_idx:self._start_idx + batch_size]]

            self._start_idx += batch_size

            return np.asarray(x_list), np.asarray(y_list), x_name, y_name			

def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def main():

    file_dir = "./data"

    train_in = file_dir + '/train'
    train_out = file_dir + '/train/labels'

    test_in = file_dir + '/test'

    aa = DataReader(train_in, train_out)
    bb, cc, dd = aa.next_batch(2)
    print("aa")

if __name__ == "__main__":
    main()
