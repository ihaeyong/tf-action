import tensorflow as tf
import numpy as np
import utils as utils
import re
from tensorflow.contrib import rnn

clip_th = 0.1
lr = 0.001
data_step=16

class Model(object):

    def __init__(self, batch_size, reuse=None, is_training=True):
        self.batch_size = batch_size
        self.feat_dim = 4096
        self.num_steps = data_step
        self.num_layers = 1
        self.num_cell = 256
        self.num_cell_time=128
        self.num_class = 101
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.feat_dim],
                                              name="inputs")
        self.labels = tf.placeholder(tf.int64, shape=[None], name="labels")
        self.is_training = is_training
        self.lr = lr
        self.global_step = tf.Variable(0, trainable=False)
        self.seq_len = self.length(self.inputs)

        sp_out, mask = self.sp_attention(self.inputs)

        concat_features = self.calc_f_v_a(self.inputs)
        t_out = self.time_attention(concat_features, reuse)
     #   term1=self.entropy_s(mask)
    #    term2=self.entropy(t_out)
        attended_inputs = t_out * sp_out
        logits = self.inference(attended_inputs, reuse)
       	
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1),
        #                                            self.labels), tf.float32))

        self.accuracy = self.top_k_error(logits, self.labels, k=1)

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                             labels=self.labels))

        self.train_op = self.optimize(self.cost)
        #self.minize1 = self.minimize(term1)
       # self.minize2 = self.minimize(term2)

    def calc_f_v_a(self, inputs):
        v, a = self.calc_delta(inputs)
      #  a = self.calc_delta_delta(v)

        concat_features =tf.concat([v, a, inputs], axis=2)
        return concat_features

    @staticmethod
    def calc_delta(inputs):
        inputs = tf.stop_gradient(inputs)
        padding = tf.constant([[0, 0], [1, 0], [0, 0]], dtype=tf.int32)
        padded_inputs = tf.pad(inputs, padding, "CONSTANT")
        delta = inputs - padded_inputs[:, :-1, :]
        #delta = tf.stop_gradient(delta)
        padding = tf.constant([[0, 0], [0, 1], [0, 0]], dtype=tf.int32)
        padded_inputs1 = tf.pad(delta, padding, "CONSTANT")
        delta_2 = delta - padded_inputs1[:, 1:, :]
       # print(delta.shape))
        #print(len(delta_2.shape))		
        return delta, delta_2
  

    def entropy(self, vector, axis=1):

        # act4= tf.exp(vector)
        # act5=tf.reduce_sum(act4, axis=1)
        # term = tf.div(x=tf.squeeze(act4), y=tf.squeeze(act5))
        # last = -tf.reduce_mean(vector1, axis=1)

        vector =tf.squeeze(vector, axis=-1)
        sm_out = tf.nn.softmax(vector, axis=1)
        self_entropy = -tf.reduce_mean(tf.multiply(sm_out, tf.log(sm_out)), axis=1)

        return self_entropy
		
    def entropy_s(self, vector, axis=2):

        # act4= tf.exp(vector)
        # act5=tf.reduce_sum(act4, axis=1)
        # term = tf.div(x=tf.squeeze(act4), y=tf.squeeze(act5))
        # last = -tf.reduce_mean(vector1, axis=1)

        #vector =tf.squeeze(vector, axis=-1)
        sm_out = tf.nn.softmax(vector, axis=2)
        self_entropy_s = -tf.reduce_mean(tf.multiply(sm_out, tf.log(sm_out)), axis=2)
       # self_entropy_s = tf.reduce_mean(self_entropy_s, axis=1)
        #self_entropy_s  =tf.squeeze(self_entropy_s)
       # print(self_entropy_s.shape)		
        return self_entropy_s	
		
		

    def minimize(self, loss_):
        minimize= tf.train.AdamOptimizer(self.lr).minimize(loss=loss_)	
        return minimize

    def config(self, feat_dim=4096, num_steps=data_step, num_layers=1, num_cell=data_step):
        self.feat_dim = feat_dim
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.num_cell = num_cell

    def optimize(self, cost):
        var_list = tf.trainable_variables()
        grads = tf.gradients(cost, var_list)
        grads, _ = tf.clip_by_global_norm(grads, clip_th)
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)
        return train_op

    def inference(self, inputs, reuse=None):

        with tf.variable_scope("classification", reuse=reuse):
            cell_fw = [tf.contrib.rnn.LSTMCell(size, initializer=tf.contrib.layers.xavier_initializer(),
                                               reuse=reuse) for size in [self.num_cell]*self.num_layers]

            cell_bw = [tf.contrib.rnn.LSTMCell(size, initializer=tf.contrib.layers.xavier_initializer(),
                                               reuse=reuse) for size in [self.num_cell]*self.num_layers]

            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=cell_fw, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=cell_bw, state_is_tuple=True)

            cell_out_list = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                            sequence_length=self.length(inputs), dtype=tf.float32)[0]
            cell_out_fw = self.last_relevant(cell_out_list[0], self.length(inputs))
            cell_out_bw = self.last_relevant(tf.reverse(cell_out_list[1], axis=[1]), self.length(inputs))

            cell_output = tf.concat([cell_out_fw, cell_out_bw], 1)
            logits = utils.affine_transform(cell_output, self.num_class, seed=0, name='softmax_logits')
            return logits

    def top_k_error(self, predictions, labels, k=1):
        batch_size = self.batch_size  # tf.shape(predictions)[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size -
                num_correct) / batch_size

    def sp_attention(self, inputs):

        inputs = tf.expand_dims(inputs, axis=3)
        conv_shape = [1, 1, 1, 1]
        kernels = utils.weight_variable(conv_shape, stddev=0.02, name='conv_W')
        bias = utils.bias_variable([conv_shape[-1]], name='conv_b')
        conv_out = utils.conv2d_basic(inputs, kernels, bias, stride=1)
       # attention = tf.nn.softmax(conv_out, axis=2)
        attention = tf.nn.sigmoid(conv_out)
        attended_inputs = tf.squeeze(inputs * attention, axis=-1)
        return attended_inputs, attention

    def time_attention(self, inputs, reuse=None):
        with tf.variable_scope("attention", reuse=reuse):
            cell_fw = [tf.contrib.rnn.LSTMCell(size, initializer=tf.contrib.layers.xavier_initializer(),
                                               reuse=reuse) for size in [self.num_cell_time]*self.num_layers]

            cell_bw = [tf.contrib.rnn.LSTMCell(size, initializer=tf.contrib.layers.xavier_initializer(),
                                               reuse=reuse) for size in [self.num_cell_time]*self.num_layers]

            cell_fw = tf.nn.rnn_cell.MultiRNNCell(cells=cell_fw, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(cells=cell_bw, state_is_tuple=True)

            cell_out_list = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                            sequence_length=self.length(inputs), dtype=tf.float32)[0]
            cell_output = tf.reshape(tf.concat([cell_out_list[0], cell_out_list[1]], 2), shape=[-1,
                                                                                                data_step*self.num_cell_time*2])

            t_attention = tf.expand_dims(tf.sigmoid(utils.affine_transform(cell_output, data_step,
                                                                           seed=0, name='sigmoid')), axis=2)
            return t_attention

    @staticmethod
    def length(sequence):
        '''
        :param sequence: (batch_size, num_steps, feat_dims) 
        :return: the sequence length
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def main():
    print("aa")
    aa = Model(2, reuse=None, is_training=True)
    print("aa")

if __name__ == "__main__":
    main()