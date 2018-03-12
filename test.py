import numpy as np
import utils as utils
import tensorflow as tf
import model as model
import datareader as dr
import os
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "test", "mode : train/ test [default : train]")
''' set directory '''

file_dir = "./data"

train_in = file_dir + '/train'
train_out = file_dir + '/train/labels'

test_in = file_dir + '/test'
test_out = file_dir + '/test/labels'

logs_dir = file_dir + '/logs'
save_dir = "./model/"
ckpt_name = ''

'''set parameters'''
device = '/gpu:1'
batch_size =1
max_len = 16
max_epoch = int(1e4)
dropout_rate = 0.5
iter_freq = 10
val_freq = 2

#                               Graph Part                                 #
accmax=1
print("Graph initialization...")
with tf.device(device):
    with tf.variable_scope("model", reuse=None):
        m_train = model.Model(batch_size=batch_size, reuse=None, is_training=True)
with tf.device(device):
    with tf.variable_scope("model", reuse=True):
        m_valid = model.Model(batch_size=batch_size, reuse=True, is_training=False)

print("Done")

#                               Model Save Part                            #
print("Setting up Saver...")
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(logs_dir)
print("Done")

#                               Session Part                               #
sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

if ckpt and ckpt.model_checkpoint_path:  # model restore
    print("Model restored...")
    print(logs_dir+ckpt_name)
    saver.restore(sess, logs_dir+ckpt_name)
    print("Done")
else:
    sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization
test_dataset = dr.DataReader(test_in, test_out, max_len=max_len, is_shuffle=False)

acc =utils.evaluation_last(m_valid, sess, test_dataset)
print('test_acc = %f' % acc)





