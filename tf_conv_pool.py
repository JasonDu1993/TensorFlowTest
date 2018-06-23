import tensorflow as tf
import numpy as np
from math import ceil

tf.set_random_seed(1)
n = 14
f = 6
s = 5
startw = 0
endw = 6
starth = 0
endh = 6

arr = np.arange(n * n * 3).reshape((1, n, n, 3))
pad_total = ((n - 1) * s + f - n)
if pad_total % 2 != 0:
    pad_total += 1
print('pad_total', pad_total)
arr = np.pad(arr,
             ((0, 0), (pad_total // 2, pad_total // 2), (pad_total // 2, pad_total // 2), (0, 0)),
             'constant')
print('arr shape', arr.shape)
arr1 = arr[:, starth:endh, startw:endw, 0]
arr2 = arr[:, starth:endh, startw:endw, 1]
arr3 = arr[:, starth:endh, startw:endw, 2]
print('convolution sum', arr1.sum() + arr2.sum() + arr3.sum())
print('arr1', arr1)
# input = tf.Variable(tf.random_normal([1, 51, 50, s]))
inp = tf.constant(arr, dtype=tf.float32)
print('inp', inp)
# filter = tf.Variable(tf.random_normal([3, 3, 3, 1]))
fil = tf.ones([f, f, 3, 1])
op = tf.nn.conv2d(inp, fil, strides=[1, s, s, 1], padding='VALID')
pool = tf.nn.max_pool(inp, [1, f, f, 1], [1, s, s, 1], padding='VALID')

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    inp_run = sess.run(inp)
    fil_run = sess.run(fil)
    op_run = sess.run(op)
    pool_run = sess.run(pool)
    # print('fil_run', fil_run, type(fil_run))
    h = 0
    w = 0
    print('op_run one:', op_run[:, h, w, 0], op_run.shape)
    # print('op_run', op_run)
    print('pool_run', pool_run[:, h, w, 0], pool_run.shape)
