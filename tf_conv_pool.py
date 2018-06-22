import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
arr = np.arange(507).reshape((1, 13, 13, 3))
print(arr[:, 6, 6, 0].sum())
print(arr[:, :, :, 1])
print(arr[:, :, :, 2])

# input = tf.Variable(tf.random_normal([1, 51, 50, 5]))
inp = tf.constant(arr, dtype=tf.float32)
print('inp', inp)
# filter = tf.Variable(tf.random_normal([3, 3, 3, 1]))
fil = tf.ones([6, 6, 3, 1])
op = tf.nn.conv2d(inp, fil, strides=[1, 5, 5, 1], padding='SAME')
pool = tf.nn.max_pool(inp, [1, 6, 6, 1], [1, 5, 5, 1], padding='SAME')

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    inp_run = sess.run(inp)
    fil_run = sess.run(fil)
    op_run = sess.run(op)
    pool_run = sess.run(pool)
    # print('fil_run', fil_run, type(fil_run))
    print('op_run', op_run, op_run.shape)
    print('pool_run', pool_run, pool_run.shape)
