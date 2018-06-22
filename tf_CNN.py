import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取图片数据集
print(type(mnist.train.next))
class VGG19(object):
    def __init__(self, learn_rating=2.5e-4):
        self.learn_rating = learn_rating
        with tf.name_scope('inputs'):
            self.img = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input_img')
            self.gt_maps = tf.placeholder(dtype=tf.float32, shape=[None, 10])
            self.output = self.generate_model()

    def generate_model(self):
        print('s')
