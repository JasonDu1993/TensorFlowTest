import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()

'''
此处我加载的数据目录如下：
-- img1.jpg
     img2.jpg
     img3.jpg
     ...
-- img1.jpg
     img2.jpg
     ...
-- ...
...
'''
writer = tf.python_io.TFRecordWriter("train.tfrecords")
classes = ['img1', 'img2', 'img3']
for index, name in enumerate(classes):
    class_path = os.path.join(cwd, "image")
    img_path = os.path.join(class_path, name + ".jpg")
    # print(img_path)
    img = Image.open(img_path)
    # print(img)
    img = img.resize((224, 224))
    # print(img)
    img_raw = img.tobytes()  # 将图片转化为原生bytes
    # print(img_raw)
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    # example2=tf.train.Example(features=tf.train.Features(featu={"l":}))
    writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()

for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    img_raw = example.features.feature['img_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    # print(label, img_raw)
    # print(label)


# 使用队列读取
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


img, label = read_and_decode("train.tfrecords")
# 使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l = sess.run([img_batch, label_batch])
        # 我们也可以根据需要对val， l进行处理
        # l = to_categorical(l, 12)
        print(val.shape, l)
