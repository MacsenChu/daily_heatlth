import tensorflow.compat.v1 as tf
import glob
import pandas as pd
import numpy as np
import os
tf.disable_v2_behavior()
dir_path = os.path.dirname(os.path.abspath(__file__))


# 1）读取图片数据filename -> 标签值
def read_picture():
    """
    读取验证码图片
    :return:
    """
    # 1、构造文件名队列
    file_list = glob.glob(os.path.join(dir_path, "Genpics", "*.jpg"))
    # print("file_list:\n", file_list)
    file_queue = tf.train.string_input_producer(file_list)

    # 2、读取与解码
    # 读取
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)

    # 解码
    image_decode = tf.image.decode_jpeg(image)

    # 更新图片形状
    image_decode.set_shape([24, 60, 3])
    # print("image_decode:\n", image_decode)
    # 修改图片类型
    image_cast = tf.cast(image_decode, tf.float32)

    # 3、构造批处理队列
    filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=2, capacity=100)

    return filename_batch, image_batch


# 2）解析csv文件，将标签值NZPP->[13, 25, 15, 15]
def parse_csv():
    # 解析CSV文件, 建立文件名和标签值对应表格
    csv_data = pd.read_csv(os.path.join(dir_path, "Genpics", "labels.csv"), names=["file_num", "digit"], index_col="file_num")

    labels = []
    for label in csv_data["digit"]:
        tmp = []
        for letter in str(label):
            tmp.append(ord(letter) - ord("0"))
        labels.append(tmp)

    csv_data["labels"] = labels

    return csv_data


def filename2label(filenames, csv_data):
    """
    将filename和标签值联系起来
    :param filenames:
    :param csv_data:
    :return:
    """
    labels = []

    # 将b'文件名中的数字提取出来并索引相应的标签值

    for filename in filenames:
        digit_str = "".join(list(filter(str.isdigit, str(filename))))
        label = csv_data.loc[int(digit_str), "labels"]
        labels.append(label)

    # print("labels:\n", labels)

    return np.array(labels)


# 4）构建卷积神经网络->y_predict
def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))


# 5）构造损失函数
# 6）优化损失
# 7）计算准确率
# 8）开启会话、开启线程
def create_model(x):
    """
    构建卷积神经网络
    :param x:[None, 24, 60, 3]
    :return:
    """
    # 1）第一个卷积大层
    with tf.variable_scope("conv1"):
        # 卷积层
        # 定义filter和偏置
        conv1_weights = create_weights(shape=[5, 5, 3, 32])
        conv1_bias = create_weights(shape=[32])
        conv1_x = tf.nn.conv2d(input=x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2）第二个卷积大层
    with tf.variable_scope("conv2"):
        # [None, 24, 60, 3] --> [None, 12, 30, 32]
        # 卷积层
        # 定义filter和偏置
        conv2_weights = create_weights(shape=[5, 5, 32, 64])
        conv2_bias = create_weights(shape=[64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3）全连接层
    with tf.variable_scope("full_connection"):
        # [None, 12, 30, 32] -> [None, 6, 15, 64]
        # [None, 6, 15, 64] -> [None, 6 * 15 * 64]
        # [None, 6 * 15 * 64] * [6 * 15 * 64, 4 * 10] = [None, 4 * 10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 6 * 15 * 64])
        weights_fc = create_weights(shape=[6 * 15 * 64, 4 * 10])
        bias_fc = create_weights(shape=[4 * 10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


def captcha_train():
    filename, image = read_picture()
    csv_data = parse_csv()

    # 1、准备数据
    x = tf.placeholder(tf.float32, shape=[None, 24, 60, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 4 * 10])

    # 2、构建模型
    y_predict = create_model(x)

    # 3、构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 4、优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 5、计算准确率
    equal_list = tf.reduce_all(
        tf.equal(tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 10]), axis=2),
                 tf.argmax(tf.reshape(y_true, shape=[-1, 4, 10]), axis=2)), axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:

        # 初始化变量
        sess.run(init)

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(os.path.join(dir_path, "ckpt", "checkpoint")):
            saver.restore(sess, os.path.join(dir_path, "ckpt", "model"))

        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            # print("filename_value:\n", filename_value)
            # print("image_value:\n", image_value)

            labels = filename2label(filename_value, csv_data)
            # 将标签值转换成one-hot
            labels_value = tf.reshape(tf.one_hot(labels, depth=10), [-1, 4 * 10]).eval()

            _, error, accuracy_value = sess.run([optimizer, loss, accuracy],
                                                feed_dict={x: image_value, y_true: labels_value})

            print("第%d次训练后损失为%f，准确率为%f" % (i + 1, error, accuracy_value))
            if i % 100 == 0:
                saver.save(sess, "./ckpt/model")

        # 回收线程
        coord.request_stop()
        coord.join(threads)


def read_predict_captcha(file_name):
    # 1、构造文件名队列
    file_list = glob.glob(file_name)
    # print("file_list:\n", file_list)
    file_queue = tf.train.string_input_producer(file_list)

    # 2、读取与解码
    # 读取
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)

    # 解码
    image_decode = tf.image.decode_jpeg(image)

    # 更新图片形状
    image_decode.set_shape([24, 60, 3])
    # print("image_decode:\n", image_decode)
    # 修改图片类型
    image_cast = tf.cast(image_decode, tf.float32)

    # 3、构造批处理队列
    image_batch = tf.train.batch([image_cast], batch_size=1, num_threads=1, capacity=1)
    # print(image_cast, image_batch)
    return image_batch


def captcha_predict(file_name):
    image = read_predict_captcha(file_name)

    # 1、准备数据
    x = tf.placeholder(tf.float32, shape=[None, 24, 60, 3])

    y_predict = create_model(x)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if os.path.exists(os.path.join(dir_path, "ckpt", "checkpoint")):
            saver.restore(sess, os.path.join(dir_path, "ckpt", "model"))

        image_value = sess.run(image)

        captcha_list = tf.argmax(
                sess.run(tf.reshape(y_predict, shape=[1, 4, 10]), feed_dict={x: image_value}), axis=2
        ).eval()[0]

        captcha = "".join(str(i) for i in captcha_list)
        # print(test)

        # 回收线程
        coord.request_stop()
        coord.join(threads)

        return captcha
