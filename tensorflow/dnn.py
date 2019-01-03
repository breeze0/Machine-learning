# -*- coding: UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    mnist = input_data.read_data_sets("MNIST_data/")

    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    X = tf.placeholder(tf.float32, shape= (None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,name= 'hidden1')

        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',activation= tf.nn.relu)

        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = logits)
        loss = tf.reduce_mean(xentropy, name='loss')#所有值求平均

    learning_rate = 0.01

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits ,y ,1)#是否与真值一致 返回布尔值
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0,1序列

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 50
    执行训练阶段
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op,feed_dict={X:X_batch,y: y_batch})
            acc_train = accuracy.eval(feed_dict={X:X_batch,y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./my_model_final.ckpt")

    #利用生存的model来进行预测
    # with tf.Session() as sess:
    #     saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    #     X_new_scaled = mnist.test.images[:20]
    #     Z = logits.eval(feed_dict={X: X_new_scaled})
    #     y_pred = np.argmax(Z, axis=1)
    #     print(y_pred)