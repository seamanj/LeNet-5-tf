# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import time

FLAGS = None

current_milli_time = lambda: int(round(time.time() * 1000))

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # tj : (?, 28, 28, 1)
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        # tj : [filter_height, filter_width, in_channels, out_channels]
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # about the sequence size :
        # https://stackoverflow.com/questions/48951622/wrong-output-size-after-conv2d-function?rq=1
        # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        # About the name:
        # With "SAME" padding, if you use a stride of 1, the layer's outputs will have the same spatial dimensions
        # as its inputs.
        # With "VALID" padding, there's no "made-up" padding inputs. The layer only uses valid input data. size can be calculated
        # by the formula in pyTorch, which is : floor((28 - 5) / 1  + 1) = 24
        # tj : But here we use the "SAME" padding : 28 * 28 * 1 -> 28 * 28 * 32

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
        # tj : 28 * 28 * 32 -> 14 * 14 * 32
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # tj : 14 * 14 * 32 -> 14 * 14 * 64

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
        # tj : 14 * 14 * 64 -> 7 * 7 * 64

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, FLAGS.num_fc1])
        b_fc1 = bias_variable([FLAGS.num_fc1])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # tj : flat(7 * 7 * 64) -> FLAGS.num_fc1

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FLAGS.num_fc1, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # tj : FLAGS.num_fc1 -> 10
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # tj : https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, validation_size=0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    # tj : each image is 28 * 28, but for how many images we don't know yet.

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    # tj : each image has 10 classes (0-10), but for how many images we don't know yet.

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)
    # tj : build the graph first, then feed the data

    with tf.name_scope('loss'):

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True




    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # tj : visualize in Tensorboard
        # https://blog.csdn.net/dcrmg/article/details/83017118\
        # detailed one : https://blog.csdn.net/helei001/article/details/51842531
        summary_writer = tf.summary.FileWriter('./log/', sess.graph)

        for v in tf.trainable_variables():
            print(v.name + " " + str(v.get_shape()) + " " + str(np.prod(v.get_shape().as_list())))
        print("Total Number of Parameters: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        
        for i in range(FLAGS.num_iterations):
            batch = mnist.train.next_batch(100)
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.test_keep_prob})
                # tj : calculate accuracy
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.train_keep_prob})
            # tj : optimization, note there is no loss step, cuz loss is included in adam_optimizer

        test_batch_size = 100
        test_sample_count = 10000
        test_iterations = np.int(test_sample_count / test_batch_size)
        test_accuracies = np.zeros((test_iterations, 1))

        for i in range(test_iterations):
            test_accuracies[i] = accuracy.eval(feed_dict={x: mnist.test.images[i * test_batch_size:(i + 1) * test_batch_size],
                                                          y_: mnist.test.labels[i * test_batch_size:(i + 1) * test_batch_size],
                                                          keep_prob: FLAGS.test_keep_prob})
            # tj : calculate the accuracy in test step.
            # print("test iter {} accuracy {}".format(i, test_accuracies[i]))
        print("test accuracy: {}".format(np.mean(test_accuracies)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnist', help='Directory of the input data is')
    parser.add_argument('--train_keep_prob', type=float, default=0.5, help='Training Keep Probability (1 - Dropout Ratio)')
    parser.add_argument('--test_keep_prob', type=float, default=1.0, help='Testing Keep Probability (1 - Dropout Ratio)')
    parser.add_argument('--num_fc1', type=int, default=512, help='Number of FC1 Output Units')
    parser.add_argument('--num_iterations', type=int, default=2000, help='Number of training iterations')
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
