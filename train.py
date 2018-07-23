import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os

# adding seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

FLAGS=None
beta = 0.01

batch_size = 11

# preparing input data
classes = ['human', 'nothuman']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'training_data'

# we load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Input data read. Printing macro-details")
print("Number of images in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of images in Validation-set:\t\t{}".format(len(data.valid.labels)))
print('\n')

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# keep_prob = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='keep_prob')

# Network graph parameters
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):

    # we shall define the weights that will be trained using create_weights function
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # we create biases using the create_biases function which are also trained
    biases = create_biases(num_filters)


    # creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases

    # we use max-pooling on the convolved layer
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # output of pooling is fed to the ReLU activation function
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):

    # we know the shape of the layer will be [batch_size, img_size, img_size, num_channels]
    # but we get it from the previous layer
    layer_shape = layer.get_shape()

    # number of features will be img_height * img_width * num_channels, but we calculate it instead of hard-coding
    num_features = layer_shape[1:4].num_elements()

    # now we flatten the layer so we have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    # we define trainable weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

# Fully connected layer takes input x and produces wx+b. Since these are matrices, we use the matmul function in Tensorflow
  #  drop_out = tf.nn.dropout(input, keep_prob)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters_conv1, conv_filter_size=filter_size_conv2, num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filters_conv2, conv_filter_size=filter_size_conv3, num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_layer_size, use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_layer_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())

with tf.name_scope('cost'):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
    with tf.name_scope('total'):
        cost = tf.reduce_mean(cross_entropy)
        # regularizer = tf.nn.l2_loss(layer_fc2)
        # cost = tf.reduce_mean(cost + beta * regularizer)
tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/home/ananye/PycharmProjects/mark1/summaries/train', session.graph)
valid_writer = tf.summary.FileWriter('/home/ananye/PycharmProjects/mark1/summaries/valid', session.graph)

session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))
            # block to iteratively write summary operations to files
            summary, acc = session.run([merged, accuracy], feed_dict=feed_dict_val)
            valid_writer.add_summary(summary, i)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_tr, options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_tr)
            train_writer.add_summary(summary, i)

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, '/home/ananye/PycharmProjects/mark1/human-detection-model')

        total_iterations += num_iteration
    train_writer.close()
    valid_writer.close()

train(num_iteration=2666)
