# Developed by Ananye Patel. 2018-07-30.
import tensorflow as tf
import numpy as np
import os
import glob
import cv2 as cv
import sys
import argparse

# to ignore warning display
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def verify(img_path):
    image_size = 128
    num_channels = 3

    images = []
    # Reading the image using OpenCV
    image = cv.imread(img_path)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv.resize(image, (image_size, image_size), 0, 0, cv.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0/255.0)
    images.append(image)
    images = np.array(images)

    # we restore the saved model
    sess = tf.Session()
    # Step 1: recreate the network graph (only graph is created at this step)
    saver = tf.train.import_meta_graph('human-detection-model.meta')
    # Step 2: now we load the weights saved using the restore method
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, lets get hold of the op that we can process to get the output
    # in the original network, y_pred is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    # classification threshold (converts probability to binary output)
    threshold = 70.0

    # the input to the network is of shape [None, image_size, image_size, num_channels], so we reshape
    x_batch = image.reshape(1, image_size, image_size, num_channels)
    # lets feed the image to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    # creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of format [probability(human), probability(not_human)]
    result = result*100
    sys_pred = result[0][0]
    # human: 0, nothuman: 1
    print('\n')
    if (sys_pred > threshold):
        outcome = 0
    else:
        outcome = 1
    return sys_pred, outcome;
