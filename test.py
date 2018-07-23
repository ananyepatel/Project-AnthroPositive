import tensorflow as tf
import numpy as np
import os
import glob
import cv2 as cv
import sys
import argparse
from sklearn.utils import shuffle

# to ignore warning display
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes = ['human', 'nothuman']
num_classes = len(classes)

image_size = 128
num_channels = 3
test_path = 'testing_data'

images = []
labels = []
img_names = []
cls = []

for fields in classes:
    index = classes.index(fields)
    print('Reading {} files (Index: {})'.format(fields, index))
    path = os.path.join(test_path, fields, '*g')
    files = glob.glob(path)
    for fl in files:
            image = cv.imread(fl)
            image = cv.resize(image, (image_size, image_size),0,0, cv.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0/255.0)
            images.append(image)
            cls.append(fields)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
images = np.array(images)
img_names = np.array(img_names)
cls = np.array(cls)

# we shuffle the data to ensure non-linearity in model
images, img_names = shuffle(images, img_names)
print('Shuffling the input data \n')
# print(img_names)

# lets restore the saved model
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

TP = 0
FP = 0
TN = 0
FN = 0
# classification threshold (converts probability to binary output)
threshold = 80.0
# image iterator
img_count = 0

sum_human = 0
sum_not = 0
fp_img_names = []
fn_img_names = []

for img in images:
    # the input to the network is of shape [None, image_size, image_size, num_channels], so we reshape
    x_batch = img.reshape(1, image_size, image_size, num_channels)
    # lets feed the image to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 2))

    # creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of format [probability(human) probability(not_human)]
    result = result*100
    sys_pred = result[0][0]

    # assumes test data is named appropriately (eg. person007.jpg, notperson069.jpg etc)
    if (img_names[img_count].startswith('person')):
        label = 0
    else:
        label = 1

    np.set_printoptions(precision=5, suppress=True)
    print('P(Human) = ' + str(sys_pred))
    print(img_names[img_count])
    print('Image label: ' + str(label) + '\n')
    # human: 0, nothuman: 1
    if (sys_pred > threshold):
        outcome = 0
        if label == 0:
            reality = 'TP'
            TP += 1
        elif label == 1:
            reality = 'FP'
            FP += 1
            fp_img_names.append(img_names[img_count])
    else:
        outcome = 1
        if label == 1:
            reality = 'TN'
            TN += 1
        elif label == 0:
            reality = 'FN'
            FN += 1
            fn_img_names.append(img_names[img_count])
    img_count += 1


print('True Positives = ' + str(TP))
print('False Positives = ' + str(FP))
print('True Negatives = ' + str(TN))
print('False Negatives = ' + str(FN))
print('\n')

avg_human_prob = sum_human / 288.0
avg_not_prob = sum_not / 300.0

accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = (2 * precision * recall) / (precision + recall)
print('accuracy = ' + str(accuracy*100) + '% of images correctly identified by model')
print('precision = ' + str(precision*100) + '% of positively identified images (human) that are actually correct')
print('recall = ' + str(recall*100) + '% of actual positives (human images) identified correctly')
print('f1 score = ' + str(f1*100) + ' : weighted average of precision and recall')
# print('avg probability(actually human) = ' + str(avg_human_prob) + ' -> choose as rough classification threshold')
# print('avg not human prob = ' + str(avg_not_prob))
print('\nList of False Positive Images')
print(fp_img_names)
print('List of False Negative Images')
print(fn_img_names)
