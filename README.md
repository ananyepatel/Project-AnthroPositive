# human-detection-model
Binary image classifier using deep convolutional network (Tensorflow)
Machine Learning model to identify a human in a given image. I've used Python 3.6 and Tensorflow 1.9.0 along with appropirate NumPy, SciPy and Scikit-learn modules to train the model. The network consists of 3 convolutional layers, a flatten layer and a fully connected layer to classify an input image into either of two categories: human or not-human.
The model assumes two directories in the root python directory: training_data and testing_data with labeled images.
