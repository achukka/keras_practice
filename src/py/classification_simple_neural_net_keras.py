
# coding: utf-8
# Importing third party libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# Importing keras modules
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD

from keras.layers import Dense
from keras.utils import np_utils

from imutils import paths

import numpy as np
import argparse
import cv2
import os

# Image to Feature  Vectors by takes as input 'image' & 'size'
def image_to_feature_vector(image, size=(32, 32)):
    '''
    Resizes the given image to a fixed size and then flattens it.
    This ensures that every image has the same feature vector size
    '''
    return cv2.resize(image, size).flatten()


# Now we construct an arguemnt parser to parse the given arguments
''' Un-comment the below code if you are not using notebook'''
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, 
#                 help="path to input dataset")
# args = vars(ap.parse_args())



def load_data(data_folder='data/dogs_vs_cats/train/'):

    print '[INFO] Gathering images..'
    imagePaths = list(paths.list_images(data_folder))

    # Init data matrix and list of labels
    data = []
    labels = []

        # Looping over all the images
    for (index, imPath) in enumerate(imagePaths):
        # Now we load the image and extract the class labels
        image = cv2.imread(imPath)
        label = imPath.split(os.path.sep)[-1].split(".")[0]

        # Feature vector (raw pixel intensities) for images
        features = image_to_feature_vector(image)
        data.append(features)
        labels.append(label)

        ## print update for every 5000 images
        if index % 5000 == 0:
            print '[INFO] Processed',index,'out of',len(imagePaths),'images'
    
    # Normalize data
    data = np.array(data) / 255.0
    return data, labels



def write_output(imageIds, predictions, filename='output/predictions.csv'):
    fp = open(filename, 'wb')
    fp.write('id,label\n')
    for (imageId,label) in zip(imageIds, predictions):
        fp.write(imageId +","+str(label)+'\n')
    fp.close()



data_folder = 'data/dogs_vs_cats/train/'
data, labels = load_data(data_folder=data_folder)

# Encoding the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# Transform labels to vectors in range[0, num_classes]
''' 'np_utils' - module from keras library, 'to-categorical' - function that converts
    class vectors (integers from 0 to 'num_classes') into binary matrix.
    Ex: bm[10, 'dog' ]= 1, if image 10 contains a dog. '''
num_classes = 2
labels = np_utils.to_categorical(labels, num_classes)



# Split the data into testing and training using 'sklearn.cross_validation' module
print '[INFO] Constructing Training and Validation Split'
train_x, valid_x, train_y, valid_y =  train_test_split(data, labels, test_size=0.3, random_state=21)


def init_model(weight_init='uniform'):
    # Define the simple neural network architecture
    model = Sequential()

    # Fully Connected layer
    model.add(Dense(768, input_dim=3072, init=weight_init, activation='relu'))

    # Second Fully Connected Layer
    model.add(Dense(384, init=weight_init, activation='relu'))

    # Third Connected layer with no activation
    model.add(Dense(2))

    # Final Soft Max Layer
    model.add(Activation("softmax"))
    return model


def train_model(model, train_x, train_y, epochs, mini_batch_size):
    # Training the model using SGD
    print '[INFO] training the model using SGD..'

    sgd = SGD(lr=0.01)
    # Configure the model for training
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

    # Training the model for 'epochs' and 'batch_size'
    model.fit(train_x, train_y, nb_epoch=epochs, batch_size=mini_batch_size)
    
    return model


epochs = 50
mini_batch_size = 128
model = init_model('lecun_uniform')
train_model(model, train_x, train_y, epochs, mini_batch_size)


# Evaluate the model on validation data
model.evaluate(valid_x, valid_y, batch_size=mini_batch_size)


epochs = 50
mini_batch_size = 128
model = init_model('glorot_normal')
train_model(model, train_x, train_y, epochs, mini_batch_size)


# Evaluate the model on validation data
print '[INFO] Evaluating on validation data..'
(valid_loss, valid_accuracy) = model.evaluate(valid_x, valid_y, 
                                               batch_size=mini_batch_size, verbose=1)
print '\n[INFO] loss:',valid_loss,'accuracy:',valid_accuracy*100.0/1.0,'%'



data_folder = 'data/dogs_vs_cats/test/'
test_data, imageIds = load_data(data_folder=data_folder)



test_predictions = model.predict(test_data, batch_size=mini_batch_size, verbose=1)



test_predictions_classes = model.predict_classes(test_data, batch_size=mini_batch_size, verbose=1)



write_output(imageIds, test_predictions_classes, filename='output/predictions_ffn.csv')

