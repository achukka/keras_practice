
# coding: utf-8

# In[1]:

# Importing third party libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# Importing keras modules
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD

from keras.utils import np_utils

from imutils import paths

import numpy as np
import argparse
import cv2
import os
import csv


# In[23]:

class Mnist:
    @staticmethod
    def build(width, height, depth, classes, 
              nb_filters=32, kernel_size=(3, 3), pool_size=(2,2), weightsPath=None):
        # Initialzing the model
        model = Sequential()
        
        ''' First set of CONV => RELU'''
        # Add Convolution Layers '20' filters and receptive filed of size (5 x 5)
        ''' Note - You need to provide the 'input_shape' only for the first layer in keras
           BORDER_MODE - 'valid' => No zero padding to the input,
                         'same'  => Padding such that input_dim=output_dim  '''
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], 
                                border_mode="valid", input_shape=(depth, height, width)))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Then Max Pooling with 'pool size (2 x 2)'
        model.add(MaxPooling2D(pool_size=pool_size))
        
        ''' Second set of CONV => RELU => pooling'''
        # Add Convolution Layers '20' filters and receptive filed of size (5 x 5)
        model.add(Convolution2D(50, kernel_size[0], kernel_size[1], border_mode="same"))
        # Add Recitified Linear Units
        model.add(Activation('relu'))        
        # Then Max Pooling with 'pool size (2 x 2)'
        model.add(MaxPooling2D(pool_size=pool_size))
        # We add a drop out layer here - with 0.25 dropout
        model.add(Dropout(0.25))
                
        ''' Fully Connected Layers, followed by 'RELU' layer '''
        # First flatten the input 
        model.add(Flatten())
        # Add FC (Dense) Layer with 'output_dim' - 500
        model.add(Dense(128))
        # Add Recitified Linear Units
        model.add(Activation('relu'))
        # Again add a drop out layer here - with 0.5 dropout
        model.add(Dropout(0.5))
        
        ''' Final Soft Max Layer '''
        # FC Layer with 'output_dim' - number_of_classes
        model.add(Dense(classes))
        # Add Final Soft Max Activation
        model.add(Activation("softmax"))
        
        # Load weights if given
        if weightsPath is not None:
            model.load_weights(weightsPath)
            
        # Return the constructed model
        return model


# In[10]:

# Initialize the optimizer and model
def train_model(train_x, train_y, valid_x, valid_y,
                weightsPath='../weights/lenet_weights.hdf5', opt='adadelta', epochs=20,
                mini_batch_size=128, loss="categorical_crossentropy"):
    # Create Model
    model = Mnist.build(width=28, height=28, depth=1, classes=10, 
                         weightsPath=weightsPath)
    # Configure the model for training
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(train_x, train_y, batch_size=mini_batch_size, nb_epoch=epochs, verbose=1)
    
    # show the accuracy on validation data
    print '[INFO] Evaluating Validation data..'
    loss, accuracy = model.evaluate(valid_x, valid_y, batch_size=mini_batch_size,
                                    verbose=1)
    print '[INFO] Accuracy:{:.2f}%'.format(accuracy*100)
    return model


# In[11]:

from sklearn import datasets
# Fetch the data from sklearn datasets ( You can also use keras datasets for the same)
dataset = datasets.fetch_mldata("MNIST Original")


# In[12]:

# Reshape the data from 784 dim-vector to 28 x 28 pixel images
data = dataset.data.reshape((dataset.data.shape[0], 28 ,28))


# In[13]:

# Expand dimensions using np newaxis
data = data[:, np.newaxis, :, :]
# Scale the data into range [0,1]
data = data/255.0

# Split data into train and test
train_x, valid_x, train_y, valid_y = train_test_split(data, dataset.target.astype("int"),
                                                      test_size=0.3)


# In[14]:

# We should transform the labels in range[0, classes] into vectors of size(classes)
# Where the index of label is set to '1' and all other entries are '0's.
train_y = np_utils.to_categorical(train_y, 10)
valid_y = np_utils.to_categorical(valid_y, 10)


# In[24]:


model = train_model(train_x, train_y, valid_x, valid_y, weightsPath=None, opt='adadelta')


# In[25]:

model_rmsprop = train_model(train_x, train_y, valid_x, valid_y, weightsPath=None, opt='rmsprop')


# In[18]:

import csv
def load_test_data(filename='../data/test.csv'):
    fp = open(filename, 'rb')
    data = csv.reader(fp, delimiter=',')
    data_x = []
    data_y = []
    index = 0
    next(data)
    for row in data:
        cols = []
        for col in row:
            cols.append(float(col)/255.0)
        data_x.append(cols)
    fp.close()
    return np.asarray(data_x)

def write_output(filename, test_labels):
    fp = open(filename, 'wb')
    fp.write('ImageId,Label\n')
    imageId = 1
    for label in test_labels:
        fp.write('{0},{1}\n'.format(imageId,label))
        imageId += 1
    fp.close()


# In[19]:

test_x = load_test_data()
test_x = test_x.reshape((test_x.shape[0], 28 ,28))
test_x = test_x[:, np.newaxis, :, :]
test_predictions = model.predict_classes(test_x, verbose=1)


# In[20]:

write_output('../output/addelta_mnist.csv', test_predictions)


# In[21]:

test_predictions_rmsprop = model_rmsprop.predict_classes(test_x, verbose=1)


# In[22]:

write_output('../output/rmsprop_mnist.csv', test_predictions)


# In[ ]:



