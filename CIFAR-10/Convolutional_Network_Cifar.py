

import pickle
import numpy
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm

"""
# To download the dataset from the keras database
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""

def train_data():
    img = []
    labels = []
    for i in range(1,6):
        i = str(i)
        file = open(('data_batch_'+ i), 'rb')
        tuple_data = pickle.load(file, encoding = 'bytes')
        array = numpy.asmatrix(tuple_data[b'data'])
        if len(img) == 0:
            img = array
        else:
            img = numpy.concatenate((img, array))
        label = numpy.asmatrix(tuple_data[b'labels'])
        label = numpy.transpose(label)
        if len(labels) == 0:
            labels = label
        else:
            labels = numpy.concatenate((labels, label))
        file.close()
    data = []
    for i in range(0,50000):
        single_img = numpy.array(img[i])
        data.append(numpy.transpose(numpy.reshape(single_img,(3, 32,32)), (1,2,0)))
    data = numpy.asarray(data)
    return (data, labels)

def test_data():
    file = open('test_batch', 'rb')
    tuple_data = pickle.load(file, encoding = 'bytes')
    img = numpy.asmatrix(tuple_data[b'data'])
    data = []
    for i in range(0,10000):
        single_img = numpy.array(img[i])
        data.append(numpy.transpose(numpy.reshape(single_img,(3, 32,32)), (1,2,0)))
    data = numpy.asarray(data)
    labels = numpy.asmatrix(tuple_data[b'labels'])
    labels = numpy.transpose(labels)
    return (data, labels)

def image_check(img):
    import random
    single_img = numpy.array(img[random.randint(0, img.shape[1])])
    single_img_reshaped = numpy.transpose(numpy.reshape(single_img,(3, 32,32)), (1,2,0))
    plt.imshow(single_img_reshaped)
    return

[X_train, y_train] = train_data()
[X_test, y_test] = test_data()

# Normalization
X_train = X_train / 255
X_test = X_test / 255

# Dealing with Categorical Data
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]


# Creating Model
def model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3),
                          activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation = 'relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu', kernel_constraint = maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

classifier = model()
print(classifier.summary())


classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

