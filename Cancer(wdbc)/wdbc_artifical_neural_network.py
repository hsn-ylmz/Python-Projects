# Artificial Neural Network Approach

import numpy as np
import pandas as pd

dataset = pd.read_csv('D:/Cancer/wdbc.csv', header = None)
X = dataset.iloc[:,2:].values
y = dataset.iloc[:, 1]

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the Artificial Neural Network
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(10, input_shape=(30,), kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the Artificial Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting the Artificial Neural Network to the trainig set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Accuracy of test set
score = classifier.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))

# Makig the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

weights = classifier.get_weights()

# Visualising Model
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='D:/Cancer/model_plot.png', show_shapes=True, show_layer_names=True)

# Visualising Network
from ann_visualizer.visualize import ann_viz
ann_viz(classifier, title="wdbc_cancer_ann_model")
