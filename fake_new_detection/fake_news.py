
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read data
data = pd.read_csv('news.csv')

# Get the shape of data
data.shape
data.head()

# Get labels
labels = data.label
labels.head()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    data['text'], labels, test_size=0.2, random_state=7)

# Initialize vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train and test dataset
train = vectorizer.fit_transform(x_train)
test = vectorizer.transform(x_test)

# Initilize PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(train, y_train)

# Make prediction and calculate accuracy_score
y_pred = classifier.predict(test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Confusion matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
