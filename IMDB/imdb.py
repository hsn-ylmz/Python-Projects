

import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

batch_size = 32
    
    
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words = None,
                                                      skip_top = 0,
                                                      maxlen = None,
                                                      seed = 113,
                                                      start_char = 1,
                                                      oov_char = 2,
                                                      index_from = 3)

x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)



classifier = Sequential()    

classifier.add(Embedding(25000, 128))  #Embedding step -> Max feature node to 128 node 

classifier.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))   # Long Short-Term Memory Approach with 128 output node  

classifier.add(Dense(1, activation='sigmoid'))

# Compiling the Artificial Neural Network
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics=['accuracy'])

 # Fitting the Artificial Neural Network to the trainig set

classifier.fit(x_train, y_train, 
                   batch_size = batch_size, 
                   epochs = 100, 
                   validation_data=(x_test, y_test))

score, acc = classifier.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Score', score)
print('Accuracy', acc)