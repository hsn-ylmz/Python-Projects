from keras.models import Sequential
from keras.layers.core import Dense, Activation


class NetWork:
    @ staticmethod
    def build():
        model = Sequential()
        
        
        model.add(Dense(30, input_dim = 9))
        model.add(Activation("relu"))
        model.add(Dense(30))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        return model
    