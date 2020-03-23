import numpy as np
import keras_applications
import keras
from keras.datasets import mnist                 
from keras.models import Sequential              
from keras.layers.core import Dense, Activation  
from keras.optimizers import SGD                 
from keras.utils import np_utils  



class FashionMNISTModel:

    def __init__(self):

        self.model = Sequential()
        self.model.add(Dense(500,input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(300))
        self.model.add(Activation('relu'))
        self.model.add(Dense(4))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=["accuracy"])

    def preprocess_training_data(self, X, y):

        X = X.reshape(20000, 784)
        X = X.astype('float32')
        X /= 255
        y = np_utils.to_categorical(y, 4)

        return X, y

    def fit(self, X, y):


        self.model.fit(X, y,
          batch_size=100,
          epochs=10,
          verbose=2,
          validation_data = (X, y))

    def preprocess_unseen_data(self, X):

        
        X_test = X.reshape(10, 784)
        X_test = X.astype('float32')
        X_test /= 255

        return X_test


    def predict(self, X):


        return self.model.predict(X)

    
