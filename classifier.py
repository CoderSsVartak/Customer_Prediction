from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

import numpy as np


class build_model:
    
    """
    train -> tuple of (x_train, y_train). Shape of x_train must be (train_eg, feature_count, 1)
    test -> tuple of (x_test, y_test). Shape of x_test must be (test_eg, feature_count, 1)
    units -> number of neurons in each LSTM layer
    layer_count -> total number of hidden layers required
    dropout_rate -> number of inactive neurons in each layer (0 <= dropout_rate < 1)
    optimizer -> optimizer to be used for training the LSTM network
    loss -> loss function to be used for training the network.
    metrics -> list of metrics to be used to validate the model
    epochs -> number of time network must be trained for the entire dataset
    op_activation -> activation function to be used in the output layer
    """
    
    def __init__(self, train, test, units, layer_count, dropout_rate, optimizer, loss, metrics, epochs,batch_size, op_activation):
        
        self.train = train
        self.test = test
        self.units = units
        self.layer_count = layer_count
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.op_activation = op_activation

    
    def generate_model(self):
        
        model = Sequential()
        
        #Add the 1st LSTM Layer
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(self.train[0].shape[1], 1)))
        #Add the dropout Layer for the 1st LSTM Layer
        model.add(Dropout(rate=self.dropout_rate))

        for layers in range(self.layer_count-2):

            #Add further LSTM layers
            model.add(LSTM(units=self.units))
            #Add the dropout Layer for the 1st LSTM Layer
            model.add(Dropout(rate=self.dropout_rate))

        #Output Layer
        model.add(Dense(units=1, activation=self.op_activation))
        return model
    
    
    def train_validate_model(self, model):
        
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)        
        history = model.fit(self.train[0], self.train[1], batch_size=self.batch_size, epochs=self.epochs, validation_data=self.test)
        
        return model, history


"""
#Driver Code
        
#Reshape X_train
x_train, x_test = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)), np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

units = 50
layers = 3
drop_rate = 0.2
optimizer ='rmsprop'
loss = 'binary_crossentropy'
metrics = ['acc']
epochs = 50
batch_size = 50
op_activation = 'sigmoid'

classifier = build_model((x_train, y_train), (x_test, y_test), units, layers, drop_rate,
                         optimizer, loss, metrics, epochs, batch_size, op_activation')

model = classifier.generate_model()
model, history = classifier.train_validate_model(model)
"""

