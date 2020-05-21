import numpy as np
from hyperopt import hp, tpe, fmin, Trials

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
tf.autograph_verbosity = 0

from tensorflow.keras.backend import clear_session
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


class OptimizableModel():
    
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test, model_type):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        tf.random.set_seed(1)
        
        if model_type.lower() == 'cnn':
            self.build_model = self.CNN_build
        if model_type.lower() == 'lstm':
            self.build_model = self.LSTM_build
        if model_type.lower() == 'convlstm':
            self.build_model = self.ConvLSTM_build
        else:
            raise NameError(model_type + 'not found')
        
        
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    
    
    def CNN_build(self, params):
        model = models.Sequential()           
        for layer in range(params['cnn_layers']):
            model.add(layers.Conv2D(filters=params['cnn_filters'], 
                                    kernel_size=[params['cnn_kernel_height'], params['cnn_kernel_width']], 
                                    strides=(1, 1), 
                                    padding=params['cnn_padding'],  
                                    activation=params['cnn_activation']))
        model.add(layers.Flatten())   
        for layer in range(params['hidden_layers']):
            model.add(layers.Dense(units = params['layers_neurons'], activation = params['layers_activation']))
        model.add(layers.Dense(units = 1, activation = 'linear'))
        return model
    
    
    def LSTM_build(self, params):
        model = models.Sequential()           
        for layer in range(params['lstm_layers']):
            returns_seq = True if layer < params['lstm_layers']-1 else False
            model.add(layers.LSTM(units = int(params['lstm_units']), activation = params['lstm_activation'], 
                                  return_sequences=returns_seq, recurrent_dropout=params['lstm_dropout']))
        model.add(layers.Flatten())   
        for layer in range(params['hidden_layers']):
            model.add(layers.Dense(units = params['layers_neurons'], activation = params['layers_activation']))
        model.add(layers.Dense(units = 1, activation = 'linear'))
        return model
    
    
    def ConvLSTM_build(self, params):        
        model = models.Sequential()           
        for layer in range(params['lstm_layers']):
            returns_seq = True if layer < params['lstm_layers']-1 else False
            model.add(layers.ConvLSTM2D(filters=int(params['lstm_filters']), 
                                        kernel_size=params['lstm_filter_shape'],
                                        activation=params['lstm_activation'],
                                        padding='same',
                                        return_sequences=returns_seq))
        model.add(layers.Flatten())   
        for layer in range(params['hidden_layers']):
            model.add(layers.Dense(units = params['layers_neurons'], activation = params['layers_activation']))
        model.add(layers.Dense(units = 1, activation = 'linear'))
        return model
    
    
    def model_train(self, params):
        print(params)
        clear_session()

        model = self.build_model(params)
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                                verbose=0, mode='min')
        earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                            verbose=0, restore_best_weights=False, mode='min')
        
        model.fit(self.x_train, self.y_train, batch_size = 256, epochs = 250, 
                  validation_data=(self.x_valid, self.y_valid), 
                  verbose=0, callbacks=[reduce_lr, earlystop])  
        return model
    
    
    def optimizable_model_train(self, space):

        model = self.model_train(space)
        y_pred = model.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_pred)
        print('RMSE: {:.2f}'.format(rmse))
        return rmse
    
    
    def optimize(self, search_space, iters=100):

        trials = Trials()
        best = fmin(fn=self.optimizable_model_train,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials = trials)
        
        return best, trials