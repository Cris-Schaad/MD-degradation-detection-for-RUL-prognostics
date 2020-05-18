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


class ConvLSTMModel():
    
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        tf.random.set_seed(1)
        
        
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    
    
    def ConvLSTM_build(self, params):
        print(params)
        clear_session()
        
        model = models.Sequential()           
        for layer in range(params['lstm_layers']):
            returns_seq = True if layer < params['lstm_layers']-1 else False
            model.add(layers.ConvLSTM2D(filters=int(params['lstm_filters']), 
                                        kernel_size=params['lstm_filter_shape']))
        model.add(layers.Flatten())   
        for layer in range(params['hidden_layers']):
            model.add(layers.Dense(units = params['layers_neurons'], activation = params['layers_activation']))
        model.add(layers.Dense(units = 1, activation = 'linear'))
    
    
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                            verbose=0, restore_best_weights=False, mode='min')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                                verbose=0, mode='min')
        return model, [earlystop, reduce_lr]
    
    
    def model_train(self, params):
        model, callbacks = self.ConvLSTM_build(params)
        model.fit(self.x_train, self.y_train, batch_size = 256, epochs = 250, 
                  validation_data=(self.x_valid, self.y_valid), 
                  verbose=0, callbacks=callbacks)  
        return model
    
    
    def optimizable_model_train(self, space):

        model = self.model_train(space)
        y_pred = model.predict(self.x_test)
        rmse = self.rmse(self.y_test, y_pred)
        print('RMSE: {:.2f}'.format(rmse))
        return rmse
    
    
    def optimize(self, iters=100):
        search_space = {'lstm_layers': hp.choice('lstm_layers', [1,2]),
                   'lstm_filters': hp.quniform('lstm_filters', 8, 256, 1), 
                   'lstm_filter_shape': hp.choice('lstm_filter_shape', [[2,2], [3,3], [4,4], [5,5]]), 
                   
                   'hidden_layers': hp.choice('hidden_layers', [1,2,3]), 
                   'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1), 
                   'layers_activation': hp.choice('layers_activation', ['tanh', 'relu'])}

        trials = Trials()
        best = fmin(fn=self.optimizable_model_train,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials = trials)
        
        return best, trials