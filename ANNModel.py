import numpy as np
from hyperopt import tpe, fmin, Trials

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
tf.autograph_verbosity = 0

from tensorflow.keras.backend import clear_session
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import save_model


class ANNModel():
    
    def __init__(self, x_train, y_train, x_valid, y_valid, model_type='CNN', x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = list(self.x_train.shape[1:])
        tf.random.set_seed(1)
        
        if model_type.lower() == 'cnn':
            self.build_model = self.CNN_build
        elif model_type.lower() == 'lstm':
            self.build_model = self.LSTM_build
        elif model_type.lower() == 'convlstm':
            self.build_model = self.ConvLSTM_build
        else:
            raise NameError(model_type + ' not found')
        
        
    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    
    
    def CNN_build(self, params):
        inputs = Input(self.input_shape)      
        x = inputs
        for layer in range(int(params['cnn_layers'])):
            x = layers.Conv2D(filters=int(params['cnn_filters']), 
                                    kernel_size=[int(params['cnn_kernel_height']),int( params['cnn_kernel_width'])], 
                                    strides=(1, 1), 
                                    padding=params['cnn_padding'],  
                                    activation=params['cnn_activation'])(x)
        x = layers.Flatten()(x)   
        x = layers.Dropout(params['dropout'])(x) 
        for layer in range(int(params['hidden_layers'])):
            x = layers.Dense(units = params['layers_neurons'], 
                             activation = params['layers_activation'])(x)
        y = layers.Dense(units = 1, activation = 'linear')(x)    
        return Model(inputs=inputs, outputs=y)
    
    
    def LSTM_build(self, params):
        inputs = Input(self.input_shape)      
        x = inputs      
        for layer in range(int(params['lstm_layers'])):
            returns_seq = True if layer < params['lstm_layers']-1 else False
            x = layers.LSTM(units = int(params['lstm_units']),
                            activation = params['lstm_activation'],
                            return_sequences=returns_seq)(x)
        x = layers.Flatten()(x)   
        x = layers.Dropout(params['dropout'])(x) 
        for layer in range(int(params['hidden_layers'])):
            x = layers.Dense(units = params['layers_neurons'], 
                             activation = params['layers_activation'])(x)
        y = layers.Dense(units = 1, activation = 'linear')(x)    
        return Model(inputs=inputs, outputs=y)
    
    
    def ConvLSTM_build(self, params):        
        inputs = Input(self.input_shape)      
        x = inputs              
        for layer in range(params['convlstm_layers']):
            returns_seq = True if layer < params['convlstm_layers']-1 else False
            x = layers.ConvLSTM2D(filters=int(params['convlstm_filters']), 
                                        kernel_size=[int(params['convlstm_kernel_height']),
                                                     int(params['convlstm_kernel_width'])],
                                        activation=params['convlstm_activation'],
                                        padding='same',
                                        return_sequences=returns_seq)(x)
        x = layers.Flatten()(x)   
        x = layers.Dropout(params['dropout'])(x) 
        for layer in range(int(params['hidden_layers'])):
            x = layers.Dense(units = params['layers_neurons'], 
                             activation = params['layers_activation'])(x)
        y = layers.Dense(units = 1, activation = 'linear')(x)    
        return Model(inputs=inputs, outputs=y)
    
    
    def model_train(self, params):
        print(params)
        clear_session()

        model = self.build_model(params)        
        adam = optimizers.Adam(lr=params['LR'])
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])    
        
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                                patience=params['LR_patience'],
                                                verbose=0, mode='min')
        earlystop = callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=params['ES_patience'], 
                                            verbose=0, restore_best_weights=False, mode='min')
        
        model.fit(self.x_train, self.y_train, batch_size = 256, epochs = 250, 
                  validation_data=(self.x_valid, self.y_valid), 
                  verbose=0, callbacks=[reduce_lr, earlystop])  
        return model
    
    
    def model_plot(self, model, save_path, model_name='model.png'):
        plot_model(model, to_file=os.path.join(save_path, model_name), 
                   expand_nested=True,
                   show_shapes=True,
                   show_layer_names=False)    
    
    
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
    
    
    def model_save(self, model, path):
        save_model(model, path, overwrite=True, include_optimizer=True, save_format='h5')
        
        