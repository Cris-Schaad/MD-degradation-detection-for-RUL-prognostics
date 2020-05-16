from hyperopt import hp, tpe, fmin, Trials

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers


class CMAPSSBayesianOptimizer():
    
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        
    def LSTM_model(self, space):
        print(space)

        model = models.Sequential()           
        for layer in range(space['lstm_layers']):
            returns_seq = True if layer < space['lstm_layers']-1 else False
            model.add(layers.LSTM(units = int(space['lstm_units']), activation = space['lstm_activation'], 
                                  return_sequences=returns_seq, recurrent_dropout=space['lstm_dropout']))
        model.add(layers.Flatten())   
        for layer in range(space['hidden_layers']):
            model.add(layers.Dense(units = space['layers_neurons'], activation = space['layers_activation']))
        model.add(layers.Dense(units = 1, activation = 'linear'))
    
    
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=space['EarlyS_patience'], 
                                            verbose=0, restore_best_weights=False, mode='min')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=space['LRswitch_patience'],
                                                verbose=0, mode='min')
        

        model.fit(self.x_train, self.y_train, batch_size = 256, epochs = 250, validation_data=(self.x_valid, self.y_valid), 
                  verbose=0, callbacks=[earlystop,reduce_lr])
        
        metrics = model.evaluate(self.x_test, self.y_test, return_dict = True)
        return metrics['loss']
    
    def optimize(self, iters=100):
        search_space = {'lstm_layers': hp.choice('lstm_layers', [1,2,3]),
                   'lstm_units': hp.quniform('lstm_units', 8, 256, 1), 
                   'lstm_activation': hp.choice('lstm_activation', ['tanh', 'relu']),
                   'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.75),
                   'hidden_layers': hp.choice('hidden_layers', [1,2,3]), 
                   'layers_neurons': hp.quniform('layers_neurons', 8, 256, 1), 
                   'layers_activation': hp.choice('layers_activation', ['tanh', 'relu']),
                   'LRswitch_patience': hp.quniform('LRswitch_patience', 10, 50, 1), 
                   'EarlyS_patience': hp.quniform('EarlyS_patience', 10, 50, 1)}

        trials = Trials()
        best = fmin(fn=self.LSTM_model,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=iters,
                    trials = trials)
        
        return best, trials