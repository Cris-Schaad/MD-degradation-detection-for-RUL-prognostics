import os
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers, callbacks

from utilities import dataset_importer
from utilities import training_functions as functions


model_name = 'LSTM_RUL_test'
results_dir =  os.path.join(os.getcwd(), 'training_results', 'C-MAPSS', model_name)
dataset_dir = os.path.join(os.getcwd(), 'datasets', 'C-MAPSS', 'CMAPSS_dataset.npz')
dataset_npz = dict(np.load(dataset_dir, allow_pickle=True))['dataset'][()]
dataset_loader = dataset_importer.CMAPSS_health_importer(dataset_dir)
functions.plt_close()

scaler_x = functions.MinMaxScaler(feature_range=(0,1), feature_axis=2)
scaler_y = functions.MinMaxScaler(feature_range=(0,1))

for sub_dataset in ['FD001']: #dataset_npz.keys():
    print('\n'+sub_dataset)
    x_train, x_valid, y_train, y_valid =  dataset_loader.get_train_set(sub_dataset, valid_size=0.1)
    x_test, y_test = dataset_loader.get_test_set_for_metrics(sub_dataset, rul_end_index=0)

    x_train = scaler_x.fit_transform(x_train)
    x_valid = scaler_x.transform(x_valid)
    x_test = scaler_x.transform(x_test)  

    iters = 1
    save_path = os.path.join(results_dir, sub_dataset)
    functions.save_folder(save_path)          
    for i in range(iters):
        
        y_train = scaler_y.fit_transform(y_train)
        y_valid = scaler_y.transform(y_valid)
        y_test = scaler_y.transform(y_test)
        
        model = Sequential()
        model.add(LSTM(units = 16, activation = 'relu', return_sequences=False, recurrent_dropout=0.2))
        model.add(Dense(units = 64, activation = 'relu'))
        model.add(Dense(units = 1, activation = 'linear'))
        
        adam = optimizers.adam(lr=0.001)
        model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
        
        earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=0, restore_best_weights=False, mode='min')
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='min')
        model_history = model.fit(x_train, y_train, batch_size = 256, epochs = 250, validation_data=(x_valid, y_valid), verbose=0, callbacks=[earlystop,reduce_lr])
        

        y_pred = model.predict(x_test)
        y_train = scaler_y.inverse_transform(y_train)
        y_valid = scaler_y.inverse_transform(y_valid)
        y_test = scaler_y.inverse_transform(y_test)
        y_pred = scaler_y.inverse_transform(y_pred)

        
        plot_name = str('C-MAPSS RUL '+sub_dataset+' prediction iter '+str(i+1))
        functions.prediction_plots(y_test, y_pred, plot_name, save_dir=save_path, xlabel='Engne sample', ylabel='RUL')
        functions.print_rmse(y_test, y_pred, sub_dataset)
#        functions.loss_plot(model_history.history['loss'], model_history.history['val_loss'])
        
        y_pred_test = []
        x_test_complete, y_test_true = dataset_loader.get_test_samples(sub_dataset)    
        for sample in x_test_complete:
            x_sample = scaler_x.transform(sample)
            y_pred_test.append(scaler_y.inverse_transform(model.predict(x_sample)))

        functions.save_results(y_test_true, np.asarray(y_pred_test), save_path, name='iter_'+str(i+1))