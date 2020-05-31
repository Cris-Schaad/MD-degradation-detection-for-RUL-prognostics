import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import sys
sys.path.append("..")
from utils.dataset_importer import CMAPSS_importer
from utils.training_functions import MinMaxScaler



dataset_name = 'CMAPSS'
model = 'CNN'


dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', dataset_name, 'processed_data')
models_dir = os.path.join(os.getcwd(), dataset_name, model)

dataset_loader = CMAPSS_importer(dataset_name)



for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
    saved_models_list = [i for i in os.listdir(os.path.join(models_dir, dataset)) if '.' not in i]
    loaded_model = load_model(os.path.join(models_dir, dataset, saved_models_list[0]))
    print(loaded_model)