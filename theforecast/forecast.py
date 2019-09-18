# -*- coding: utf-8 -*-
"""
    theforecast.forecast
    ~~~~~
    
    
"""
from collections import OrderedDict
from configparser import ConfigParser
import logging
import os
from theforecast import neuralnetwork
from theforecast.neuralnetwork import NeuralNetwork
from .database import CsvDatabase
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import time

logger = logging.getLogger(__name__)


class Forecast:

    def __init__(self, configs):
        self.databases = self.__init_databases__(configs)
        self.neuralnetwork = self.__init_neuralnetwork__(configs)
        
        settingsfile = os.path.join(configs, 'settings.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        self.interval = settings.getint('General', 'interval') * 60

    def __init_databases__(self, configs):
        # Read the systems database settings
        settingsfile = os.path.join(configs, 'database.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        timezone = settings.get('General', 'timezone')
        
        enabled = OrderedDict()
        for database in settings.get('General', 'enabled').split(','):
            if database.lower() == 'csv':
                enabled[database] = CsvDatabase(configs, timezone)
        
        return enabled

    def __init_neuralnetwork__(self, configs):
        return NeuralNetwork(configs)
        
    def execute(self, pred_start, k, f_retrain, status_predict):
        logger.info("Starting th-e-forecast")
        # get new data from CSV-file
        # data = self.databases['CSV'].read_file('C:\\Users\\sf\\Software\\eclipse\\PyWorkspace\\th-e-forecast\\bin\\lib\\BI_jul_aug.csv')
        X_pred = []
        y = []
        theNN = self.neuralnetwork
        data = self.databases['CSV'].data
        # TODO: norm data here!?
        data = [data[0][:pred_start + k],
                data[1][:pred_start + k]]
        n_training_days = 30
        
        # retrain model        
        if k % f_retrain == 190:
            data_input_retrain = [data[0][-1440 * n_training_days:],
                                  data[1][-1440 * n_training_days:]]
            X_train, Y_train = theNN.getInputVector(data_input_retrain,
                                                    theNN.lookBack,
                                                    theNN.lookAhead,
                                                    theNN.fMin,
                                                    training=True)
            theNN.model.fit(X_train, Y_train[:, 0, :], epochs=1, batch_size=64, verbose=2)
            theNN.model.save('myModel2')
        
        # prediction 
        if status_predict == True:    
            data_input_pred = [data[0][-1440 * 7:],
                               data[1][-1440 * 7:]]
            X_pred = theNN.getInputVector(data_input_pred,
                                          theNN.lookBack,
                                          theNN.lookAhead,
                                          theNN.fMin,
                                          training=False)
            y = theNN.predict_recursive(data_input_pred)
   
        return X_pred, y

    def persist(self, result):
        for database in reversed(self.databases.values()):
            database.persist(result)
    
        
class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

