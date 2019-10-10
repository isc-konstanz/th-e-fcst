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
from scipy import signal

logger = logging.getLogger(__name__)


class Forecast:

    def __init__(self, configs):
        self.databases = self.__init_databases__(configs)
        self.neuralnetwork = self.__init_neuralnetwork__(configs)
        self.forecast = []
        self.forecast_unfiltered = []
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
        
    def execute(self):
        logger.info("Starting th-e-forecast")
        data = self.databases['CSV'].data
#         retrain model        
#         if k % 200 == 180:
#             X_train, Y_train = self.neuralnetwork.getInputVector(data, training=True)
#             self.neuralnetwork.train(X_train, Y_train[:, 0, :], epochs = 1)
#             self.neuralnetwork.model.save(logging + '\myModel')
        
        # prediction     
        data_input_pred = [data[0][-1440 * 4:],
                           data[1][-1440 * 4:]]
        y = self.neuralnetwork.predict_recursive(data_input_pred)
        self.forecast_unfiltered = y
        b, a = signal.butter(8, 0.022)
        self.forecast = signal.filtfilt(b, a, y, method='pad', padtype='even', padlen=150)
        

class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

