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
from scipy import signal

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
        
    def execute(self):
        logger.info("Starting th-e-forecast")
        theNN = self.neuralnetwork
        data = self.databases['CSV'].data  # 1. get input data
        
        # TODO: hier muesste es sein: databases.get() neuste daten aus der DB abrufen
        b, a = signal.butter(8, 0.01)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
        data[0] = signal.filtfilt(b, a, data[0], padlen=150)
        
        # TODO: train the newest data
        X, Y = theNN.getInputVector(data, theNN.lookBack, theNN.lookAhead, theNN.fMin, training=True)
        theNN.model.fit(X, Y, epochs=2, batch_size=64, verbose=2)
        # TODO: create input vector and predict
        X, Y = theNN.getInputVector(data, theNN.lookBack, theNN.lookAhead, theNN.fMin, training=False)
        prediction = theNN.model.predict(X[0:1])
        
        # plot results:
        plt.plot(np.linspace(0, 7, 328), X[0].transpose())
        plt.plot(np.linspace(7, 8, 288), prediction[0, :, :].transpose())
        
        return X, prediction

    def persist(self, result):
        for database in reversed(self.databases.values()):
            database.persist(result)


class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

