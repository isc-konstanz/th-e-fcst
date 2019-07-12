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

from .database import CsvDatabase

logger = logging.getLogger(__name__)


class Forecast:

    def __init__(self, configs):
        self.databases = self.__init_databases__(configs)
        
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

    def execute(self, NeuralNetwork=None, configs):
        logger.info("Starting th-e-forecast")
        
        # load trained neural network
        if NeuralNetwork == None:
            NeuralNetwork = neuralnetwork(configs)
            myModel = NeuralNetwork.get()
            
        # 1. get input data in reshaped form
        inputData = []
        # 2. retrain model 
        NeuralNetwork.train(myModel, inputData)
        # 3. predict recursive
        PredictionResult = NeuralNetwork.predict(myModel, inputData)

        return PredictionResult

    def persist(self, result):
        for database in reversed(self.databases.values()):
            database.persist(result)


class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

