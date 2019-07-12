# -*- coding: utf-8 -*-
"""
    theforecast.forecast
    ~~~~~
    
    
"""
from collections import OrderedDict
from configparser import ConfigParser
import logging
import os

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

    def execute(self, NeuralNetwork = None):
        logger.info("Starting th-e-forecast")
        
        if NeuralNetwork == None:
            # load trained neural network
            
        # TODO: Do the forecast
        
        # 1. get input data in reshaped form
        # 2. retrain model 
        # 3. predict recursive
        

        return

    def persist(self, result):
        for database in reversed(self.databases.values()):
            database.persist(result)


class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

