# -*- coding: utf-8 -*-
"""
    theforecast.forecast
    ~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import os

from configparser import ConfigParser
from collections import OrderedDict

from .database import CsvDatabase


class Forecast:

    def __init__(self, configs):
        self.databases = self.__init_databases__(configs)
        
        settingsfile = os.path.join(configs, 'settings.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        self.interval = settings.getint('General','interval')*60


    def __init_databases__(self, configs):
        # Read the systems database settings
        settingsfile = os.path.join(configs, 'database.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        timezone = settings.get('General','timezone')
        
        enabled = OrderedDict()
        for database in settings.get('General','enabled').split(','):
            if database.lower() == 'csv':
                enabled[database] = CsvDatabase(configs, timezone)
        
        return enabled


    def execute(self):
        #TODO: Do the forecast
        logger.info("Starting th-e-forecast")
        
        return


    def persist(self, result):
        for database in reversed(self.databases.values()):
            database.persist(result)


class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """

