# -*- coding: utf-8 -*-
"""
    theforecast.forecast
    ~~~~~
    
    
"""
from collections import OrderedDict
from configparser import ConfigParser
import logging
import os
from theforecast.neuralnetwork import NeuralNetwork
from .database import CsvDatabase
from scipy import signal
import pandas
import datetime

logger = logging.getLogger(__name__)


class Forecast:

    def __init__(self, configs):
        settingsfile = os.path.join(configs, 'settings.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        self.interval = settings.getint('General', 'interval')
        self.start = settings.get('General', 'start')
        
        self.databases = self.__init_databases__(configs)
        self.neuralnetwork = self.__init_neuralnetwork__(configs)
        self.forecast = []

    def __init_databases__(self, configs):
        settingsfile = os.path.join(configs, 'database.cfg')
        settings = ConfigParser()
        settings.read(settingsfile)
        
        timezone = settings.get('General', 'timezone')
        
        enabled = OrderedDict()
        for database in settings.get('General', 'enabled').split(','):
            if database.lower() == 'csv':
                enabled[database] = CsvDatabase(configs, self.start, timezone)
        
        return enabled

    def __init_neuralnetwork__(self, configs):
        return NeuralNetwork(configs)
        
    def execute(self):
        try:
            logger.info("Starting th-e-forecast")
            y = self.neuralnetwork.predict_recursive(self.databases['CSV'].data)
            self.forecast_unfiltered = y
            b, a = signal.butter(8, 0.022)
            self.forecast = signal.filtfilt(b, a, y, method='pad', padtype='even', padlen=150)
        except:
            raise ForecastException()
        
    def get_kpi_mpc(self, mpc, req_start, req_end):
        ''' This function calculates the KPI of the MPC 
        '''
        data = self.databases['CSV'].data
        data = data[req_start <= data.index]

        n = int((req_end - req_start).seconds / 60)
        control = mpc.history[-n:]
        
        try:
            mpc.execute(data['bi'].get_values())
            erg = mpc.control
        except:
            logger.error('Fatal optimization error at calculating KPI')
    
    def get_kpi_forecast(self, time):    
        self.databases['CSV'].read_file(self.databases['CSV'].datafile, time + datetime.timedelta(0, 1440 * 60))
        data = self.databases['CSV'].data
        data = data[time:]
        forecast = self.forecast
        
        self.databases['CSV'].read_file(self.databases['CSV'].datafile, time)
        
        
class ForecastException(Exception):
    """
    Raise if any error occurred during the forecast.
    
    """
