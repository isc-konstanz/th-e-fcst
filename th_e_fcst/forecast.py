# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast
    ~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import datetime as dt

from configparser import ConfigParser
from th_e_core import Forecast as ForecastCore
from th_e_fcst import NeuralNetwork, NeuralForecast, NeuralPrediction


class Forecast(ForecastCore):

    @staticmethod
    def from_configs(context, configs, **kwargs):
        return Forecast(configs, context, **kwargs)

    def __init__(self, configs, context, **kwargs):
        if not isinstance(configs, ConfigParser):
            raise ValueError('Invalid configuration type: {}'.format(type(configs)))
        
        self._context = context
        self._activate(context, configs, **kwargs)

    def _activate(self, context, configs, **kwargs): #@UnusedVariable
        type = configs.get('General', 'type', fallback='none').lower() #@ReservedAssignment
        if type == 'none':
            self._weather = None
            self._model = NeuralPrediction.from_forecast(self, configs, **kwargs)
        else:
            config_weather = ConfigParser()
            config_weather.read_dict(configs)
            for section in NeuralNetwork.SECTIONS + ['Import']:
                config_weather.remove_section(section)
            
            self._weather = ForecastCore.from_configs(context, config_weather, **kwargs)
            self._model = NeuralForecast.from_forecast(context, configs, **kwargs)
            
        self._context = context

    def _get(self, *args, **kwargs):
        data = self._get_data(*args, **kwargs)
        return self._get_range(self._model.run(data, **kwargs), 
                               kwargs.get('start', None), 
                               kwargs.get('end', None))

    def _get_data(self, start, end=None, **kwargs):
        prior_start = start - self._model.time_prior
        prior_end = start - dt.timedelta(minutes=self._model.resolution)
        
        data = self._get_history(prior_start, prior_end, **kwargs)
        
        if self._weather is not None:
            weather = self._weather.get(start, end, **kwargs)
            data = pd.concat([data, weather], axis=0)
        
        return data

    def _get_history(self, start, end, **kwargs):
        data = self._system._database.get(start, end, **kwargs)
        data = data[data.columns.drop(list(data.filter(regex='_energy')))]
        
        if self._weather is not None:
            weather = self._weather._database.get(start, end, **kwargs)
            data = pd.concat([data, weather], axis=1)
        
        # TODO: Pass resample type to database
        resolution = self._model.resolution*60
        if resolution > 1:
            offset = (start - start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() % resolution
            data = data.resample(str(int(resolution))+'s', offset=str(int(offset))+'s').mean()
        
        return data

