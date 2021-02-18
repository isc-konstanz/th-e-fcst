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
from th_e_fcst import NeuralNetwork


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
        if configs.get('General', 'type', fallback='default').lower() == 'default':
            self._weather = None
        else:
            config_weather = ConfigParser()
            config_weather.read_dict(configs)
            for section in NeuralNetwork.SECTIONS + ['Import']:
                config_weather.remove_section(section)
            
            self._weather = ForecastCore.from_configs(context, config_weather, **kwargs)

        model = configs.get('NeuralNetwork', 'model', fallback='default').lower()
        if model in ['conv', 'default']:
            from th_e_fcst.neural_network import ConvLSTM
            self._model = ConvLSTM.from_forecast(context, configs, **kwargs)

        elif model is 'lstm':
            from th_e_fcst.neural_network import StackedLSTM
            self._model = StackedLSTM.from_forecast(context, configs, **kwargs)

        elif model is 'mlp':
            from th_e_fcst.neural_network import MultiLayerPerceptron
            self._model = MultiLayerPerceptron.from_forecast(context, configs, **kwargs)
        else:
            raise ValueError('Unknown ANN model : '+model)

        self._context = context

    def _get(self, *args, **kwargs):
        data = self._get_data(*args, **kwargs)
        return self._get_range(self._model.run(data, **kwargs), 
                               kwargs.get('start', None), 
                               kwargs.get('end', None))

    def _get_data(self, start, end=None, **kwargs):
        resolution = self._model._resolutions[0]
        prior_end = start - resolution.time_step
        prior_start = start - resolution.time_prior\
                            - resolution.time_step\
                            + dt.timedelta(seconds=1)
        
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
        
        return data


# class Basic(Model):
# 
#     def load(self, start, earliest, latest, interval):
#         gen_pv = self.load_generation(earliest, latest, interval)
#         cons_el = self.load_consumption_electrical(earliest, latest, interval)
#         cons_th = self.load_consumption_thermal(earliest, latest, interval)
#         data = pd.concat([gen_pv, cons_el, cons_th], axis=1)
#         
#         if len(data) > 1:
#             data = self._correction(earliest, start, interval, data)
#         
#         return data
# 
# 
#     def load_generation(self, start, end, interval):
#         keys = [FORECAST_GENERATION_PV]
#         
#         # Get historic data of the last day and extrapolate it for the coming days
#         end_day = start + dt.timedelta(days=1)
#         if end_day > end:
#             end_day = end
#         
#         data = self._weighted_mean(keys, start, end_day, interval, step=1)
#         
#         while data.index[-1] < end:
#             tmp = data.copy()
#             tmp.index = tmp.index + dt.timedelta(days=1)
#             data = pd.concat([data, tmp[data.index[-1]+dt.timedelta(seconds=interval):end]], axis=0)
#         
#         return data
# 
# 
#     def load_consumption_electrical(self, start, end, interval):
#         keys = [FORECAST_CONSUMPTION_ELECTRICAL]
#         data = self._weighted_mean(keys, start, end, interval, step=7)
#         
#         return data
# 
# 
#     def load_consumption_thermal(self, start, end, interval):
#         keys = [STORAGE_THERMAL_TEMPERATURE, HEAT_PUMP_THERMAL, COGENERATOR_THERMAL]
#         data = self._weighted_mean(keys, start, end, interval, step=7)
#         
#         # Calculate thermal energy consumption from the storages temperature, minus known generation
#         storage_th_delta = (data[STORAGE_THERMAL_TEMPERATURE] - data[STORAGE_THERMAL_TEMPERATURE].shift(1)).fillna(0)
#         storage_th_loss = data[STORAGE_THERMAL_TEMPERATURE] - data[STORAGE_THERMAL_TEMPERATURE]*self.components.storage_thermal.loss(interval)
#         data[FORECAST_CONSUMPTION_THERMAL] = - ((storage_th_delta - storage_th_loss)*self.components.storage_thermal.capacity() \
#                                              - data[HEAT_PUMP_THERMAL] - data[COGENERATOR_THERMAL])
#         
#         return data.loc[:,[FORECAST_CONSUMPTION_THERMAL]]
# 
# 
#     def _weighted_mean(self, keys, start, end, interval, step=1):
#         historic = pd.DataFrame(columns=keys)
#         h = 3
#         
#         # Create weighted consumption average of the three last timesteps
#         for i in range(h):
#             data = self.server.get(keys, start-dt.timedelta(days=1+i*step), end-dt.timedelta(days=1+i*step), interval)
#             data.index = data.index + dt.timedelta(days=1+i*step)
#             data = data[start:end]
#             
#             historic = historic.add(data.astype(float).multiply(h-i), fill_value=0)
#         
#         return historic/6
# 
# 
#     def _correction(self, start, end, interval, data):
#         keys = data.columns
#         
#         if end - start < dt.timedelta(seconds=2*interval):
#             end = end - dt.timedelta(seconds=2*interval)
#         
#         result = self.server.get(keys, start, end, interval)
#         data = result.combine_first(data)
#         last = result.iloc[-1,:]
#         
#         offset = data.index.get_loc(end) - 1
#         for i in range(len(keys)):
#             key = keys[i]
#             
#             # TODO: calculate factor through e-(j/x)
#             for j in range(1, 4):
#                 data.iloc[j+offset, i] = (data.iloc[j+offset, i]*j/4 + last[key]*(1-j/4))/2
#         
#         return data

