# -*- coding: utf-8 -*-
"""
    th_e_fcst.model
    ~~~~~~~~~~~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import os
import json
import datetime as dt

from configparser import ConfigParser
from th_e_core import Model


class NeuralNetwork(Model):

    SECTIONS = ['NeuralNetwork', 'Model', 'Features']

    @classmethod
    def from_forecast(cls, context, forecast_configs, **kwargs):
        configs = ConfigParser()
        configs.add_section('General')
        
        root_dir = forecast_configs.get('General', 'root_dir')
        configs.set('General', 'root_dir', root_dir)
        
        lib_dir = forecast_configs.get('General', 'lib_dir')
        configs.set('General', 'lib_dir', lib_dir if os.path.isabs(lib_dir) else os.path.join(root_dir, lib_dir))
        
        tmp_dir = forecast_configs.get('General', 'tmp_dir')
        configs.set('General', 'tmp_dir', tmp_dir if os.path.isabs(tmp_dir) else os.path.join(root_dir, tmp_dir))
        
        data_dir = forecast_configs.get('General', 'data_dir')
        configs.set('General', 'data_dir', data_dir if os.path.isabs(data_dir) else os.path.join(root_dir, data_dir))
        
        configs.set('General', 'config_dir', forecast_configs.get('General', 'config_dir'))
        configs.set('General', 'config_file', forecast_configs.get('General', 'config_file'))
        
        if forecast_configs.has_section('NeuralNetwork'):
            for key, value in forecast_configs.items('NeuralNetwork'):
                configs.set('General', key, value)
        
        for section in forecast_configs.sections():
            if section in NeuralNetwork.SECTIONS and not section == 'NeuralNetwork':
                if not configs.has_section(section):
                    configs.add_section(section)
                for key, value in forecast_configs.items(section):
                    configs.set(section, key, value)
        
        configs.add_section('Import')
        configs.set('Import', 'class', cls.__name__)
        configs.set('Import', 'module', '.'.join(cls.__module__.split('.')[:-1]))
        configs.set('Import', 'package', cls.__module__.split('.')[-1])
        
        return cls(configs, context, **kwargs)

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)
        
        self.resolution = configs.getint('General', 'resolution')
        self.steps_prior = configs.getint('General', 'steps_prior')
        self.steps_horizon = configs.getint('General', 'steps_horizon', fallback=None)
        
        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')
        
        self.features = {}
        for (key, value) in configs.items('Features'):
            self.features[key] = json.loads(value)

    @property
    def time_prior(self):
        return dt.timedelta(minutes=self.resolution*self.steps_prior)

    @property
    def time_horizon(self):
        if self.steps_horizon is None:
            return None
        
        return dt.timedelta(minutes=self.resolution*(self.steps_horizon-1))

# class BasicModel(Model):
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

