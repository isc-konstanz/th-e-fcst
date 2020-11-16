# -*- coding: utf-8 -*-
"""
    th_e_fcst.model
    ~~~~~~~~~~~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import os
import json
import numpy as np
import pandas as pd
import datetime as dt

from configparser import ConfigParser
from pandas.tseries.frequencies import to_offset
from pvlib.solarposition import get_solarposition
from keras.callbacks import History, EarlyStopping
from keras.models import model_from_json
from th_e_core import Model
from abc import abstractmethod

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        
        self.dir = os.path.join(configs['General']['lib_dir'], 'model')
        
        self.resolution = configs.getint('General', 'resolution')
        self.steps_prior = configs.getint('General', 'steps_prior')
        self.steps_horizon = configs.getint('General', 'steps_horizon', fallback=None)
        
        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')
        
        self.features = {}
        for (key, value) in configs.items('Features'):
            self.features[key] = json.loads(value)

    def _build(self, context, configs, **kwargs):
        super()._build(context, configs, **kwargs)
        
        self.history = History()
        self.callbacks = [self.history, EarlyStopping(patience=32, restore_best_weights=True)]
        
        #TODO: implement date based backups and naming scheme
        if self.exists():
            self.model = self._load_model()
        
        else:
            self.model = self._build_model(configs['Model'])
        
        self.model.compile(optimizer=configs.get('Model', 'optimizer'), 
                           loss=configs.get('Model', 'loss'), 
                           metrics=configs.get('Model', 'metrics', fallback=[]))

    @abstractmethod
    def _build_model(self, configs):
        pass

    def _load_model(self, inplace=False):
        logger.debug("Loading model from file")
        
        with open(os.path.join(self.dir, 'model.json'), 'r') as f:
            model = model_from_json(f.read())
            model.load_weights(os.path.join(self.dir, 'model.h5'))
            
            if inplace:
                self.model = model
            
            return model

    def _save_model(self):
        logger.debug("Saving model to file")
        
        # Serialize model to JSON
        with open(os.path.join(self.dir, 'model.json'), 'w') as f:
            f.write(self.model.to_json())
        
        # Serialize weights to HDF5
        self.model.save_weights(os.path.join(self.dir, 'model.h5'))

    def exists(self):
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            return False
        
        return os.path.isfile(os.path.join(self.dir, 'model.json')) and \
               os.path.isfile(os.path.join(self.dir, 'model.h5'))

    def run(self, date, data, *args):
        results = list()
        if len(args) > 0 and isinstance(args[0], pd.DataFrame):
            forecast = args.pop(0)
            data = pd.concat([data, forecast], axis=1)
        
        if self.steps_horizon is None:
            end = data.index[-1]
        else:
            end = date + self.time_horizon
        
        features = self._parse_features(data)
        time = date - self.time_prior
        while time < end:
            X = self._parse_inputs(features, time)
            y = self._run_step(X)
            
            # Add predicted output to features of next iteration
            features.loc[time, self.features['target']] = y
            results.append(y)
        
        return results

    def _run_step(self, X):
        if len(X.shape) < 3:
            X = X.reshape(1, X.shape[0], X.shape[1])
            
        result = float(self.model.predict(X, verbose=LOG_VERBOSE))
        if result < 1e-3:
            result = 0
        
        return result

    def train(self, data):
        features = self._parse_features(data)
        return self._train(features)

    def _train(self, features):
        X, y = self._parse_data(features)
        logger.debug("Built input of %s, %s", X.shape, y.shape)
        
        split = int(len(y) / 10.0)
        result = self.model.fit(X[split:], y[split:], batch_size=self.batch, epochs=self.epochs, callbacks=self.callbacks, 
                                validation_data=(X[:split], y[:split]), verbose=LOG_VERBOSE)
        
        self._save_model()
        return result

    def _parse_data(self, features, X=list(), y=list()):
        end = features.index[-1]
        time = features.index[0] + self.time_prior
        while time <= end:
            try:
                inputs = self._parse_inputs(features, time)
                target = self._parse_target(features, time)
                
                # If no exception was raised, add the validated data to the set
                X.append(inputs)
                y.append(target)
                
            except ValueError:
                logger.debug("Skipping %s", time)
                
            time += dt.timedelta(minutes=self.resolution)
        
        return np.array(X), np.array(y)

    @abstractmethod
    def _parse_inputs(self, features, time):
        pass

    @abstractmethod
    def _parse_target(self, features, time):
        pass

    @property
    def time_prior(self):
        return dt.timedelta(minutes=self.resolution*self.steps_prior)

    @property
    def time_horizon(self):
        if self.steps_horizon is None:
            return None
        
        return dt.timedelta(minutes=self.resolution*(self.steps_horizon-1))

    def _parse_features(self, data):
        columns = self.features['target'] + self.features['input']
        
        #TODO: use weather pressure for solar position
        solar = get_solarposition(pd.date_range(data.index[0], data.index[-1], freq='min'), 
                                  self._system.location.latitude, 
                                  self._system.location.longitude, 
                                  altitude=self._system.location.altitude)
        solar = solar.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
        solar.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']
        
        data = data[np.intersect1d(data.columns, columns)]
        data.loc[:, 'doy'] = data.index.dayofyear
        
        features = pd.concat([data, solar], axis=1).resample('{}min'.format(self.resolution), closed='right').mean()
        features.index += to_offset('{}min'.format(self.resolution))
        
        features = self._parse_horizon(features)
        features = self._parse_cyclic(features)
        
        return features

    def _parse_horizon(self, data):
        if self.steps_horizon is not None:
            data['horizon'] = (data.index - data.index[0]) / np.timedelta64(1, 'm') % (self.resolution*self.steps_horizon)
        else:
            data['horizon'] = self.resolution*range(data.index)
        
        return data

    def _parse_cyclic(self, data):
        for feature, bound in self.features['cyclic'].items():
            data[feature + '_sin'] = np.sin(2.0*np.pi*data[feature] / bound)
            data[feature + '_cos'] = np.cos(2.0*np.pi*data[feature] / bound)
        
        return data

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

