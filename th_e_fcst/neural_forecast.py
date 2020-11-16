# -*- coding: utf-8 -*-
"""
    th-e-fcst.neural_forecast
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import os

import numpy as np
import pandas as pd
import datetime as dt

from pandas.tseries.frequencies import to_offset
from pvlib.solarposition import get_solarposition
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, LeakyReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import History, EarlyStopping

from th_e_fcst.model import NeuralNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOG_VERBOSE = 0


class NeuralForecast(NeuralNetwork):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)
        
        self.dir = os.path.join(configs['General']['lib_dir'], 'model')

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

    def _build_model(self, configs):
        model = Sequential()
        model.add(Conv1D(int(configs['filter_size']), 
                         int(configs['kernel_size']), 
                         input_shape=(self.steps_prior+1, len(self.features['target'] + self.features['input'])), 
                         activation=configs['activation'],  
                         dilation_rate=1, 
                         padding='causal', 
                         kernel_initializer='he_uniform'))
        
        for n in range(int(configs['layers_conv'])-1):
            model.add(Conv1D(int(configs['filter_size']), 
                             int(configs['kernel_size']), 
                             activation=configs['activation'], 
                             dilation_rate=2**(n+1), 
                             padding='causal', 
                             kernel_initializer='he_uniform'))
        
        model.add(MaxPooling1D(int(configs['pool_size'])))
        model.add(LSTM(int(configs['layers_hidden']), activation=configs['activation']))
        
        neurons = int(configs['neurons'])
        model.add(Dense(neurons, activation=configs['activation'], kernel_initializer='he_uniform'))
        model.add(Dense(neurons, activation=configs['activation'], kernel_initializer='he_uniform'))
        model.add(Dense(neurons, activation=configs['activation'], kernel_initializer='he_uniform'))
        model.add(Dense(len(self.features['target'])))
        model.add(LeakyReLU(alpha=0.001))
        
        return model

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
            os.mkdir(self.dir)
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

    def _parse_inputs(self, features, time):
        inputs = self._extract_inputs(features, time)
        
        # TODO: Replace interpolation with prediction of ANN
        inputs.interpolate(method='linear', inplace=True)
        #inputs.interpolate(method='akima', inplace=True)
        #inputs.interpolate(method='nearest', fill_value='extrapolate', inplace=True)
        
        return np.squeeze(inputs.values)

    def _extract_inputs(self, features, time):
        start = time - self.time_prior
        end = time - dt.timedelta(minutes=self.resolution)
        
        data = features.loc[start:end, self.features['target'] + self.features['input']]
        if data.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % time)
        
        data.loc[time] = np.append([np.NaN]*len(self.features['target']), features.loc[end, self.features['input']].values)
        return data

    def _parse_target(self, features, time):
        #return np.squeeze(self._extract_target(features, time).values)
        return float(self._extract_target(features, time))

    def _extract_target(self, features, time):
        #end = time + self.time_horizon
        #data = features.loc[time:end, self.features['target']]
        data = features.loc[time, self.features['target']]
        if data.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % time)
        
        return data

    def _parse_features(self, data):
        #TODO: use weather pressure for solar position
        solar = get_solarposition(pd.date_range(data.index[0], data.index[-1], freq='min'), 
                                  self._system.location.latitude, 
                                  self._system.location.longitude, 
                                  altitude=self._system.location.altitude)
        solar = solar.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']] \
                     .resample('{}min'.format(self.resolution), closed='right').mean()
        solar.index += to_offset('{}min'.format(self.resolution))
        solar.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']
        
        data['doy'] = data.index.dayofyear
        
        features = pd.concat([data, solar], axis=1)
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

