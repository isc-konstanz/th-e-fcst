# -*- coding: utf-8 -*-
"""
    th-e-fcst.neural_forecast
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
import logging
import numpy as np
import datetime as dt

from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D

from th_e_fcst.model import NeuralNetwork

logger = logging.getLogger(__name__)


class NeuralForecast(NeuralNetwork):

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

    def _parse_inputs(self, features, time):
        inputs = self._extract_inputs(features, time)
        return np.squeeze(inputs.values)

    def _extract_inputs(self, features, time):
        start = time - self.time_prior
        end = time - dt.timedelta(minutes=self.resolution)
        
        data = features.loc[start:end, self.features['target'] + self.features['input']]
        if data.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % time)
        
        data.loc[time] = np.append([np.NaN]*len(self.features['target']), features.loc[end, self.features['input']].values)
        
        # TODO: Replace interpolation with prediction of ANN
        data.interpolate(method='linear', inplace=True)
        #data.interpolate(method='akima', inplace=True)
        #data.interpolate(method='nearest', fill_value='extrapolate', inplace=True)
        
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

