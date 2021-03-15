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
from keras.callbacks import History, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, LeakyReLU, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import model_from_json
from tensorflow.summary import create_file_writer, scalar
from th_e_core import Model
from abc import abstractmethod

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(Model):

    SECTIONS = ['Resolution', 'Model', 'Features']

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
            if section.startswith(tuple(NeuralNetwork.SECTIONS)):
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
        
        self.dir = os.path.join(configs['General']['data_dir'], 'model')
        
        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')
        
        self._resolutions = list()
        for resolution in [s for s in configs.sections() if s.lower().startswith('resolution')]:
            self._resolutions.append(Resolution(**dict(configs.items(resolution))))
        
        if len(self._resolutions) < 1:
            raise ValueError("Invalid control configurations without specified step resolutions")
        
        self.features = {}
        for (key, value) in configs.items('Features'):
            try:
                self.features[key] = json.loads(value)
                
            except json.decoder.JSONDecodeError:
                pass

    def _build(self, context, configs, **kwargs):
        super()._build(context, configs, **kwargs)
        
        self.history = History()
        self.callbacks = [self.history,
                          TensorBoard(log_dir=self.dir, histogram_freq=1),
                          EarlyStopping(patience=32, restore_best_weights=True)]
        
        #TODO: implement date based backups and naming scheme
        if self.exists():
            self.model = self._load()
        
        else:
            self.model = self.build(configs['Model'])
        
        self.model.compile(optimizer=configs.get('Model', 'optimizer'), 
                           loss=configs.get('Model', 'loss'), 
                           metrics=configs.get('Model', 'metrics', fallback=[]))

    @abstractmethod
    def build(self, configs):
        pass

    def _load(self, inplace=False):
        logger.debug("Loading model from file")
        
        with open(os.path.join(self.dir, 'model.json'), 'r') as f:
            model = model_from_json(f.read())
            model.load_weights(os.path.join(self.dir, 'model.h5'))
            
            if inplace:
                self.model = model
            
            return model

    def _save(self):
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
        #if result < 1e-3:
        #    result = 0
        
        return result

    def train(self, data):
        features = self._parse_features(data)
        self.data_distributions(features, train=True)
        return self._train(features)

    def _train(self, features):
        inputs, targets = self._parse_data(features)
        logger.debug("Built input of %s, %s", inputs.shape, targets.shape)
        
        split = int(len(targets) / 10.0)
        result = self.model.fit(inputs[split:], targets[split:], batch_size=self.batch, epochs=self.epochs,
                                validation_data=(inputs[:split], targets[:split]), callbacks=self.callbacks,
                                verbose=LOG_VERBOSE)
        
        # write normed loss to tensorboard
        train_summary_writer = create_file_writer(os.path.join(self.dir, 'custom_metric'))
        norm_loss = result.history['loss']/features['pv_power'].max()
        for epoch in range(len(result.history['loss'])):
            with train_summary_writer.as_default():
                scalar('norm_loss', norm_loss[epoch], step=epoch)
        
        self._save()
        return result

    def _shuffle_data(self, inputs, targets):
        import random

        data = []
        for i in range(len(inputs)):
            data.append(list(inputs[i]) + list([targets[i]]))

        random.shuffle(data)

        inputs = []
        targets = []
        for i in range(len(data)):
            inputs.append(data[i][:len(self.features['input'])])
            targets.append(data[i][len(self.features['input']):][0])

        return inputs, targets

    def _parse_data(self, features, inputs, targets, shuffle=True):
        end = features.index[-1]
        time = features.index[0] + self._resolutions[0].time_prior
        while time <= end:
            try:
                input = self._parse_inputs(features, time)
                target = self._parse_target(features, time)

                # If no exception was raised, add the validated data to the set
                inputs.append(input)
                targets.append(target)
                
            except ValueError:
                logger.debug("Skipping %s", time)
                
            time += dt.timedelta(minutes=self._resolutions[-1].minutes)

        inputs = np.array(inputs)
        targets = np.array(targets)

        if shuffle:
            return self._shuffle_data(inputs, targets)

        return inputs, targets

    def _parse_inputs(self, features, time):
        inputs = self._extract_inputs(features, time)
        return np.squeeze(inputs.values)

    def _extract_inputs(self, features, time):
        data = pd.DataFrame()
        data.index.name = 'time'
        for resolution in self._resolutions:
            resolution_end = time - resolution.time_step
            resolution_start = time - resolution.time_prior
            resolution_data = features.loc[(resolution_start - resolution.time_step + dt.timedelta(seconds=1)):resolution_end, 
                                           self.features['target'] + self.features['input']]
            
            data = resolution.resample(resolution_data).combine_first(data[:resolution_start])
        
        if data.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % time)
        
        return data

    def _parse_target(self, features, time):
        #return np.squeeze(self._extract_target(features, time).values)
        return float(self._extract_target(features, time))

    def _extract_target(self, features, time):
        # TODO: Implement horizon resolutions
        resolution = self._resolutions[-1]
        resolution_target = resolution.resample(features.loc[time - resolution.time_step + dt.timedelta(seconds=1): time, 
                                                             self.features['target']])
        
        data = resolution_target.loc[time,:]
        if data.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % time)
        
        return data

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
        data['day_of_year'] = data.index.dayofyear
        data['day_of_week'] = data.index.dayofweek
        
        features = pd.concat([data, solar], axis=1)
        features = self._parse_horizon(features)
        features = self._parse_cyclic(features)

        features = self.rescale(features, scale=True)
        return features

    def rescale(self, data, scale=True):
        assert isinstance(scale, bool)

        def gen_trafos(data, features):
            trafos = {}
            for item in features['scaling']:
                if features['trafo'].lower() == 'standard':
                    mean = data[item[0]].mean()
                    std = data[item[0]].std()

                    def trafo(x, mean=mean, std=std):  # force early binding
                        return (x - mean) / std

                    def inv_trafo(x, mean=mean, std=std):
                        return x * std + mean

                elif features['trafo'].lower() == 'norm':
                    feature_max = data[item[0]].max()

                    def trafo(x, max=feature_max):  # force early binding
                        return x / max

                    def inv_trafo(x, max=feature_max):
                        return x * max
                else:
                    raise ValueError('The transformation {}'.format(self.features['trafo'])
                                     + ' is not defined in the function gen_trafos.')

                trafo_tuple = (trafo, inv_trafo)
                trafos[trafo_tuple] = item
            return trafos

        if scale == True:
            self.trafos = gen_trafos(data, self.features)  # retrieve trafos for features
            for trafo_tuple, item in self.trafos.items():
                for column in item:
                    if column in data.columns:
                        data[column] = trafo_tuple[0](data[column])

        if scale == False:
            for trafo_tuple, item in self.trafos.items():
                for column in item:
                    if column in data.columns:
                        data[column] = trafo_tuple[1](data[column])
        return data

    def data_distributions(self, features, train=True): #ToDo change path to generalize to multiple scaling trafos
        assert isinstance(train, bool)
        import matplotlib.pyplot as plt
        bin_num = 100 #desired number of bins in each plot
        for feature in features.columns: #create 100 equal space bin vals per feat.
            bins = []
            domain = features[feature].max()-features[feature].min()
            bin_step = domain/bin_num
            counter = features[feature].min()
            for i in range(bin_num):
                bins.append(counter)
                counter = counter + bin_step
            bins.append(counter) #for the last value of counter

            plt_info = plt.hist(features[feature], bins=bins)
            bin_values, bins = plt_info[0], plt_info[1]
            count_range = max(bin_values)-min(bin_values)
            sorted_values = list(bin_values)
            sorted_values.sort(reverse=True)

            for i in range(len(sorted_values)-1): #scale plots by step through sorted bins

                if abs(sorted_values[i]-sorted_values[i+1])/count_range < 0.80:
                    continue
                else:
                    plt.ylim([0, sorted_values[i+1]+10])
                    break

            if train is True: #save histogram to appropriate folder.
                path = os.path.join(self.dir, '../distributions/train/{}.png'.format(feature))
                plt.savefig(path)
                plt.clf()

            elif train is False:
                path = os.path.join(self.dir, '../distributions/test/{}.png'.format(feature))
                plt.savefig(path)
                plt.clf()


    def _parse_horizon(self, data):
        resolution = self._resolutions[0]
        if resolution.steps_horizon is not None:
            data['horizon'] = (data.index - data.index[0]) / np.timedelta64(1, 'm') % (resolution.minutes*resolution.steps_horizon)
        else:
            data['horizon'] = self.resolution*range(data.index)
        
        return data

    def _parse_cyclic(self, data):
        for feature, bound in self.features['cyclic'].items():
            data[feature + '_sin'] = np.sin(2.0*np.pi*data[feature] / bound)
            data[feature + '_cos'] = np.cos(2.0*np.pi*data[feature] / bound)
            data = data.drop(columns=[feature])
        return data


class ConvLSTM(NeuralNetwork):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)
        self._estimate = kwargs.get('estimate') if 'estimate' in kwargs else \
                         configs.get('Features', 'estimate', fallback='true').lower() == 'true'

    def build(self, configs):
        if not self._estimate:
            steps = 0
        else:
            steps = 1
        
        for resolution in self._resolutions:
            steps += resolution.steps_prior
        
        model = Sequential()
        model.add(Conv1D(int(configs['filters']),
                         int(configs['kernel_size']), 
                         input_shape=(steps, len(self.features['target'] + self.features['input'])), 
                         activation=configs['conv_activation'],
                         kernel_initializer=configs['conv_kernel'],
                         dilation_rate=int(configs['dilation']),
                         padding='causal'))
        
        for n in range(int(configs['layers_conv'])-1):
            model.add(Conv1D(int(configs['filters']),
                             int(configs['kernel_size']), 
                             activation=configs['conv_activation'],
                             kernel_initializer=configs['conv_kernel'],
                             dilation_rate=2**(n+1), # ToDo consider how to handle dilation in .cfgs
                             padding='causal')) # ToDo handle padding in configs
        
        model.add(MaxPooling1D(int(configs['pool_size'])))

        lstm_units = json.loads(configs['lstm_units'])
        while lstm_units:
            model.add(LSTM(lstm_units.pop(0),
                           activation=configs['lstm_activation']))

        dense_units = json.loads(configs['dense_units'])
        while dense_units:
            model.add(Dense(dense_units.pop(0),
                            activation=configs['dense_activation'],
                            kernel_initializer=configs['dense_kernel']))

        model.add(Dense(len(self.features['target'])))

        if configs['dense_activation']=='relu':
            model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))
        
        return model

    def _extract_inputs(self, features, time):
        inputs = super()._extract_inputs(features, time)
        
        if self._estimate:
            resolution = self._resolutions[-1]
            resolution_inputs = resolution.resample(features.loc[(time - resolution.time_step + dt.timedelta(seconds=1)):time, 
                                                                 self.features['input']])
            
            inputs.loc[time] = np.append([np.NaN]*len(self.features['target']), resolution_inputs.values)
            
            # TODO: Replace interpolation with prediction of ANN
            inputs.interpolate(method='linear', inplace=True)
            #inputs.interpolate(method='akima', inplace=True)
            #inputs.interpolate(method='nearest', fill_value='extrapolate', inplace=True)
        
        return inputs

class MultiLayerPerceptron(NeuralNetwork):

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)
        self._estimate = kwargs.get('estimate') if 'estimate' in kwargs else \
                         configs.get('Features', 'estimate', fallback='true').lower() == 'true'

    def _extract_inputs(self, features, time):
        inputs = super()._extract_inputs(features, time)

        if self._estimate:
            resolution = self._resolutions[-1]
            resolution_inputs = resolution.resample(
                features.loc[(time - resolution.time_step + dt.timedelta(seconds=1)):time,
                self.features['input']])

            inputs.loc[time] = np.append([np.NaN] * len(self.features['target']), resolution_inputs.values)

            # TODO: Replace interpolation with prediction of ANN
            inputs.interpolate(method='linear', inplace=True)
            # inputs.interpolate(method='akima', inplace=True)
            # inputs.interpolate(method='nearest', fill_value='extrapolate', inplace=True)

        return inputs

    def build(self, configs):
        if not self._estimate:
            steps = 0
        else:
            steps = 1

        for resolution in self._resolutions:
            steps += resolution.steps_prior

        model = Sequential()
        num_channels = len(self.features['target'] + self.features['input'])
        units = steps*num_channels
        model.add(Flatten(input_shape=(steps, num_channels)))
        model.add(Dense(units, activation=configs['dense_activation'],
                  kernel_initializer=configs['dense_kernel']))

        dense_units = json.loads(configs['dense_units'])
        while dense_units:
            model.add(Dense(dense_units.pop(0),
                            activation=configs['dense_activation'],
                            kernel_initializer=configs['dense_kernel']))

        model.add(Dense(len(self.features['target']),
                        activation=configs['dense_activation'],
                        kernel_initializer=configs['dense_kernel']))

        if configs['dense_activation'] == 'relu':
            model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))

        return model


class StackedLSTM(NeuralNetwork):

    def build(self, configs):
        steps = 0
        for resolution in self._resolutions:
            steps += resolution.steps_prior
        
        model = Sequential()

        model.add(LSTM(32,
                       input_shape=(steps, len(self.features['target'] + self.features['input'])),
                       activation=configs['activation'],
                       kernel_initializer=configs['lstm_kernel'],
                       return_sequences=True))

        lstm_units = json.loads(configs['lstm_units'])
        while lstm_units:
            model.add(LSTM(lstm_units.pop(0),
                           activation=configs['lstm_activation'],
                           kernel_initializer=configs['lstm_kernel'],
                           return_sequences=True))

        model.add(LSTM(32,
                       input_shape=(steps, len(self.features['target'] + self.features['input'])),
                       activation=configs['lstm_activation'],
                       kernel_initializer=configs['lstm_kernel']))
        
        neurons = json.loads(configs['dense_units'])
        while neurons:
            model.add(Dense(neurons.pop(0),
                            activation=configs['dense_activation'],
                            kernel_initializer=configs['dense_kernel']))

        model.add(Dense(len(self.features['target'])))

        if configs['dense_activation'] == 'relu':
            model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))
        
        return model


class Resolution:

    def __init__(self, minutes, steps_prior=None, steps_horizon=None):
        self.minutes = int(minutes)
        self.steps_prior = int(steps_prior) if steps_prior else None
        self.steps_horizon = int(steps_horizon) if steps_horizon else None

    @property
    def time_step(self):
        return dt.timedelta(minutes=self.minutes)

    @property
    def time_prior(self):
        if self.steps_prior is None:
            return None
        
        return dt.timedelta(minutes=self.minutes*self.steps_prior)

    @property
    def time_horizon(self):
        if self.steps_horizon is None:
            return None
        
        return dt.timedelta(minutes=self.minutes*(self.steps_horizon-1))

    def resample(self, features):
        data = features.resample('{}min'.format(self.minutes), closed='right').mean()
        data.index += to_offset('{}min'.format(self.minutes))
        
        return data

