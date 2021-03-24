# -*- coding: utf-8 -*-
"""
    th_e_fcst.model
    ~~~~~~~~~~~~~~~
    
    
"""
import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import logging

from pandas.tseries.frequencies import to_offset
from pvlib.solarposition import get_solarposition
from keras.callbacks import History, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.models import model_from_json
from tensorflow import summary
from th_e_core import Model

logger = logging.getLogger(__name__)

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(Model):

    @staticmethod
    def from_configs(context, configs, **kwargs):
        model = configs.get('General', 'model', fallback='default').lower()

        if model in ['mlp', 'ann', 'dense', 'default']:
            return NeuralNetwork(configs, context, **kwargs)

        elif model in ['convdilated', 'conv', 'cnn']:
            return ConvDilated(configs, context, **kwargs)

        elif model == 'convlstm':
            return ConvLSTM(configs, context, **kwargs)

        elif model == 'lstm':
            return StackedLSTM(configs, context, **kwargs)

        else:
            return Model.from_configs(context, configs, **kwargs)

    def _configure(self, configs, **kwargs):
        super()._configure(configs, **kwargs)

        self.dir = os.path.join(configs['General']['data_dir'], 'model')

        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')

        self.features = {}
        for (key, value) in configs.items('Features'):
            try:
                self.features[key] = json.loads(value)

            except json.decoder.JSONDecodeError:
                pass

        resolutions_range = pd.date_range(dt.datetime.now().replace(minute=0, second=0, microsecond=0), periods=0)

        self.resolutions = list()
        for resolution_configs in [s for s in configs.sections() if s.lower().startswith('resolution')]:
            resolution = Resolution(**dict(configs.items(resolution_configs)))

            resolution_start = dt.datetime.now().replace(minute=0, second=0, microsecond=0) - resolution.time_prior
            resolutions_range = resolutions_range.union(pd.date_range(resolution_start,
                                                                      periods=resolution.steps_prior,
                                                                      freq='{}min'.format(resolution.minutes)))

            self.resolutions.append(resolution)

        if len(self.resolutions) < 1:
            raise ValueError("Invalid control configurations without specified step resolutions")

        self._estimate = kwargs.get('estimate') if 'estimate' in kwargs else \
            configs.get('Features', 'estimate', fallback='true').lower() == 'true'

        input_steps = len(resolutions_range)

        if self._estimate:
            input_steps += 1

        self._input_shape = (input_steps,
                             len(self.features['target'] + self.features['input']))
        self._target_shape = len(self.features['target'])

    def _build(self, context, configs, **kwargs):
        super()._build(context, configs, **kwargs)

        self.history = History()
        self.callbacks = [self.history,
                          TensorBoard(log_dir=self.dir, histogram_freq=1),
                          EarlyStopping(patience=self.epochs/4, restore_best_weights=True)]

        # TODO: implement date based backups and naming scheme
        if self.exists():
            self.model = self._load()

        else:
            self.model = self._build_model(configs)

        self.model.compile(optimizer=configs.get('General', 'optimizer'), loss=configs.get('General', 'loss'),
                           metrics=configs.get('General', 'metrics', fallback=[]))

    def _build_model(self, configs):
        model = Sequential()
        model = self._build_dense(model, configs['Dense'], first=True)

        return model

    def _build_dense(self, model, configs, first=False, flatten=False):
        dropout = configs.getfloat('dropout', fallback=0)
        units = configs.get('units')
        if units.isdigit():
            units = [int(units)] * configs.getint('layers', fallback=1)
        else:
            units = json.loads(units)

        length = len(units)
        for i in range(length):
            kwargs = self._parse_kwargs(configs, 'activation', 'kernel_initializer')

            if first and i == 0:
                kwargs['input_dim'] = self._input_shape[1]

            if flatten:
                model.add(Flatten())

            model.add(Dense(units[i], **kwargs))

            if dropout > 0.:
                model.add(Dropout(dropout))

        model.add(Dense(self._target_shape,
                        activation=configs['activation'],
                        kernel_initializer=configs['kernel_initializer']))

        if configs['activation'] == 'relu':
            model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))

        return model

    def _load(self, inplace=False):
        logger.debug("Loading model for system {} from file".format(self._system.name))

        with open(os.path.join(self.dir, 'model.json'), 'r') as f:
            model = model_from_json(f.read())
            model.load_weights(os.path.join(self.dir, 'model.h5'))

            if inplace:
                self.model = model

            return model

    def _save(self):
        logger.debug("Saving model for system {} to file".format(self._system.name))

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

        time = date - self.time_prior

        features = self._parse_features(data)

        # Remove target values from features, as those will be recursively filled with predictions
        features.loc[time:, self.features['target']] = np.NaN

        while time < end:
            time_next = time + dt.timedelta(minutes=self.resolutions[0].minutes)

            inputs = self._parse_inputs(features, time)
            result = self._run_step(inputs)

            results.append(result)
            result_range = features[(features.index >= time) & (features.index < time_next)].index

            # Add predicted output to features of next iteration
            features.loc[result_range, forecast.features['target']] = result

            time = time_next

        return results

    def _run_step(self, inputs):
        if len(inputs.shape) < 3:
            inputs = inputs.reshape(1, inputs.shape[0], inputs.shape[1])

        return float(self.model.predict(inputs, verbose=LOG_VERBOSE))

    def train(self, data):
        features = self._parse_features(data)
        return self._train(features)

    def _train(self, features):
        inputs, targets = self._parse_data(features)
        logger.debug("Built input of %s, %s", inputs.shape, targets.shape)

        split = int(len(targets) / 10.0)
        result = self.model.fit(inputs[split:], targets[split:], batch_size=self.batch, epochs=self.epochs,
                                validation_data=(inputs[:split], targets[:split]), callbacks=self.callbacks,
                                verbose=LOG_VERBOSE)

        # Write normed loss to TensorBoard
        writer = summary.create_file_writer(os.path.join(self.dir, 'loss'))
        for target in self.features['target']:
            loss = result.history['loss'] / features[target].max()
            loss_name = 'epoch_loss_norm' if len(self.features['target']) == 1 else '{}_norm'.format(target)
            for epoch in range(len(result.history['loss'])):
                with writer.as_default():
                    summary.scalar(loss_name, loss[epoch], step=epoch)

        self._write_distributions(features)
        self._save()
        return result

    def _shuffle_data(self, inputs, targets):
        import random

        data = []
        for i in range(len(inputs)):
            data.append({
                'input': inputs[i],
                'target': targets[i]
            })
        random.shuffle(data)

        inputs = []
        targets = []
        for i in range(len(data)):
            inputs.append(data[i]['input'])
            targets.append(data[i]['target'])

        return inputs, targets

    def _parse_data(self, features, shuffle=True):
        targets = []
        inputs = []

        end = features.index[-1]
        time = features.index[0] + self.resolutions[0].time_prior
        while time <= end:
            try:
                input = self._parse_inputs(features, time)
                target = self._parse_target(features, time)

                # If no exception was raised, add the validated data to the set
                inputs.append(input)
                targets.append(target)

            except ValueError:
                logger.debug("Skipping %s", time)

            time += dt.timedelta(minutes=self.resolutions[-1].minutes)

        if shuffle:
            inputs, targets = self._shuffle_data(inputs, targets)

        return np.array(inputs, dtype=float), np.array(targets, dtype=float)

    def _parse_inputs(self, features, time):
        inputs = self._extract_inputs(features, time)
        return np.squeeze(inputs.values)

    def _extract_inputs(self, features, time):
        data = pd.DataFrame()
        data.index.name = 'time'
        for resolution in self.resolutions:
            resolution_end = time - resolution.time_step
            resolution_start = time - resolution.time_prior
            resolution_data = features.loc[resolution_start:resolution_end,
                                           self.features['target'] + self.features['input']]

            data = resolution.resample(resolution_data).combine_first(data)

        if data.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % time)

        if self._estimate:
            resolution = self.resolutions[-1]
            resolution_end = time - resolution.time_step
            resolution_inputs = resolution.resample(features.loc[resolution_end:time,
                                                                 self.features['input']])

            data.loc[time] = np.append([np.NaN] * len(self.features['target']), resolution_inputs.values)

            # TODO: Replace interpolation with prediction of ANN
            data.interpolate(method='linear', inplace=True)

        return data

    def _parse_target(self, features, time):
        # return np.squeeze(self._extract_target(features, time).values)
        return float(self._extract_target(features, time))

    def _extract_target(self, features, time):
        # TODO: Implement horizon resolutions
        resolution = self.resolutions[-1]
        resolution_target = resolution.resample(
            features.loc[time - resolution.time_step + dt.timedelta(seconds=1): time,
            self.features['target']])

        data = resolution_target.loc[time, :]
        if data.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % time)

        return data

    def _parse_features(self, data):
        columns = self.features['target'] + self.features['input']

        # TODO: use weather pressure for solar position
        solar = get_solarposition(pd.date_range(data.index[0], data.index[-1], freq='min'),
                                  self._system.location.latitude,
                                  self._system.location.longitude,
                                  altitude=self._system.location.altitude)
        solar = solar.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
        solar.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']

        data = data[np.intersect1d(data.columns, columns)].copy()
        data['day_of_year'] = data.index.dayofyear
        data['day_of_week'] = data.index.dayofweek

        features = pd.concat([data, solar], axis=1)
        features = self._parse_cyclic(features)
        features = self._parse_horizon(features)
        features = self._scale_features(features)

        return features

    def _scale_features(self, data):
        if 'scaling' not in self.features:
            return data

        def _scale_transformations(invert=False):
            trafos = {}

            for feature, trafo in self.features.get('scaling', {}).items():
                if feature not in data.columns:
                    continue

                if trafo.lower() == 'norm' or str(trafo).isdigit():
                    trafo_value = float(trafo) if not str(trafo).isdigit() else data[feature].max()

                    def transform(value):
                        if not invert:
                            return value / trafo_value
                        else:
                            return value * trafo_value

                elif trafo.lower() == 'std':
                    mean = data[feature].mean()
                    std = data[feature].std()

                    def transform(value):
                        if not invert:
                            return (value - mean) / std
                        else:
                            return value * std + mean
                else:
                    raise ValueError('The transformation {} is not defined in the function gen_trafos.'.format(trafo))

                trafos[feature] = transform

            return trafos

        for feature, transform in _scale_transformations():
            data[feature] = transform(data[feature])

        return data

    def _parse_horizon(self, data):
        resolution = self.resolutions[0]
        data['horizon'] = (data.index - data.index[0]) / np.timedelta64(1, 'm') % (
                resolution.minutes * resolution.steps_horizon)

        return data

    def _parse_cyclic(self, data):
        for feature, bound in self.features['cyclic'].items():
            data[feature + '_sin'] = np.sin(2.0 * np.pi * data[feature] / bound)
            data[feature + '_cos'] = np.cos(2.0 * np.pi * data[feature] / bound)
            data = data.drop(columns=[feature])

        return data

    @staticmethod
    def _parse_kwargs(configs, *args):
        kwargs = {}
        for arg in args:
            kwargs[arg] = configs[arg]

        return kwargs

    # TODO: Move function to evaluation package
    @staticmethod
    def _write_distributions(features, path=''):
        import matplotlib.pyplot as plt

        # Desired number of bins in each plot
        bin_num = 100
        for feature in features.columns:  # create 100 equal space bin vals per feat.
            bins = []
            bin_domain = features[feature].max() - features[feature].min()
            bin_step = bin_domain / bin_num

            counter = features[feature].min()
            for i in range(bin_num):
                bins.append(counter)
                counter = counter + bin_step

            # Add the last value of the counter
            bins.append(counter)

            plt_info = plt.hist(features[feature], bins=bins)
            bin_values, bins = plt_info[0], plt_info[1]
            count_range = max(bin_values) - min(bin_values)
            sorted_values = list(bin_values)
            sorted_values.sort(reverse=True)

            # Scale plots by step through sorted bins
            for i in range(len(sorted_values) - 1):
                if abs(sorted_values[i] - sorted_values[i + 1]) / count_range < 0.80:
                    continue
                else:
                    plt.ylim([0, sorted_values[i + 1] + 10])
                    break

            # Save histogram to appropriate folder
            path_dist = os.path.join(path, 'dist')
            path_file = os.path.join(path_dist, '{}.png'.format(feature))
            if not os.path.isdir(path_dist):
                os.makedirs(path_dist, exist_ok=True)

            plt.savefig(path_file)
            plt.clf()


class StackedLSTM(NeuralNetwork):

    def _build_model(self, configs):
        model = Sequential()
        model = self._build_lstm(model, configs['LSTM'], first=True)
        model = self._build_dense(model, configs['Dense'])

        return model

    def _build_lstm(self, model, configs, first=False):
        units = configs.get('units')
        if units.isdigit():
            units = [int(units)] * configs.getint('layers', fallback=1)
        else:
            units = json.loads(units)

        length = len(units)
        for i in range(length):
            kwargs = self._parse_kwargs(configs, 'activation', 'kernel_initializer')

            if i == 0 and first:
                kwargs['input_shape'] = self._input_shape

            elif i < length-1:
                kwargs['return_sequences'] = True

            model.add(LSTM(units[i], **kwargs))

        return model


class ConvDilated(NeuralNetwork):

    def _build_model(self, configs):
        model = Sequential()
        model = self._build_conv(model, configs['Conv1D'], first=True)
        model = self._build_dense(model, configs['Dense'], flatten=True)

        return model

    def _build_conv(self, model, configs, first=False):
        filters = configs.get('filters')
        if filters.isdigit():
            filters = [int(filters)] * configs.getint('layers', fallback=1)
        else:
            filters = json.loads(filters)

        # TODO: Handle padding and dilation in configuration
        length = len(filters)
        for i in range(length):
            kwargs = self._parse_kwargs(configs, 'activation', 'kernel_initializer')
            kwargs['padding'] = 'causal'

            if first and i == 0:
                kwargs['input_shape'] = self._input_shape
                kwargs['dilation_rate'] = 1
            else:
                kwargs['dilation_rate'] = 2**(i+1)

            model.add(Conv1D(filters[i], int(configs['kernel_size']), **kwargs))

        model.add(MaxPooling1D(int(configs['pool_size'])))

        return model


class ConvLSTM(ConvDilated, StackedLSTM):

    def _build_model(self, configs):
        model = Sequential()
        model = self._build_conv(model, configs['Conv1D'], first=True)
        model = self._build_lstm(model, configs['LSTM'])
        model = self._build_dense(model, configs['Dense'])

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

        return dt.timedelta(minutes=self.minutes * self.steps_prior)

    @property
    def time_horizon(self):
        if self.steps_horizon is None:
            return None

        return dt.timedelta(minutes=self.minutes * (self.steps_horizon - 1))

    def resample(self, features):
        data = features.resample('{}min'.format(self.minutes), closed='right').mean()
        data.index += to_offset('{}min'.format(self.minutes))

        return data
