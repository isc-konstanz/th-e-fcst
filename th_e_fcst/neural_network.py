# -*- coding: utf-8 -*-
"""
    th_e_fcst.neural_network
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
                self.features[key] = value

        # TODO: Save and load these from database
        self.features['covariance_pa'] = {}
        self.features['covariance_std'] = {}

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
        self.callbacks = [self.history, TensorBoard(log_dir=self.dir, histogram_freq=1)]

        self._early_stopping = configs.get('General', 'early_stopping', fallback='True').lower() == 'true'
        if self._early_stopping:
            self.callbacks.append(EarlyStopping(patience=self.epochs/2, restore_best_weights=True))

        # TODO: implement date based backups and naming scheme
        if self.exists():
            self._load()

        else:
            self._build_layers(configs)

        self.model.compile(optimizer=configs.get('General', 'optimizer'), loss=configs.get('General', 'loss'),
                           metrics=configs.get('General', 'metrics', fallback=[]))

    def _build_layers(self, configs):
        self.model = Sequential()
        self._add_dense(configs['Dense'], first=True)

    def _add_dense(self, configs, first=False, flatten=False):
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
                self.model.add(Flatten())

            self.model.add(Dense(units[i], **kwargs))

            if dropout > 0.:
                self.model.add(Dropout(dropout))

        self.model.add(Dense(self._target_shape,
                            activation=configs['activation'],
                            kernel_initializer=configs['kernel_initializer']))

        if configs['activation'] == 'relu':
            self.model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))

    def _load(self, inplace=True):
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
        results = pd.DataFrame(columns=self.features['target'])

        if len(self.resolutions) == 1:
            resolution_min = self.resolutions[0]
        else:
            for i in range(len(self.resolutions) - 1, 0, -1):
                resolution_min = self.resolutions[i]
                if resolution_min.steps_horizon is not None:
                    break

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
            time_next = time + resolution_min.time_step

            inputs = self._parse_inputs(features, time)
            result = self._run_step(inputs)

            results.loc[time, self.features['target']] = result
            result_range = features[(features.index >= time) & (features.index < time_next)].index

            # Add predicted output to features of next iteration
            features.loc[result_range, self.features['target']] = result

            time = time_next

        return self._scale_features(result, invert=True)

    def _run_step(self, inputs):
        if len(inputs.shape) < 3:
            inputs = inputs.reshape(1, inputs.shape[0], inputs.shape[1])

        return float(self.model.predict(inputs, verbose=LOG_VERBOSE))

    def train(self, data):
        features = self._parse_features(data)
        return self._train(features)

    def _train(self, features, shuffle=True):
        inputs, targets = self._parse_data(features)
        logger.debug("Built input of %s, %s", inputs.shape, targets.shape)

        kwargs = {
            'verbose': LOG_VERBOSE
        }
        if self._early_stopping:
            if shuffle:
                inputs, targets = self._shuffle_data(inputs, targets)

            validation_split = int(len(targets) / 10.0)
            validation_inputs = inputs[:validation_split]
            validation_targets = targets[:validation_split]

            kwargs['validation_data'] = (validation_inputs, validation_targets)

            inputs = inputs[validation_split:]
            targets = targets[validation_split:]

        result = self.model.fit(inputs, targets, batch_size=self.batch, epochs=self.epochs, callbacks=self.callbacks,
                                **kwargs)

        # Write normed loss to TensorBoard
        # writer = summary.create_file_writer(os.path.join(self.dir, 'loss'))
        # for target in self.features['target']:
        #     loss = result.history['loss'] / features[target].max()
        #     loss_name = 'epoch_loss_norm' if len(self.features['target']) == 1 else '{}_norm'.format(target)
        #     for epoch in range(len(result.history['loss'])):
        #         with writer.as_default():
        #             summary.scalar(loss_name, loss[epoch], step=epoch)

        self._save()
        return result

    @staticmethod
    def _shuffle_data(inputs, targets):
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

        return np.array(inputs, dtype=float), \
               np.array(targets, dtype=float)

    def _parse_data(self, features):
        targets = []
        inputs = []

        end = features.index[-1]
        time = features.index[0] + self.resolutions[-1].time_prior
        while time <= end:
            try:
                input = np.squeeze(self._parse_inputs(features, time, update=False))
                target = float(self._parse_target(features, time))

                # If no exception was raised, add the validated data to the set
                inputs.append(input)
                targets.append(target)

            except ValueError as e:
                logger.debug("Skipping %s: %s", time, str(e))

            time += dt.timedelta(minutes=self.resolutions[-1].minutes)

        return np.array(inputs, dtype=float), \
               np.array(targets, dtype=float)

    def _parse_inputs(self, features, time, update=True):
        resolution_min = self.resolutions[-1]
        resolution_end = time - resolution_min.time_step
        resolution_range = features[(features.index > resolution_end) & (features.index <= time)].index

        features_target = self.features['target']
        features.loc[resolution_range, features_target] = np.NaN

        # TODO: Replace interpolation with prediction of ANN
        features[features_target] = features[features_target].interpolate(method='linear')

        if update:
            # Calculate the doubt for the current time step
            # This is necessary for the recursive iteration
            features = self._calc_doubt(features, resolution_range)

        data = pd.DataFrame()
        data.index.name = 'time'
        for resolution in self.resolutions:
            resolution_end = time - resolution.time_step if not self._estimate else time
            resolution_start = time - resolution.time_prior
            resolution_data = features.loc[resolution_start:resolution_end,
                                           self.features['target'] + self.features['input']]

            data = resolution.resample(resolution_data).combine_first(data)

        if data.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % time)

        return data

    def _parse_target(self, features, time):
        # TODO: Implement horizon resolutions
        resolution = self.resolutions[-1]
        resolution_target = resolution.resample(features.loc[time - resolution.time_step + dt.timedelta(seconds=1):time,
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
        features.index.name = 'time'

        features = self._parse_cyclic(features)
        features = self._scale_features(features)
        features = self._calc_doubt(features)

        return features[columns]

    def _scale_features(self, features, invert=False):
        if 'scaling' not in self.features:
            return features

        for feature, transformation in self.features.get('scaling', {}).items():
            if feature not in features.columns:
                continue

            if str(transformation).isdigit():
                if not invert:
                    features[feature] /= float(transformation)
                else:
                    features[feature] *= float(transformation)

            # TODO: Save the maximum or std value, to allow the live scaling for small feature sets
            # elif transformation.lower() == 'norm':
            #     if not invert:
            #         features[feature] /= features[feature].max()
            #     else:
            #         features[feature] *= features[feature].max()
            #
            # elif transformation.lower() == 'std':
            #     mean = features[feature].mean()
            #     std = features[feature].std()
            #
            #     if not invert:
            #         features[feature] = (features[feature] - mean) / std
            #     else:
            #         features[feature] = features[feature] * std + mean
            else:
                raise ValueError('The transformation "{}" is not defined.'.format(transformation))

        return features

    def _calc_doubt(self, features, times=None):
        if 'doubt' not in self.features:
            return features

        if times is None:
            times = features.index

        time_interval = dt.timedelta(minutes=self.features.get('doubt_interval', 60))
        time_periods = len(features[times[0]:times[0] + time_interval].index) - 1
        time_range = features[(features.index > times[0] - time_interval) & (features.index <= times[-1])].index

        for feature, feature_cor in self.features['doubt'].items():
            features_cov_key = '{}_{}_cov'.format(feature, feature_cor)
            features_cov = features.loc[time_range, [feature, feature_cor]] \
                .rolling(time_periods, min_periods=time_periods) \
                .cov().unstack()[feature][feature_cor]

            # Overall covariance of the series series per annum
            if features_cov_key not in self.features['covariance_pa']:
                features_cov_pa = features[feature].cov(features[feature_cor])
                self.features['covariance_pa'][features_cov_key] = features_cov_pa

            features_cov_pa = self.features['covariance_pa'][features_cov_key]

            # Std of sample covariance from population covariance estimate
            if features_cov_key not in self.features['covariance_std']:
                features_cov_std = np.sqrt(((features_cov - features_cov_pa) ** 2).sum() / (len(features_cov) - 1))
                self.features['covariance_std'][features_cov_key] = features_cov_std

            features_cov_std = self.features['covariance_std'][features_cov_key]

            features.loc[times, features_cov_key] = features_cov[times]
            features.loc[times, feature+'_doubt'] = (abs(features[feature] - features[feature_cor])) / \
                                                    (abs(features_cov[times] - features_cov_pa) / features_cov_std)

        return features

    def _parse_cyclic(self, features):
        for feature, bound in self.features['cyclic'].items():
            features[feature + '_sin'] = np.sin(2.0 * np.pi * features[feature] / bound)
            features[feature + '_cos'] = np.cos(2.0 * np.pi * features[feature] / bound)
            features = features.drop(columns=[feature])

        return features

    @staticmethod
    def _parse_kwargs(configs, *args):
        kwargs = {}
        for arg in args:
            kwargs[arg] = configs[arg]

        return kwargs


class StackedLSTM(NeuralNetwork):

    def _build_layers(self, configs):
        self.model = Sequential()
        self._add_lstm(configs['LSTM'], first=True)
        self._add_dense(configs['Dense'])

    def _add_lstm(self, configs, first=False):
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

            self.model.add(LSTM(units[i], **kwargs))


class ConvDilated(NeuralNetwork):

    def _build_layers(self, configs):
        self.model = Sequential()
        self._add_conv(configs['Conv1D'], first=True)
        self._add_dense(configs['Dense'], flatten=True)

    def _add_conv(self, configs, first=False):
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

            self.model.add(Conv1D(filters[i], int(configs['kernel_size']), **kwargs))

        self.model.add(MaxPooling1D(int(configs['pool_size'])))


class ConvLSTM(ConvDilated, StackedLSTM):

    def _build_layers(self, configs):
        self.model = Sequential()
        self._add_conv(configs['Conv1D'], first=True)
        self._add_lstm(configs['LSTM'])
        self._add_dense(configs['Dense'])


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
