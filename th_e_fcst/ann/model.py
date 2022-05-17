# -*- coding: utf-8 -*-
"""
    th_e_fcst.ann.model
    ~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import os
import json
import random
import numpy as np
import pandas as pd
import datetime as dt
import holidays as hl
import logging

from glob import glob
from copy import deepcopy
from typing import Any, Optional, Tuple, List, Dict
from configparser import ConfigParser as Configurations
from configparser import SectionProxy as ConfigurationSection
from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas.tseries.frequencies import to_offset

from tensorflow.keras.callbacks import History, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.models import model_from_json
from th_e_core import System, Model

logger = logging.getLogger(__name__)

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(Model):

    @classmethod
    def read(cls, system: System, **kwargs) -> NeuralNetwork:
        configs = cls._read_configs(system, **kwargs)
        model = configs.get('General', 'model', fallback='default').lower()

        if model in ['mlp', 'ann', 'dense', 'default']:
            return NeuralNetwork(system, configs, **kwargs)

        elif model in ['convdilated', 'conv', 'cnn']:
            return ConvDilated(system, configs, **kwargs)

        elif model == 'convlstm':
            return ConvLSTM(system, configs, **kwargs)

        elif model == 'lstm':
            return StackedLSTM(system, configs, **kwargs)

        raise TypeError('Invalid model: {}'.format(type))

    def _configure(self, configs: Configurations) -> None:
        super()._configure(configs)

        self.dir = os.path.join(configs['General']['data_dir'], 'model')

        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')

        self.features = {}
        for (key, value) in configs.items('Features'):
            try:
                self.features[key] = json.loads(value)

            except json.decoder.JSONDecodeError:
                self.features[key] = value

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

        input_steps = len(resolutions_range)

        self._estimate = configs.get('Features', 'estimate', fallback='true').lower() == 'true'
        if self._estimate:
            input_steps += 1

        input_features = self.features['target'] + self.features['input']
        input_count = len(input_features)
        for feature, _ in self.features['cyclic'].items():
            if feature in input_features:
                input_count += 1

        self._input_shape = (input_steps, input_count)

        target_steps = 1
        target_count = len(self.features['target'])
        self._target_shape = (target_steps, target_count)

    def _build(self, system: System, configs: Configurations) -> None:
        super()._build(system, configs)

        self.history = History()
        self.callbacks = [self.history, TensorBoard(log_dir=self.dir, histogram_freq=1)]

        self._early_stopping = configs.get('General', 'early_stopping', fallback='True').lower() == 'true'
        if self._early_stopping:
            self._early_stopping_split = configs.getint('General', 'early_stopping_split', fallback=7)
            self._early_stopping_patience = configs.getint('General', 'early_stopping_patience', fallback=self.epochs/4)
            self.callbacks.append(EarlyStopping(patience=self._early_stopping_patience, restore_best_weights=True))

        self._build_model(configs)

    def _build_model(self, configs: Configurations) -> None:
        # TODO: implement date based backups and naming scheme
        if self.exists():
            self._load()
        else:
            self._build_layers(configs)

        self.model.compile(optimizer=self._parse_optimizer(configs),
                           metrics=self._parse_metrics(configs),
                           loss=self._parse_loss(configs))

    def _build_layers(self, configs: Configurations) -> None:
        pass

    def __getstate__(self):
        state = self.__dict__.copy()

        # Do not serialize model
        del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # TODO: Build model back since it was not serialized
        # self._build_model(self._configs)
        self.model = None

    @staticmethod
    def _parse_optimizer(configs: Configurations):
        optimizer = configs.get('General', 'optimizer')
        if optimizer == 'adam' and configs.has_option('General', 'learning_rate'):
            optimizer = Adam(learning_rate=configs.getfloat('General', 'learning_rate'))

        return optimizer

    @staticmethod
    def _parse_metrics(configs: Configurations):
        return configs.get('General', 'metrics', fallback=[])

    @staticmethod
    def _parse_loss(configs: Configurations):
        return configs.get('General', 'loss')

    def _load(self, from_json: bool = False) -> None:
        logger.debug("Loading model for system {} from file".format(self._system.name))
        self.model = None
        try:
            if os.path.isfile(os.path.join(self.dir, 'model.json')) and from_json:
                with open(os.path.join(self.dir, 'model.json'), 'r') as f:
                    self.model = model_from_json(f.read())

        except ValueError as e:
            logger.warning(str(e))

        if self.model is None:
            self._build_layers(self._configs)
            logger.info("Built model after failure to read serialized graph")

        if (glob(os.path.join(self.dir, 'checkpoint*')) and
                glob(os.path.join(self.dir, 'model.data*')) and
                os.path.isfile(os.path.join(self.dir, 'model.index'))):
            self.model.load_weights(os.path.join(self.dir, 'model'))
        else:
            self.model.load_weights(os.path.join(self.dir, 'model.h5'))

    def _save(self) -> None:
        logger.debug("Saving model for system {} to file".format(self._system.name))

        # Serialize model to JSON
        with open(os.path.join(self.dir, 'model.json'), 'w') as f:
            f.write(self.model.to_json())

        # Serialize weights to HDF5
        self.model.save_weights(os.path.join(self.dir, 'model.h5'))

        # Serialize weights checkpoint
        self.model.save_weights(os.path.join(self.dir, 'model'))

    def exists(self):
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            return False

        exists = ((glob(os.path.join(self.dir, 'checkpoint*')) and
                   glob(os.path.join(self.dir, 'model.data*')) and
                  os.path.isfile(os.path.join(self.dir, 'model.index'))) or
                  os.path.isfile(os.path.join(self.dir, 'model.h5')))

        return exists

    def predict(self,
                data: pd.DataFrame,
                date: pd.Timestamp | dt.datetime,
                *args: Optional[Any]) -> pd.DataFrame:

        if len(args) > 0 and isinstance(args[0], pd.DataFrame):
            forecast = list(args).pop(0)
            data = pd.concat([data, forecast], axis=1)

        features = self._parse_features(deepcopy(data))
        return self._predict(features, date)

    def _predict(self,
                 features: pd.DataFrame,
                 date: pd.Timestamp | dt.datetime):

        # TODO: Implement horizon resolutions
        resolution = self.resolutions[-1]
        targets = pd.DataFrame(columns=self.features['target'], dtype=float)

        if resolution.steps_horizon is None:
            end = features.index[-1]
        else:
            end = date + resolution.time_horizon

        # Work on copy to retain original input vector to be saved.
        features = deepcopy(features)

        # Remove target values from features, as those will be recursively filled with predictions
        features.loc[features.index > date, self.features['target']] = np.NaN

        while date < end:
            date_fcst = date + resolution.time_step
            if date_fcst not in features.index:
                break

            input = self._parse_inputs(features, date)
            input = self._scale_features(input)
            target = self._predict_step(input)
            targets.loc[date_fcst, self.features['target']] = target

            # Add predicted output to features of next iteration
            features.loc[(features.index >= date_fcst) &
                         (features.index < date_fcst + resolution.time_step), self.features['target']] = target

            date = date_fcst

        return self._scale_features(targets, invert=True)

    def _predict_step(self, input: pd.DataFrame) -> np.ndarray | float:
        input_shape = (1, self._input_shape[0], self._input_shape[1])
        input = self._reshape_data(input, input_shape)
        target = self.model(input)  # .predict(input, verbose=LOG_VERBOSE)
        target = self._reshape_data(target, self._target_shape[1])

        return target

    def train(self, data: pd.DataFrame) -> History:
        features = self._parse_features(data)
        return self._train(features)

    def _train(self,
               features: pd.DataFrame,
               shuffle: bool = True) -> History:

        kwargs = {
            'verbose': LOG_VERBOSE
        }
        data = self._parse_data(features)
        inputs = []
        targets = []
        if not self._early_stopping:
            for d in data.values():
                if shuffle:
                    random.shuffle(d)
                for i in range(len(d)):
                    inputs.append(d[i]['input'])
                    targets.append(d[i]['target'])
        else:
            validation_inputs = []
            validation_targets = []

            for d in data.values():
                split = len(d) - int(len(d)/self._early_stopping_split)
                if shuffle:
                    random.shuffle(d)

                for i in range(len(d)):
                    if i <= split:
                        inputs.append(d[i]['input'])
                        targets.append(d[i]['target'])
                    else:
                        validation_inputs.append(d[i]['input'])
                        validation_targets.append(d[i]['target'])

            kwargs['validation_data'] = (np.array(validation_inputs, dtype=float),
                                         np.array(validation_targets, dtype=float))

        result = self.model.fit(np.array(inputs, dtype=float),
                                np.array(targets, dtype=float),
                                callbacks=self.callbacks,
                                batch_size=self.batch,
                                epochs=self.epochs,
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
    def _reshape_data(data: pd.Dataframe | np.ndarray, shape: Tuple | int) -> np.ndarray | float:
        if isinstance(data, pd.DataFrame):
            data = data.values

        return np.squeeze(data).reshape(shape)

    def _parse_data(self, features: pd.DataFrame) -> Dict[int, List[Dict[str, np.ndarray]]]:
        # TODO: implement additional method to bin data
        index = np.unique(features.index.isocalendar().week)
        dates = dict((i, []) for i in index)
        date = features.index[0] + self.resolutions[-1].time_prior
        while date <= features.index[-1]:
            date_index = date.week
            dates[date_index].append(date)
            date += dt.timedelta(minutes=self.resolutions[-1].minutes)

        data = {}
        steps = []
        with ProcessPoolExecutor() as executor:
            for step_index, step_dates in dates.items():
                step_features = features[step_dates[0] - self.resolutions[-1].time_prior:step_dates[-1]]
                steps.append(executor.submit(self._parse_step, step_features, step_index, step_dates))

            for step in as_completed(steps):
                step_index, step_data = step.result()
                data[step_index] = step_data

        return data

    def _parse_step(self, features: pd.DataFrame, index: int,
                    dates: List[pd.Timestamp]) -> Tuple[int, List[Dict[str, np.ndarray]]]:
        data = []
        for date in dates:
            try:
                input = self._parse_inputs(features, date)
                input = self._scale_features(input)

                target = self._parse_targets(features, date)
                target = self._scale_features(target)

                # If no exception was raised, add the validated data to the set
                data.append({
                    'input': self._reshape_data(input, self._input_shape),
                    'target': self._reshape_data(target, self._target_shape[1])
                })
            except (KeyError, ValueError) as e:
                logger.debug("Skipping %s: %s", date, str(e))

        return index, data

    def _parse_inputs(self,
                      features: pd.DataFrame,
                      index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]
        data = self.resolutions[-1].resample(deepcopy(features))

        # TODO: Optionally replace the estimate with the prediction of an ANN
        if self._estimate:
            # Make sure that no future target values exist
            data.loc[index[0]:, self.features['target']] = np.NaN

            # TODO: Implement horizon resolutions
            estimate = data.loc[index, self.features['target']]
            for estimate_step in estimate.index:
                estimate_value = data.loc[(data.index < index[0]) &
                                          (data.index.hour == estimate_step.hour) &
                                          (data.index.minute == estimate_step.minute),
                                          self.features['target']].mean(axis=0)
                if estimate_value.empty:
                    continue

                estimate.loc[estimate_step, :] = estimate_value.values

            # Estimate the value of targets for times corresponding to the hour to the time to be predicted
            if self._system.contains_type('pv') and 'pv_yield' in self.features['input']:
                # Use calculated PV yield values as PV estimate
                estimate.loc[index, 'pv_power'] = data.loc[index, 'pv_yield']

            if estimate.isnull().values.any():
                estimate = features[self.features['target']].interpolate(method='linear').loc[index]

            data.loc[index, self.features['target']] = estimate

        data = self._add_doubt(data, index)

        inputs = pd.DataFrame()
        inputs.index.name = 'time'
        for resolution in self.resolutions:
            resolution_end = index[0] - resolution.time_step if not self._estimate else index[-1]
            resolution_start = index[0] - resolution.time_step - resolution.time_prior + dt.timedelta(minutes=1)
            resolution_inputs = resolution.resample(data[resolution_start:resolution_end])

            inputs = resolution_inputs.combine_first(inputs)

        inputs = self._add_meta(inputs)[self.features['target'] + self.features['input']]
        inputs = self._parse_cyclic(inputs)

        if inputs.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % index)

        return inputs

    def _parse_targets(self,
                       features: pd.DataFrame,
                       index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]

        # TODO: Implement horizon resolutions
        data = self.resolutions[-1].resample(deepcopy(features))

        targets = data.loc[index, self.features['target']]
        if targets.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % index)

        return targets

    def _parse_features(self, data):
        columns = self.features['target'] + self.features['input']
        features = deepcopy(data[np.intersect1d(data.columns, columns)])

        return features[np.intersect1d(data.columns, columns)]

    def _scale_features(self, features, invert=False):
        scaled_features = deepcopy(features)
        if 'scaling' not in self.features:
            return scaled_features

        scaling = self.features.get('scaling', {})
        if 'doubt' in self.features:
            for feature in self.features['doubt'].keys():
                scaling[feature+'_doubt'] = scaling[feature]

        for feature, transformation in scaling.items():
            if feature not in scaled_features.columns:
                continue

            if str(transformation).isdigit():
                if not invert:
                    scaled_features[feature] /= float(transformation)
                else:
                    scaled_features[feature] *= float(transformation)

            # TODO: Save the maximum or std value, to allow the live scaling for small feature sets
            # elif transformation.lower() == 'norm':
            #     if not invert:
            #         scaled_featuers[feature] /= scaled_featuers[feature].max()
            #     else:
            #         scaled_featuers[feature] *= scaled_featuers[feature].max()
            #
            # elif transformation.lower() == 'std':
            #     mean = scaled_featuers[feature].mean()
            #     std = scaled_featuers[feature].std()
            #
            #     if not invert:
            #         scaled_featuers[feature] = (scaled_featuers[feature] - mean) / std
            #     else:
            #         scaled_featuers[feature] = scaled_featuers[feature] * std + mean
            else:
                raise ValueError('The transformation "{}" is not defined.'.format(transformation))

        return scaled_features

    def _add_doubt(self, features, times=None):
        if 'doubt' not in self.features:
            return features

        if times is None:
            times = features.index

        for feature, feature_cor in self.features['doubt'].items():
            features.loc[times, feature+'_doubt'] = abs(features.loc[times, feature] - features.loc[times, feature_cor])

        return features

    def _add_meta(self, features):
        features['hour_of_day'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['day_of_year'] = features.index.dayofyear - 1

        if self._system.location.country is None:
            return features

        features['holiday'] = 0

        holiday_args = {
            'years': list(dict.fromkeys(features.index.year))
        }
        if self._system.location.state is not None:
            holiday_args['subdiv'] = self._system.location.state

        holidays = hl.country_holidays(self._system.location.country, **holiday_args)
        features.loc[np.isin(features.index.date, list(holidays.keys())), 'holiday'] = 1

        return features

    def _parse_cyclic(self, features):
        cyclic_features = deepcopy(features)
        if 'cyclic' not in self.features:
            return cyclic_features

        for feature, bound in self.features['cyclic'].items():
            if feature not in cyclic_features.columns:
                continue

            cyclic_index = cyclic_features.columns.get_loc(feature)
            cyclic_features.insert(cyclic_index, feature + '_cos', np.cos(2.0 * np.pi * features[feature] / bound))
            cyclic_features.insert(cyclic_index, feature + '_sin', np.sin(2.0 * np.pi * features[feature] / bound))
            cyclic_features = cyclic_features.drop(columns=[feature])

        return cyclic_features

    @staticmethod
    def _parse_kwargs(configs, *args):
        kwargs = {}
        for arg in args:
            kwargs[arg] = configs[arg]

        return kwargs


class MultiLayerPerceptron(NeuralNetwork):

    def _build_layers(self, configs: Configurations) -> None:
        self.model = Sequential(name='MultiLayerPerceptron')
        self._add_dense(configs['Dense'], first=True)

    def _add_dense(self, configs: ConfigurationSection, first: bool = False) -> None:
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

            self.model.add(Dense(units[i], **kwargs))

            if dropout > 0.:
                self.model.add(Dropout(dropout))

        self.model.add(Dense(self._target_shape[1],
                             activation=configs['activation'],
                             kernel_initializer=configs['kernel_initializer']))

        if configs['activation'] == 'relu':
            self.model.add(LeakyReLU(alpha=float(configs['leaky_alpha'])))


class StackedLSTM(MultiLayerPerceptron):

    def _build_layers(self, configs: Configurations) -> None:
        self.model = Sequential(name='StackedLSTM')
        self._add_lstm(configs['LSTM'], first=True)
        self._add_dense(configs['Dense'])

    def _add_lstm(self, configs: ConfigurationSection, first: bool = False) -> None:
        units = configs.get('units')
        if units.isdigit():
            units = [int(units)] * configs.getint('layers', fallback=1)
        else:
            units = json.loads(units)

        length = len(units)
        for i in range(length):
            kwargs = self._parse_kwargs(configs, 'activation')

            if i == 0 and first:
                kwargs['input_shape'] = self._input_shape

            elif i < length-1:
                kwargs['return_sequences'] = True

            self.model.add(LSTM(units[i], **kwargs))


class ConvDilated(MultiLayerPerceptron):

    def _build_layers(self, configs: Configurations) -> None:
        self.model = Sequential(name='ConvolutionalDilation')
        self._add_conv(configs['Conv1D'], first=True)
        self._add_dense(configs['Dense'])

    def _add_conv(self, configs: ConfigurationSection, flatten:bool = True, first: bool = False) -> None:
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
                kwargs['dilation_rate'] = 2**i

            self.model.add(Conv1D(filters[i], int(configs['kernel_size']), **kwargs))

        if 'pool_size' in configs:
            pool_size = int(configs['pool_size'])
            self.model.add(MaxPooling1D(pool_size))

        if flatten:
            self.model.add(Flatten())


class ConvLSTM(ConvDilated, StackedLSTM):

    def _build_layers(self, configs: Configurations) -> None:
        self.model = Sequential(name='ConvolutionalLSTM')
        self._add_conv(configs['Conv1D'], first=True)
        self._add_lstm(configs['LSTM'])
        self._add_dense(configs['Dense'])


class Resolution:

    def __init__(self, minutes, steps_prior=None, steps_horizon=None):
        self.minutes = int(minutes)
        self.steps_prior = int(steps_prior) if steps_prior else None
        self.steps_horizon = int(steps_horizon) if steps_horizon else None

    @property
    def time_step(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.minutes)

    @property
    def time_prior(self) -> dt.timedelta | None:
        if self.steps_prior is None:
            return None

        return dt.timedelta(minutes=self.minutes * self.steps_prior)

    @property
    def time_horizon(self) -> dt.timedelta | None:
        if self.steps_horizon is None:
            return None

        return dt.timedelta(minutes=self.minutes * self.steps_horizon)

    def resample(self, features: pd.DataFrame) -> pd.DataFrame:
        data = features.resample('{}min'.format(self.minutes), closed='right').mean()
        data.index += to_offset('{}min'.format(self.minutes))

        return data
