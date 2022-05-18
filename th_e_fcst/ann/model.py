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
import logging

from glob import glob
from copy import deepcopy
from typing import Any, Optional, Tuple, List, Dict
from configparser import ConfigParser as Configurations
from configparser import SectionProxy as ConfigurationSection
from concurrent.futures import ProcessPoolExecutor, as_completed

from tensorflow.keras.callbacks import History, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.models import model_from_json
from th_e_fcst.ann import Features
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
            return NeuralNetwork(system, configs)

        elif model in ['convdilated', 'conv', 'cnn']:
            return ConvDilated(system, configs)

        elif model == 'convlstm':
            return ConvLSTM(system, configs)

        elif model == 'lstm':
            return StackedLSTM(system, configs)

        raise TypeError('Invalid model: {}'.format(type))

    def _configure(self, configs: Configurations) -> None:
        super()._configure(configs)

        self.dir = os.path.join(configs['General']['data_dir'], 'model')

        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')

    def _build(self, system: System, configs: Configurations) -> None:
        super()._build(system, configs)

        self.history = History()
        self.callbacks = [self.history, TensorBoard(log_dir=self.dir, histogram_freq=1)]

        self._early_stopping = configs.get('General', 'early_stopping', fallback='True').lower() == 'true'
        if self._early_stopping:
            self._early_stopping_split = configs.getint('General', 'early_stopping_split', fallback=7)
            self._early_stopping_patience = configs.getint('General', 'early_stopping_patience', fallback=self.epochs/4)
            self.callbacks.append(EarlyStopping(patience=self._early_stopping_patience, restore_best_weights=True))

        self.features = Features(system, configs)

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

        features = self.features.extract(data)
        return self._predict(features, date)

    def _predict(self,
                 features: pd.DataFrame,
                 date: pd.Timestamp | dt.datetime):

        # TODO: Implement horizon resolutions
        resolution = self.features.resolutions[-1]
        targets = pd.DataFrame(columns=self.features.target_keys, dtype=float)

        if resolution.steps_horizon is None:
            end = features.index[-1]
        else:
            end = date + resolution.time_horizon

        # Work on copy to retain original input vector to be saved.
        features = deepcopy(features)

        # Remove target values from features, as those will be recursively filled with predictions
        features.loc[features.index > date, self.features.target_keys] = np.NaN

        while date < end:
            date_fcst = date + resolution.time_step
            if date_fcst not in features.index:
                break

            input = self.features.input(features, date)
            input = self.features.scale(input)
            target = self._predict_step(input)
            targets.loc[date_fcst, self.features.target_keys] = target

            # Add predicted output to features of next iteration
            features.loc[(features.index >= date_fcst) &
                         (features.index < date_fcst + resolution.time_step), self.features.target_keys] = target

            date = date_fcst

        return self.features.scale(targets, invert=True)

    def _predict_step(self, input: pd.DataFrame) -> np.ndarray | float:
        input_shape = (1, self.features.input_shape[0], self.features.input_shape[1])
        input = self._reshape(input, input_shape)
        target = self.model(input)  # .predict(input, verbose=LOG_VERBOSE)
        target = self._reshape(target, self.features.target_shape[1])

        return target

    def train(self, data: pd.DataFrame) -> History:
        features = self.features.extract(data)
        return self._train(features)

    def _train(self,
               features: pd.DataFrame,
               shuffle: bool = True) -> History:

        kwargs = {
            'verbose': LOG_VERBOSE
        }
        data = self._parse_features(features)
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
        # for target in self.features.target_keys:
        #     loss = result.history['loss'] / features[target].max()
        #     loss_name = 'epoch_loss_norm' if len(self.features.target_keys) == 1 else '{}_norm'.format(target)
        #     for epoch in range(len(result.history['loss'])):
        #         with writer.as_default():
        #             summary.scalar(loss_name, loss[epoch], step=epoch)

        self._save()
        return result

    @staticmethod
    def _reshape(data: pd.Dataframe | np.ndarray, shape: Tuple | int) -> np.ndarray | float:
        if isinstance(data, pd.DataFrame):
            data = data.values

        return np.squeeze(data).reshape(shape)

    def _parse_batch(self, features: pd.DataFrame, index: int,
                     dates: List[pd.Timestamp]) -> Tuple[int, List[Dict[str, np.ndarray]]]:
        data = []
        for date in dates:
            try:
                input = self.features.input(features, date)
                input = self.features.scale(input)

                target = self.features.target(features, date)
                target = self.features.scale(target)

                # If no exception was raised, add the validated data to the set
                data.append({
                    'input': self._reshape(input, self.features.input_shape),
                    'target': self._reshape(target, self.features.target_shape[1])
                })
            except (KeyError, ValueError) as e:
                logger.debug("Skipping %s: %s", date, str(e))

        return index, data

    def _parse_features(self, features: pd.DataFrame) -> Dict[int, List[Dict[str, np.ndarray]]]:
        # TODO: implement additional method to bin data
        index = np.unique(features.index.isocalendar().week)
        dates = dict((i, []) for i in index)
        date = features.index[0] + self.features.resolutions[-1].time_prior
        while date <= features.index[-1]:
            date_index = date.week
            dates[date_index].append(date)
            date += dt.timedelta(minutes=self.features.resolutions[-1].minutes)

        data = {}
        steps = []
        with ProcessPoolExecutor() as executor:
            for step_index, step_dates in dates.items():
                step_features = features[step_dates[0] - self.features.resolutions[-1].time_prior:step_dates[-1]]
                steps.append(executor.submit(self._parse_batch, step_features, step_index, step_dates))

            for step in as_completed(steps):
                step_index, step_data = step.result()
                data[step_index] = step_data

        return data

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
                kwargs['input_dim'] = self.features.input_shape[1]

            self.model.add(Dense(units[i], **kwargs))

            if dropout > 0.:
                self.model.add(Dropout(dropout))

        self.model.add(Dense(self.features.target_shape[1],
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
                kwargs['input_shape'] = self.features.input_shape

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
                kwargs['input_shape'] = self.features.input_shape
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
