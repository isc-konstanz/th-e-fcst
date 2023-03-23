# -*- coding: utf-8 -*-
"""
    th_e_fcst.ann.model
    ~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Any, Optional, Tuple, List, Dict

import os
import json
import random
import numpy as np
import pandas as pd
import datetime as dt
import logging

from glob import glob
from copy import deepcopy
from corsys import System, Configurations
from concurrent.futures import ProcessPoolExecutor, as_completed

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Concatenate, Flatten, Reshape, Dense, Dropout, \
                                    LeakyReLU, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from ..base import Forecast
from . import Features

logger = logging.getLogger(__name__)

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork(Forecast):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

        self.dir = os.path.join(configs.dirs.data, 'model')

        self.epochs = configs.getint('General', 'epochs')
        self.batch = configs.getint('General', 'batch')

    def __build__(self, system: System, configs: Configurations) -> None:
        super().__build__(system, configs)

        self.history = History()
        self.callbacks = [self.history]

        self._early_stopping = configs.get('General', 'early_stopping', fallback='True').lower() == 'true'
        if self._early_stopping:
            self._early_stopping_bins = configs.get('General', 'early_stopping_bins', fallback=None)
            if self._early_stopping_bins is not None:
                self._early_stopping_bins = self._early_stopping_bins.lower()
                if self._early_stopping_bins not in ['hour', 'day_of_year', 'day_of_week', 'week']:
                    raise ValueError("Unknown early stopping batch method: " + self._early_stopping_bins)

            self._early_stopping_split = configs.getint('General', 'early_stopping_split', fallback=10)
            self._early_stopping_patience = configs.getint('General', 'early_stopping_patience', fallback=self.epochs/4)
            self.callbacks.append(EarlyStopping(patience=self._early_stopping_patience, restore_best_weights=True))

        self._tensorboard = configs.getboolean('General', 'tensorboard', fallback=False)
        if self._tensorboard:
            self.callbacks.append(TensorBoard(log_dir=self.dir, histogram_freq=1))

        self.features = Features.read(system)

        # TODO: implement date based backups and naming scheme
        if self.exists():
            self.model = self._load_model()
        else:
            self.model = self._build_model(configs)

        self.model.compile(optimizer=self._parse_optimizer(configs),
                           metrics=self._parse_metrics(configs),
                           loss=self._parse_loss(configs))

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

    def _build_model(self, configs: Configurations) -> Model:
        input_series = Input(shape=self.features.input_shape[0])
        input_values = Input(shape=self.features.input_shape[1])

        x = input_series
        if configs.has_section('Conv1D'):
            x = self._build_conv(input_series, **configs['Conv1D'])

        if configs.has_section('LSTM'):
            x = self._build_lstm(x, **configs['LSTM'])

        x = Concatenate()([Flatten()(x),
                           Flatten()(input_values)])

        x = self._build_dense(x, **configs['Dense'])

        outputs = self._build_output(x, self.features.target_shape)
        model = Model(inputs=[input_series, input_values],
                      outputs=outputs,
                      name='TH-E-Forecast')

        # TODO: implement date based backups and naming scheme
        return model

    @staticmethod
    def _build_conv(x,
                    filters: int | str | List[int] = 32,
                    layers: int = 3,
                    padding: str = 'causal',
                    activation: str = 'relu',
                    kernel_size: int = 2,
                    pool_size: int = 0,
                    **kwargs):

        if isinstance(filters, str):
            if filters.isdigit():
                filters = int(filters)
            else:
                filters = json.loads(filters)
        if isinstance(filters, int):
            filters = [filters] * int(layers)

        if isinstance(pool_size, str):
            pool_size = int(pool_size)

        kwargs['activation'] = activation
        kwargs['padding'] = padding

        # TODO: Handle dilation in configuration
        length = len(filters)
        for i in range(length):
            if i == 0:
                kwargs['dilation_rate'] = 1
            else:
                kwargs['dilation_rate'] = 2**i

            x = Conv1D(filters[i], int(kernel_size), **kwargs)(x)

        if pool_size > 0:
            x = MaxPooling1D(pool_size)(x)

        return x

    @staticmethod
    def _build_lstm(x,
                    units: int | str | List[int] = 32,
                    layers: int = 1,
                    activation: str = 'relu',
                    **kwargs):
        if isinstance(units, str):
            if units.isdigit():
                units = int(units)
            else:
                units = json.loads(units)
        if isinstance(units, int):
            units = [units] * int(layers)

        kwargs['activation'] = activation

        length = len(units)
        for i in range(length):
            if i < length-1:
                kwargs['return_sequences'] = True

            x = LSTM(units[i], **kwargs)(x)
        return x

    @staticmethod
    def _build_dense(x,
                     units: int | str | List[int] = 32,
                     layers: int = 3,
                     dropout: int = 0,
                     activation: str = 'relu',
                     **kwargs):
        if isinstance(units, str):
            if units.isdigit():
                units = int(units)
            else:
                units = json.loads(units)
        if isinstance(units, int):
            units = [units] * int(layers)

        if isinstance(dropout, str):
            dropout = float(dropout)

        kwargs['activation'] = activation

        length = len(units)
        for i in range(length):
            x = Dense(units[i], **kwargs)(x)

            if dropout > 0.:
                x = Dropout(dropout)(x)
        return x

    @staticmethod
    def _build_output(x,
                      shape: tuple,
                      activation: str = 'relu',
                      leaky_alpha: float = 1e-6,
                      **kwargs):
        if isinstance(leaky_alpha, str):
            leaky_alpha = float(leaky_alpha)

        kwargs['activation'] = activation

        x = Dense(np.prod(shape), **kwargs)(x)

        if activation == 'relu':
            x = LeakyReLU(alpha=leaky_alpha)(x)
        return Reshape(shape)(x)

    def _load_model(self, from_json: bool = False) -> Model:
        logger.debug("Loading model for system {} from file".format(self._system.name))
        if (glob(os.path.join(self.dir, 'variables', 'variables.data*')) and
                os.path.isfile(os.path.join(self.dir, 'variables', 'variables.index')) and
                os.path.isfile(os.path.join(self.dir, 'saved_model.pb')) and
                os.path.isfile(os.path.join(self.dir, 'keras_metadata.pb'))):

            model = load_model(os.path.join(self.dir))
        else:
            model = self._build_model(self._configs)

            if (glob(os.path.join(self.dir, 'checkpoint*')) and
                    glob(os.path.join(self.dir, 'model.data*')) and
                    os.path.isfile(os.path.join(self.dir, 'model.index'))):
                model.load_weights(os.path.join(self.dir, 'model'))
            else:
                model.load_weights(os.path.join(self.dir, 'model.h5'))
        return model

    def _save(self) -> None:
        logger.debug("Saving model for system {} to file".format(self._system.name))

        # Serialize weights checkpoint
        self.model.save(self.dir)

    def exists(self):
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            return False

        if (glob(os.path.join(self.dir, 'variables', 'variables.data*')) and
                os.path.isfile(os.path.join(self.dir, 'variables', 'variables.index')) and
                os.path.isfile(os.path.join(self.dir, 'saved_model.pb')) and
                os.path.isfile(os.path.join(self.dir, 'keras_metadata.pb'))):
            return True

        if ((glob(os.path.join(self.dir, 'checkpoint*')) and
                glob(os.path.join(self.dir, 'model.data*')) and
                os.path.isfile(os.path.join(self.dir, 'model.index'))) or
                os.path.isfile(os.path.join(self.dir, 'model.h5'))):
            return True

        return False

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
            date_step = date + resolution.time_step
            if date_step not in features.index:
                break

            input = self.features.input(features, date)
            input = self.features.scale(input)
            target = self._predict_step(input)
            target = self.features.scale(pd.DataFrame(target, columns=self.features.target_keys), invert=True)
            targets.loc[date_step, self.features.target_keys] = target.values

            # Add predicted output to features of next iteration
            features.loc[(features.index >= date_step) &
                         (features.index < date_step + resolution.time_step), self.features.target_keys] = target.values

            date = date_step

        return targets

    def _predict_step(self, input: pd.DataFrame) -> np.ndarray | float:
        # input_shape = (1, self.features.input_shape[0], self.features.input_shape[1])
        input = self._parse_input(input, self.features.input_shape)

        target = self.model(input)  # .predict(input, verbose=LOG_VERBOSE)
        target = self._parse_target(target, self.features.target_shape)

        return target

    def train(self, data: pd.DataFrame) -> History:
        features = self.features.extract(data)
        return self._train(features)

    def _train(self, features: pd.DataFrame, validation_features: pd.DataFrame = None) -> History:
        kwargs = {
            'verbose': LOG_VERBOSE
        }
        input_series = []
        input_values = []
        target_values = []
        if not self._early_stopping or \
                (validation_features is not None and not validation_features.empty):

            data = self._parse_features(features)
            random.shuffle(data)
            for i in range(len(data)):
                input_series.append(data[i]['input'][0])
                input_values.append(data[i]['input'][1])
                target_values.append(data[i]['target'])

            if validation_features is not None and not validation_features.empty:
                validation_data = self._parse_features(validation_features)
                validation_input_series = []
                validation_input_values = []
                validation_target_values = []

                random.shuffle(validation_data)
                for i in range(len(validation_data)):
                    validation_input_series.append(validation_data[i]['input'][0])
                    validation_input_values.append(validation_data[i]['input'][1])
                    validation_target_values.append(validation_data[i]['target'])

                kwargs['validation_data'] = ([np.array(validation_input_series, dtype=float),
                                              np.array(validation_input_values, dtype=float)],
                                             np.array(validation_target_values, dtype=float))
        else:
            data = self._parse_batches(features)

            validation_input_series = []
            validation_input_values = []
            validation_target_values = []

            for d in data:
                split = len(d) - int(len(d)/self._early_stopping_split)
                random.shuffle(d)

                for i in range(len(d)):
                    if i <= split:
                        input_series.append(d[i]['input'][0])
                        input_values.append(d[i]['input'][1])
                        target_values.append(d[i]['target'])
                    else:
                        validation_input_series.append(d[i]['input'][0])
                        validation_input_values.append(d[i]['input'][1])
                        validation_target_values.append(d[i]['target'])

            kwargs['validation_data'] = ([np.array(validation_input_series, dtype=float),
                                          np.array(validation_input_values, dtype=float)],
                                         np.array(validation_target_values, dtype=float))

        result = self.model.fit([np.array(input_series, dtype=float),
                                 np.array(input_values, dtype=float)],
                                np.array(target_values, dtype=float),
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

    def _parse_input(self,
                     data: pd.Dataframe,
                     date: pd.Timestamp | dt.datetime,
                     shape: Tuple | int) -> List[np.ndarray, np.ndarray]:
        input_series = data.loc[data.index <= date, self.features.target_keys + self.features.input_series_keys]
        input_values = data.loc[data.index > date, self.features.input_value_keys]
        return [np.squeeze(input_series.values).reshape(shape[0]),
                np.squeeze(input_values.values).reshape(shape[1])]

    def _parse_target(self,
                      data: pd.Dataframe,
                      date: pd.Timestamp | dt.datetime,
                      shape: Tuple | int) -> np.ndarray | float:
        target = data.loc[data.index > date, self.features.target_keys]
        return np.squeeze(target.values).reshape(shape)

    def _parse_features(self,
                        features: pd.DataFrame,
                        dates: List[pd.Timestamp] = None) -> List[Dict[str, np.ndarray]]:
        data = []
        if dates is None:
            dates = self._parse_dates(features)
        for date in dates:
            try:
                input = self.features.input(features, date)
                input = self.features.scale(input)

                target = self.features.target(features, date)
                target = self.features.scale(target)

                # If no exception was raised, add the validated data to the set
                data.append({
                    'input': self._parse_input(input, date, self.features.input_shape),
                    'target': self._parse_target(target, date, self.features.target_shape[1])
                })
            except (KeyError, ValueError) as e:
                logger.debug("Skipping %s: %s", date, str(e))

        return data

    def _parse_batches(self, features: pd.DataFrame) -> List[List[Dict[str, np.ndarray]]]:

        # noinspection PyShadowingNames
        def parse_batch_index(date: pd.Timestamp) -> int:
            method = self._early_stopping_bins
            if method is None:
                return 0
            elif method == 'hour':
                return date.hour
            elif method == 'day_of_year':
                return date.dayofyear
            elif method == 'day_of_week':
                return date.dayofweek
            elif method == 'week':
                return date.week

        data = {}
        batches = {}
        for date in self._parse_dates(features):
            index = parse_batch_index(date)
            if index not in batches.keys():
                batches[index] = []
            batches[index].append(date)

        # noinspection PyShadowingNames
        def batch_append(batch_index: int, batch_data: List[Dict[str, np.ndarray]]) -> None:
            if len(batch_data) > 0:
                data[batch_index] = batch_data
            else:
                logger.debug("Skipping empty batch index: %s", batch_index)

        # noinspection PyShadowingNames
        def extract_features(batch_dates: List[dt.datetime]) -> pd.DataFrame:
            extract_resolution = self.features.resolutions[-1]
            return features[batch_dates[0] - extract_resolution.time_prior:
                            batch_dates[-1] + extract_resolution.time_step]

        threading = True  # FIXME: Make configurable
        if threading:
            futures = {}
            with ProcessPoolExecutor(max(os.cpu_count() - 1, 1)) as executor:
                for batch_index, batch_dates in batches.items():
                    future = executor.submit(self._parse_features, extract_features(batch_dates), batch_dates)
                    futures[future] = batch_index

                for future in as_completed(futures):
                    batch_index = futures[future]
                    batch_append(batch_index, future.result())
        else:
            for batch_index, batch_dates in batches.items():
                batch_append(batch_index, self._parse_features(extract_features(batch_dates), batch_dates))

        return list(data.values())

    def _parse_dates(self, features: pd.DataFrame) -> List[pd.Timestamp]:
        date = features.index[0] + self.features.resolutions[-1].time_prior
        dates = []
        while date <= features.index[-1]:
            dates.append(date)
            date += dt.timedelta(minutes=self.features.resolutions[-1].minutes)
        return dates
