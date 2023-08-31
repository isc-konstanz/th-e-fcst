# -*- coding: utf-8 -*-
"""
    th_e_fcst.forecast.ann.tensorflow
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Tuple, List, Dict

import os
import json
import random
import numpy as np
import pandas as pd
import datetime as dt
import logging

from glob import glob
from copy import deepcopy
from itertools import chain
from corsys import System, Configurations
from corsys.tools import to_bool, to_float
from concurrent.futures import ProcessPoolExecutor, as_completed

from kerasbeats import NBeatsModel
from keras.optimizers import Adam
from keras.callbacks import History, EarlyStopping, TensorBoard
from keras.layers import Input, Concatenate, Flatten, Reshape, Dropout, LeakyReLU,\
                         Dense, Conv1D, MaxPooling1D, TimeDistributed, LSTM
from keras.models import load_model
from keras import Model
from ..base import Forecast
from . import Features

logger = logging.getLogger(__name__)

LOG_VERBOSE = 0

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TensorForecast(Forecast):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

        self.dir = os.path.join(configs.dirs.data, 'model')

        self.epochs = configs.getint(Configurations.GENERAL, 'epochs')
        self.batch = configs.getint(Configurations.GENERAL, 'batch')

    def __activate__(self, system: System) -> None:
        super().__activate__(system)

        self.features = Features.build(system)

        self.history = History()
        self.callbacks = [self.history]

        self._early_stopping = to_bool(self.configs.get(Configurations.GENERAL, 'early_stopping', fallback='True'))
        if self._early_stopping:
            self._early_stopping_bins = self.configs.get(Configurations.GENERAL, 'early_stopping_bins', fallback=None)
            if self._early_stopping_bins is not None:
                self._early_stopping_bins = self._early_stopping_bins.lower()
                if self._early_stopping_bins not in ['hour', 'day_of_year', 'day_of_week', 'week']:
                    raise ValueError("Unknown early stopping batch method: " + self._early_stopping_bins)

            self._early_stopping_split = self.configs.getint(Configurations.GENERAL,
                                                             'early_stopping_split', fallback=10)
            self._early_stopping_patience = self.configs.getint(Configurations.GENERAL,
                                                                'early_stopping_patience', fallback=self.epochs/4)
            self.callbacks.append(EarlyStopping(patience=self._early_stopping_patience, restore_best_weights=True))

        self._tensorboard = self.configs.getboolean(Configurations.GENERAL, 'tensorboard', fallback=False)
        if self._tensorboard:
            self.callbacks.append(TensorBoard(log_dir=self.dir, histogram_freq=1))

        # TODO: implement date based backups and naming scheme
        if self.is_trained():
            self.model = self._load_model()
        else:
            self.model = self._build_model(self.configs)

        self.model.compile(optimizer=self._parse_optimizer(self.configs),
                           metrics=self._parse_metrics(self.configs),
                           loss=self._parse_loss(self.configs))

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
        optimizer = configs.get(Configurations.GENERAL, 'optimizer')
        if optimizer == 'adam' and configs.has_option(Configurations.GENERAL, 'learning_rate'):
            optimizer = Adam(learning_rate=configs.getfloat(Configurations.GENERAL, 'learning_rate'))

        return optimizer

    @staticmethod
    def _parse_metrics(configs: Configurations):
        metrics = configs.get(Configurations.GENERAL, 'metrics', fallback=None)
        if not metrics:
            metrics = []
        elif metrics.isdigit():
            metrics = [int(metrics)]
        else:
            metrics = json.loads(metrics)
        return metrics

    @staticmethod
    def _parse_loss(configs: Configurations):
        return configs.get(Configurations.GENERAL, 'loss')

    # noinspection SpellCheckingInspection
    def _build_model(self, configs: Configurations | Dict) -> Model:
        # Loosely based on towardsdatascience guide: CNN+LSTM for Forecasting
        # https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826

        def has_configs(section: str) -> bool:
            if section in configs:
                if 'enabled' in configs[section]:
                    return to_bool(configs[section]['enabled'])
                return True
            return False

        def get_configs(section: str) -> dict:
            if section in configs:
                section = dict(configs[section])
                section.pop("enabled", None)
                return section
            return {}

        target_count = len(self.features.target_keys)

        inputs = []
        input_targets = []
        for i in range(target_count):
            input_shape = self.features.input_shape[i]
            input_target = Input(shape=input_shape)
            input_targets.append(input_target)
            inputs.append(input_target)

        input_series = Input(shape=self.features.input_shape[target_count])
        inputs.append(input_series)

        tensors = []
        if has_configs('Conv1D'):
            inputs_cnn = []
            for input_target in input_targets:
                inputs_cnn.append(Concatenate()([Reshape((1, *input_target.shape[1:], 1))(input_target),
                                                 Reshape((1, *input_series.shape[1:]))(input_series)]))
            inputs_cnn = Concatenate(axis=1)(inputs_cnn) if target_count > 1 else inputs_cnn[0]
            tensors_cnn = self._build_cnn(inputs_cnn, **get_configs('Conv1D'))
            tensors.append(Flatten(tensors_cnn))

            if has_configs('LSTM'):
                tensors_lstm = self._build_lstm(tensors_cnn, **get_configs('LSTM'))
                tensors.append(tensors_lstm)

        if has_configs('N-BEATS'):
            for input_target in input_targets:
                beats_shape = (*input_target.shape[1:], self.features.target_shape[0])
                tensors_beats = self._build_nbeats(input_target, *beats_shape, **get_configs('N-BEATS'))
                tensors.append(tensors_beats)

        input_values = Input(shape=self.features.input_shape[target_count+1])
        inputs.append(input_values)
        tensors.append(input_values)
        tensors = Concatenate()(tensors)

        tensors = self._build_dense(tensors, **get_configs('Dense'))

        outputs = self._build_output(tensors, self.features.target_shape, **get_configs('Output'))
        model = Model(inputs=inputs, outputs=outputs, name='th-e-forecast')

        # TODO: implement date based backups and naming scheme
        return model

    @staticmethod
    def _build_cnn(x,
                   flatten: bool = True,
                   filters: int | str | List[int] = 64,
                   layers: int = 3,
                   padding: str = 'causal',
                   activation: str = 'relu',
                   kernel_size: int = 3,
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
            cnn_args = deepcopy(kwargs)
            if i == 0:
                cnn_args['dilation_rate'] = 1
                cnn_args['batch_input_shape'] = tuple(x.shape)
            else:
                cnn_args['dilation_rate'] = 2**i

            x = TimeDistributed(Conv1D(filters[i], int(kernel_size), **cnn_args))(x)

        if pool_size > 0:
            x = TimeDistributed(MaxPooling1D(pool_size))(x)

        if flatten:
            x = TimeDistributed(Flatten())(x)

        return x

    @staticmethod
    def _build_lstm(x,
                    units: int | str | List[int] = 64,
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
        # kwargs['stateful'] = True

        length = len(units)
        for i in range(length):
            lstm_args = deepcopy(kwargs)
            if i < length-1:
                lstm_args['return_sequences'] = True

            x = LSTM(units[i], **lstm_args)(x)

        return x

    # noinspection SpellCheckingInspection
    @staticmethod
    def _build_nbeats(x,
                      steps: int,
                      horizon: int,
                      model: str = 'generic',
                      **kwargs):
        nbeats = NBeatsModel(model_type=model,
                             lookback=int(steps/horizon),
                             horizon=horizon, **kwargs)
        nbeats.build_layer()
        return nbeats.model_layer(x)

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
                      leaky_alpha: float = 1e-3,
                      **kwargs):

        x = LeakyReLU(alpha=to_float(leaky_alpha))(x)
        outputs = [Dense(shape[0], **kwargs)(x)]

        return outputs

    def _load_model(self, from_json: bool = False) -> Model:
        logger.debug("Loading model for system {} from file".format(self.system.name))
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
        save_date = dt.datetime.now(self.system.location.timezone)
        logger.debug("Saving model for system {} to file".format(self.system.name))

        # Serialize weights checkpoint
        self.model.save(self.dir)
        self.model.save_weights(os.path.join(self.dir, f"model-SNAPSHOT{save_date.strftime('%Y%m%d')}.h5"))

    @property
    def active(self) -> bool:
        return self.is_trained()

    def predict(self,
                data: pd.DataFrame,
                date: pd.Timestamp | dt.datetime) -> pd.DataFrame:

        features = self.features.validate(data)
        return self._predict(features, date)

    # noinspection PyShadowingBuiltins
    def _predict(self,
                 features: pd.DataFrame,
                 date: pd.Timestamp | dt.datetime):

        # resolution_max = self.features.resolutions[0]
        resolution_min = self.features.resolutions[0]
        if len(self.features.resolutions) > 1:
            for i in range(len(self.features.resolutions) - 1, 0, -1):
                resolution_min = self.features.resolutions[i]
                if resolution_min.steps_horizon is not None:
                    break

        targets = pd.DataFrame(columns=self.features.target_keys, dtype=float)

        if resolution_min.steps_horizon is None:
            end = features.index[-1]
        else:
            end = date + resolution_min.time_horizon

        # Work on copy to retain original input vector to be saved.
        features = deepcopy(features)

        # Remove target values from features, as those will be recursively filled with predictions
        features.loc[features.index > date, self.features.target_keys] = np.NaN

        while date < end:
            date_step = date + resolution_min.time_step
            if date_step not in features.index:
                break

            input = self.features.input(features, date)
            input = self.features.scale(input)
            target = self._predict_step(input, date)

            # Do not normalize targets!
            # target = self.features.scale(pd.DataFrame(target, columns=self.features.target_keys), invert=True)

            targets.loc[date_step, self.features.target_keys] = target  # .values

            # Add predicted output to features of next iteration
            features.loc[(features.index >= date_step) &
                         (features.index < date_step + resolution_min.time_step),
                         self.features.target_keys] = target  # .values

            date = date_step

        return targets

    # noinspection PyShadowingBuiltins
    def _predict_step(self,
                      input: pd.DataFrame,
                      date: pd.Timestamp | dt.datetime) -> np.ndarray | float:
        input_shape = [tuple(chain.from_iterable(
            (list([1]), list(self.features.input_shape[i]))))
            for i in range(len(self.features.input_shape))]
        input = self._parse_input(input, date, input_shape)

        target = self.model(input)  # .predict(input, verbose=LOG_VERBOSE)
        target = np.squeeze(target).reshape(self.features.target_shape)

        return target

    def is_trained(self) -> bool:
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

    def train(self, data: pd.DataFrame) -> History:
        features = self.features.validate(data)
        return self._train(features)

    def _train(self,
               training_features: pd.DataFrame,
               validation_features: pd.DataFrame = None,
               threading: bool = False) -> History:
        kwargs = {
            'verbose': LOG_VERBOSE
        }
        inputs = [[] for _ in range(len(self.features.input_shape))]
        targets = []
        if not self._early_stopping or \
                (validation_features is not None and not validation_features.empty):

            data = self._parse_features(training_features)
            random.shuffle(data)
            for i in range(len(data)):
                for s in range(len(data[i]['input'])):
                    inputs[s].append(data[i]['input'][s])
                targets.append(data[i]['target'])

            if validation_features is not None and not validation_features.empty:
                validation_data = self._parse_features(validation_features)
                validation_inputs = [[] for _ in range(len(inputs))]
                validation_targets = []

                random.shuffle(validation_data)
                for i in range(len(validation_data)):
                    for s in range(len(validation_data[i]['input'])):
                        validation_inputs[s].append(validation_data[i]['input'][s])
                    validation_targets.append(validation_data[i]['target'])

                validation_inputs = [np.array(validation_input, dtype=float) for validation_input in validation_inputs]
                validation_targets = np.array(validation_targets, dtype=float)
                kwargs['validation_data'] = (validation_inputs, validation_targets)
        else:
            validation_inputs = [[] for _ in range(len(inputs))]
            validation_targets = []

            data = self._parse_batches(training_features, threading)
            for d in data:
                split = len(d) - int(len(d)/self._early_stopping_split)
                random.shuffle(d)

                for i in range(len(d)):
                    if i <= split:
                        for s in range(len(d[i]['input'])):
                            inputs[s].append(d[i]['input'][s])
                        targets.append(d[i]['target'])
                    else:
                        for s in range(len(d[i]['input'])):
                            validation_inputs[s].append(d[i]['input'][s])
                        validation_targets.append(d[i]['target'])

            validation_inputs = [np.array(validation_input, dtype=float) for validation_input in validation_inputs]
            validation_targets = np.array(validation_targets, dtype=float)
            kwargs['validation_data'] = (validation_inputs, validation_targets)

        result = self.model.fit([np.array(input, dtype=float) for input in inputs],
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

    def _parse_input(self,
                     data: pd.Dataframe,
                     date: pd.Timestamp | dt.datetime,
                     shape: List | Tuple | int) -> List[np.ndarray]:

        def squeeze(input_data: pd.DataFrame, input_shape: Tuple) -> np.ndarray:
            return np.squeeze(input_data.values).reshape(input_shape)

        inputs = []
        input_count = 0
        for target in self.features.target_keys:
            inputs.append(squeeze(data.loc[data.index <= date, target], shape[input_count]))
            input_count += 1
        inputs.append(squeeze(data.loc[data.index <= date, self.features.input_series_keys], shape[input_count]))
        inputs.append(squeeze(data.loc[data.index > date, self.features.input_value_keys], shape[input_count+1]))
        return inputs

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
                # Do not normalize targets!
                # target = self.features.scale(target)

                # If no exception was raised, add the validated data to the set
                data.append({
                    'input': self._parse_input(input, date, self.features.input_shape),
                    'target': self._parse_target(target, date, self.features.target_shape[1])
                })
            except (KeyError, ValueError) as e:
                logger.warning("Skipping %s: %s", date, str(e))

        return data

    def _parse_batches(self, features: pd.DataFrame, threading: bool = False) -> List[List[Dict[str, np.ndarray]]]:

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
            return features[batch_dates[0] - self.features.resolutions[0].time_prior:
                            batch_dates[-1] + self.features.resolutions[-1].time_step]

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
        date = features.index[0] + self.features.resolutions[0].time_prior
        dates = []
        while date <= features.index[-1]:
            dates.append(date)
            date += dt.timedelta(minutes=self.features.resolutions[-1].minutes)
        return dates
