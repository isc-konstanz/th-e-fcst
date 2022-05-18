# -*- coding: utf-8 -*-
"""
    th_e_fcst.ann.features
    ~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import datetime as dt
import holidays as hl
import logging

from copy import deepcopy
from typing import Any, Tuple, List, Dict
from pandas.tseries.frequencies import to_offset
from configparser import ConfigParser as Configurations
from th_e_core import Configurable, System

logger = logging.getLogger(__name__)


class Features(Configurable):

    def __init__(self, system: System, configs: Configurations) -> None:
        Configurable.__init__(self, configs)
        self._system = system

    def _configure(self, configs: Configurations, **kwargs) -> None:
        super()._configure(configs)

        self.resolutions = list()
        resolutions_range = pd.date_range(dt.datetime.now().replace(minute=0, second=0, microsecond=0), periods=0)
        for resolution_configs in [s for s in configs.sections() if s.lower().startswith('resolution')]:
            resolution = Resolution(**dict(configs.items(resolution_configs)))

            resolution_start = dt.datetime.now().replace(minute=0, second=0, microsecond=0) - resolution.time_prior
            resolutions_range = resolutions_range.union(pd.date_range(resolution_start,
                                                                      periods=resolution.steps_prior,
                                                                      freq='{}min'.format(resolution.minutes)))

            self.resolutions.append(resolution)

        def parse_feature(key: str, fallback=None) -> Any:
            config = configs.get('Features', key, fallback=None)
            if config is None:
                return fallback
            return json.loads(config)

        self._doubt_interval = configs.getint('Features', 'doubt_interval', fallback=60)
        self._doubt = parse_feature('doubt', fallback={})
        self._cyclic = parse_feature('cyclic', fallback={})
        self._scaling = parse_feature('scaling', fallback={})

        for feature in self._doubt.keys():
            if feature in self._scaling.keys():
                self._scaling[feature+'_doubt'] = self._scaling[feature]

        self.input_keys = parse_feature('input')
        self.target_keys = parse_feature('target')

        input_steps = len(resolutions_range)

        self._estimate = configs.get('Features', 'estimate', fallback='true').lower() == 'true'
        if self._estimate:
            input_steps += 1

        input_features = self.target_keys + self.input_keys
        input_count = len(input_features)
        for feature, _ in self._cyclic.items():
            if feature in input_features:
                input_count += 1

        self.input_shape = (input_steps, input_count)

        target_steps = 1
        target_count = len(self.target_keys)
        self.target_shape = (target_steps, target_count)

    def input(self,
              features: pd.DataFrame,
              index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]
        data = self.resolutions[-1].resample(deepcopy(features))

        # TODO: Optionally replace the estimate with the prediction of an ANN
        if self._estimate:
            # Make sure that no future target values exist
            data.loc[index[0]:, self.target_keys] = np.NaN

            # TODO: Implement horizon resolutions
            estimate = data.loc[index, self.target_keys]
            for estimate_step in estimate.index:
                estimate_value = data.loc[(data.index < index[0]) &
                                          (data.index.hour == estimate_step.hour) &
                                          (data.index.minute == estimate_step.minute),
                                          self.target_keys].mean(axis=0)
                if estimate_value.empty:
                    continue

                estimate.loc[estimate_step, :] = estimate_value.values

            # Estimate the value of target for times corresponding to the hour to the time to be predicted
            if self._system.contains_type('pv') and 'pv_yield' in self.input_keys:
                # Use calculated PV yield values as PV estimate
                estimate.loc[index, 'pv_power'] = data.loc[index, 'pv_yield']

            if estimate.isnull().values.any():
                estimate = features[self.target_keys].interpolate(method='linear').loc[index]

            data.loc[index, self.target_keys] = estimate

        data = self._add_doubt(data, index)

        inputs = pd.DataFrame()
        inputs.index.name = 'time'
        for resolution in self.resolutions:
            resolution_end = index[0] - resolution.time_step if not self._estimate else index[-1]
            resolution_start = index[0] - resolution.time_step - resolution.time_prior + dt.timedelta(minutes=1)
            resolution_inputs = resolution.resample(data[resolution_start:resolution_end])

            inputs = resolution_inputs.combine_first(inputs)

        inputs = self._add_meta(inputs)[self.target_keys + self.input_keys]
        inputs = self._parse_cyclic(inputs)

        if inputs.isnull().values.any():
            raise ValueError("Input data incomplete for %s" % index)

        return inputs

    def target(self,
               features: pd.DataFrame,
               index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]

        # TODO: Implement horizon resolutions
        data = self.resolutions[-1].resample(deepcopy(features))

        targets = data.loc[index, self.target_keys]
        if targets.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % index)

        return targets

    def extract(self, data):
        columns = self.target_keys + self.input_keys
        features = deepcopy(data[np.intersect1d(data.columns, columns)])

        return features

    def scale(self, features, invert=False):
        if len(self._scaling) == 0:
            return features

        scaled_features = deepcopy(features)
        for feature, transformation in self._scaling.items():
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
            #         scaled_features[feature] /= scaled_features[feature].max()
            #     else:
            #         scaled_features[feature] *= scaled_features[feature].max()
            #
            # elif transformation.lower() == 'std':
            #     mean = scaled_features[feature].mean()
            #     std = scaled_features[feature].std()
            #
            #     if not invert:
            #         scaled_features[feature] = (scaled_features[feature] - mean) / std
            #     else:
            #         scaled_features[feature] = scaled_features[feature] * std + mean
            else:
                raise ValueError('The transformation "{}" is not defined.'.format(transformation))

        return scaled_features

    def _add_doubt(self, features, times=None):
        if len(self._doubt) == 0:
            return features

        if times is None:
            times = features.index

        for feature, feature_cor in self._doubt.items():
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
        if len(self._cyclic) == 0:
            return features

        cyclic_features = deepcopy(features)
        for feature, bound in self._cyclic.items():
            if feature not in cyclic_features.columns:
                continue

            cyclic_index = cyclic_features.columns.get_loc(feature)
            cyclic_features.insert(cyclic_index, feature + '_cos', np.cos(2.0 * np.pi * features[feature] / bound))
            cyclic_features.insert(cyclic_index, feature + '_sin', np.sin(2.0 * np.pi * features[feature] / bound))
            cyclic_features = cyclic_features.drop(columns=[feature])

        return cyclic_features


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
