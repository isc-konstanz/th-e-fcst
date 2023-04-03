# -*- coding: utf-8 -*-
"""
    th_e_fcst.ann.features
    ~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Any, List, Callable

import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import holidays as hl
import logging

from copy import deepcopy
from corsys import Configurations, Configurable
from corsys.tools import to_bool
from ... import System, Resolutions

logger = logging.getLogger(__name__)


class Features(Configurable):

    @classmethod
    def build(cls, system: System, conf_file: str = 'features.cfg') -> Features:
        configs = Configurations.from_configs(system.configs, conf_file)
        features = cls(system, configs)
        features.__build__(configs)
        return features

    def __init__(self, system: System, configs: Configurations, *args, **kwargs) -> None:
        super().__init__(configs, *args, **kwargs)
        self._system = system

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

        self._estimate = to_bool(configs.get('General', 'estimate', fallback='true'))

        # TODO: Make this part of a wrapper Resolutions list class
        self._autoregressive = to_bool(configs.get('General', 'autoregressive', fallback='true'))

    def __build__(self, configs: Configurations) -> None:

        def parse_section(section: str, cast: Callable = str) -> dict[str, Any]:
            if configs.has_section(section):
                return dict({k: cast(v) for k, v in configs.items(section)})
            return {}

        self._scaling = parse_section('Scaling', float)
        self._cyclic = parse_section('Cyclic', float)
        self._doubt = parse_section('Doubt')
        for feature in self._doubt.keys():
            if feature in self._scaling.keys():
                self._scaling[feature+'_doubt'] = self._scaling[feature]

        self.target_keys = configs.get('Target', 'values').splitlines()
        self.input_keys = []

        def parse_cyclic_keys(config_keys: List[str]) -> List[str]:
            input_keys = []
            for key in config_keys:
                if key in self._cyclic.keys():
                    input_keys.append(key + '_cos')
                    input_keys.append(key + '_sin')
                else:
                    input_keys.append(key)
                self.input_keys.append(key)
            return input_keys

        self.input_series_keys = parse_cyclic_keys(configs.get('Input', 'series').splitlines())
        self.input_value_keys = parse_cyclic_keys(configs.get('Input', 'values').splitlines())

        if self._estimate:
            self.input_value_keys += self.input_series_keys

        input_series_count = len(self.input_series_keys + self.target_keys)
        input_value_count = len(self.input_value_keys)
        input_steps = self.resolutions.steps_prior

        self.input_shape = [(input_steps, input_series_count), (input_value_count,)]

        if self._autoregressive:
            target_steps = 1
        else:
            target_steps = self.resolutions.steps_horizon

        target_count = len(self.target_keys)

        self.target_shape = (target_steps, target_count)

    @property
    def resolutions(self) -> Resolutions:
        return self._system.forecast.resolutions

    @property
    def system(self) -> System:
        return self._system

    def input(self,
              features: pd.DataFrame,
              index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]

        # Convert to UTC before resampling, to avoid offset issues
        timezone = features.index.tzinfo
        data = self.resolutions[-1].resample(deepcopy(features.tz_convert(tz.utc)))
        data = self._add_doubt(data)
        data = self._add_meta(data)
        data = self._extract(data)
        data = self._parse_cyclic(data)

        inputs = pd.DataFrame()
        inputs.index.name = 'time'
        for resolution in self.resolutions:
            if resolution.steps_prior is None:
                continue

            resolution_end = index[-1].tz_convert(tz.utc)
            resolution_start = (index[0].tz_convert(tz.utc)
                                - resolution.time_step
                                - resolution.time_prior + dt.timedelta(minutes=1))
            resolution_offset = resolution_end.hour * 60 + resolution_end.minute
            resolution_inputs = resolution.resample(data[resolution_start:resolution_end],
                                                    offset=resolution_offset)

            inputs = resolution_inputs.combine_first(inputs)

        if inputs.isna().values.any() or len(inputs) < len(index):
            raise ValueError("Input data incomplete for %s" % index)

        # Make sure that no future target values exist
        inputs.loc[index[0]:, self.target_keys] = np.NaN

        return inputs.tz_convert(timezone)

    def target(self,
               features: pd.DataFrame,
               index: pd.DatetimeIndex | pd.Timestamp | dt.datetime) -> pd.DataFrame:

        if not isinstance(index, pd.DatetimeIndex):
            index = [self.resolutions[-1].time_step + index]

        # TODO: Implement horizon resolutions
        # Convert to UTC before resampling, to avoid offset issues
        timezone = features.index.tzinfo
        data = self.resolutions[-1].resample(deepcopy(features.tz_convert(tz.utc)))

        if self._autoregressive:
            targets = data.loc[index, self.target_keys]
        else:
            targets = pd.DataFrame()
            targets.index.name = 'time'
            for resolution in self.resolutions:
                if resolution.steps_horizon is None:
                    continue

                resolution_end = index[0].tz_convert(tz.utc) + resolution.time_horizon - dt.timedelta(minutes=1)
                resolution_start = index[0].tz_convert(tz.utc)
                resolution_offset = resolution_end.hour * 60 + resolution_end.minute
                resolution_targets = resolution.resample(data[resolution_start:resolution_end],
                                                         offset=resolution_offset)

                targets = resolution_targets.combine_first(targets)

        if targets.isnull().values.any():
            raise ValueError("Target data incomplete for %s" % index)

        return targets.tz_convert(timezone)

    def extract(self, data):
        return deepcopy(self._extract(data))

    def _extract(self, data):
        columns = self.target_keys + self.input_keys
        return data[np.intersect1d(data.columns, columns)]

    def scale(self, features, invert=False):
        if len(self._scaling) == 0:
            return features

        scaled_features = deepcopy(features)
        for feature, transformation in self._scaling.items():
            if feature not in scaled_features.columns:
                continue

            # if str(transformation).isdigit():
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
            # else:
            #     raise ValueError('The transformation "{}" is not defined.'.format(transformation))

        return scaled_features

    def _add_doubt(self, features):
        if len(self._doubt) == 0:
            return features

        for feature, feature_cor in self._doubt.items():
            if feature+'_doubt' in self.input_keys:
                features[feature+'_doubt'] = abs(features[feature] - features[feature_cor])
        return features

    def _add_meta(self, features):
        features['hour_of_day'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['day_of_year'] = features.index.dayofyear - 1

        if self.system.location.country is None:
            return features

        features['holiday'] = 0

        holiday_args = {
            'years': list(dict.fromkeys(features.index.year))
        }
        if self.system.location.state is not None:
            holiday_args['subdiv'] = self.system.location.state

        holidays = hl.country_holidays(self.system.location.country, **holiday_args)
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
