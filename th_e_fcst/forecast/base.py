# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast.base
    ~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd
import datetime as dt
from corsys import System, Configurable, Configurations
from corsys.tools import floor_date, ceil_date
from .. import Resolutions


class Forecast(ABC, Configurable):

    @classmethod
    def read(cls, system: System, conf_file: str = 'forecast.cfg') -> Forecast:
        configs = Configurations.from_configs(system.configs, conf_file)
        model = configs.get('General', 'model', fallback='default').lower()
        if model in ['ann', 'neuralnetwork']:
            from .ann import TensorForecast
            return TensorForecast(system, configs)
        if model in ['pv', 'solar']:
            from .pv import PVForecast
            return PVForecast(system, configs)
        if model == 'database':
            from .db import DatabaseForecast
            return DatabaseForecast(system, configs)
        if model == 'default':
            from .default import DefaultForecast
            return DefaultForecast(system, configs)

        raise TypeError('Invalid forecast model: {}'.format(model))

    def __init__(self, system: System, configs: Configurations, *args, **kwargs) -> None:
        super().__init__(configs, *args, **kwargs)
        self.system = system
        self._active = False

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self._resolutions = Resolutions.read(configs)

    def __activate__(self, system: System) -> None:
        self._active = True

    def __call__(self,
                 start: pd.Timestamp | dt.datetime = pd.Timestamp.now(),
                 end:   Optional[pd.Timestamp | dt.datetime] = None,
                 data:  Optional[pd.DataFrame] = None,
                 *args, **kwargs) -> pd.DataFrame:
        """
        Retrieves the forecasted data for a specified time interval

        :param start:
            the start time for which forecasted data will be looked up for.
            For many applications, passing datetime.datetime.now() will suffice.
        :type start:
            :class:`pandas.Timestamp` or datetime

        :param end:
            the end time for which forecasted data will be looked up for.
        :type end:
            :class:`pandas.Timestamp` or datetime

        :param data:
            Optional data, a prediction can be based on, e.g. a weather forecast,
            if available.
        :type data:
            :class:`pandas.DataFrame` or None

        :returns:
            the forecasted data, indexed in a specific time interval.

        :rtype:
            :class:`pandas.DataFrame`
        """
        if data is not None and not data.empty:
            data = self._validate_resolution(data)
        forecast = self.predict(start, end, data=data, *args, **kwargs)

        if data is not None and not data.empty:
            forecast = pd.concat([forecast, data], axis='columns')
        return self._get_range(forecast, start, end)

    @property
    def resolutions(self) -> Resolutions:
        return self._resolutions

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self.__activate__(self.system)

    def _validate_resolution(self, data: pd.DataFrame) -> pd.DataFrame:
        horizon = self.resolutions.get_horizon(how='min')
        horizon_freq = f'{horizon.minutes}T'
        horizon_start = floor_date(data.index[0], freq=horizon_freq)
        horizon_end = ceil_date(data.index[-1], freq=horizon_freq)
        horizon_index = pd.date_range(horizon_start, horizon_end, freq=horizon_freq)

        if not all([i in data.index for i in horizon_index]):
            data = data.combine_first(pd.DataFrame(index=horizon_index))
            if data.isna().values.any():
                data.interpolate(method='linear', inplace=True)
                # data.interpolate(method='akima', inplace=True)
            return data.loc[horizon_index, :]
        return data

    def _get_range(self,
                   forecast: pd.DataFrame,
                   start:    pd.Timestamp | dt.datetime,
                   end:      pd.Timestamp | dt.datetime) -> pd.DataFrame:
        if start is None or start < forecast.index[0]:
            start = forecast.index[0]
        if end is None or end > forecast.index[-1]:
            end = forecast.index[-1]
        return forecast[(forecast.index >= start) & (forecast.index <= end)]

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        pass


class ForecastException(Exception):
    """
    Raise if an error occurred while forecasting.

    """
    pass
