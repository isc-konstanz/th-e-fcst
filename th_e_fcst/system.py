# -*- coding: utf-8 -*-
"""
    th-e-fcst.system
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Optional, Dict

import logging
import numpy as np
import pandas as pd
import datetime as dt
import pvsys
from corsys import Component, Configurations
from corsys.io import DatabaseUnavailableException
from corsys.weather import WeatherUnavailableException
from corsys.tools import floor_date
from corsys.cmpt import Photovoltaics
from .forecast import Forecast

logger = logging.getLogger(__name__)


class System(pvsys.System):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self._forecast = Forecast.read(self)

    def __activate__(self, components: Dict[str, Component]) -> None:
        super().__activate__(components)
        self._forecast.activate()

    def __build__(self, **kwargs) -> Optional[pd.DataFrame]:
        data = super().__build__(**kwargs)
        if data is None or data.empty:
            return None

        if Photovoltaics.POWER not in data.columns:
            cmpts_input = super()._validate_input(data)
            cmpts_pv = [cmpt for cmpt in self.values() if cmpt.type == 'pv']
            if len(cmpts_pv) > 0:
                data[Photovoltaics.POWER] = 0
                for cmpt in cmpts_pv:
                    input_pv = self._get_solar_yield(cmpt, cmpts_input)
                    data[Photovoltaics.POWER] += input_pv[Photovoltaics.POWER].abs()

                data_time = pd.DataFrame(index=data.index, data=data.index)
                data_time.columns = ['date']
                data_time['hours'] = ((data_time['date'] - data_time['date'].shift(1)) / np.timedelta64(1, 'h')).bfill()
                data[Photovoltaics.ENERGY] = (data[Photovoltaics.POWER] / 1000 * data_time['hours']).cumsum()

        return data

    # noinspection PyShadowingBuiltins
    def __call__(self, date: pd.Timestamp | dt.datetime = None, **kwargs) -> pd.DataFrame:
        if date is None:
            date = pd.Timestamp.now(tz=self.location.timezone)
        date = floor_date(date)

        data_start = date - self.forecast.resolutions.get_prior(how='max').time_prior + dt.timedelta(minutes=1)
        data_end = date
        data = self._get_data(data_start, data_end)

        forecast_start = data_end + self.forecast.resolutions.get_horizon(how='min').time_step
        forecast_end = forecast_start + self.forecast.resolutions.get_horizon(how='max').time_horizon
        try:
            input = self._get_input(forecast_start, forecast_end, **kwargs)
            data = pd.concat([data, input], axis=1)

        except WeatherUnavailableException as e:
            logger.debug(str(e))

        forecast = self.forecast.get(forecast_start, forecast_end, data)
        return forecast.combine_first(data)

    @property
    def forecast(self) -> Forecast:
        return self._forecast

    # noinspection PyUnresolvedReferences, PyShadowingBuiltins
    def _get_input(self, *args, **kwargs) -> pd.DataFrame:
        input = super()._get_input(*args, **kwargs)

        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_solar_yield(cmpt, input)
                result['pv_yield'] += result_pv['pv_power'].abs()

        return input

    # noinspection PyTypeChecker, PyShadowingBuiltins
    def _get_data(self,
                  start: pd.Timestamp | dt.datetime,
                  end:   pd.Timestamp | dt.datetime, **kwargs) -> pd.DataFrame:
        try:
            data = self.database.read(start, end)
            data = data[data.columns.drop(list(data.filter(regex='_energy')))]

        except DatabaseUnavailableException as e:
            data = pd.DataFrame()
            logger.warning(str(e))

        try:
            weather = self.weather.database.read(start, end, **kwargs)
            input = self._validate_input(weather)
            data = pd.concat([data, input], axis=1)

        except WeatherUnavailableException as e:
            logger.debug(str(e))
        except DatabaseUnavailableException as e:
            logger.warning(str(e))

        data.index.name = 'time'
        return data[(data.index >= start) & (data.index <= end)]

    # noinspection PyUnresolvedReferences, PyTypeChecker, PyShadowingBuiltins
    def _validate_input(self, weather: pd.DataFrame) -> pd.DataFrame:
        input = super()._validate_input(weather)
        cmpts_pv = [cmpt for cmpt in self.values() if cmpt.type == 'pv']
        if len(cmpts_pv) > 0:
            input['pv_yield'] = 0
            for cmpt in cmpts_pv:
                input_pv = self._get_solar_yield(cmpt, input)
                input['pv_yield'] += input_pv['pv_power'].abs()
        return input
