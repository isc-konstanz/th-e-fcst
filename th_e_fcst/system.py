# -*- coding: utf-8 -*-
"""
    th-e-fcst.system
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
import logging

import pandas as pd
import datetime as dt
import pvsys
from corsys import Component, Configurations
from corsys.io import DatabaseUnavailableException
from corsys.weather import WeatherUnavailableException
from typing import Dict
from .forecast import Forecast

logger = logging.getLogger(__name__)


class System(pvsys.System):

    def __activate__(self, components: Dict[str, Component], configs: Configurations) -> None:
        super().__activate__(components, configs)
        self._forecast = Forecast.read(self)

    # noinspection PyShadowingBuiltins
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        input = self._get_input(*args, **kwargs)
        result = self._forecast.get(*args, **kwargs)
        return pd.concat([result, input], axis=1)

    @property
    def forecast(self) -> Forecast:
        return self._forecast

    # noinspection PyUnresolvedReferences
    def _get_input(self, *args, **kwargs) -> pd.DataFrame:
        input = super()._get_input(*args, **kwargs)

        for cmpt in self.values():
            if cmpt.type == 'pv':
                result_pv = self._get_solar_yield(cmpt, input)
                result['pv_yield'] += result_pv['pv_power'].abs()

        return input

    # noinspection PyUnresolvedReferences
    def _get_data(self,
                  start: pd.Timestamp | dt.datetime,
                  end:   pd.Timestamp | dt.datetime = None, **kwargs) -> pd.DataFrame:
        resolution = self.forecast.features.resolutions[0]
        prior_end = start - resolution.time_step
        prior_start = start - resolution.time_prior\
                            - resolution.time_step + dt.timedelta(minutes=1)

        data = self._get_data_history(prior_start, prior_end)
        try:
            input = self._get_input(start, end, **kwargs)
            data = pd.concat([data, input], axis=1)

        except WeatherUnavailableException as e:
            logger.debug(str(e))

        return data

    # noinspection PyTypeChecker
    def _get_data_history(self,
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
        return data

    # noinspection PyUnresolvedReferences, PyTypeChecker, SpellCheckingInspection
    def _validate_input(self, weather: pd.DataFrame) -> pd.DataFrame:
        input = super()._validate_input(weather)
        cmpts_pv = [cmpt for cmpt in self.values() if cmpt.type == 'pv']
        if len(cmpts_pv) > 0:
            input['pv_yield'] = 0
            for cmpt in cmpts_pv:
                input_pv = self._get_solar_yield(cmpt, input)
                input['pv_yield'] += input_pv['pv_power'].abs()
        return input
