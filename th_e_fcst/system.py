# -*- coding: utf-8 -*-
"""
    th-e-fcst.system
    ~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
import logging

import pytz as tz
import pandas as pd
import datetime as dt
import th_e_core
from th_e_fcst import Forecast
from th_e_core.tools import convert_timezone
from configparser import ConfigParser as Configurations
from typing import Dict
from copy import deepcopy

logger = logging.getLogger(__name__)


class System(th_e_core.System):

    def _activate(self, components: Dict[str, th_e_core.Component], configs: Configurations) -> None:
        super()._activate(components, configs)

        self.forecast = Forecast.read(self)

    @property
    def _component_types(self):
        return super()._component_types + ['solar', 'array', 'modules']

    # noinspection PyShadowingBuiltins
    def _component(self, configs, type):
        if type in ['pv', 'solar', 'array', 'modules']:
            try:
                # noinspection PyUnresolvedReferences
                from th_e_core.cmpt import Photovoltaics
                return Photovoltaics(self, configs)

            except ImportError as e:
                logger.debug("Unable to instance PV configuration: {}".format(str(e)))

        return super()._component(configs, type)

    # noinspection PyProtectedMember
    def build(self,
              start: pd.Timestamp | dt.datetime = None,
              end:   pd.Timestamp | dt.datetime = None, **_) -> None:

        from th_e_data import build
        database = deepcopy(self._database)
        database.enabled = False

        # noinspection SpellCheckingInspection
        bldargs = dict()
        bldargs['start'] = convert_timezone(start, self.location.pytz)
        bldargs['end'] = convert_timezone(end, self.location.pytz)

        weather = self.forecast.build(**bldargs)

        if (weather is None or weather.empty) and not database.exists(**bldargs):
            weather = self.forecast._get_weather(**bldargs)

        data = build(self.configs, database, weather=weather, **bldargs)
        if data is not None and not data.empty:
            from th_e_core.cmpt import Photovoltaics
            if self.contains_type('pv') and \
                    (Photovoltaics.POWER not in data.columns or
                     Photovoltaics.ENERGY not in data.columns):
                data[Photovoltaics.POWER] = self.forecast._get_solar_yield(weather[data.index[0]:data.index[-1]]).pv_yield
                data[Photovoltaics.POWER].fillna(0, inplace=True)
                data[Photovoltaics.ENERGY] = (data[Photovoltaics.POWER] / 1000 *
                                              (data.index[1] - data.index[0]).seconds / 3600).cumsum()

            if self._database.enabled:
                self._database.write(data, split_data=True, **bldargs)

    def run(self, date=None, **kwargs):
        if date is None:
            date = dt.datetime.now(tz.utc)
        
        return self._run(date, **kwargs)

    def _run(self, date, **kwargs):
        data = self.forecast.get(date, **kwargs)
        
        if self._database is not None:
            self._database.persist(data, **kwargs)
        
        return data
