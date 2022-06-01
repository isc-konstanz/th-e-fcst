# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast
    ~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import logging
import pandas as pd
import datetime as dt
import th_e_core

from th_e_core import System
from th_e_fcst import NeuralNetwork

logger = logging.getLogger(__name__)


class Forecast(th_e_core.Forecast):

    @classmethod
    def read(cls, system: System, **kwargs) -> Forecast:
        return cls(system, cls._read_configs(system, **kwargs))

    def _activate(self, system, configs):
        super()._activate(system, configs)
        if configs.get('General', 'type', fallback='default').lower() == 'default':
            self._weather = None
        else:
            self._weather = super().read(system)

        # TODO: implement ARIMAX prediction
        self._model = NeuralNetwork.read(system)

    # noinspection PyProtectedMember
    def build(self, **kwargs) -> pd.Dataframe:
        from th_e_data import build
        return build(self.configs, self._weather._database, location=self._system.location, **kwargs)

    def _get(self, *args, **kwargs):
        data = self._get_data(*args, **kwargs)
        return self._get_range(self._model.predict(data, *args, **kwargs),
                               kwargs.get('start', None), 
                               kwargs.get('end', None))

    def _get_data(self,
                  start: pd.Timestamp | dt.datetime,
                  end:   pd.Timestamp | dt.datetime = None, **_) -> pd.DataFrame:
        resolution = self._model.features.resolutions[0]
        prior_end = start - resolution.time_step
        prior_start = start - resolution.time_prior\
                            - resolution.time_step + dt.timedelta(minutes=1)

        data = self._get_data_history(prior_start, prior_end)

        if self._weather is not None:
            weather = self._weather.get(start, end)

            if self._system.contains_type('pv'):
                solar_yield = self._get_solar_yield(weather)
                data = pd.concat([data, solar_yield], axis=1)
            data = pd.concat([data, weather], axis=1)
        data = pd.concat([data, self._get_solar_position(data.index)], axis=1)

        return data

    # noinspection PyProtectedMember
    def _get_data_history(self,
                          start: pd.Timestamp | dt.datetime,
                          end:   pd.Timestamp | dt.datetime, **_) -> pd.DataFrame:
        data = self._system._database.read(start, end)
        data = data[data.columns.drop(list(data.filter(regex='_energy')))]

        if self._weather is not None:
            weather = self._get_weather(start, end)

            if self._system.contains_type('pv'):
                solar_yield = self._get_solar_yield(weather)
                data = pd.concat([data, solar_yield], axis=1)
            data = pd.concat([data, weather], axis=1)
        data = pd.concat([data, self._get_solar_position(data.index)], axis=1)

        return data

    # noinspection PyProtectedMember
    def _get_weather(self,
                     start: pd.Timestamp | dt.datetime,
                     end:   pd.Timestamp | dt.datetime, **_) -> pd.DataFrame:
        return self._weather._database.read(start, end)

    def _get_solar_position(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        data = pd.DataFrame(index=index)
        # minutes = pd.date_range(weather.index[0], weather.index[-1], tz=weather.index.tz, freq='min')
        try:
            # noinspection PyUnresolvedReferences
            from pvlib.solarposition import get_solarposition

            # TODO: use weather pressure for solar position
            data = get_solarposition(index,
                                     self._system.location.latitude,
                                     self._system.location.longitude,
                                     altitude=self._system.location.altitude)
            data = data.loc[:, ['azimuth', 'apparent_zenith', 'apparent_elevation']]
            data.columns = ['solar_azimuth', 'solar_zenith', 'solar_elevation']

        except ImportError as e:
            logger.warning("Unable to generate solar position: {}".format(str(e)))

        return data

    def _get_solar_yield(self, weather: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(index=weather.index)
        try:
            # noinspection PyUnresolvedReferences
            from th_e_yield.model import Model

            data['pv_yield'] = 0
            for array in self._system.get_type('pv'):
                model = Model(self._system, array, self._model.configs, section='Yield')
                data.pv_yield += model.run(weather)['p_ac']

        except ImportError as e:
            logger.warning("Unable to calculate PV yield: {}".format(str(e)))

        return data


# class Basic(Model):
# 
#     def load(self, start, earliest, latest, interval):
#         gen_pv = self.load_generation(earliest, latest, interval)
#         cons_el = self.load_consumption_electrical(earliest, latest, interval)
#         cons_th = self.load_consumption_thermal(earliest, latest, interval)
#         data = pd.concat([gen_pv, cons_el, cons_th], axis=1)
#         
#         if len(data) > 1:
#             data = self._correction(earliest, start, interval, data)
#         
#         return data
# 
# 
#     def load_generation(self, start, end, interval):
#         keys = [FORECAST_GENERATION_PV]
#         
#         # Get historic data of the last day and extrapolate it for the coming days
#         end_day = start + dt.timedelta(days=1)
#         if end_day > end:
#             end_day = end
#         
#         data = self._weighted_mean(keys, start, end_day, interval, step=1)
#         
#         while data.index[-1] < end:
#             tmp = data.copy()
#             tmp.index = tmp.index + dt.timedelta(days=1)
#             data = pd.concat([data, tmp[data.index[-1]+dt.timedelta(seconds=interval):end]], axis=0)
#         
#         return data
# 
# 
#     def load_consumption_electrical(self, start, end, interval):
#         keys = [FORECAST_CONSUMPTION_ELECTRICAL]
#         data = self._weighted_mean(keys, start, end, interval, step=7)
#         
#         return data
# 
# 
#     def load_consumption_thermal(self, start, end, interval):
#         keys = [STORAGE_THERMAL_TEMPERATURE, HEAT_PUMP_THERMAL, COGENERATOR_THERMAL]
#         data = self._weighted_mean(keys, start, end, interval, step=7)
#         
#         # Calculate thermal energy consumption from the storages temperature, minus known generation
#         storage_th_delta = (data[STORAGE_THERMAL_TEMPERATURE] - data[STORAGE_THERMAL_TEMPERATURE].shift(1)).fillna(0)
#         storage_th_loss = data[STORAGE_THERMAL_TEMPERATURE] - data[STORAGE_THERMAL_TEMPERATURE]*self.components.storage_thermal.loss(interval)
#         data[FORECAST_CONSUMPTION_THERMAL] = - ((storage_th_delta - storage_th_loss)*self.components.storage_thermal.capacity() \
#                                              - data[HEAT_PUMP_THERMAL] - data[COGENERATOR_THERMAL])
#         
#         return data.loc[:,[FORECAST_CONSUMPTION_THERMAL]]
# 
# 
#     def _weighted_mean(self, keys, start, end, interval, step=1):
#         historic = pd.DataFrame(columns=keys)
#         h = 3
#         
#         # Create weighted consumption average of the three last timesteps
#         for i in range(h):
#             data = self.server.get(keys, start-dt.timedelta(days=1+i*step), end-dt.timedelta(days=1+i*step), interval)
#             data.index = data.index + dt.timedelta(days=1+i*step)
#             data = data[start:end]
#             
#             historic = historic.add(data.astype(float).multiply(h-i), fill_value=0)
#         
#         return historic/6
# 
# 
#     def _correction(self, start, end, interval, data):
#         keys = data.columns
#         
#         if end - start < dt.timedelta(seconds=2*interval):
#             end = end - dt.timedelta(seconds=2*interval)
#         
#         result = self.server.get(keys, start, end, interval)
#         data = result.combine_first(data)
#         last = result.iloc[-1,:]
#         
#         offset = data.index.get_loc(end) - 1
#         for i in range(len(keys)):
#             key = keys[i]
#             
#             # TODO: calculate factor through e-(j/x)
#             for j in range(1, 4):
#                 data.iloc[j+offset, i] = (data.iloc[j+offset, i]*j/4 + last[key]*(1-j/4))/2
#         
#         return data

