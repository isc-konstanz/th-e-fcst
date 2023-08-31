# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast.default
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import pytz as tz
import pandas as pd
import datetime as dt

from dateutil.relativedelta import relativedelta
from corsys.tools import to_date
from corsys.cmpt import Photovoltaic
from corsys.system import Configurations, System
from .base import ForecastException
from .db import DatabaseForecast
from .pv import PVForecast


class DefaultForecast(DatabaseForecast, PVForecast):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.prior_weeks = configs.getint(Configurations.GENERAL, 'prior_weeks', fallback=1)

    def __activate__(self, system: System) -> None:
        super().__activate__(system)

    # noinspection PyProtectedMember
    def predict(self,
                start: pd.Timestamp | dt.datetime | str = None,
                end:   pd.Timestamp | dt.datetime | str = None,
                data:  pd.DataFrame = None,
                format: str = '%d.%m.%Y',
                **kwargs) -> pd.DataFrame:

        start = to_date(start, timezone=self.system.location.timezone, format=format)
        end = to_date(end, timezone=self.system.location.timezone, format=format)

        forecast = self._predict_persistance(start, end, **kwargs)

        if self.system.contains_type(Photovoltaic.TYPE):
            if data is None or data.index[0] > start or data.index[-1] < end:
                data = self.system.weather.get(start, end)
                data = self.system._validate_input(data)
                data = self._validate_resolution(data)
            forecast_pv = self._predict_solar_yield(data)
            forecast = forecast.drop(labels=forecast_pv.columns, axis='columns', errors='ignore')
            forecast = pd.concat([forecast, forecast_pv], axis='columns')
        return forecast

    # noinspection SpellCheckingInspection
    def _predict_persistance(self,
                             start: pd.Timestamp | dt.datetime,
                             end: pd.Timestamp | dt.datetime,
                             **kwargs) -> pd.DataFrame:

        data = pd.DataFrame()
        weights = 0

        # Create weighted average of the last week(s)
        for week in range(self.prior_weeks):
            week_offset = pd.DateOffset(weeks=week+1)
            week_start = start - week_offset
            week_end = end - week_offset

            if not self.database.exists(start=week_start, end=week_end, **kwargs):
                if data.empty:
                    raise ForecastException("Unable to read prior data from "
                                            f"{week_start.strftime('%d.%m.%Y %H:%M')} to "
                                            f"{week_end.strftime('%d.%m.%Y %H:%M')}")
                continue

            week_data = self.database.read(start=week_start, end=week_end, **kwargs)
            week_data = self._validate_resolution(week_data)
            week_timezone = week_data.index.tzinfo
            week_data.index = (week_data.index.tz_convert(tz.utc) + week_offset).tz_convert(week_timezone)

            # TODO: calculate factor through e-(j/x)
            weight = self.prior_weeks - week
            weights += weight
            data = data.add(week_data * weight, fill_value=0)

        if data.empty:
            raise ForecastException("Unable to read prior data")

        return data/weights
