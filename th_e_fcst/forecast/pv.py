# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast.default
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import pandas as pd
import datetime as dt

from corsys.tools import to_date
from corsys.cmpt import Photovoltaic
from .base import Forecast


class PVForecast(Forecast):

    # noinspection PyProtectedMember
    def predict(self,
                start: pd.Timestamp | dt.datetime | str = None,
                end:   pd.Timestamp | dt.datetime | str = None,
                data:  pd.DataFrame = None,
                format: str = '%d.%m.%Y') -> pd.DataFrame:

        start = to_date(start, timezone=self.system.location.timezone, format=format)
        end = to_date(end, timezone=self.system.location.timezone, format=format)

        if data is None:
            data = self.system.weather.get(start, end)
            data = self.system._validate_input(data)
            data = self._validate_resolution(data)
        return self._predict_solar_yield(data)

    # noinspection PyProtectedMember
    def _predict_solar_yield(self, weather: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(index=weather.index, data=0, columns=[Photovoltaic.POWER])
        for pv in self.system.get_type(Photovoltaic.TYPE):
            pv_data = self.system._get_solar_yield(pv, weather)
            data.loc[:, Photovoltaic.POWER] += pv_data[Photovoltaic.POWER].abs()
        return data
