# -*- coding: utf-8 -*-
"""
    th-e-fcst.forecast.db
    ~~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import pandas as pd
import datetime as dt

from corsys.tools import to_date
from corsys.system import System
from corsys.io import DatabaseUnavailableException
from . import Forecast


class DatabaseForecast(Forecast):

    def __activate__(self, system: System) -> None:
        super().__activate__(system)
        self._database = system.database

    @property
    def database(self):
        if not self._database.enabled:
            raise DatabaseUnavailableException(f"Forecast \"{self.system.name}\" database is disabled")
        return self._database

    # noinspection PyShadowingBuiltins
    def predict(self,
                start:  pd.Timestamp | dt.datetime | str = None,
                end:    pd.Timestamp | dt.datetime | str = None,
                format: str = '%d.%m.%Y',
                **kwargs) -> pd.DataFrame:

        start = to_date(start, timezone=self.system.location.timezone)
        end = to_date(end, timezone=self.system.location.timezone)

        return self.database.read(start=start, end=end, **kwargs)
