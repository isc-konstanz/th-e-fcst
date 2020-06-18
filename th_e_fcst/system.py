# -*- coding: utf-8 -*-
"""
    th-e-fcst.system
    ~~~~~~~~~~~~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import pytz as tz
import datetime as dt

from th_e_core import System as SystemCore
from th_e_fcst import Forecast


class System(SystemCore):

    def _activate(self, componens, *args, **kwargs):
        super()._activate(componens, *args, **kwargs)
        
        self.forecast = Forecast.read(self, **kwargs)

    def run(self, date=None, **kwargs):
        if date is None:
            date = dt.datetime.now(tz.utc)
        
        return self._run(date, **kwargs)

    def _run(self, date, **kwargs):
        data = self.forecast.get(date, **kwargs)
        
        if self._database is not None:
            self._database.persist(data, **kwargs)
        
        return data

