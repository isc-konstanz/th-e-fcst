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

    @property
    def _component_types(self):
        return super()._component_types + ['solar', 'modules', 'configs']

    def _component(self, configs, type, **kwargs):
        if type in ['pv', 'solar', 'modules', 'configs']:
            try:
                from th_e_yield.system import Configurations
                return Configurations(configs, self, **kwargs)

            except ImportError as e:
                logger.debug("Unable to instance PV configuration: {}".format(str(e)))

        return super()._component(configs, type, **kwargs)

    def run(self, date=None, **kwargs):
        if date is None:
            date = dt.datetime.now(tz.utc)
        
        return self._run(date, **kwargs)

    def _run(self, date, **kwargs):
        data = self.forecast.get(date, **kwargs)
        
        if self._database is not None:
            self._database.persist(data, **kwargs)
        
        return data

