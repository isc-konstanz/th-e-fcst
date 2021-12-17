# -*- coding: utf-8 -*-
"""
    th-e-fcst.system
    ~~~~~~~~~~~~~~~~
    
    
"""
import logging
import pytz as tz
import datetime as dt
import th_e_core
from th_e_fcst import Forecast

logger = logging.getLogger(__name__)


class System(th_e_core.System):

    def _activate(self, components, *args, **kwargs):
        super()._activate(components, *args, **kwargs)
        
        self.forecast = Forecast.read(self, **kwargs)

    @property
    def _component_types(self):
        return super()._component_types + ['solar', 'array', 'modules', 'configs']

    # noinspection PyShadowingBuiltins
    def _component(self, configs, type, **kwargs):
        if type in ['pv', 'solar', 'array', 'modules', 'configs']:
            try:
                from th_e_core.pvsystem import PVSystem
                return PVSystem(self, configs, **kwargs)

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
