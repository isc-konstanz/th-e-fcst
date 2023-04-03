# -*- coding: utf-8 -*-
"""
    th_e_fcst.resolution
    ~~~~~~~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Iterator
from collections.abc import Sequence

import pandas as pd
import datetime as dt
import warnings

from corsys import Configurations
from pandas.tseries.frequencies import to_offset


class Resolutions(Sequence):

    @classmethod
    def read(cls, configs: Configurations) -> Resolutions:
        prior_range = pd.date_range(dt.datetime.now().replace(minute=0, second=0, microsecond=0), periods=0)
        horizon_range = pd.date_range(dt.datetime.now().replace(minute=0, second=0, microsecond=0), periods=0)

        resolutions = list()
        resolution_index = 0
        for resolution_configs in [s for s in configs.sections() if s.lower().startswith('resolution')]:
            resolution = Resolution(resolution_index, **dict(configs.items(resolution_configs)))

            if resolution.steps_prior:
                prior_start = dt.datetime.now().replace(minute=0, second=0, microsecond=0) - resolution.time_prior
                prior_range = prior_range.union(pd.date_range(prior_start,
                                                              periods=resolution.steps_prior,
                                                              freq='{}min'.format(resolution.minutes)))

            if resolution.steps_horizon:
                horizon_start = dt.datetime.now().replace(minute=0, second=0, microsecond=0) + resolution.time_step
                horizon_range = horizon_range.union(pd.date_range(horizon_start,
                                                                  periods=resolution.steps_horizon,
                                                                  freq='{}min'.format(resolution.minutes)))

            resolutions.append(resolution)
            resolution_index += 1
        return cls(*resolutions,
                   steps_prior=len(prior_range) if not prior_range.empty else None,
                   steps_horizon=len(horizon_range) if not horizon_range.empty else None)

    def __init__(self, *resolutions: Resolution, steps_prior=None, steps_horizon=None) -> None:
        self._resolutions = list()
        self._resolutions.extend(resolutions)
        self.steps_prior = steps_prior
        self.steps_horizon = steps_horizon

    def __repr__(self):
        return '\n\n'.join(f'{res}' for res in self._resolutions)

    def __getitem__(self, index: int) -> Resolution:
        return self._resolutions[index]

    def __iter__(self) -> Iterator[Resolution]:
        return iter(self._resolutions)

    def __len__(self) -> int:
        return len(self._resolutions)

    def get_prior(self, how='max') -> Resolution:

        def _get_prior(resolution: Resolution, resolution_range: Iterator[int]) -> Resolution:
            if len(self._resolutions) > 1:
                for i in resolution_range:
                    resolution = self._resolutions[i]
                    if resolution.steps_prior is not None:
                        break
            return resolution

        if how.lower() == 'max':
            return _get_prior(self._resolutions[-1], range(0, len(self._resolutions)-1))
        elif how.lower() == 'min':
            return _get_prior(self._resolutions[0], range(len(self._resolutions)-1, 0, -1))
        else:
            raise ValueError(f'Unable to retrieve "{how}" prior time')

    def time_prior(self, how='max') -> dt.timedelta | None:
        return self.get_prior(how).time_prior

    def get_horizon(self, how: str = 'max') -> Resolution:

        def _get_horizon(resolution: Resolution, resolution_range: Iterator[int]) -> Resolution:
            if len(self._resolutions) > 1:
                for i in resolution_range:
                    resolution = self._resolutions[i]
                    if resolution.steps_horizon is not None:
                        break
            return resolution

        if how.lower() == 'max':
            return _get_horizon(self._resolutions[-1], range(len(self._resolutions)))
        elif how.lower() == 'min':
            return _get_horizon(self._resolutions[0], range(len(self._resolutions)-1, 0, -1))
        else:
            raise ValueError(f'Unable to retrieve "{how}" time horizon')

    def time_horizon(self, how: str = 'max') -> dt.timedelta | None:
        return self.get_horizon(how).time_horizon


class Resolution:

    def __init__(self, index, minutes, steps_prior=None, steps_horizon=None):
        self.index = int(index)
        self.minutes = int(minutes)
        self.steps_prior = int(steps_prior) if steps_prior else None
        self.steps_horizon = int(steps_horizon) if steps_horizon else None

    def __repr__(self):
        attrs = {
            'minutes': self.minutes
        }
        if self.steps_prior:
            attrs['prior'] = self.time_prior
        if self.steps_horizon:
            attrs['horizon'] = self.time_horizon
        return (f'Resolution {self.index+1}: \n  ' + '\n  '.join(
            f'{key}: {str(val)}' for key, val in attrs.items()))

    @property
    def time_step(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.minutes)

    @property
    def time_prior(self) -> dt.timedelta | None:
        if self.steps_prior is None:
            return dt.timedelta(minutes=0)

        return dt.timedelta(minutes=self.minutes * self.steps_prior)

    @property
    def time_horizon(self) -> dt.timedelta | None:
        if self.steps_horizon is None:
            return dt.timedelta(minutes=0)

        return dt.timedelta(minutes=self.minutes * self.steps_horizon)

    def resample(self, data: pd.DataFrame, offset=None) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            data = data.resample('{}min'.format(self.minutes), closed='right', base=offset).mean()
            data.index += to_offset('{}min'.format(self.minutes))

        return data
