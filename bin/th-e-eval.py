#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-eval
    ~~~~~~~~~
    
    To learn how to configure specific settings, see "th-e-eval --help"

"""
import os
import time
import inspect
import pytz as tz
import pandas as pd
import datetime as dt

from copy import deepcopy
from typing import Optional, Union

import scisys.io as io
from scisys import Results, Evaluation
from corsys import Settings
from corsys.tools import floor_date, ceil_date, to_date, to_bool
from argparse import ArgumentParser, RawTextHelpFormatter
from tensorboard.program import TensorBoard
from th_e_fcst import System


def main(**kwargs) -> None:
    verbose = kwargs.pop('verbose', 'false').lower() == 'true'

    systems = System.read(settings)
    results = []
    error = False

    tensorboard = _launch_tensorboard(systems, **kwargs)

    for system in systems:
        logger.info('Starting TH-E Evaluation of system {}'.format(system.name))

        system_results = Results(system, verbose=verbose)
        system_results.durations.start('Simulation')
        try:
            if not system.forecast.exists():
                train(system, system_results, **kwargs)
            predict(system, system_results, **kwargs)

        except Exception as e:
            error = True
            logger.error("Error simulating system %s: %s", system.name, str(e))
            logger.exception(e)
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

        finally:
            system_results.durations.stop('Simulation')
            system_results.close()

        results.append(system_results)

    if not error:
        evaluations = Evaluation.read(settings)
        evaluations(results)

        logger.info("Finished TH-E Evaluation{0}".format('s' if len(systems) > 1 else ''))

        if tensorboard:
            # keep_running = input('Keep tensorboard running? [y/n]').lower()
            # keep_running = keep_running in ['j', 'ja', 'y', 'yes', 'true']
            keep_running = settings.getint('General', 'keep_running', fallback=15)
            while keep_running > 0:
                try:
                    time.sleep(60)
                    keep_running -= 1

                except KeyboardInterrupt:
                    keep_running = 0

            # tensorboard.stop()


# noinspection PyProtectedMember
def train(system, results, verbose=False, **kwargs):
    logger.debug("Beginning training of neural network for system: {}".format(system.name))
    results.durations.start('Training')

    timezone = system.location.pytz
    start = to_date(settings['Training']['start'], timezone)
    end = to_date(settings['Training']['end'], timezone)
    end = ceil_date(end, system.location.pytz)

    bldargs = dict(kwargs)
    bldargs['start'] = start
    bldargs['end'] = end

    data = system.build(**bldargs)

    features_dir = os.path.join(system.configs.dirs.data, 'model')
    features_path = os.path.join(features_dir, 'features')
    features = _load_features(system, start, end, data, features_dir)

    if verbose:
        io.write_csv(system, features, features_path)
        io.print_distributions(features, path=system.forecast.dir)

    system.forecast._train(features, threading=settings.getboolean('General', 'threading', fallback=True))

    results.durations.stop('Training')
    logger.debug("Training of neural network for system {} complete after {:.2f} minutes"
                 .format(system.name, results.durations['Training']))


# noinspection PyProtectedMember, PyShadowingBuiltins
def predict(system, results, verbose=False, **kwargs):
    timezone = system.location.timezone
    start = to_date(settings['General']['start'], timezone)
    end = to_date(settings['General']['end'], timezone)
    end = ceil_date(end, timezone)

    bldargs = dict(kwargs)
    bldargs['start'] = start
    bldargs['end'] = end

    data = system.build(**bldargs)

    features_dir = os.path.join(system.configs.dirs.data, 'results')
    features_path = os.path.join(features_dir, 'features')
    features = _load_features(system, start, end, data, features_dir)

    if verbose:
        io.write_csv(system, features, features_path)

    logger.debug("Starting predictions for system: {}".format(system.name))
    results.durations.start('Prediction')

    resolution_max = system.forecast.features.resolutions[0]
    resolution_min = system.forecast.features.resolutions[0]
    if len(system.forecast.features.resolutions) > 1:
        for i in range(len(system.forecast.features.resolutions)-1, 0, -1):
            resolution_min = system.forecast.features.resolutions[i]
            if resolution_min.steps_horizon is not None:
                break

    resolution_data = resolution_min.resample(features)

    # Reactivate this, when multiprocessing will be implemented
    # global logger
    # if process.current_process().name != 'MainProcess':
    #    logger = process.get_logger()

    interval = settings.getint('General', 'interval')
    end = ceil_date(features.index[-1] - resolution_min.time_horizon, timezone=system.location.timezone)
    start = floor_date(features.index[0] + resolution_max.time_prior, timezone=system.location.timezone)
    date = start

    training_recursive = settings.getboolean('Training', 'recursive', fallback=True)
    if training_recursive:
        training_interval = settings.getint('Training', 'interval', fallback=24)*60
        training_date = _next_date(date, training_interval*2)
        training_last = date
        date = _next_date(date, training_interval)

    while date <= end:
        date_path = date.strftime('%Y-%m-%d/%H-%M-%S')
        if date_path in results:
            try:
                # If this step was simulated already, load the results and skip the prediction
                results.load(date_path+'/output')

                date = _next_date(date, interval)
                continue

            except Exception as e:
                logger.debug("Error loading %s: %s", date, str(e))
        try:
            date_start = date + resolution_min.time_step
            date_prior = date_start - resolution_max.time_prior
            date_horizon = date_start + resolution_min.time_horizon
            date_features = deepcopy(resolution_data[date_prior:date_horizon])
            date_range = date_features[date_start:date_horizon].index

            input = system.forecast.features.input(date_features, date_range)
            target = system.forecast.features.target(date_features, date_range)
            target.rename(columns={t: t + '_ref' for t in system.forecast.features.target_keys},
                          inplace=True)

            prediction = system.forecast._predict(date_features, date)

            result = pd.concat([prediction, target], axis=1)

            # Add error columns for all targets
            for target_key in system.forecast.features.target_keys:
                result[target_key + '_err'] = result[target_key] - result[target_key]

            result = pd.concat([result, resolution_data.loc[result.index,
                                                            [column for column in resolution_data.columns
                                                             if column not in result.columns]]], axis=1)

            results[date_path+'/output'] = result
            result.index.name = 'time'
            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)

            results[date_path+'/output'] = result
            results[date_path+'/input'] = input
            results[date_path+'/target'] = target

            if training_recursive and date >= training_date:
                if abs((training_date - date).total_seconds()) <= training_interval:
                    training_features = deepcopy(resolution_data[training_last - resolution_max.time_prior:training_date])
                    validation_features = deepcopy(resolution_data[training_last - dt.timedelta(days=7):training_last])

                    system.forecast._train(training_features, validation_features)

                training_last = training_date
                training_date = _next_date(date, training_interval)

            date = _next_date(date, interval)

        except ValueError as e:
            logger.warning("Skipping %s: %s", date, str(e))
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            date = _next_date(date, interval)

    results.durations.stop('Prediction')
    logger.debug("Predictions for system {} complete after {:.2f} minutes"
                 .format(system.name, results.durations['Prediction']))


# noinspection PyProtectedMember, PyUnresolvedReferences
def _load_features(system: System,
                   start: Union[pd.Timestamp, dt.datetime],
                   end: Union[pd.Timestamp, dt.datetime],
                   data: pd.DataFrame,
                   features_dir: str) -> pd.DataFrame:
    features_path = os.path.join(features_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    if os.path.isfile(features_path + '.h5'):
        with pd.HDFStore(features_path + '.h5', mode='r') as hdf:
            features = hdf.get('features')
    else:
        features = system.forecast.features.extract(data)
        features = system.forecast.features._add_meta(features)
        features.to_hdf(features_path + '.h5', 'features', mode='w')

    return features.loc[start:end]


def _launch_tensorboard(systems, **kwargs) -> Optional[TensorBoard]:
    # noinspection PyProtectedMember
    tensorboard_launch = any([system.forecast._tensorboard for system in systems])
    if tensorboard_launch and 'tensorboard' in kwargs:
        tensorboard_launch = to_bool(kwargs['tensorboard'])

    tensorboard = None
    if tensorboard_launch:
        logging.getLogger('MARKDOWN').setLevel(logging.ERROR)
        logging.getLogger('tensorboard').setLevel(logging.ERROR)
        logger_werkzeug = logging.getLogger('werkzeug')
        logger_werkzeug.setLevel(logging.ERROR)
        logger_werkzeug.disabled = True

        tensorboard = TensorBoard()
        tensorboard.configure(argv=[None, '--logdir', settings.dirs.data])
        tensorboard_url = tensorboard.launch()
        logger.info("Started TensorBoard at {}".format(tensorboard_url))

    return tensorboard


def _next_date(date: pd.Timestamp, interval: int) -> pd.Timestamp:
    date += dt.timedelta(minutes=interval)
    if interval >= 60:
        timezone = date.tzinfo

        # FIXME: verify pandas version includes fix in commit from 17. Nov 20221:
        # https://github.com/pandas-dev/pandas/pull/44357
        date = date.astimezone(tz.utc).round('{minutes}s'.format(minutes=interval))\
                   .astimezone(timezone)

    return date


if __name__ == "__main__":
    from th_e_fcst import __version__

    run_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(run_dir) == 'bin':
        run_dir = os.path.dirname(run_dir)

    os.chdir(run_dir)
    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    settings = Settings('th-e-fcst', parser=parser)

    import logging

    logger = logging.getLogger('th-e-fcst')
    main(**settings.general)
