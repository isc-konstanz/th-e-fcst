#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-simulation
    ~~~~~~~~~~~~~~~
    
    
    To learn how to configure specific settings, see "th-e-simulation --help"

"""
import os
import time
import shutil
import inspect
import traceback
import pytz as tz
import pandas as pd
import datetime as dt

from copy import deepcopy
from argparse import ArgumentParser, RawTextHelpFormatter
from tensorboard import program

from th_e_data import Results, Evaluation
from th_e_core import configs
from th_e_core.tools import floor_date, ceil_date
import th_e_data.io as io


# noinspection PyProtectedMember, SpellCheckingInspection
def simulate(args):
    from th_e_fcst import System

    settings = configs.read('settings.cfg', **vars(args))

    verbose = settings.getboolean('General', 'verbose', fallback=False)

    kwargs = vars(args)
    kwargs.update(settings.items('General'))
    kwargs['verbose'] = verbose

    systems = System.read(**kwargs)
    results = []
    error = False

    tensorboard = _launch_tensorboard(systems, **kwargs)

    for system in systems:
        logger.info('Starting TH-E Simulation of system {}'.format(system.name))

        system_results = Results(system, verbose=verbose)
        system_results.durations.start('Simulation')
        try:
            _simulate_training(settings, system, system_results, **kwargs)
            _simulate_prediction(settings, system, system_results, **kwargs)

        except Exception as e:
            error = True
            logger.error("Error simulating system %s: %s", system.name, str(e))
            logger.info("%s: %s", type(e).__name__, traceback.format_exc())

        finally:
            system_results.durations.stop('Simulation')
            system_results.close()

        results.append(system_results)

    if not error:
        evaluation = Evaluation.read(**kwargs)
        evaluation.run(results)

        logger.info("Finished TH-E Simulation{0}".format('s' if len(systems) > 1 else ''))

        if tensorboard:
            # keep_running = input('Keep tensorboard running? [y/n]').lower()
            # keep_running = keep_running in ['j', 'ja', 'y', 'yes', 'true']
            keep_running = settings.getboolean('General', 'keep_running', fallback=False)
            while keep_running:
                try:
                    time.sleep(100)

                except KeyboardInterrupt:
                    keep_running = False


# noinspection PyProtectedMember
def _simulate_training(settings, system, results, verbose=False, **kwargs):
    forecast = system.forecast._model

    if forecast.exists():
        return

    logger.debug("Beginning training of neural network for system: {}".format(system.name))
    results.durations.start('Training')

    timezone = system.location.pytz
    start = _get_date(settings['Training']['start'], timezone)
    end = _get_date(settings['Training']['end'], timezone)
    end = ceil_date(end, system.location.pytz)

    bldargs = dict(kwargs)
    bldargs['start'] = start
    bldargs['end'] = end

    system.build(**bldargs)

    features_path = os.path.join(system.configs.get('General', 'data_dir'), 'model', 'features')
    if os.path.isfile(features_path + '.h5'):
        with pd.HDFStore(features_path + '.h5', mode='r') as hdf:
            features = hdf.get('features').loc[start:end]
    else:
        data = system.forecast._get_data_history(start, end)
        features = system.forecast._model.features.extract(data)
        features = system.forecast._model.features._add_meta(features)
        features.to_hdf(features_path + '.h5', 'features', mode='w')

        if verbose:
            io.write_csv(system, features, features_path)
            io.print_distributions(features, path=system.forecast._model.dir)

    system.forecast._model._train(features)

    results.durations.stop('Training')
    logger.debug("Training of neural network for system {} complete after {:.2f} minutes"
                 .format(system.name, results.durations['Training']))


# noinspection PyProtectedMember
def _simulate_prediction(settings, system, results, verbose=False, **kwargs):
    forecast = system.forecast._model

    timezone = system.location.pytz
    start = _get_date(settings['General']['start'], timezone)
    end = _get_date(settings['General']['end'], timezone)
    end = ceil_date(end, system.location.pytz)

    bldargs = dict(kwargs)
    bldargs['start'] = start
    bldargs['end'] = end

    system.build(**bldargs)

    features_dir = os.path.join(system.configs.get('General', 'data_dir'), 'results')
    features_path = os.path.join(features_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    if os.path.isfile(features_path + '.h5'):
        with pd.HDFStore(features_path + '.h5', mode='r') as hdf:
            features = hdf.get('features')
    else:
        data = system._database.read(start, end)
        weather = system.forecast._weather._database.read(start, end)
        if system.contains_type('pv'):
            solar_yield = system.forecast._get_solar_yield(weather)
            data = pd.concat([data, solar_yield], axis=1)

        solar_position = system.forecast._get_solar_position(data.index)
        data = pd.concat([data, weather, solar_position], axis=1)

        features = system.forecast._model.features.extract(data)
        features = system.forecast._model.features._add_meta(features)
        features.to_hdf(features_path + '.h5', 'features', mode='w')

        if verbose:
            io.write_csv(system, features, features_path)

    logger.debug("Beginning predictions for system: {}".format(system.name))
    results.durations.start('Prediction')

    resolution_min = forecast.features.resolutions[0]
    if len(forecast.features.resolutions) > 1:
        for i in range(len(forecast.features.resolutions)-1, 0, -1):
            resolution_min = forecast.features.resolutions[i]
            if resolution_min.steps_horizon is not None:
                break

    resolution_max = forecast.features.resolutions[0]
    resolution_data = resolution_min.resample(features)

    # Reactivate this, when multiprocessing will be implemented
    # global logger
    # if process.current_process().name != 'MainProcess':
    #    logger = process.get_logger()

    interval = settings.getint('General', 'interval')
    end = ceil_date(features.index[-1] - resolution_max.time_horizon, timezone=system.location.tz)
    start = floor_date(features.index[0] + resolution_max.time_prior, timezone=system.location.tz)
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
            # If this step was simulated already, load the results and skip the prediction
            results.load(date_path+'/output')

            date = _next_date(date, interval)
            continue

        try:
            date_prior = date - resolution_max.time_prior
            date_start = date + resolution_max.time_step
            date_horizon = date + resolution_max.time_horizon
            date_features = deepcopy(resolution_data[date_prior:date_horizon])
            date_range = date_features[date_start:date_horizon].index

            input = forecast.features.input(date_features, date_range)
            target = forecast.features.target(date_features, date_range)
            prediction = forecast._predict(date_features, date)
            prediction.rename(columns={target: target + '_est' for target in forecast.features.target_keys}, inplace=True)

            result = pd.concat([target, prediction], axis=1)

            # Add error columns for all targets
            for target_key in forecast.features.target_keys:
                result[target_key + '_err'] = result[target_key + '_est'] - result[target_key]

            result = pd.concat([result, resolution_data.loc[result.index,
                                                            [column for column in resolution_data.columns
                                                             if column not in result.columns]]], axis=1)

            result.index.name = 'time'
            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)

            results[date_path+'/output'] = result
            results[date_path+'/input'] = input
            results[date_path+'/target'] = target

            if training_recursive and date >= training_date:
                if abs((training_date - date).total_seconds()) <= training_interval:
                    training_features = deepcopy(resolution_data[training_last - resolution_max.time_prior:training_date])
                    validation_features = deepcopy(resolution_data[training_last - dt.timedelta(days=7):training_last])

                    forecast._train(training_features, validation_features)

                training_last = training_date
                training_date = _next_date(date, training_interval)

            date = _next_date(date, interval)

        except ValueError as e:
            logger.debug("Skipping %s: %s", date, str(e))
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            date = _next_date(date, interval)

    results.durations.stop('Prediction')
    logger.debug("Predictions for system {} complete after {:.2f} minutes"
                 .format(system.name, results.durations['Prediction']))


def _launch_tensorboard(systems, **kwargs):
    # noinspection PyProtectedMember
    launch = any([system.forecast._model._tensorboard for system in systems])
    if launch and 'tensorboard' in kwargs:
        launch = kwargs['tensorboard'] if isinstance(kwargs['tensorboard'], bool) \
                                       else str(kwargs['tensorboard']).lower() == 'true'
    if launch:
        logging.getLogger('MARKDOWN').setLevel(logging.ERROR)
        logging.getLogger('tensorboard').setLevel(logging.ERROR)
        logger_werkzeug = logging.getLogger('werkzeug')
        logger_werkzeug.setLevel(logging.ERROR)
        logger_werkzeug.disabled = True

        tensorboard = program.TensorBoard()
        tensorboard.configure(argv=[None, '--logdir', kwargs['data_dir']])
        tensorboard_url = tensorboard.launch()

        logger.info("Started TensorBoard at {}".format(tensorboard_url))

    return launch


def _next_date(date: pd.Timestamp, interval: int) -> pd.Timestamp:
    date += dt.timedelta(minutes=interval)
    if interval >= 60:
        timezone = date.tzinfo

        # FIXME: verify pandas version includes fix in commit from 17. Nov 20221:
        # https://github.com/pandas-dev/pandas/pull/44357
        date = date.astimezone(tz.utc).round('{minutes}s'.format(minutes=interval))\
                   .astimezone(timezone)

    return date


def _get_date(time_str: str, timezone: tz.timezone) -> pd.Timestamp:
    return pd.Timestamp(timezone.localize(dt.datetime.strptime(time_str, '%d.%m.%Y')))


def _get_parser(root_dir: str) -> ArgumentParser:
    from th_e_fcst import __version__

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument('-r', '--root-directory',
                        dest='root_dir',
                        help="directory where the package and related libraries are located",
                        default=root_dir,
                        metavar='DIR')

    parser.add_argument('-c', '--config-directory',
                        dest='config_dir',
                        help="directory to expect configuration files",
                        default='conf',
                        metavar='DIR')

    parser.add_argument('-d', '--data-directory',
                        dest='data_dir',
                        help="directory to expect and write result files to",
                        default='data',
                        metavar='DIR')

    return parser


if __name__ == "__main__":
    run_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(simulate)))
    if os.path.basename(run_dir) == 'bin':
        run_dir = os.path.dirname(run_dir)

    os.chdir(run_dir)

    os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())

    if not os.path.exists('log'):
        os.makedirs('log')

    logging_file = os.path.join(os.path.join(run_dir, 'conf'), 'logging.cfg')
    if not os.path.isfile(logging_file):
        logging_default = logging_file.replace('logging.cfg', 'logging.default.cfg')
        if os.path.isfile(logging_default):
            shutil.copy(logging_default, logging_file)
        else:
            raise FileNotFoundError("Unable to open logging.cfg in: " +
                                    os.path.join(os.path.join(run_dir, 'conf')))

    # Load the logging configuration
    import logging
    import logging.config
    logging.config.fileConfig(logging_file)
    logging.getLogger('h5py').setLevel(logging.WARN)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('tensorflow').setLevel(logging.WARN)

    logger = logging.getLogger('th-e-simulation')

    simulate(_get_parser(run_dir).parse_args())
