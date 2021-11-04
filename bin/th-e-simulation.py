#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-simulation
    ~~~~~~~~~~~~~~~
    
    
    To learn how to configure specific settings, see "th-e-simulation --help"

"""
import os
import sys
import time
import inspect
import traceback
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import calendar as cal

from copy import deepcopy
from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser
from tensorboard import program
from typing import Union

from th_e_core.tools import floor_date, ceil_date

from tables import NaturalNameWarning
import warnings
warnings.filterwarnings('ignore', category=NaturalNameWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))

TARGETS = {
    'pv': 'Photovoltaics',
    'el': 'Electrical',
    'th': 'Thermal'
}


# noinspection PyProtectedMember
def main(args):
    from th_e_sim.iotools import write_csv
    from th_e_sim import preparation
    from th_e_fcst import System

    logger.info("Starting TH-E Simulation")

    settings_file = os.path.join(args.config_dir, 'settings.cfg')
    if not os.path.isfile(settings_file):
        raise ValueError('Unable to open simulation settings: {}'.format(settings_file))

    settings = ConfigParser()
    settings.read(settings_file)

    error = False
    kwargs = vars(args)
    kwargs.update(dict(settings.items('General')))

    tensorboard = _launch_tensorboard(**kwargs)

    start = _get_date(settings['General']['start'])
    end = _get_date(settings['General']['end']) + dt.timedelta(hours=23, minutes=59)

    systems = System.read(**kwargs)
    for system in systems:
        logger.info('Starting TH-E-Simulation of system {}'.format(system.name))
        durations = {
            'simulation': {
                'start': dt.datetime.now()
            }
        }
        zentrale = os.path.join('\\\\zentrale', 'isc', 'abteilung-systeme')
        preparation.process_weather(system, os.path.join(zentrale, 'data', 'Meteoblue'))
        preparation.process_system(system, os.path.join(zentrale, 'data', 'OPSD'))
        try:
            if not system.forecast._model.exists():
                from th_e_sim.iotools import print_distributions

                logging.debug("Beginning training of neural network for system: {}".format(system.name))
                durations['training'] = {
                    'start': dt.datetime.now()
                }
                features_path = os.path.join(system.configs.get('General', 'data_dir'), 'model', 'features')
                if os.path.isfile(features_path + '.h5'):
                    with pd.HDFStore(features_path + '.h5', mode='r') as hdf:
                        features = hdf.get('features')
                else:
                    features = system.forecast._get_history(_get_date(settings['Training']['start']),
                                                            _get_date(settings['Training']['end'])
                                                            + dt.timedelta(hours=23, minutes=59))

                    features = system.forecast._model._parse_features(features)
                    features.to_hdf(features_path + '.h5', 'features', mode='w')
                    write_csv(system, features, features_path)

                    if settings.getboolean('General', 'verbose', fallback=False):
                        print_distributions(features, path=system.forecast._model.dir)

                system.forecast._model._train(features)

                durations['training']['end'] = dt.datetime.now()
                durations['training']['minutes'] = (durations['training']['end'] -
                                                    durations['training']['start']).total_seconds() / 60.0

                logging.debug("Training of neural network for system {} complete after {} minutes"
                              .format(system.name, durations['training']['minutes']))

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
                    solar = system.forecast._get_yield(weather)
                    data = pd.concat([data, solar], axis=1)

                features = system.forecast._model._parse_features(pd.concat([data, weather], axis=1))
                features.to_hdf(features_path + '.h5', 'features', mode='w')
                write_csv(system, features, features_path)

            durations['prediction'] = {
                'start': dt.datetime.now()
            }
            logging.debug("Beginning predictions for system: {}".format(system.name))

            results = simulate(settings, system, features)

            durations['prediction']['end'] = dt.datetime.now()
            durations['prediction']['minutes'] = (durations['prediction']['end'] -
                                                  durations['prediction']['start']).total_seconds() / 60.0

            for results_err in [c for c in results.columns if c.endswith('_err')]:
                results_file = os.path.join('results', results_err.replace('_err', '').replace('_power', ''))
                write_csv(system, results, results_file)

            logging.debug("Predictions for system {} complete after {} minutes"
                          .format(system.name, durations['prediction']['minutes']))

            durations['simulation']['end'] = dt.datetime.now()
            durations['simulation']['minutes'] = (durations['simulation']['end'] -
                                                  durations['simulation']['start']).total_seconds() / 60.0

            # Store results with its system, to summarize it later on
            system.simulation = {
                'results': results,
                'durations': durations
            }

        except Exception as e:
            error = True
            logger.error("Error simulating system %s: %s", system.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

    logger.info("Finished TH-E Simulation{0}".format('s' if len(systems) > 1 else ''))

    if not error:
        evaluate(settings, systems)

        if tensorboard:
            logger.info("TensorBoard will be kept running")

    while tensorboard and not error:
        try:
            time.sleep(100)

        except KeyboardInterrupt:
            tensorboard = False


# noinspection PyProtectedMember
def simulate(settings, system, features, **kwargs):
    forecast = system.forecast._model

    resolution_min = forecast.resolutions[0]
    if len(forecast.resolutions) > 1:
        for i in range(len(forecast.resolutions)-1, 0, -1):
            resolution_min = forecast.resolutions[i]
            if resolution_min.steps_horizon is not None:
                break

    resolution_max = forecast.resolutions[0]
    resolution_data = resolution_min.resample(features)

    system_dir = system._configs['General']['data_dir']
    database = deepcopy(system._database)
    database.dir = os.path.join(system_dir, 'results')
    # database.format = '%Y%m%d'
    database.enabled = True
    datastore = pd.HDFStore(os.path.join(system_dir, 'results', 'results.h5'))

    # Reactivate this, when multiprocessing will be implemented
    # global logger
    # if process.current_process().name != 'MainProcess':
    #    logger = process.get_logger()

    verbose = settings.getboolean('General', 'verbose', fallback=False)

    interval = settings.getint('General', 'interval')
    date = floor_date(features.index[0] + resolution_max.time_prior, timezone=system.location.tz)
    end = ceil_date(features.index[-1] - resolution_max.time_horizon, timezone=system.location.tz)

    # training_recursive = settings.getboolean('Training', 'recursive', fallback=False)
    # training_interval = settings.getint('Training', 'interval')
    # training_last = time

    # results = {}
    results = pd.DataFrame()
    while date <= end:
        # Check if this step was simulated already and load the results, if so
        date_str = date.strftime('%Y%m%d_%H%M%S')
        date_dir = os.path.join(system_dir, 'results', date.strftime('%Y%m%d'))
        date_path = '/{0}.'.format(date)
        if date_path in datastore:
            result = datastore.get(date_path+'/outputs')
            # inputs = datastore.get(date_path+'/inputs')
            # targets = datastore.get(date_path+'/targets')

            # results[date] = (inputs, targets, prediction)
            results = pd.concat([results, result], axis=0)

            date = _increment_date(date, interval)
            continue

        try:
            date_prior = date - resolution_max.time_prior
            date_start = date + resolution_max.time_step
            date_horizon = date + resolution_max.time_horizon
            date_features = deepcopy(resolution_data[date_prior:date_horizon])
            date_range = date_features[date_start:date_horizon].index

            inputs = forecast._parse_inputs(date_features, date_range)
            targets = forecast._parse_targets(date_features, date_range)
            prediction = forecast._predict(date_features, date)

            # results[date] = (inputs, targets, prediction)
            result = pd.concat([targets,
                                prediction.rename(columns={
                                    target: target+'_est' for target in forecast.features['target']
                                })],
                               axis=1)

            for target in forecast.features['target']:
                result[target + '_err'] = result[target + '_est'] - result[target]

            result = pd.concat([result, resolution_data.loc[result.index, np.setdiff1d(forecast.features['input'],
                                                                                       forecast.features['target'],
                                                                                       assume_unique=True)]], axis=1)

            result.index.name = 'time'
            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)
            results = pd.concat([results, result], axis=0)

            result.to_hdf(datastore, date_path+'/outputs')
            inputs.to_hdf(datastore, date_path+'/inputs')
            targets.to_hdf(datastore, date_path+'/targets')
            if verbose:
                os.makedirs(date_dir, exist_ok=True)
                database.write(result,  file=date_str+'_outputs.csv', subdir=date_dir)
                database.write(inputs,  file=date_str+'_inputs.csv',  subdir=date_dir)
                database.write(targets, file=date_str+'_targets.csv', subdir=date_dir)

            date = _increment_date(date, interval)

        except ValueError as e:
            logger.debug("Skipping %s: %s", date, str(e))
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            date = _increment_date(date, interval)

    database.close()
    datastore.close()

    return results


# noinspection PyProtectedMember
def evaluate(settings, systems):
    from th_e_sim.iotools import print_boxplot, write_excel

    # noinspection PyProtectedMember
    def apollo(data, data_target):
        data_doubt = data_target + '_doubt'
        data_doubts = system.forecast._model.features.get('doubt', {})
        if data_target not in data_doubts:
            return

        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('evaluation', data_name + '_apollo')

        doubt = data[data_doubt]
        data.loc[:, data_column] = data[data_column] - data[data_column] * doubt

        data_dates = pd.DataFrame(index=list(set(data.index.date)))
        for date in data_dates.index:
            data_dates.loc[date, data_doubt] = data.loc[data.index.date == date, data_doubt].mean()
            data_dates.loc[date, data_target] = data.loc[data.index.date == date, data_target].mean()
        data_dates = data_dates.loc[data_dates[data_target] > data_dates[data_target].quantile(0.75), data_doubt]

        logger.debug("Most accurately forecasted days: \n%s", "\n".join(
                    ["{}. {}".format(i+1, d) for i, d in enumerate(data_dates.abs().sort_values().head(10).index)]))

        data_desc = _evaluate_data(system, data, data.index.hour, data_column, data_file,
                                   label='Hours', title='Apollo')

        data_rmse = data_desc.transpose().loc[['rmse']]
        data_rmse.columns = ['Hour {}'.format(int(c) + 1) for c in data_rmse.columns]
        data_rmse.index = [system.name]

        return (data[data_column] ** 2).mean() ** .5, data_rmse

    # noinspection PyProtectedMember
    def astraea(data, data_target):
        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('evaluation', data_name + '_astraea')
        data_desc = _evaluate_data(system, data, data.index.hour, data_column, data_file,
                                   label='Hours', title='Astraea')

        data_mae = data_desc.transpose().loc[['mae']]
        data_mae.columns = ['Hour {}'.format(int(c) + 1) for c in data_mae.columns]
        data_mae.index = [system.name]

        data_rmse = pd.Series(index=range(7), dtype='float64')
        for day in data_rmse.index:
            day_data = data[data.index.day_of_week == day]
            day_file = os.path.join('evaluation', data_name + '_astraea_{}'.format(day + 1))
            day_desc = _evaluate_data(system, day_data, day_data.index.hour, data_column, day_file,
                                      label='Hours', title='Astraea ({})'.format(cal.day_name[day]))

            data_rmse[day] = (day_desc['mean'] ** 2).mean() ** .5

        return (data_rmse ** 2).mean() ** .5, data_mae

    # noinspection PyProtectedMember
    def prometheus(data, data_target):
        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('evaluation', data_name + '_prometheus')

        data_mae = pd.DataFrame(index=[system.name])
        data_mae_weighted = []
        for horizon in range(24):
            horizon_mae = data.loc[data['horizon'] == horizon + 1, data_column].abs().mean()
            data_mae.loc[system.name, 'Horizon {}'.format(horizon + 1)] = horizon_mae
            data_mae_weighted.append(horizon_mae * (0.75 ** horizon))

        horizons = {}
        for horizon in [1, 3, 6, 12, 24]:
            horizon_data = data[data['horizon'] == horizon].assign(horizon=horizon)
            horizon_file = os.path.join('evaluation', data_name + '_prometheus_{}'.format(horizon))
            # horizon_desc = describe_data(horizon_data, horizon_data['horizon'], data_column, horizon_file)
            horizon_desc = _evaluate_data(system, horizon_data, horizon_data.index.hour, data_column, horizon_file,
                                          label='Horizons', title='Prometheus ({})'.format(horizon))

            horizons[horizon] = horizon_data

        try:
            horizons_data = pd.concat([horizons[1], horizons[12], horizons[24]])
            print_boxplot(system, horizons_data, horizons_data.index.hour, data_column, data_file,
                          label='Horizons', title='Prometheus', hue='horizon', colors=5)

        except ImportError as e:
            logger.debug(
                "Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(data_file), system.name,
                                                                        str(e)))

        return (np.array(data_mae_weighted) ** 2).mean() ** .5, data_mae

    # def weights(data, data_target):
    #     trainable_count = int(
    #         np.sum([K.count_params(p) for p in system.forecast._model.model.trainable_weights]))
    #     non_trainable_count = int(
    #         np.sum([K.count_params(p) for p in system.forecast._model.model.non_trainable_weights]))
    #     total_count = trainable_count + non_trainable_count
    #
    #     return None, total_count

    summary = pd.DataFrame(index=[s.name for s in systems],
                           columns=pd.MultiIndex.from_tuples([('Durations [min]', 'Simulation'),
                                                              ('Durations [min]', 'Prediction')]))

    evaluations = {}
    for system in systems:
        results = system.simulation['results']
        durations = system.simulation['durations']

        # index = pd.IndexSlice
        summary.loc[system.name, ('Durations [min]', 'Simulation')] = round(durations['simulation']['minutes'])
        summary.loc[system.name, ('Durations [min]', 'Prediction')] = round(durations['prediction']['minutes'])

        if 'training' in durations.keys():
            summary.loc[system.name, ('Durations [min]', 'Training')] = round(durations['training']['minutes'])

        def concat_evaluation(name, header, data):
            if data is None:
                return

            data.columns = pd.MultiIndex.from_product([[header], data.columns])
            if name not in evaluations.keys():
                evaluations[name] = pd.DataFrame(columns=data.columns)

            evaluations[name] = pd.concat([evaluations[name], data], axis=0)

        def add_evaluation(name, header, kpi, data=None):
            concat_evaluation(name, header, data)
            if kpi is not None:
                summary.loc[system.name, (header, name)] = kpi

        for target in system.forecast._model.features['target']:
            target_id = target.replace('_power', '')
            target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

            if target+'_err' not in results.columns:
                continue

            if target_id in ['pv']:
                columns_daylight = np.intersect1d(results.columns, ['ghi', 'dni', 'dhi', 'solar_elevation'])
                if len(columns_daylight) > 0:
                    results = results[(results[columns_daylight] > 0).any(axis=1)]

            add_evaluation('Apollo', target_name, *apollo(deepcopy(results), target))
            add_evaluation('Astraea', target_name, *astraea(deepcopy(results), target))
            add_evaluation('Prometheus', target_name, *prometheus(deepcopy(results), target))

    write_excel(settings, summary, evaluations)


def _evaluate_data(system, data, index, column, file, **kwargs):
    from th_e_sim.iotools import print_boxplot
    try:

        print_boxplot(system, data, index, column, file, **kwargs)

    except ImportError as e:
        logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                             str(e)))

    return _describe_data(system, data, index, column, file)


def _describe_data(system, data, index, column, file):
    from th_e_sim.iotools import write_csv

    data = data[column]
    group = data.groupby([index])
    median = group.median()
    median.name = 'median'
    mae = data.abs().groupby([index]).mean()
    mae.name = 'mae'
    rmse = (data ** 2).groupby([index]).mean() ** .5
    rmse.name = 'rmse'
    description = pd.concat([rmse, mae, median, group.describe()], axis=1)
    description.index.name = 'index'
    del description['count']

    write_csv(system, description, file)

    return description


def _launch_tensorboard(**kwargs):
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


def _increment_date(date: Union[dt.datetime, pd.Timestamp], interval: int) -> Union[dt.datetime, pd.Timestamp]:
    freq = '{}min'.format(interval)
    delta = dt.timedelta(minutes=interval)
    date_inc = (date + delta).floor(freq)
    if date_inc <= date:
        date_inc = (date + 2*delta).floor(freq)
    return date_inc


def _get_date(date_str: str) -> dt.datetime:
    return tz.utc.localize(dt.datetime.strptime(date_str, '%d.%m.%Y'))


def _get_parser(root_dir: str) -> ArgumentParser:
    from th_e_fcst import __version__

    def _to_bool(v: str) -> bool:
        return v.lower() in ("yes", "true", "1")

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

    parser.add_argument('-tb', '--tensorboard',
                        dest='tensorboard',
                        help="Launches TensorBoard at the selected data directory",
                        type=_to_bool,
                        default=False)

    return parser


if __name__ == "__main__":
    run_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(run_dir) == 'bin':
        run_dir = os.path.dirname(run_dir)

    os.chdir(run_dir)

    if not os.path.exists('log'):
        os.makedirs('log')

    # Load the logging configuration
    import logging
    import logging.config
    logging_file = os.path.join(os.path.join(run_dir, 'conf'), 'logging.cfg')
    logging.config.fileConfig(logging_file)
    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(run_dir).parse_args())
