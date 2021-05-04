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
import copy
import shutil
import inspect
import logging
import traceback
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import calendar as cal

from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser
from tensorboard import program

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))

TARGETS = {
    'pv': 'Photovoltaics',
    'el': 'Electrical',
    'th': 'Thermal'
}


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

    start = _get_time(settings['General']['start'])
    end = _get_time(settings['General']['end']) + dt.timedelta(hours=23, minutes=59)

    systems = System.read(**kwargs)
    for system in systems:
        logger.info('Starting TH-E-Simulation of system {}'.format(system.name))
        durations = {
            'simulation': {
                'start': dt.datetime.now()
            }
        }
        preparation.process_weather(system, os.path.join('\\\\zentrale', 'isc', 'abteilung-systeme', 'data', 'Meteoblue'))
        preparation.process_system(system, os.path.join('\\\\zentrale', 'isc', 'abteilung-systeme', 'data', 'OPSD'))
        try:
            if not system.forecast._model.exists():
                from th_e_sim.iotools import print_distributions

                logging.debug("Beginning training of neural network for system: {}".format(system.name))
                durations['training'] = {
                    'start': dt.datetime.now()
                }
                features = system.forecast._get_history(_get_time(settings['Training']['start']),
                                                        _get_time(settings['Training']['end'])
                                                        + dt.timedelta(hours=23, minutes=59))

                print_distributions(features, path=system.forecast._model.dir)

                system.forecast._model.train(features)

                durations['training']['end'] = dt.datetime.now()
                durations['training']['minutes'] = (durations['training']['end'] -
                                                    durations['training']['start']).total_seconds() / 60.0

                logging.debug("Training of neural network for system {} complete after {} minutes"
                              .format(system.name, durations['training']['minutes']))

            data = system._database.get(start, end)
            weather = system.forecast._weather._database.get(start, end)
            features = system.forecast._model._parse_features(pd.concat([data, weather], axis=1))
            features_file = os.path.join('validation', 'features')
            write_csv(system, features, features_file)
            durations['prediction'] = {
                'start': dt.datetime.now()
            }
            logging.debug("Beginning predictions for system: {}".format(system.name))

            durations['prediction']['end'] = dt.datetime.now()
            durations['prediction']['minutes'] = (durations['prediction']['end'] -
                                                  durations['prediction']['start']).total_seconds() / 60.0

            results = simulate(settings, system, features)

            for results_err in [c for c in results.columns if c.endswith('_err')]:
                results_file = os.path.join('results', results_err.replace('_err', '').replace('_power', ''))
                write_csv(system, results, results_file)

            logging.debug("Predictions for system {} complete {} minutes".format(system.name,
                                                                                 durations['prediction']['minutes']))

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
    logger.debug("Finished {0} Simulation{1} after: {2}".format(len(systems), 's' if len(systems) > 1 else '',
                                                                durations['simulation']['minutes']))

    logger.info("TH-E Simulation{0} finished".format('s' if len(systems) > 1 else ''))

    if not error:
        validate(settings, systems)

        if tensorboard:
            logger.info("TensorBoard will be kept running")

    while tensorboard and not error:
        try:
            time.sleep(100)

        except KeyboardInterrupt:
            tensorboard = False


def simulate(settings, system, features, **kwargs):
    forecast = system.forecast._model
    results = pd.DataFrame()

    if len(forecast.resolutions) == 1:
        resolution_min = forecast.resolutions[0]
    else:
        for i in range(len(forecast.resolutions)-1, 0, -1):
            resolution_min = forecast.resolutions[i]
            if resolution_min.steps_horizon is not None:
                break

    resolution_max = forecast.resolutions[0]
    resolution_data = resolution_min.resample(features)

    system_dir = system._configs['General']['data_dir']
    database = copy.deepcopy(system._database)
    database.dir = system_dir
    # database.format = '%Y%m%d'
    database.enabled = True

    # Reactivate this, when multiprocessing will be implemented
    # global logger
    # if process.current_process().name != 'MainProcess':
    #    logger = process.get_logger()

    verbose = settings.getboolean('General', 'verbose', fallback=False)
    interval = settings.getint('General', 'interval')
    time = features.index[0] + resolution_max.time_prior
    end = features.index[-1] - resolution_max.time_horizon

    training_recursive = settings.getboolean('Training', 'recursive', fallback=False)
    # training_interval = settings.getint('Training', 'interval')
    # training_last = time

    while time <= end:
        # Check if this step was simulated already and load the results, if so
        if database.exists(time, subdir='outputs'):
            result = database.get(time, subdir='outputs')
            results = pd.concat([results, result], axis=0)

            time += dt.timedelta(minutes=interval)
            continue

        try:
            step_result = list()
            step_prior = time - resolution_max.time_prior - resolution_max.time_step + dt.timedelta(seconds=1)
            step_horizon = time + resolution_max.time_horizon
            step_features = copy.deepcopy(features[step_prior:step_horizon])

            # Extract target feature reference values and scale them for evaluation later
            step_reference = resolution_data.loc[time:time+resolution_min.time_horizon, forecast.features['target']]
            step_reference = forecast._scale_features(step_reference, invert=True)

            # Remove target values from features, as those will be recursively filled with predictions
            step_features.loc[time:, forecast.features['target']] = np.NaN

            step = time
            step_index = step_features[step:step+resolution_min.time_horizon].index
            while step in step_index:
                step_next = step + resolution_min.time_step
                step_inputs = forecast._parse_inputs(step_features, step)

                if verbose:
                    database.persist(step_inputs,
                                     subdir='inputs',
                                     file=step.strftime('%Y%m%d_%H%M%S') + '.csv')

                inputs = np.squeeze(step_inputs.fillna(0).values)
                result = forecast._run_step(inputs)

                # Add predicted output to features of next iteration
                step_range = step_features[(step_features.index >= step) & (step_features.index < step_next)].index
                step_features.loc[step_range, forecast.features['target']] = result

                step_result.append(result)
                step = step_next

            if training_recursive:
                training_features = features[step_prior:step_horizon]

                forecast._train(training_features)
                forecast._save_model()

            result = pd.DataFrame(data=step_result, index=step_reference.index, columns=forecast.features['target'])
            result = forecast._scale_features(result, invert=True)
            result.rename('{}_est'.format, axis=1, inplace=True)
            result = pd.concat([step_reference, result], axis=1)

            for target in forecast.features['target']:
                result[target + '_err'] = result[target + '_est'] - result[target]

            result = pd.concat([result, step_features[forecast.features['input']]], axis=1)

            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)
            result.index.name = 'time'

            database.persist(result, subdir='outputs')

            results = pd.concat([results, result], axis=0)

        except ValueError as e:
            logger.debug("Skipping %s: %s", time, str(e))
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

        time += dt.timedelta(minutes=interval)

    return results


def validate(settings, systems):
    from th_e_sim.iotools import print_boxplot, write_excel

    def apollo(data, data_target):
        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('validation', data_name + '_apollo')

        doubt = data[data_target + '_doubt']
        data_cor = data.copy()
        data_cor.loc[:, data_column] = data_cor[data_column] - doubt
        data_cor.loc[data_cor[data_column] < 0, data_column] = 0
        data_cor = data_cor.drop(doubt[doubt >= data_cor[data_target].max()].index)

        data_desc = _validate_data(system, data_cor, data_cor.index.hour, data_column, data_file,
                                   label='Hours', title='Apollo')

        data_rmse = data_desc.transpose().loc[['rmse']]
        data_rmse.columns = ['Hour {}'.format(c + 1) for c in data_rmse.columns]
        data_rmse.index = [system.name]

        return (data_cor[data_column] ** 2).mean() ** .5, data_rmse

    def astraea(data, data_target):
        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('validation', data_name + '_astraea')
        data_desc = _validate_data(system, data, data.index.hour, data_column, data_file,
                                   label='Hours', title='Astraea')

        data_mae = data_desc.transpose().loc[['mae']]
        data_mae.columns = ['Hour {}'.format(c + 1) for c in data_mae.columns]
        data_mae.index = [system.name]

        data_rmse = pd.Series(index=range(7), dtype='float64')
        for day in data_rmse.index:
            day_data = data[data.index.day_of_week == day]
            day_file = os.path.join('validation', data_name + '_astraea_{}'.format(day + 1))
            day_desc = _validate_data(system, day_data, day_data.index.hour, data_column, day_file,
                                      label='Hours', title='Astraea ({})'.format(cal.day_name[day]))

            data_rmse[day] = (day_desc['mean'] ** 2).mean() ** .5

        return (data_rmse ** 2).mean() ** .5, data_mae

    def prometheus(data, data_target):
        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('validation', data_name + '_prometheus')

        data_mae = pd.DataFrame(index=[system.name])
        data_mae_weighted = []
        for horizon in range(24):
            horizon_mae = data.loc[data['horizon'] == horizon + 1, data_column].abs().mean()
            data_mae.loc[system.name, 'Horizon {}'.format(horizon + 1)] = horizon_mae
            data_mae_weighted.append(horizon_mae * (0.75 ** horizon))

        horizons = {}
        for horizon in [1, 3, 6, 12, 24]:
            horizon_data = data[data['horizon'] == horizon].assign(horizon=horizon)
            horizon_file = os.path.join('validation', data_name + '_prometheus_{}'.format(horizon))
            # horizon_desc = describe_data(horizon_data, horizon_data['horizon'], data_column, horizon_file)
            horizon_desc = _validate_data(system, horizon_data, horizon_data.index.hour, data_column, horizon_file,
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

    validations = {}
    for system in systems:
        results = system.simulation['results']
        durations = system.simulation['durations']

        # index = pd.IndexSlice
        summary.loc[system.name, ('Durations [min]', 'Simulation')] = round(durations['simulation']['minutes'])
        summary.loc[system.name, ('Durations [min]', 'Prediction')] = round(durations['prediction']['minutes'])

        if 'training' in durations.keys():
            summary.loc[system.name, ('Durations [min]', 'Training')] = round(durations['training']['minutes'])

        def concat_validation(name, header, data):
            if data is None:
                return

            data.columns = pd.MultiIndex.from_product([[header], data.columns])
            if name not in validations.keys():
                validations[name] = pd.DataFrame(columns=data.columns)

            validations[name] = pd.concat([validations[name], data], axis=0)

        def add_validation(name, header, kpi, data=None):
            concat_validation(name, header, data)
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
                    results = results[(results[columns_daylight] > 0).any()]

            add_validation('Apollo', target_name, *apollo(results, target))
            add_validation('Astraea', target_name, *astraea(results, target))
            add_validation('Prometheus', target_name, *prometheus(results, target))

    write_excel(settings, summary, validations)


def _validate_data(system, data, index, column, file, **kwargs):
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


def _get_time(time_str):
    return tz.utc.localize(dt.datetime.strptime(time_str, '%d.%m.%Y'))


def _get_parser(root_dir):
    from th_e_fcst import __version__

    def _to_bool(v):
        return v.lower() in ("yes", "true", "1")

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version',
                         action='version',
                         version='%(prog)s {version}'.format(version=__version__))

    parser.add_argument('-r','--root-directory',
                        dest='root_dir',
                        help="directory where the package and related libraries are located",
                        default=root_dir,
                        metavar='DIR')

    parser.add_argument('-c','--config-directory',
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
    root_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(root_dir) == 'bin':
        root_dir = os.path.dirname(root_dir)

    os.chdir(root_dir)

    if not os.path.exists('log'):
        os.makedirs('log')

    # Load the logging configuration
    import logging.config
    logging_file = os.path.join(os.path.join(root_dir, 'conf'), 'logging.cfg')
    logging.config.fileConfig(logging_file)
    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(root_dir).parse_args())

