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
import numpy as np
import pandas as pd
import datetime as dt
import calendar as cal

from copy import deepcopy
from argparse import ArgumentParser, RawTextHelpFormatter
from tensorboard import program
from typing import Union

from th_e_core import configs
from th_e_core.tools import floor_date, ceil_date
import th_e_data.io as io

TARGETS = {
    'pv': 'Photovoltaics',
    'el': 'Electrical',
    'th': 'Thermal'
}


# noinspection PyProtectedMember, SpellCheckingInspection
def main(args):
    from th_e_fcst import System
    from th_e_core.tools import ceil_date

    logger.info("Starting TH-E Simulation")

    settings = configs.read('settings.cfg', **vars(args))

    error = False
    kwargs = vars(args)
    kwargs.update(settings.items('General'))

    tensorboard = _launch_tensorboard(**kwargs)

    verbose = settings.getboolean('General', 'verbose', fallback=False)

    systems = System.read(**kwargs)
    for system in systems:
        logger.info('Starting TH-E-Simulation of system {}'.format(system.name))
        timezone = system.location.pytz
        durations = {
            'simulation': {
                'start': dt.datetime.now()
            }
        }
        start = _get_time(settings['General']['start'], timezone)
        end = _get_time(settings['General']['end'], timezone)
        end = ceil_date(end, system.location.pytz)

        start_train = _get_time(settings['Training']['start'], timezone)
        end_train = _get_time(settings['Training']['end'], timezone)
        end_train = ceil_date(end_train, system.location.pytz)

        bldargs = dict(kwargs)
        bldargs['start'] = start_train
        bldargs['end'] = end

        system.build(**bldargs)
        try:
            if not system.forecast._model.exists():
                logger.debug("Beginning training of neural network for system: {}".format(system.name))
                durations['training'] = {
                    'start': dt.datetime.now()
                }
                features_path = os.path.join(system.configs.get('General', 'data_dir'), 'model', 'features')
                if os.path.isfile(features_path + '.h5'):
                    with pd.HDFStore(features_path + '.h5', mode='r') as hdf:
                        features = hdf.get('features').loc[start_train:end_train]
                else:
                    features = system.forecast._get_data_history(start_train, end_train)
                    features = system.forecast._model._parse_features(features)
                    features = system.forecast._model._add_meta(features)
                    features.to_hdf(features_path + '.h5', 'features', mode='w')

                    if verbose:
                        io.write_csv(system, features, features_path)
                        io.print_distributions(features, path=system.forecast._model.dir)

                system.forecast._model._train(features)

                durations['training']['end'] = dt.datetime.now()
                durations['training']['minutes'] = (durations['training']['end'] -
                                                    durations['training']['start']).total_seconds() / 60.0

                logger.debug("Training of neural network for system {} complete after {:.2f} minutes"
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
                    solar_yield = system.forecast._get_solar_yield(weather)
                    data = pd.concat([data, solar_yield], axis=1)

                solar_position = system.forecast._get_solar_position(data.index)
                data = pd.concat([data, weather, solar_position], axis=1)

                features = system.forecast._model._parse_features(data)
                features = system.forecast._model._add_meta(features)
                features.to_hdf(features_path + '.h5', 'features', mode='w')

                if verbose:
                    io.write_csv(system, features, features_path)

            durations['prediction'] = {
                'start': dt.datetime.now()
            }
            logger.debug("Beginning predictions for system: {}".format(system.name))

            results = simulate(settings, system, features)

            durations['prediction']['end'] = dt.datetime.now()
            durations['prediction']['minutes'] = (durations['prediction']['end'] -
                                                  durations['prediction']['start']).total_seconds() / 60.0

            for results_err in [c for c in results.columns if c.endswith('_err')]:
                results_file = os.path.join('results', results_err.replace('_err', '').replace('_power', ''))
                io.write_csv(system, results, results_file)

            logger.debug("Predictions for system {} complete after {:.2f} minutes"
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
            # keep_running = input('Keep tensorboard running? [y/n]').lower()
            # keep_running = keep_running in ['j', 'ja', 'y', 'yes', 'true']
            keep_running = settings.getboolean('General', 'keep_running', fallback=False)
            while keep_running:
                try:
                    time.sleep(100)

                except KeyboardInterrupt:
                    keep_running = False


# noinspection PyProtectedMember
def simulate(settings, system, features, **kwargs):
    forecast = system.forecast._model

    resolution_min = forecast.features.resolutions[0]
    if len(forecast.features.resolutions) > 1:
        for i in range(len(forecast.features.resolutions)-1, 0, -1):
            resolution_min = forecast.features.resolutions[i]
            if resolution_min.steps_horizon is not None:
                break

    resolution_max = forecast.features.resolutions[0]
    resolution_data = resolution_min.resample(features)

    system_dir = system.configs['General']['data_dir']
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

    # training_recursive = settings.getboolean('Training', 'recursive', fallback=True)
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
            # input = datastore.get(date_path+'/input')
            # target = datastore.get(date_path+'/target')

            # results[date] = (input, target, prediction)
            results = pd.concat([results, result], axis=0)

            date = _increment_date(date, interval)
            continue

        try:
            date_prior = date - resolution_max.time_prior
            date_start = date + resolution_max.time_step
            date_horizon = date + resolution_max.time_horizon
            date_features = deepcopy(resolution_data[date_prior:date_horizon])
            date_range = date_features[date_start:date_horizon].index

            inputs = forecast.features.input(date_features, date_range)
            targets = forecast.features.target(date_features, date_range)
            prediction = forecast._predict(date_features, date)
            prediction.rename(columns={target: target + '_est' for target in forecast.features.target_keys}, inplace=True)

            # results[date] = (input, target, prediction)
            result = pd.concat([targets, prediction], axis=1)

            for target in forecast.features.target_keys:
                result[target + '_err'] = result[target + '_est'] - result[target]

            result = pd.concat([result, resolution_data.loc[result.index,
                                                            [column for column in resolution_data.columns
                                                             if column not in result.columns]]], axis=1)

            result.index.name = 'time'
            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)
            results = pd.concat([results, result], axis=0)

            result.to_hdf(datastore, date_path+'/outputs')
            inputs.to_hdf(datastore, date_path+'/input')
            targets.to_hdf(datastore, date_path+'/target')
            if verbose:
                os.makedirs(date_dir, exist_ok=True)
                database.write(result,  file=date_str+'_outputs.csv', subdir=date_dir, rename=False)
                database.write(inputs,  file=date_str+'_inputs.csv',  subdir=date_dir, rename=False)
                database.write(targets, file=date_str+'_targets.csv', subdir=date_dir, rename=False)

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

    # noinspection PyProtectedMember
    def apollo(data, data_target):
        data_doubt = data_target + '_doubt'
        data_doubts = system.forecast._model.features._doubt
        if data_target not in data_doubts:
            return None, None

        data_column = data_target + '_err'
        data_name = data_target.replace('_power', '')
        data_file = os.path.join('evaluation', data_name + '_apollo')

        doubt = (data[data_doubt]/data[data_target]).replace(np.inf, 0).abs().clip(upper=1)
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
            io.print_boxplot(system, horizons_data, horizons_data.index.hour, data_column, data_file,
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

    # TODO: implement durations metadata json file and load them e.g. if H5 file already exists
    headers = [('Durations [min]', 'Simulation'), ('Durations [min]', 'Prediction')]
    if any('training' in system.durations.keys() for system in systems):
        headers.append(('Durations [min]', 'Training'))

    summary = pd.DataFrame(index=[s.name for s in systems],
                           columns=pd.MultiIndex.from_tuples(headers))

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

        interval = settings.getint('General', 'interval')
        for target in system.forecast._model.features.target_keys:
            target_id = target.replace('_power', '')
            target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

            if target+'_err' not in results.columns:
                continue

            target_results = results[target+'_err']
            target_iqr = target_results.quantile(0.75) - target_results.quantile(0.25)
            results = results[np.abs((target_results - target_results.median()) / target_iqr) < 2.22]
#            results = results[np.abs(stats.zscore(results[target+'_err'])) < 3]
#            results = results[np.logical_or(target_results < target_results.quantile(0.99),
#                                            target_results > target_results.quantile(0.01))]

            if target_id in ['pv']:
                columns_daylight = np.intersect1d(results.columns, ['ghi', 'dni', 'dhi', 'solar_elevation'])
                if len(columns_daylight) > 0:
                    results = results[(results[columns_daylight] > 0).any(axis=1)]

            if target_id in ['pv', 'el']:
                import calendar

                power_file = os.path.join('evaluation', target)
                power_colors = ['#004F9E', '#4f779e']
                power_data = results[results.horizon < interval/60]
                power_data.rename(columns={
                    target+'_est': 'Prediction',
                    target:        'Measurement'
                }, inplace=True)
                power_data = power_data.loc[:, ['Prediction', 'Measurement']]
                power_data['hours'] = power_data.index.hour
                power_melt = pd.melt(power_data, 'hours', value_name='power', var_name='results')
                _print_data(system, power_melt, 'hours', 'power', power_file, hue='results', style='results',
                            label='Hours of the day', title='{} power'.format(TARGETS[target_id]), colors=power_colors)

                for month in [m+1 for m in range(12)]:
                    month_file = os.path.join('evaluation', target_id + '_month{}'.format(month))
                    month_title = '{} power ({})'.format(TARGETS[target_id], calendar.month_name[month])
                    month_data = power_data[power_data.index.month == month]
                    month_melt = pd.melt(month_data, 'hours', value_name='power', var_name='results')
                    _print_data(system, month_melt, 'hours', 'power', month_file, hue='results', style='results',
                                label='Hours of the day', title=month_title, colors=power_colors)

                if target_id in ['el']:
                    for day in range(7):
                        day_file = os.path.join('evaluation', target_id + '_day{}'.format(day+1))
                        day_title = '{} power ({})'.format(TARGETS[target_id], calendar.day_name[day])
                        day_data = power_data[(power_data.index.day_of_week == day)]
                        day_melt = pd.melt(day_data, 'hours', value_name='power', var_name='results')
                        _print_data(system, day_melt, 'hours', 'power', day_file, hue='results', style='results',
                                    label='Hours of the day', title=day_title, colors=power_colors)

            add_evaluation('Apollo', target_name, *apollo(deepcopy(results), target))
            add_evaluation('Astraea', target_name, *astraea(deepcopy(results), target))
            add_evaluation('Prometheus', target_name, *prometheus(deepcopy(results), target))

    io.write_excel(settings, summary, evaluations)


def _evaluate_data(system, data, index, column, file, **kwargs):
    try:
        io.print_boxplot(system, data, index, column, file, **kwargs)

    except ImportError as e:
        logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                             str(e)))

    return _describe_data(system, data, index, column, file)


def _describe_data(system, data, index, column, file, write_file=True):
    from th_e_data.io import write_csv

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

    if write_file:
        write_csv(system, description, file)

    return description


def _print_data(system, data, index, column, file, **kwargs):
    from th_e_data.io import print_lineplot
    try:
        print_lineplot(system, data, index, column, file, **kwargs)

    except ImportError as e:
        logger.debug("Unable to plot lineplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                              str(e)))


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
    increment_freq = '{}min'.format(interval)
    increment_delta = dt.timedelta(minutes=interval)
    increment_count = 1

    increment_date = pd.NaT
    while increment_date is pd.NaT or increment_date <= date:
        increment_date = (date + increment_count*increment_delta).floor(increment_freq, ambiguous='NaT')
        increment_count += 1

    return increment_date


def _get_time(time_str: str, timezone: tz.timezone) -> pd.Timestamp:
    return pd.Timestamp(timezone.localize(dt.datetime.strptime(time_str, '%d.%m.%Y')))


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
    logging.getLogger('tensorflow').setLevel(logging.INFO)

    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(run_dir).parse_args())
