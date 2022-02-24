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
import json

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

    verbose = settings.getboolean('General', 'verbose', fallback=False)

    start = _get_date(settings['General']['start'])
    end = _get_date(settings['General']['end']) + dt.timedelta(hours=23, minutes=59)

    systems = System.read(**kwargs)
    for system in systems:

        # Initialize dictionary to save info for later evaluation.
        system.simulation = {
            'results': 0,
            'durations': 0,
            'evaluation': 0
        }

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

                    if verbose:
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

                if verbose:
                    write_csv(system, features, features_path)

            durations['prediction'] = {
                'start': dt.datetime.now()
            }

            logging.debug("Beginning predictions for system: {}".format(system.name))

            system.simulation['results'] = simulate(settings, system, features)

            durations['prediction']['end'] = dt.datetime.now()
            durations['prediction']['minutes'] = (durations['prediction']['end'] -
                                                  durations['prediction']['start']).total_seconds() / 60.0

            logging.debug("Predictions for system {} complete after {} minutes"
                          .format(system.name, durations['prediction']['minutes']))

            durations['simulation']['end'] = dt.datetime.now()
            durations['simulation']['minutes'] = (durations['simulation']['end'] -
                                                  durations['simulation']['start']).total_seconds() / 60.0

            # Store results with its system, to summarize it later on
            system.simulation['durations'] = durations

        except Exception as e:
            error = True
            logger.error("Error simulating system %s: %s", system.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

    logger.info("Finished TH-E Simulation{0}".format('s' if len(systems) > 1 else ''))

    # OUTPUT OF THE PROGRAM IS GENERATED HERE
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

    results = pd.DataFrame()
    while date <= end:
        # Check if this step was simulated already and load the results, if so
        date_str = date.strftime('%Y%m%d_%H%M%S')
        date_dir = os.path.join(system_dir, 'results', date.strftime('%Y%m%d'))
        date_path = '/{0}.'.format(date)
        if date_path in datastore:
            result = datastore.get(date_path+'/outputs')
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

def evaluate(settings, systems):
    from th_e_sim.iotools import write_excel

    def _parse_eval(eval_config):

        sections = {'targets': str, 'metrics': str, 'conditions': 'condition',
                    'groups': str, 'group_bins': int, 'summary': str, 'boxplot': bool}

        values = [list() for i in range(len(sections))]
        eval_dict = dict(zip(sections.keys(), values))

        for key, parameters in eval_config.items():

            parameters = parameters.split(', ')
            pi = sections[key]

            if pi == str:

                for parameter in parameters:
                    eval_dict[key].append(pi(parameter))

            elif pi == bool:

                for parameter in parameters:
                    eval_dict[key].append(pi(parameter))

            elif pi == int:

                for parameter in parameters:
                    eval_dict[key].append(pi(parameter))

            elif pi == "condition":

                for parameter in parameters:
                    parameter = parameter.split(" ")
                    parameter[2] = json.loads(parameter[2])
                    eval_dict[key].append(parameter)

        return eval_dict

    def concat_evaluation(system, name, header, data):
        if data is None:
            return

        cols = [(header, system.name, col) for col in data.columns]
        data.columns = pd.MultiIndex.from_tuples(cols, names=['target', 'system', 'metrics'])
        if name not in evaluations.keys():
            evaluations[name] = data
        else:
            evaluations[name] = pd.concat([evaluations[name], data], axis=1)

    def add_evaluation(system, name, header, kpi, data=None):
        concat_evaluation(system, name, header, data)
        if kpi is not None:
            summary_tbl.loc[system.name, (header, name)] = kpi

    def _evaluate_data(system, data, level, column, file, **kwargs):
        from th_e_sim.iotools import print_boxplot
        try:
            _data = deepcopy(data)
            _data.index = data.index.get_level_values(level)
            print_boxplot(system, _data, _data.index, column, file, **kwargs)

        except ImportError as e:
            logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                                 str(e)))

        return _describe_data(system, data, data.index, column, file)

    def _print_boxplot(system, labels, data, file, **kwargs):
        from th_e_sim.iotools import print_boxplot
        try:
            colors = len(set(labels))
            print_boxplot(system, None, labels, data, file, colors=colors, **kwargs)

        except ImportError as e:
            logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                                 str(e)))

    def _describe_data(system, data, level, column, file):
        from th_e_sim.iotools import write_csv

        data = data[column]
        group = data.groupby(level)
        median = group.median()
        median.name = 'median'
        mae = data.abs().groupby(level).mean()
        mae.name = 'mae'
        rmse = (data ** 2).groupby(level).mean() ** .5
        rmse.name = 'rmse'
        description = pd.concat([rmse, mae, median, group.describe()], axis=1)
        description.index = pd.MultiIndex.from_tuples(description.index, names=data.index.names)
        del description['count']

        write_csv(system, description, file)

        return description

    def _extract_labels(data, groups, group_bins):

        def _gitterize(data: pd.Series, steps):
            from math import floor, ceil

            total_steps = steps
            f_max = ceil(data.max())
            f_min = floor(data.min())
            big_delta = f_max - f_min

            # Round step_size down
            small_delta = floor(big_delta / total_steps * 10) / 10

            if small_delta == 0:
                raise ValueError("The axis {} cannot be analyzed with the regular grid spacing of {} between"
                                 "grid points. Please choose a smaller number of steps".format(feature, small_delta))

            to_edge = big_delta - small_delta * total_steps
            extra_steps = ceil(to_edge / small_delta)
            total_steps = total_steps + extra_steps

            discrete_axis = [round(f_min + small_delta * x, 2) for x in range(total_steps + 1)]

            return discrete_axis, small_delta

        if len(groups) != len(group_bins):
            groups = groups[-len(group_bins):]

        d_axis = zip(groups, group_bins)
        gitterized = list()

        for feature, steps in d_axis:

            data[feature + '_d'] = data[feature]
            discrete_feature, step_size = _gitterize(data[feature], int(steps))
            gitterized.append(feature)

            for i in discrete_feature:
                i_loc = data[feature + '_d'] - i
                i_loc = (i_loc >= 0) & (i_loc < step_size)
                data.loc[i_loc, feature + '_d'] = i

        return gitterized


    def discrete_metrics(name, data, target, groups, conditions, metrics, summary, group_bins=None, boxplot=False, **kwargs):

        data = deepcopy(data)

        # replace continuous groups with discretized equivalents generated in _extract_labels
        if group_bins != None:
            gitter = _extract_labels(data, groups, group_bins)
            _groups = list()
            for group in groups:
                if group in gitter:
                    _groups.append(group + '_d')
                else:
                    _groups.append(group)

        # Index is unimportant; all information contained in the index which is important for the
        # analysis should be added as a seperate column.
        data.index = [x for x in range(len(data))]
        req_cols = list()
        req_cols.append(target)
        req_cols = req_cols + _groups

        if conditions:

            for i in range(len(conditions)):
                req_cols.append(conditions[i][0])

        req_cols = set(req_cols)

        if not req_cols.issubset(set(data.columns)):
            raise ValueError("The data does not contain the necessary columns for the "
                             "evaluation as configured in the config. Please ensure that "
                             "the following configured columns are as intended: {}".format(req_cols))

        def perform_metrics(name, data, err_col, groups, metrics, boxplot):

            data = deepcopy(data)
            _metrics = []
            for metric in metrics:

                if 'mae' == metric:

                    data[err_col] = data[err_col].abs()
                    mae = data.groupby(groups).mean()
                    ae_std = data.groupby(groups).std()
                    _metrics.append(mae)
                    _metrics.append(ae_std)

                elif 'mse' == metric:

                    data[err_col] = (data[err_col] ** 2)
                    mse = data.groupby(groups).mean()
                    se_std = data.groupby(groups).std()
                    _metrics.append(mse)
                    _metrics.append(se_std)

                elif 'rmse' == metric:

                    data[err_col] = (data[err_col] ** 2)
                    rmse = data.groupby(groups).mean() ** 0.5
                    rse_std = data.groupby(groups).std() ** 0.5
                    _metrics.append(rmse)
                    _metrics.append(rse_std)

                elif 'mbe' == metric:

                    mbe = data.groupby(groups).mean()
                    be_std = data.groupby(groups).std()
                    _metrics.append(mbe)
                    _metrics.append(be_std)

                else:
                    raise ValueError("The chosen metric {} has not yet been implemented".format(metric))

                if boxplot and len(groups) == 1:
                    _print_boxplot(system, data[groups[0]], data[err_col].values, os.path.join("evaluation", name, metric))

            # introduce count to data
            n = [1 for x in range(len(data))]
            n = pd.Series(n, index=data.index, name='count')
            data = pd.concat([data, n], axis=1)

            # count points in each group
            n = data[groups + ['count']].groupby(groups).sum()

            _metrics.append(n)

            # concatenate results
            metric_data = pd.concat(_metrics, axis=1)

            # Generate appropriate column names
            metrics_c1 = [metric for metric in metrics]
            metrics_c2 = [metric + '_std' for metric in metrics]
            metric_cols = list()

            for metric, std in zip(metrics_c1, metrics_c2):

                metric_cols.append(metric)
                metric_cols.append(std)

            metric_cols.append('count')
            metric_data.columns = metric_cols

            return metric_data

        def select_data(data, conditions):

            def select_rows(data, feature, operator, value, *args):

                series = data[feature]

                if operator.lower() in ['lt', '<']:
                    rows = (series < value)

                elif operator.lower() in ['gt', '>']:
                    rows = (series > value)

                elif operator.lower() in ['leq', '<=']:
                    rows = (series <= value)

                elif operator.lower() in ['geq', '>=']:
                    rows = (series >= value)

                elif operator.lower() in ['eq', '=', '==']:
                    rows = (series == value)

                else:
                    raise ValueError('An improper condition is present in the dict defining the kpi.')

                return rows

            _ps = pd.Series([True] * len(data), index=data.index)


            for c in conditions:

                if not c:
                    continue

                rows = select_rows(data, *c)
                _ps = _ps & rows

            selected = data.iloc[_ps.values]

            return selected

        def summarize(evaluation, metric, groups, option=None):

            options = ['horizon_weighted', 'mean', 'high_load_bias',
                       'err_per_load', 'optimist']

            w = pd.Series([4 / 7, 2 / 7, 1 / 7], name='weights')

            if option == 'mean':
                return evaluation[metric].mean()

            elif option == 'horizon_weighted':

                if not 'horizon' in groups:
                    raise ValueError("This summary is not compatible with your "
                                     "chosen group index {}".format(groups))
                ri = [1, 3, 6]
                w.index = ri
                weighted_sum = evaluation.loc[ri, metric].dot(w)
                return weighted_sum

            elif option == 'high_load_bias':
                # This calculation only works as long as the following assumption
                # is true: The error scales with target load
                qs = evaluation[metric].quantile([0.75, 0.5, 0.25])
                qs.index = w.index
                weighted_sum = qs.dot(w)
                return weighted_sum

            elif option == 'err_per_load':
                watt_series = pd.Series(evaluation.index, index=evaluation.index)
                watt_series = watt_series.iloc[(watt_series != 0).values]
                err_watt = evaluation.loc[watt_series.index, metric].div(watt_series)
                err_watt = err_watt.mean()
                return err_watt

            elif option == 'optimist':
                return evaluation[metric].min()

            else:
                raise ValueError('The current option is not yet available for metric summarization '
                                  'please choose one of the following options: {}'.format(options))

        #select data pertaining to the desired feature space to be examined
        data = select_data(data, conditions)

        # select err data pertaining to desired target
        err_col = target + '_err'

        eval_cols = [err_col] + _groups
        data = data[eval_cols]

        # calculate metrics
        evaluation = perform_metrics(name, data, err_col, _groups, metrics, boxplot)
        kpi = summarize(evaluation, metrics[0], _groups, option=summary[0])

        # calculate summary
        return evaluation, kpi

    summary_tbl = pd.DataFrame(index=[s.name for s in systems],
                               columns=pd.MultiIndex.from_tuples([('Durations [min]', 'Simulation'),
                                                                  ('Durations [min]', 'Prediction')]))
    evaluations = {}
    for system in systems:

        # index = pd.IndexSlice
        durations = system.simulation['durations']
        summary_tbl.loc[system.name, ('Durations [min]', 'Simulation')] = round(durations['simulation']['minutes'])
        summary_tbl.loc[system.name, ('Durations [min]', 'Prediction')] = round(durations['prediction']['minutes'])

        if 'training' in durations.keys():
            summary_tbl.loc[system.name, ('Durations [min]', 'Training')] = round(durations['training']['minutes'])

        # Retrieve eval configs
        data_dir = system.configs['General']['data_dir']
        eval_cfg = os.path.join(data_dir, 'conf', 'evaluation.cfg')
        eval_settings = ConfigParser()
        eval_settings.read(eval_cfg)

        # Load results
        results = system.simulation['results']

        # Extract additional features
        results['day_hour'] = [t.hour for t in results.index]
        results['weekday'] = [t.weekday for t in results.index]
        results['month'] = [t.month for t in results.index]

        # Perform Evaluations
        for name in eval_settings.sections():

            if name == "DEFAULT":
                continue

            config = _parse_eval(eval_settings[name])

            for target in config['targets']:
                # calculate metric
                metric, summary = discrete_metrics(name, results, target, **config)

                target_id = target.replace('_power', '')
                target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

                add_evaluation(system, name, target_name, summary, metric)

    write_excel(settings, summary_tbl, evaluations)

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

    logging.getLogger('matplotlib')\
           .setLevel(logging.WARN)

    main(_get_parser(run_dir).parse_args())
