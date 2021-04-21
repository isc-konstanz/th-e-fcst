#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-simulation
    ~~~~~~~~~~~~~~~


    To learn how to configure specific settings, see "th-e-simulation --help"

"""
import logging

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))

import copy
import shutil
import inspect
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import dateutil.relativedelta as rd
import matplotlib.pyplot as plt
from keras import backend as K

from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser


def main(args):
    from th_e_fcst import System

    logger.info("Starting TH-E Simulation")

    settings_file = os.path.join(args.config_dir, 'settings.cfg')
    if not os.path.isfile(settings_file):
        raise ValueError('Unable to open simulation settings: {}'.format(settings_file))

    settings = ConfigParser()
    settings.read(settings_file)

    kwargs = vars(args)
    kwargs.update(dict(settings.items('General')))

    start = _get_time(settings['General']['start'])
    end = _get_time(settings['General']['end']) + dt.timedelta(hours=23, minutes=59)

    systems = System.read(**kwargs)
    for system in systems:
        logger.info('Starting TH-E-Simulation of model {}'.format(system.id))
        start_simulation = dt.datetime.now()
        _prepare_weather(system)
        _prepare_system(system)

        if not system.forecast._model.exists():
            logging.info("Beginning network training of model {}".format(system.id))
            start_training = dt.datetime.now()
            system.forecast._model.train(system.forecast._get_history(_get_time(settings['Training']['start']),
                                                                      _get_time(settings['Training']['end']) \
                                                                      + dt.timedelta(hours=23, minutes=59)))
            end_training = dt.datetime.now()
            logging.info("Network training of model {} complete".format(system.id))
            train_time = end_training - start_training
            logging.info("Network training lasted: {}".format(train_time))

        logging.info("Beginning network predictions of model {}".format(system.id))
        start_prediction = dt.datetime.now()
        val_dir = os.path.join('lib', 'validation')
        results = pd.DataFrame()
        val_features = pd.DataFrame()
        for file in os.listdir(val_dir):
            features = pd.read_csv(os.path.join(val_dir, file),
                                   index_col=0, parse_dates=True,
                                   infer_datetime_format=True)
            val_features = pd.concat([val_features, features])
            results = pd.concat([results, _simulate(settings, system, features)])

        # save feature distributions
        system.forecast._model.data_distributions(val_features, train=False)
        end_prediction = dt.datetime.now()
        pred_time = end_prediction - start_prediction
        logging.info("Network predictions of model {} complete".format(system.id))
        logging.info('Network prediction lasted: {}'.format(pred_time))

        # Do not evaluate horizon, if forecast is done in a daily or higher interval
        if settings.getint('General', 'interval') < 1440:
            results_horizon1 = _result_horizon(system, results, 1)
            results_horizon3 = _result_horizon(system, results, 3)
            results_horizon6 = _result_horizon(system, results, 6)
            results_horizon12 = _result_horizon(system, results, 12)
            results_horizon24 = _result_horizon(system, results, 24)

            # results_horizons = pd.concat([results_horizon1, results_horizon3, results_horizon6, results_horizon12, results_horizon24])
            results_horizons = pd.concat([results_horizon1, results_horizon6, results_horizon24])
            _result_boxplot(system, results_horizons, results_horizons.index.hour, name='horizons', label='Hours',
                            hue='horizon', colors=5)

        _result_hours(system, results, name='hours')

        end_simulation = dt.datetime.now()
        logger.info("Finished TH-E Simulation of model {}".format(system.id))
        logger.info("TH-E Simulation lasted: {}".format(end_simulation - start_simulation))
        sim_time = end_simulation - start_simulation

        try:
            kpi = _result_summary(system, results, sim_time, train_time, pred_time)
        except NameError:
            logging.warning('train_time set to {}:'.format(0) +
                            'Training for {}'.format(system.id) +
                            'did not occur due to preexisting model.')
            train_time = 0
            kpi = _result_summary(system, results, sim_time, train_time, pred_time)

        interval = settings.getint('General', 'interval') / 60
        results = results[results['horizon'] <= interval].sort_index()
        del results['horizon']
        _result_write(system, results)

        for key, info in kpi.items():
            _result_write(system, info, results_name=key, results_dir='results')

    _result_comparison(systems)


def _simulate(settings, system, features, **kwargs):
    forecast = system.forecast._model
    results = pd.DataFrame()

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
    time = features.index[0] + forecast._resolutions[-1].time_prior
    end = features.index[-1] - forecast._resolutions[-1].time_horizon

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

            if 'doubt' in forecast.features['input']:
                step_doubt = list()

            step_features = copy.deepcopy(features[time - forecast._resolutions[-1].time_prior:
                                                   time + forecast._resolutions[-1].time_horizon])

            step_index = step_features[time:].index
            step = step_index[0]
            while step in step_index:
                step_inputs = forecast._extract_inputs(step_features, step)

                if verbose:
                    database.persist(step_inputs,
                                     subdir='inputs',
                                     file=step.strftime('%Y%m%d_%H%M%S') + '.csv')

                inputs = np.squeeze(step_inputs.fillna(0).values)
                result = forecast._run_step(inputs)

                if 'doubt' in forecast.features['input']:
                    # retrieve prediction doubt (assumes doubt is last column in inputs)
                    if forecast._estimate == True:
                        doubt = inputs[-2, -1]
                    elif forecast._estimate == False:
                        doubt = inputs[-1, -1]

                    step_doubt.append(doubt)

                    # recalculate doubt for target hour (this value will be utilized in next prediction)
                    sm1 = step_features.loc[step - dt.timedelta(hours=23):step, 'pv_power'].mean()
                    sm2 = step_features.loc[step - dt.timedelta(hours=23):step, 'dni'].mean()

                    sf1 = step_features.loc[step - dt.timedelta(hours=23):step, 'pv_power']
                    sf2 = step_features.loc[step - dt.timedelta(hours=23):step, 'dni']

                    cov = ((sf1 - sm1) * (sf2 - sm2)).mean()
                    step_features.loc[step, 'doubt'] = abs(cov - forecast.covariance) / forecast.cov_std

                # Add predicted output to features of next iteration
                step_features.loc[step, forecast.features['target']] = result

                step_result.append(result)
                step += dt.timedelta(minutes=forecast._resolutions[-1].minutes)

            if training_recursive:
                training_features = step_features

                forecast._train(training_features)
                forecast._save_model()

            target = forecast.features['target']

            if 'doubt' in forecast.features['input']:
                result = features.loc[step_index[0]:step_index[-1], ['doubt'] + target]
                result = pd.concat(
                    [result, pd.DataFrame(step_result, result.index, columns=[t + '_est' for t in target]),
                     pd.DataFrame(step_doubt, result.index, columns=['true_doubt'])],
                    axis=1)
            else:
                result = features.loc[step_index[0]:step_index[-1], target]
                result = pd.concat(
                    [result, pd.DataFrame(step_result, result.index, columns=[t + '_est' for t in target])],
                    axis=1)

            result = system.forecast._model.rescale(result, scale=False)

            for target in forecast.features['target']:
                result[target + '_err'] = result[target + '_est'] - result[target]

            result['horizon'] = pd.Series(range(1, len(result.index) + 1), result.index)
            result.index.name = 'time'

            database.persist(result, subdir='outputs')

            results = pd.concat([results, result], axis=0)

        except ValueError as e:
            logger.debug("Skipping %s: %s", time, str(e))

        time += dt.timedelta(minutes=interval)

    return results


def _result_summary(system, results, sim_time, train_time, pred_time, doubt=False):
    err_names = ['err_hs']
    for i in range(24):
        err_names.append('err_h{}'.format(i + 1))

    # retrieve total and 'horizon_wise' mae and mse of sim targets
    targets = system.forecast._model.features['target']

    if doubt is True:
        kpi = {'times': pd.DataFrame(),
               'mse': pd.DataFrame(index=err_names, columns=targets),
               'mae': pd.DataFrame(index=err_names, columns=targets),
               'mse_cor': pd.DataFrame(index=err_names, columns=targets),
               'mae_cor': pd.DataFrame(index=err_names, columns=targets),
               'weights': pd.DataFrame({'train_weights': 1, 'nontrain_weights': 1, 'total_weights': 1}, index=[0]),
               'apollo': pd.DataFrame({'apollo': 1}, index=[0]),
               'apollo_2': pd.DataFrame(index=err_names, columns=targets),
               'horizon_doubt': pd.Series(index=err_names[1:], name='horizon_doubt')}
    else:
        kpi = {'times': pd.DataFrame(),
               'mse': pd.DataFrame(index=err_names, columns=targets),
               'mae': pd.DataFrame(index=err_names, columns=targets),
               'weights': pd.DataFrame({'train_weights': 1, 'nontrain_weights': 1, 'total_weights': 1}, index=[0]),
               'apollo': pd.DataFrame({'apollo': 1}, index=[0]),
               'apollo_2': pd.DataFrame(index=err_names, columns=targets), }

    # retrieve duration of various process durations and compile in np.array
    kpi['times']['sim_time'] = [sim_time]
    kpi['times']['train_time'] = [train_time]
    kpi['times']['pred_time'] = [pred_time]

    if doubt is True:
        for i in range(24):
            horizon_data = results.loc[results['horizon'] == i + 1]
            kpi['horizon_doubt']['err_h{}'.format(i + 1)] = horizon_data['true_doubt'].mean()

    for target in targets:
        kpi['mse'][target]['err_hs'] = (results[target + '_err'] ** 2).mean()
        kpi['mae'][target]['err_hs'] = abs(results[target + '_err']).mean()

        # calculate apollo_2
        median = results['pv_power_err'].groupby([results.index.hour]).median()
        kpi['apollo_2'][target]['err_hs'] = abs(median).mean()

        if doubt is True:
            # noise corrected error
            err_cor = results[target + '_err'] - results['doubt'] * results[target + '_err']
            err_cor.loc[err_cor < 0] = 0
            kpi['mse_cor'][target]['err_hs'] = (err_cor ** 2).mean()
            kpi['mae_cor'][target]['err_hs'] = abs(err_cor).mean()

        for i in range(24):
            # parse horizon data
            horizon_data = results.loc[results['horizon'] == i + 1]

            kpi['mse'][target]['err_h{}'.format(i + 1)] = (horizon_data[target + '_err'] ** 2).mean()
            kpi['mae'][target]['err_h{}'.format(i + 1)] = abs(horizon_data[target + '_err']).mean()

            median = horizon_data[target + '_err'].groupby([horizon_data.index.hour]).median()
            kpi['apollo_2'][target]['err_h{}'.format(i + 1)] = abs(median).mean()

            if doubt is True:
                err_cor = horizon_data[target + '_err'] - horizon_data['doubt'] * horizon_data[target + '_err']
                err_cor.loc[err_cor < 0] = 0
                kpi['mse_cor'][target]['err_h{}'.format(i + 1)] = (err_cor ** 2).mean()
                kpi['mae_cor'][target]['err_h{}'.format(i + 1)] = abs(err_cor).mean()

    trainable_count = int(
        np.sum([K.count_params(p) for p in system.forecast._model.model.trainable_weights]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in system.forecast._model.model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count

    kpi['weights']['train_weights'] = trainable_count
    kpi['weights']['nontrain_weights'] = non_trainable_count
    kpi['weights']['total_weights'] = total_count

    hourly_max = results['pv_power_err'].groupby([results.index.hour]).max()
    median = results['pv_power_err'].groupby([results.index.hour]).median()
    kpi['apollo']['apollo'] = (median[5:22] / hourly_max).mean()  # ToDo fix index, take abs of median

    return kpi


def _result_horizon(system, results, hour):
    results_dir = os.path.join('results', 'horizons')
    results_horizon = results[results['horizon'] == hour].assign(horizon=hour)
    _result_hours(system, results_horizon, name='horizon{}'.format(hour), dir=results_dir)

    return results_horizon


def _result_hours(system, results, dir='results', name=None):
    _result_describe(system, results, results.index.hour, dir=dir, name=name)
    _result_boxplot(system, results, results.index.hour, dir=dir, name=name, label='Hours')


def _result_describe(system, results, index, name=None, dir='results'):
    for error in [c for c in results.columns if c.endswith('_err')]:
        result = results[error]
        median = result.groupby([index]).median()
        median.name = 'median'
        desc = pd.concat([median, result.groupby([index]).describe()], axis=1)
        _result_write(system, desc, error.split('_err')[0], dir, name)


def _result_write(system, results, results_name='results', results_dir='', postfix=None):
    system_dir = system._configs['General']['data_dir']
    database = copy.deepcopy(system._database)
    database.dir = system_dir
    # database.format = '%Y%m%d'
    database.enabled = True
    database_dir = os.path.join(database.dir, results_dir)

    if not os.path.isdir(database_dir):
        os.makedirs(database_dir, exist_ok=True)

    if postfix is not None:
        results_name += '_{}'.format(postfix)

    results.to_csv(os.path.join(database_dir, results_name + '.csv'),
                   sep=database.separator,
                   decimal=database.decimal,
                   encoding='utf-8')


def _result_boxplot(system, results, index, label='', name=None, colors=None, dir='results', **kwargs):
    try:
        import seaborn as sns

        for error in [c for c in results.columns if c.endswith('_err')]:
            plot_name = error.split('_err')[0]
            if name is not None:
                plot_name += '_{}'.format(name)

            plot_file = os.path.join(system._configs['General']['data_dir'], dir, plot_name + '.png')

            plt.figure()
            plot_fliers = dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='lightgrey')
            plot_colors = colors if colors is not None else index.nunique()
            plot_palette = sns.light_palette('#0069B4', n_colors=plot_colors, reverse=True)
            plot = sns.boxplot(x=index, y=error, data=results, palette=plot_palette, flierprops=plot_fliers,
                               **kwargs)  # , showfliers=False)
            plot.set(xlabel=label, ylabel='Error [W]')
            plot.figure.savefig(plot_file)
            plt.show(block=False)

    except ImportError:
        pass


def _result_comparison(systems):
    import xlsxwriter

    def _write_performance_summary(xldoc):
        metrics = ['mse', 'mae', 'mse_cor', 'mae_cor',
                   'times', 'weights', 'apollo', 'apollo_2',
                   'horizon_doubt']

        def _retrieve_model_data(systems, sheets):
            data = {}

            for sheet in sheets:
                data[sheet] = pd.DataFrame()

            i = 0
            for system in systems:
                data_dir = system._configs['General']['data_dir']
                database = os.path.join(data_dir, 'results')
                for sheet in sheets:
                    csv = os.path.join(database, sheet + '.csv')
                    if os.path.isfile(csv):
                        system_data = pd.read_csv(os.path.join(database, sheet + '.csv'), index_col=0)

                        if sheet in ['mae', 'mae_cor', 'mse', 'mae_cor', 'horizon_doubt', 'hourly_doubt']:
                            system_data.columns = [name + '_{}'.format(i) for name in
                                                   system_data.columns]  # ensure unique column names
                            data[sheet] = pd.concat([data[sheet], system_data], axis=1)
                        else:
                            data[sheet] = pd.concat([data[sheet], system_data])
                i += 1

            # delete keys for which no info was extracted
            for sheet in sheets:
                if data[sheet].shape == (0, 0):
                    del data[sheet]
            return data

        def _write_MSE_MAE(systems, data, kpi, offset):
            if kpi.lower() not in metrics:
                print('Valid metrics include:')
                for kpi in metrics:
                    print(kpi)
                raise ValueError('The chosen kpi does not belong to the list metrics used.')

            worksheet.merge_range(len(systems) + 2, offset, len(systems) + 3, offset, kpi.upper(), bold_format)

            worksheet.write_column(len(systems) + 4, offset, list(data.index), bold_format)
            offset += 1

            # write system ids above data
            if len(systems) != len(data.columns):
                step = int(len(data.columns) / len(systems))
                n = 0
                for system in systems:
                    worksheet.merge_range(len(systems) + 2, offset + n * step,
                                          len(systems) + 2, offset + (n + 1) * step - 1,
                                          system.id, merge_format)
                    n += 1
            else:
                for i in range(len(systems)):
                    worksheet.write(len(systems) + 2, offset + i, systems[i].id, merge_format)

            # write data
            for column in data.columns:
                worksheet.write(len(systems) + 3, offset, column)
                worksheet.write_column(len(systems) + 4, offset, data[column])
                offset += 1

            return offset + 1

        def _write_summary_table(systems, data, offset):
            for i in range(len(systems)):
                worksheet.write(i + 1, offset, systems[i].id, bold_format)
            for i in range(len(data.columns)):
                worksheet.write(0, i + offset + 1, data.columns[i], bold_format)  # write column labels
                worksheet.write_column(1, i + offset + 1, data[data.columns[i]])  # write column data

            return len(data.columns) + offset + 2  # next offset

        if len(systems) == 25:
            raise TypeError('This methods formatting relies on the fact that len(systems) does not' +
                            ' equal 25.')

        data = _retrieve_model_data(systems, metrics)

        worksheet = workbook.add_worksheet('performance_summary')

        left_col = 0
        offset = 0
        for key, info in data.items():
            if len(info.index) == len(systems):  # ToDo results in err if len(systems) = 25
                left_col = _write_summary_table(systems, info, left_col)
            else:
                offset = _write_MSE_MAE(systems, info, key, offset)

    workbook = xlsxwriter.Workbook('data\\model_comparison.xlsx')

    bold_format = workbook.add_format({'bold': True})
    merge_format = workbook.add_format({'bold': True, 'align': 'center', 'fg_color': 'yellow'})

    _write_performance_summary(workbook)
    workbook.close()


def _get_time(time_str):
    return tz.utc.localize(dt.datetime.strptime(time_str, '%d.%m.%Y'))


def _get_parser(root_dir):
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

