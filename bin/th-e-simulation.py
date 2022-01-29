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

            features_r = system.forecast._model.resolutions[0].resample(features)
            durations['prediction'] = {
                'start': dt.datetime.now()
            }

            logging.debug("Beginning predictions for system: {}".format(system.name))

            system.simulation['results'] = simulate(settings, system, features)

            system.simulation['evaluation'] = mi_results(settings, system, features_r)

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

def mi_results(settings, system, features):

    def save_pickle(dir, name, data):
        import pickle

        with open(os.path.join(dir, name) + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(dir, name):
        import pickle
        with open(os.path.join(dir, name) + '.pkl', 'rb') as f:
            dict_frame = pickle.load(f)

        return dict_frame

    def check_bins(mi_data, data):

        # Testing Code
        sorted_data = mi_data
        ground_truth = data

        assert len(sorted_data) == len(ground_truth)

        for col in sorted_data.index.names:

            if col == 'horizon':

                labels = sorted_data.index.get_level_values(level=col)
                labels = pd.Series(labels, name=col)
                values = sorted_data[col]
                values.index = labels.index

                condition = not (values != labels).any()
                assert condition
                continue

            labels = sorted_data.index.get_level_values(level=col)
            labels = pd.Series(labels, name=col)
            values = sorted_data[col]
            values.index = labels.index

            next_labels = labels + grid_spaces[col]['step_size']

            c1 = (values < labels).any()
            c1 = not c1
            c2 = (values >= next_labels).any()
            c2 = not c2

            assert c1 & c2

    def gen_index(data, steps, features):
        from math import floor, ceil

        keys = ['step_size', 'max', 'min']
        values = 0
        step_sizes = {feature: dict.fromkeys(keys, values) for feature in features}
        mi_arrays = []

        for feature in features:

            total_steps = steps
            f_max = ceil(data[feature].max())
            f_min = floor(data[feature].min())
            big_delta = f_max - f_min

            small_delta = floor(big_delta/total_steps * 10) / 10
            if small_delta == 0:
                raise ValueError("The axis {} cannot be analyzed with the regular grid spacing of {} between"
                                 "grid points. Please choose a smaller number of steps".format(feature, small_delta))

            to_edge = big_delta - small_delta * total_steps

            if not to_edge == 0:

                # Assure logic is correct: if step_size != 0 then to_edge < small_delta
                assert to_edge < small_delta
                # Assure f_max and f_min are rounded to the nearest .2%f
                assert abs(floor(to_edge * 100) / 100 - to_edge) < 0.0001

                total_steps = total_steps + 1
                f_max = f_max + (small_delta - to_edge)/2
                f_min = f_min - (small_delta - to_edge)/2

                # Assure f_max and f_min are rounded to the nearest .2%f
                assert abs(f_max - floor(f_max * 100) / 100) < 0.0001
                assert abs(f_min == ceil(f_min * 100) / 100) < 0.0001

            mi_array = [round(f_min + small_delta*x, 1) for x in range(total_steps + 1)]

            # second check of logic (if step_size != 0 then to_edge < small_delta),
            # as well as check of intermediate operations.
            assert mi_array[-1] == f_max

            mi_arrays.append(mi_array)

            step_sizes[feature]['step_size'] = small_delta
            step_sizes[feature]['max'] = f_max
            step_sizes[feature]['min'] = f_min

        mi = pd.MultiIndex.from_product(mi_arrays, names=features)

        return mi, step_sizes


    def bin_results(results, mi, grid_info):

        binned_rs = pd.DataFrame(index=mi, columns=results.columns)

        cols = mi.names
        grid_results = results[cols]

        nd_step = []
        for col in cols:
            nd_step.append(grid_info[col]['step_size'])

        nd_step = np.array(nd_step)

        for i in mi:

            left_bound = np.array(list(i))
            right_bound = left_bound + nd_step

            c_1 = left_bound <= grid_results
            c_2 = right_bound > grid_results
            c = c_1 & c_2
            bin_condition = pd.Series([True]*len(c), name='n')
            c.index = bin_condition.index

            for col in c.columns:

                bin_condition = bin_condition & c[col]

            bin = results.iloc[bin_condition.values]

            if len(bin) < 1 and len(bin) != 0:

                resolutions = [grid_info[col]['step_size'] for col in mi.names]
                raise ValueError('The chosen resolution of the grid ({}) distributes '
                                 'the data over to many bins. Please decrease its value'.format(resolutions))

            bin.index = pd.MultiIndex.from_tuples([i]*len(bin), names=mi.names)
            binned_rs = pd.concat([binned_rs, bin])

        binned_rs = binned_rs.dropna()
        binned_rs.sort_index(level=mi.names[0])

        return binned_rs


    def regional_doubt(mi_data):

        cols = [col for col in mi_data.columns if col.endswith('doubt')]
        new_cols = [col + '_r' for col in cols]
        epsilon = 10e-7
        index = set(mi_data.index)

        for i in index:

            d_avg = mi_data.loc[i, cols].mean()
            d_std = mi_data.loc[i, cols].std()

            doubt_data = abs(mi_data.loc[i, cols] - d_avg) / (d_std + epsilon)
            mi_data.loc[i, new_cols] = doubt_data.values

        return mi_data

    system_dir = system._configs['General']['data_dir']
    eval_dir = os.path.join(system_dir, 'evaluation')

    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    if not os.path.isfile(os.path.join(eval_dir, 'evaluation_data') + '.pkl'):

        results = system.simulation['results']
        results['time'] = results.index
        grid_features = json.loads(settings.get('Evaluation', 'Features'))
        regions, grid_spaces = gen_index(data=features, steps=40, features=grid_features)
        mi_rs = bin_results(results, regions, grid_spaces)
        mi_rs = regional_doubt(mi_rs)

        reindex = list()
        reindex.append(mi_rs['horizon'].values)
        for name in mi_rs.index.names:

            values = mi_rs.index.get_level_values(level=name)
            reindex.append(values)

        names = ['horizon'] + mi_rs.index.names
        mi_rs.index = pd.MultiIndex.from_arrays(reindex, names=names)

        #check_bins(mi_rs, results)

        save_pickle(eval_dir, 'evaluation_data', mi_rs)
        save_pickle(eval_dir, 'grid_info', grid_spaces)

    else:
        dir = os.path.join(system_dir, 'evaluation')
        mi_rs = load_pickle(dir, 'evaluation_data')

    return mi_rs


def evaluate(settings, systems):
    from th_e_sim.iotools import write_excel

    def _parse_eval(name, eval_config):

        sections = ['target', 'metric', 'conditions',
                    'groups', 'summary', 'boxplot']

        eval_sections = eval_config.keys()

        if not set(sections).issubset(set(eval_sections)):

            raise ValueError('The set of sections defined in the eval_config for the metric {} '
                             'does not contain the required set of sections: \n {}'.format(name, sections))

        eval_dict = {s: None for s in eval_sections}

        for s in eval_dict:

            try:
                eval_dict[s] = json.loads(eval_config.get(s))

            except json.decoder.JSONDecodeError:
                eval_dict[s] = eval_config.get(s)

        return eval_dict

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

    def _print_boxplot_mi(system, data, level, column, file, **kwargs):
        from th_e_sim.iotools import print_boxplot
        try:
            _data = deepcopy(data)
            _data.index = data.index.get_level_values(level)
            print_boxplot(system, _data, _data.index, column, file, **kwargs)

        except ImportError as e:
            logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                                 str(e)))

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

    def mi_moments(mi_data, targets):

        err_cols = [target + '_err' for target in targets]

        mi = mi_data.index
        groups = mi.names
        
        be = mi_data[err_cols].groupby(level=groups)
        mbe = be.mean()
        be_std = be.std()
        mbe.columns = pd.MultiIndex.from_product([['mbe'], targets], names=['kpi', 'targets'])
        be_std.columns = pd.MultiIndex.from_product([['mbe_std'], targets], names=['kpi', 'targets'])
        
        ae = mi_data[err_cols].abs().groupby(level=groups)
        mae = ae.mean()
        ae_std = ae.std()
        mae.columns = pd.MultiIndex.from_product([['mae'], targets], names=['kpi', 'targets'])
        ae_std.columns = pd.MultiIndex.from_product([['mae_std'], targets], names=['kpi', 'targets'])
        
        se = (mi_data[err_cols] ** 2).groupby(level=groups)
        rmse = se.mean() ** 0.5
        rse_std = se.std() ** 0.5
        rmse.columns = pd.MultiIndex.from_product([['rmse'], targets], names=['kpi', 'targets'])
        rse_std.columns = pd.MultiIndex.from_product([['rmse_std'], targets], names=['kpi', 'targets'])

        n_col = pd.MultiIndex.from_product([['count'], targets], names=['kpi', 'targets'])
        n = pd.DataFrame(index=mi_data.index, columns=n_col)
        _n = [1 for x in range(len(mi_data))]

        for col in n_col:
            n[col] = _n
        n = n.groupby(level=groups).sum()

        mi_kpi = pd.concat([mbe, be_std, mae, ae_std, rmse, rse_std, n], axis=1)

        return mi_kpi

    def _relative_kpi(mi_kpi):

        m = mi_kpi.mean()
        std = mi_kpi.std()

        mi_rkpi = (mi_kpi - m)/std
        return round(mi_rkpi, 2)

    def discrete_metrics(name, data, target, groups, conditions: dict, metric, boxplot=False, **kwargs):


        def perform_metrics(name, data, groups, metric, boxplot):

            kpi = []
            if 'mae' == metric:
                ae = data.abs()
                mae = ae.groupby(groups).mean()
                ae_std = ae.groupby(groups).std()
                kpi.append(mae)
                kpi.append(ae_std)

                if boxplot:
                    _print_boxplot(system, groups, ae.values, os.path.join("evaluation", name))

            elif 'mse' == metric:
                se = (data ** 2)
                mse = se.groupby(groups).mean()
                se_std = se.groupby(groups).std()
                kpi.append(mse)
                kpi.append(se_std)

                if boxplot:
                    _print_boxplot(system, groups, se.values, os.path.join("evaluation", name))

            elif 'rmse' == metric:
                se = (data ** 2)
                rmse = se.groupby(groups).mean() ** 0.5
                rse_std = se.groupby(groups).std() ** 0.5
                kpi.append(rmse)
                kpi.append(rse_std)

                if boxplot:
                    _print_boxplot(system, groups, se.values, os.path.join("evaluation", name))

            elif 'mbe' == metric:
                be = data
                mbe = be.groupby(groups).mean()
                be_std = be.groupby(groups).std()
                kpi.append(mbe)
                kpi.append(be_std)

                if boxplot:
                    _print_boxplot(system, groups, be.values, os.path.join("evaluation", name))
            else:
                raise ValueError("The chosen metric {} has not yet been implemented".format(metric))

            kpi = pd.concat(kpi, axis=1)
            kpi.columns = [metric, metric + "_std"]

            return kpi

        def select_data(data, conditions):

            def select_rows(series, operator, value):

                if operator == 'lt':
                    rows = (series < value)

                elif operator == 'gt':
                    rows = (series > value)

                elif operator == 'leq':
                    rows = (series <= value)

                elif operator == 'geq':
                    rows = (series >= value)

                elif operator == 'eq':
                    rows = (series == value)

                else:
                    raise ValueError('An improper condition is present in the dict defining the kpi.')

                return rows

            _ps = pd.Series([True] * len(data), index=data.index)

            for axis, selection in conditions.items():

                rows = select_rows(data[axis], **selection)
                _ps = _ps & rows

            selected = data.iloc[_ps.values]

            return selected, _ps.values

        # select data pertaining to the desired feature space to be examined
        eval_data, pos = select_data(data, conditions)

        # define labels for grouping
        labels = eval_data[groups].values

        # select err data pertaining to desired target
        err_col = target + '_err'
        eval_data = eval_data[err_col]

        # introduce count
        _n = [1 for x in range(len(eval_data))]
        n = pd.Series(_n, name='count')
        n = n.groupby(labels).sum()

        # calculate metrics
        evaluation = perform_metrics(name, eval_data, labels, metric, boxplot)
        evaluation = pd.concat([evaluation, n], axis=1)

        return evaluation

    def sunny(mi_kpi, boxplot=False):

        if not {'solar_elevation', 'horizon'}.issubset(mi_kpi.index.names):
            raise ValueError('The regions horizon, and solar_elevation must be present in the evaluation config file'
                             'in order to calculate this kpi')

        if 'pv_power' not in mi_kpi.columns.get_level_values('targets'):
            raise ValueError('This kpi is intended for predictions made on photovoltaic modules.')

        mi = mi_kpi.index
        c = mi.get_level_values('solar_elevation') >= 33

        sunny_kpi = mi_kpi[[('mae', 'pv_power'), ('mae_std', 'pv_power')]]
        sunny_data = sunny_kpi.iloc[c]
        sunny_kpi = sunny_data.groupby(level='horizon').mean()

        n = mi_kpi[('count', 'pv_power')]
        n = n.iloc[c]
        n = n.groupby(level=['horizon']).sum()

        sunny_kpi = pd.concat([sunny_kpi, n], axis=1)
        sunny_kpi.columns = ['mae', 'mae_std', 'count']

        if boxplot:
            sunny_data.columns = ['mae', 'mae_std']
            _print_boxplot_mi(system, sunny_data, 'horizon', 'mae', 'evaluation/sunny')

        return sunny_kpi

    def astrea(mi_kpi, boxplot=False):

        if not {'horizon'}.issubset(mi_kpi.index.names):
            raise ValueError('The regions horizon, and solar_elevation must be present in the evaluation config file'
                             'in order to calculate this kpi')

        if 'pv_power' not in mi_kpi.columns.get_level_values('targets'):
            raise ValueError('This kpi is intended for predictions made on photovoltaic modules.')

        mi = mi_kpi.index
        c = mi.get_level_values('solar_elevation') >= 0

        astrea_kpi = mi_kpi[['mae', 'mae_std']]
        astrea_data = astrea_kpi.loc[c]
        astrea_kpi = astrea_data.groupby(level='horizon').mean()

        n = mi_kpi[('count', 'pv_power')]
        n = n.groupby(level='horizon').sum()

        astrea_kpi = pd.concat([astrea_kpi, n], axis=1)
        astrea_kpi.columns = ['mae', 'mae_std', 'count']

        if boxplot:

            astrea_data.columns = ['mae', 'mae_std']
            _print_boxplot_mi(system, astrea_data, 'horizon', 'mae', 'evaluation/astrea')

        return astrea_kpi

    def night(mi_kpi, boxplot=False):


        if not {'solar_elevation', 'horizon'}.issubset(mi_kpi.index.names):
            raise ValueError('The regions horizon, and solar_elevation must be present in the evaluation config file'
                             'in order to calculate this kpi')

        if 'pv_power' not in mi_kpi.columns.get_level_values('targets'):
            raise ValueError('This kpi is intended for predictions made on photovoltaic modules.')

        mi = mi_kpi.index
        c = mi.get_level_values('solar_elevation') < 0

        night_kpi = mi_kpi[[('mae', 'pv_power'), ('mae_std', 'pv_power')]]
        night_data = night_kpi.iloc[c]
        night_kpi = night_data.groupby(level='horizon').mean()

        n = mi_kpi[('count', 'pv_power')]
        n = n.iloc[c]
        n = n.groupby(level=['horizon']).sum()

        night_kpi = pd.concat([night_kpi, n], axis=1)
        night_kpi.columns = ['mae', 'mae_std', 'count']

        if boxplot:
            night_data.columns = ['mae', 'mae_std']
            _print_boxplot_mi(system, night_data, 'horizon', 'mae', 'evaluation/nights')

        return night_kpi

    def shadows(mi_kpi, boxplot=False):

        if not {'solar_elevation', 'horizon'}.issubset(mi_kpi.index.names):
            raise ValueError('The regions horizon, and solar_elevation must be present in the evaluation config file'
                             'in order to calculate this kpi')

        if 'pv_power' not in mi_kpi.columns.get_level_values('targets'):
            raise ValueError('This kpi is intended for predictions made on photovoltaic modules.')

        mi = mi_kpi.index
        c_1 = mi.get_level_values('solar_elevation') <= 6.6
        c_2 = mi.get_level_values('solar_elevation') >= 0
        c = c_1 & c_2

        shadow_kpi = mi_kpi[[('mbe', 'pv_power'), ('mbe_std', 'pv_power')]]
        shadow_data = shadow_kpi.iloc[c]
        shadow_kpi = shadow_data.groupby(level='horizon').mean()

        n = mi_kpi[('count', 'pv_power')]
        n = n.iloc[c]
        n = n.groupby(level=['horizon']).sum()

        shadow_kpi = pd.concat([shadow_kpi, n], axis=1)
        shadow_kpi.columns = ['mbe', 'mbe_std', 'count']

        if boxplot:
            shadow_data.columns = ['mbe', 'mbe_std']
            _print_boxplot_mi(system, shadow_data, 'horizon', 'mbe', 'evaluation/shadow')

        return shadow_kpi

    def filter_data(data):

        cols = [col for col in data.columns if col.endswith('doubt_r')]

        if not cols:
            raise ValueError("The data must have a doubt value present to filter the data.")

        c = data[cols] < 1
        _ps = pd.Series([True]*len(c), index=c.index)

        for col in cols:
            _ps = _ps & c[col]

        f_data = data.iloc[_ps.values]

        return f_data

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

        # retrieve results
        mi_results = system.simulation['evaluation']

        # retrieve eval config for system
        data_dir = system.configs['General']['data_dir']
        eval_cfg = os.path.join(data_dir, 'conf', 'evaluation.cfg')
        eval_settings = ConfigParser()
        eval_settings.read(eval_cfg)

        for name in eval_settings.keys():

            if name == "DEFAULT":
                continue

            config = _parse_eval(name, eval_settings[name])

            # calculate metric
            metric = discrete_metrics(name, mi_results, **config)

            # parse target name from config
            target = config['target']
            target_id = target.replace('_power', '')
            target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

            # parse summary from metric
            summary_pos = config['summary']
            summary = metric.loc[summary_pos, config['metric']]

            add_evaluation(name, target_name, summary, metric)

        #moments = mi_moments(evaluation_data, targets)


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
