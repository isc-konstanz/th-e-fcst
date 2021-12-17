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
                assert floor(to_edge * 100) / 100 == to_edge

                total_steps = total_steps + 1
                f_max = f_max + (small_delta - to_edge)/2
                f_min = f_min - (small_delta - to_edge)/2

                # Assure f_max and f_min are rounded to the nearest .2%f
                assert f_max == floor(f_max * 100) / 100
                assert f_min == ceil(f_min * 100) / 100

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

            doubt_data = (mi_data.loc[i, cols] - d_avg) / (d_std + epsilon)
            mi_data.loc[i, new_cols] = doubt_data.values

        return mi_data

    system_dir = system._configs['General']['data_dir']
    eval_dir = os.path.join(system_dir, 'evaluation')

    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    if not os.path.isfile(os.path.join(eval_dir, 'evaluation_data') + '.pkl'):

        # First group the timepoints appearing in features according to their weather conditions
        attributes = json.loads(settings.get('Evaluation', 'Features'))
        attributes.pop('regional_doubt', None)
        index = gen_index(attributes)
        system.simulation['evaluation'] = sort_data(features, index)

        # Now used these groupings to calculate the std deviation and mean value of the doubt values
        # in each of the regions defined in settings.
        system.simulation['evaluation'] = regional_doubt(system.simulation['evaluation'], features)

        # Use the std and mean values of the doubt in each region to calculate the deviation of the doubt
        # value for a singular timepoint from the average doubt value in the region it belongs, in units of
        # the regions standard deviation. This information is added to a new column named regional doubt in
        # the features dataframe.
        resolve_doubt(system.simulation['evaluation'], features)

        # re-sort the data now into their appropriate regions now that the required information is
        # now available in the features dataframe.
        attributes = json.loads(settings.get('Evaluation', 'Features'))
        index = gen_index(attributes)
        system.simulation['evaluation'] = sort_data(features, index)

        # Now use the sorted timepoints to provide the various predictions in results
        # their appropriate indices
        system.simulation['evaluation'] = _parse_regions(system)

        # save this object for future runs.
        save_pickle(eval_dir, 'evaluation_data', system.simulation['evaluation'])

    else:
        dir = os.path.join (system_dir, 'evaluation')
        system.simulation['evaluation'] = load_pickle(dir, 'evaluation_data')

def evaluate(settings, systems):
    from th_e_sim.iotools import print_boxplot, write_excel

    # [evaluate_regions]:
    # This function takes err data of a network, labeled according to chosen features,
    # as input. The labels of this err data are described by the MultiIndex indexing it.
    # This function assumes that the bottom level is called regional_doubt.
    # With this err data the function creates a directory tree; each directory of this
    # directory tree corresponds to one value of one level of the MultiIndex grouping,
    # chosen for the error data. Therefore, the path of the bottom directory of this file tree
    # can be ordered a partial index of the the MultiIndex (including a value for each level
    # but the last, that being the index determining the regional doubt value). To
    # structure the error along the regional doubt axis of our MultiIndex grouping
    # we create a boxplot which visually displays the error given by the directory path and the
    # various doubt values which arise with this partial index. This same set of errors is further
    # summarized with a .csv describing the error's distribution.

    def evaluate_regions(system, evaluation_data, target):
        from copy import deepcopy
        assert isinstance(evaluation_data.index, pd.MultiIndex)

        # Here we create a copy of the data so that transformations potentially carried out in
        # this function don't harm the object on which we are performing our analysis.
        data = deepcopy(evaluation_data)

        # We then retrieve the partial index values which will be used to create the directory
        # tree as well as to select segments of our error data for evaluation.
        control_names = list(evaluation_data.index.names)
        control_names.remove('regional_doubt')
        control_index = {}
        for name in control_names:
            control_index[name] = data.index.get_level_values(name)

        control_index = list(zip(*list(control_index.values())))
        control_index = pd.MultiIndex.from_tuples(control_index, names=control_names)

        # Here we use the previously defined partial index to create a filetree and then at the
        # bottom of this filetree to output the boxplot and csv files described above.
        for index in control_index:

            sub_dir = os.path.join(system._configs['General']['data_dir'], 'evaluation', target)
            if not os.path.isdir(sub_dir):
                os.mkdir(sub_dir)

            for level, value in zip(control_names, list(index)):
                sub_dir = os.path.join(sub_dir, level + '_{}'.format(value))

            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
                file = os.path.join(sub_dir, 'regional_doubt')
                partial_index = index + (slice(None),)
                _evaluate_data(system, data.loc[partial_index, :], 'regional_doubt', evaluation_data.columns[-1], file)

    def standard_kpi(data, data_target):
        data_column = data_target + '_p_err'

        mbe = data[data_column].mean()
        mae = data[data_column].abs().mean()
        rmse = (data[data_column] ** 2).mean() ** 0.5
        nrmse = rmse / 10000

        return mbe, mae, rmse, nrmse

    def doubt_kpi(data, data_target):
        region_names = data.index.names
        data_column = data_target + '_p_err'

        doubt = pd.Series(data.index.get_level_values('regional_doubt'), name='regional_doubt')
        low_doubt = doubt.quantile(0.25)
        hi_doubt = doubt.quantile(0.75)

        if 'solar_elevation' in region_names:

            solar_mbe = data[data_column].groupby(level=['horizon', 'solar_elevation', 'regional_doubt']).mean()
            solar_mae = (data[data_column].abs()).groupby(level=['horizon', 'solar_elevation', 'regional_doubt']).mean()
            solar_rmse = (data[data_column] ** 2).groupby(level=['horizon', 'solar_elevation', 'regional_doubt']).mean() ** 0.5
            solar_nrmse = solar_rmse / 10000

            horizon_cond = (solar_mbe.index.get_level_values('horizon') == 1)
            peak_solar_cond = solar_mbe.index.get_level_values('solar_elevation') >= 35
            low_doubt_cond = solar_mbe.index.get_level_values('regional_doubt') <= low_doubt
            condition = horizon_cond & peak_solar_cond & low_doubt_cond

            peak_solar_mbe_low_doubt = solar_mbe.iloc[condition].mean()
            peak_solar_mae_low_doubt = solar_mae.iloc[condition].mean()
            peak_solar_rmse_low_doubt = solar_rmse.iloc[condition].mean()
            peak_solar_nrmse_low_doubt = solar_nrmse.iloc[condition].mean()

            med_doubt_cond = (low_doubt < solar_mbe.index.get_level_values('regional_doubt')) & \
                             (solar_mbe.index.get_level_values('regional_doubt') < hi_doubt)
            condition = horizon_cond & peak_solar_cond & med_doubt_cond

            peak_solar_mbe_med_doubt = solar_mbe.iloc[condition].mean()
            peak_solar_mae_med_doubt = solar_mae.iloc[condition].mean()
            peak_solar_rmse_med_doubt = solar_rmse.iloc[condition].mean()
            peak_solar_nrmse_med_doubt = solar_nrmse.iloc[condition].mean()

            hi_doubt_cond = solar_mbe.index.get_level_values('regional_doubt') >= hi_doubt
            condition = horizon_cond & peak_solar_cond & hi_doubt_cond

            peak_solar_mbe_hi_doubt = solar_mbe.iloc[condition].mean()
            peak_solar_mae_hi_doubt = solar_mae.iloc[condition].mean()
            peak_solar_rmse_hi_doubt = solar_rmse.iloc[condition].mean()
            peak_solar_nrmse_hi_doubt = solar_nrmse.iloc[condition].mean()

            solar_doubt_kpi = {'low_doubt': [peak_solar_mbe_low_doubt, peak_solar_mae_low_doubt, peak_solar_rmse_low_doubt, peak_solar_nrmse_low_doubt],
                               'med_doubt': [peak_solar_mbe_med_doubt, peak_solar_mae_med_doubt, peak_solar_rmse_med_doubt, peak_solar_nrmse_med_doubt],
                               'hi_doubt': [peak_solar_mbe_hi_doubt, peak_solar_mae_hi_doubt, peak_solar_rmse_hi_doubt, peak_solar_nrmse_hi_doubt]}

            return solar_doubt_kpi, low_doubt, hi_doubt
        return

    summary = pd.DataFrame(index=[s.name for s in systems],
                           columns=pd.MultiIndex.from_tuples([('Durations [min]', 'Simulation'),
                                                              ('Durations [min]', 'Prediction')]))

    evaluations = {}
    for system in systems:
        evaluation_data = system.simulation['evaluation']
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

            evaluate_regions(system, system.simulation['evaluation'], target)

            target_id = target.replace('_power', '')
            target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

            #if target+'_err' not in results.columns:
            #    continue

            #if target_id in ['pv']:
            #    columns_daylight = np.intersect1d(results.columns, ['ghi', 'dni', 'dhi', 'solar_elevation'])
            #    if len(columns_daylight) > 0:
            #        results = results[(results[columns_daylight] > 0).any(axis=1)]

            mbe, mae, rmse, nrmse = standard_kpi(evaluation_data, target)
            add_evaluation('mbe', target_name, mbe)
            add_evaluation('mae', target_name, mae)
            add_evaluation('rmse', target_name, rmse)
            add_evaluation('nrmse', target_name, nrmse)

            solar_doubt_kpi, low_doubt, hi_doubt = doubt_kpi(evaluation_data, target)

            add_evaluation('low_doubt', target_name, low_doubt)
            add_evaluation('hi_doubt', target_name, hi_doubt)

            metric_names = ['mbe', 'mae', 'rmse', 'nrmse']
            for doubt_level, metrics in solar_doubt_kpi.items():

                i = 0
                for metric in metrics:
                    add_evaluation(doubt_level+' '+metric_names[i], target_name, metric)
                    i += 1


    write_excel(settings, summary, evaluations)


def _evaluate_data(system, data, level, column, file, **kwargs):
    from th_e_sim.iotools import print_boxplot
    try:
        _data = copy.deepcopy(data)
        _data.index = data.index.get_level_values(level)
        print_boxplot(system, _data, _data.index, column, file, **kwargs)

    except ImportError as e:
        logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                             str(e)))

    return _describe_data(system, data, data.index, column, file)


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
