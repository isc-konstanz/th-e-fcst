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
import json

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

                features = system.forecast._model._parse_features(features)

                if settings.getboolean('General', 'verbose', fallback=False):
                    print_distributions(features, path=system.forecast._model.dir)

                system.forecast._model._train(features)

                durations['training']['end'] = dt.datetime.now()
                durations['training']['minutes'] = (durations['training']['end'] -
                                                    durations['training']['start']).total_seconds() / 60.0

                logging.debug("Training of neural network for system {} complete after {} minutes"
                              .format(system.name, durations['training']['minutes']))

            data = system._database.get(start, end)
            weather = system.forecast._weather._database.get(start, end)
            if system.contains_type('pv'):
                solar = system.forecast._get_yield(weather)
                data = pd.concat([data, solar], axis=1)

            features = system.forecast._model._parse_features(pd.concat([data, weather], axis=1))
            features_R = system.forecast._model.resolutions[0].resample(features)
            features_file = os.path.join('evaluation', 'features')
            write_csv(system, features, features_file)
            durations['prediction'] = {
                'start': dt.datetime.now()
            }

            structure_err(settings, system, features_R)
            logging.debug("Beginning predictions for system: {}".format(system.name))

            results = simulate(settings, system, features)

            durations['prediction']['end'] = dt.datetime.now()
            durations['prediction']['minutes'] = (durations['prediction']['end'] -
                                                  durations['prediction']['start']).total_seconds() / 60.0

            logging.debug("Predictions for system {} complete after {} minutes"
                          .format(system.name, durations['prediction']['minutes']))

            durations['simulation']['end'] = dt.datetime.now()
            durations['simulation']['minutes'] = (durations['simulation']['end'] -
                                                  durations['simulation']['start']).total_seconds() / 60.0

            # Store results with its system, to summarize it later on
            system.simulation['results'] = results
            system.simulation['durations'] = durations
            #system.simulation = {
            #    'results': results,
            #    'durations': durations
            #}

        except Exception as e:
            error = True
            logger.error("Error simulating system %s: %s", system.name, str(e))
            logger.debug("%s: %s", type(e).__name__, traceback.format_exc())

    logger.info("Finished TH-E Simulation{0}".format('s' if len(systems) > 1 else ''))

    # OUTPUT OF THE PROGRAM IS GENERATED HERE
    if not error:
        my_evaluation(systems)
        evaluate(settings, systems)

        if tensorboard:
            logger.info("TensorBoard will be kept running")

    while tensorboard and not error:
        try:
            time.sleep(100)

        except KeyboardInterrupt:
            tensorboard = False


def simulate(settings, system, features, **kwargs):

    def save_to_database(date, input, target, prediction, database):
        # take a prediction output by the simulate function and save it to
        # the appropriate database.
        condition_cols = [col for col in input.columns if col not in target.columns]
        target.columns = [target + '_t' for target in target.columns]
        prediction.columns = [predict + '_p' for predict in prediction.columns]

        out_cond = input.loc[date: , condition_cols]

        data_out = pd.concat([target, prediction, out_cond], axis=1)
        data_out.index.name = 'time'
        data_in = input
        data_in.index.name = 'time'

        database.persist(data_in, subdir='inputs', file=date.strftime(database.format) + '.csv')
        database.persist(data_out, subdir='outputs')

    def load_from_database(date, database):

        data = database.get(date, subdir='outputs')
        target_cols = [col for col in data.columns if 't' == col.split('_')[-1]]
        predict_cols = [col for col in data.columns if 'p' == col.split('_')[-1]]
        cols = ['_'.join(col.split('_')[:-1]) for col in target_cols]

        target = data[target_cols]
        target.columns = cols

        prediction = data[predict_cols]
        prediction.columns = cols

        input = database.get(date, subdir='inputs')

        return input, target, prediction

    forecast = system.forecast._model

    #if len(forecast.resolutions) == 1:
        #resolution_min = forecast.resolutions[0]
    #else:
        #for i in range(len(forecast.resolutions)-1, 0, -1):
            #resolution_min = forecast.resolutions[i]
            #if resolution_min.steps_horizon is not None:
                #break

    resolution_max = forecast.resolutions[0]

    system_dir = system._configs['General']['data_dir']
    database = copy.deepcopy(system._database)
    database.dir = system_dir
    #database.format = '%Y%m%d'
    database.enabled = True

    # Reactivate this, when multiprocessing will be implemented
    # global logger
    # if process.current_process().name != 'MainProcess':
    #    logger = process.get_logger()

    #verbose = settings.getboolean('General', 'verbose', fallback=False)
    interval = settings.getint('General', 'interval')
    date = features.index[0] + resolution_max.time_prior + resolution_max.time_step
    end = features.index[-1] - resolution_max.time_horizon

    # training_recursive = settings.getboolean('Training', 'recursive', fallback=False)
    # training_interval = settings.getint('Training', 'interval')
    # training_last = time

    results = {}
    while date <= end:

        if database.exists(date, subdir='outputs') and database.exists(date, subdir='inputs'):

            results[date] = load_from_database(date, database)
            date += dt.timedelta(minutes=interval)
            continue

        try:
            # Input index in features
            target_data_i = date - resolution_max.time_step + dt.timedelta(minutes=resolution_max.resolution)
            input_i_start = target_data_i - resolution_max.time_prior
            input_i_end = date + resolution_max.time_horizon

            # Strip targets
            input = copy.deepcopy(features[input_i_start:input_i_end])
            target = resolution_max.resample(input.loc[date:input_i_end, forecast.features['target']])

            # Replace targets with yield values
            target_range = input.loc[(input.index >= target_data_i) & (input.index <= input_i_end)].index
            input.loc[target_range, forecast.features['target']] = input.loc[target_range, 'pv_yield']
            forecast._calc_doubt(input, target_range)
            input = resolution_max.resample(input)

            prediction = forecast._predict(input)

            results[date] = (input, target, prediction)
            save_to_database(date, input, target, prediction, database)

            date += dt.timedelta(minutes=interval)

        except ValueError as e:
            logger.debug("Skipping %s: %s", date, str(e))
            # logger.debug("%s: %s", type(e).__name__, traceback.format_exc())
            date += dt.timedelta(minutes=interval)

    return results

def structure_err(settings, system, features):

    def save_pickle(dir, name, data):
        import pickle

        with open(os.path.join(dir, name) + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(dir, name):
        import pickle
        with open(os.path.join(dir, name) + '.pkl', 'rb') as f:
            dict_frame = pickle.load(f)

        return dict_frame

    def gen_index(grid_dict):
        assert isinstance(grid_dict, dict)

        # The first step is to create a list of tuples which iterate through the range
        # of possible label values. To this end for continuous variables I choose
        # the label to be the left end of the interval; this is a
        # a natural choice since the function range() doesn't iterate to the last element.

        axes = {}
        for attribute, info in grid_dict.items():
            axes[attribute] = [x / info['Magnitude'] for x in range(info['Min'], info['Max'], info['Width'])]

        # The next step generates the appropriate Multiindex object.
        names = []
        index = []
        for axis, values in axes.items():
            names.append(axis)
            index.append(values)

        index = pd.MultiIndex.from_product(index, names=names)

        return index

    def sort_data(features, labels):

        # Initialize the dataframe object.
        labeled_err = pd.DataFrame()


        for date in features.index:

            # For each datapoint we must extract the values by which we wish to
            # group our prediction data for evaluation.
            # ToDo: don't hardcode 24hrs or 1hr.
            label = []
            for name in labels.names:
                    condition = features.loc[date, name]
                    assert isinstance(condition, np.float64)
                    label.append(condition)
            label = tuple(label)

            # We must now utilize these values stored in the label tuple to determine,
            # the index with which we will label this timepoint. This label will be
            # chosen to be the index value whose lay to the left of the real label value
            # and whose distance to the real values is the smallest.
            find_index = list(labels)
            distance = 10000
            for ind in find_index:
                lower_bound = all(x <= y for x, y in zip(ind, label))
                distance_condition = (sum(np.subtract(label, ind)) == min(distance, sum(np.subtract(label, ind))))
                if lower_bound and distance_condition:
                    distance = min(distance, sum(np.subtract(label, ind)))
                    _index = ind
                else:
                    continue

            # The real value of the timepoint is replaced with the appropriate label
            # determined above. The result (label, timepoint) is saved in a
            # pd.DataFrame.
            label = _index
            labeled_err = pd.concat([labeled_err, pd.DataFrame({'dates': date},
                                                   index=pd.MultiIndex.from_tuples([label],
                                                   names=labels.names))]).dropna()
        return labeled_err

    # This function must always be carried out following sort_data. This function
    # calculates the average doubt values as well as their standard deviation in
    # the various regions defined in settings. This information is saved in the
    # evaluation dataframe output by sort_data.
    def regional_doubt(evaluation, features):

        for index in evaluation.index:
            region_dates = evaluation.loc[index, :]
            doubt_avg = features.loc[list(region_dates['dates']), 'pv_power_doubt'].mean()
            doubt_std = features.loc[list(region_dates['dates']), 'pv_power_doubt'].std()
            evaluation.loc[index, 'doubt_avg'] = doubt_avg
            evaluation.loc[index, 'doubt_std'] = doubt_std

        return evaluation

    # This function utilizes the regional doubt
    def resolve_doubt(evaluation, features):

        for date in list(evaluation['dates']):
            # Retrieve the conditions forecasted for the current timepoint.
            weather_conditions = evaluation.index[evaluation['dates'] == date]
            weather_conditions = weather_conditions[0]

            # Retrieve the doubt and calculate the regional doubt of this timepoint
            # utilizing the std and mean of the doubt values for the region to which
            # this timepoint belongs.
            doubt = float(features.loc[date, 'pv_power_doubt'])
            doubt_avg = float(evaluation.loc[weather_conditions, 'doubt_avg'].iloc[0])
            doubt_std = float(evaluation.loc[weather_conditions, 'doubt_std'].iloc[0])

            if doubt_std == 0:
                features.loc[date, 'regional_doubt'] = 0
            else:
                features.loc[date, 'regional_doubt'] = abs(doubt - doubt_avg) / doubt_std

    system_dir = system._configs['General']['data_dir']
    eval_dir = os.path.join(system_dir, 'evaluation')

    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)

    if not os.path.isfile(os.path.join(eval_dir, 'sorted_times') + '.pkl'):

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

        # save the sorted database object for future runs.
        save_pickle(eval_dir, 'sorted_times', system.simulation['evaluation'])

    else:
        dir = os.path.join (system_dir, 'evaluation')
        system.simulation['evaluation'] = load_pickle(dir, 'sorted_times')


def my_evaluation(systems):

    def save_pickle(dir, name, data):
        import pickle

        with open(os.path.join(dir, name) + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(dir, name):
        import pickle
        with open(os.path.join(dir, name) + '.pkl', 'rb') as f:
            dict_frame = pickle.load(f)

        return dict_frame

    def _parse_regions(system):

        # Initialize dataframe to which all forecasts with the appropriate indices will
        # be appended.
        evaluation_data = pd.DataFrame()

        # Iterate through all results and concatenate target and predictions into one
        # DataFrame object (output).
        for date, info in system.simulation['results'].items():

            input, targets, predictions = info

            targets.columns = [target + '_t' for target in targets.columns]
            predictions.columns = [predict + '_p' for predict in predictions.columns]

            # Concatenate results into one dataframe and assign the appropriate index
            # values.
            output = pd.concat([targets, predictions], axis=1)

            # Extract the index from the evaluation for each hourly prediction made in the
            # forecast.
            regions = []
            i = 1

            for date_2 in output.index:

                conditions = system.simulation['evaluation'].index[system.simulation['evaluation']['dates'] == date_2]
                conditions = conditions[0]

                # Manually add the forecast horizon to the extracted index for the current
                # prediction.
                region = (i, ) + conditions
                regions.append(region)
                i += 1

            region_names = ['horizon'] + system.simulation['evaluation'].index.names
            output.index = pd.MultiIndex.from_tuples(regions, names=region_names)

            # Calculate the error of the prediction for this forecast
            for target, prediction in zip(targets.columns, predictions.columns):
                output[prediction + '_err'] = output[target] - output[prediction]

            evaluation_data = pd.concat([evaluation_data, output])

        # Save evaluation object to allow for the purpose of future evaluations.
        save_pickle(os.path.join(system_dir, 'evaluation'), 'evaluation_data', evaluation_data)
        assert isinstance(evaluation_data.index, pd.MultiIndex)

        return evaluation_data

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

    def evaluate_regions(system, evaluation_data, dir):
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
            sub_dir = dir
            for level, value in zip(control_names, list(index)):
                sub_dir = os.path.join(sub_dir, level + '_{}'.format(value))

            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
                file = os.path.join(sub_dir, 'regional_doubt')
                partial_index = index + (slice(None),)
                _evaluate_data(system, data.loc[partial_index, :], 'regional_doubt', evaluation_data.columns[-1], file)

    for system in systems:
        # Extract important variables from system.
        system_dir = system._configs['General']['data_dir']
        eval_dir = os.path.join(system_dir, 'evaluation')
        targets = system.forecast._model.features['target']

        # Create data object summarizing network performance error in the various regions defined
        # in settings.cfg.

        evaluation_file = os.path.join(eval_dir, 'evaluation_data.pkl')

        if not os.path.isfile(evaluation_file):
            system.simulation['evaluation'] = _parse_regions(system)
        else:
            system.simulation['evaluation'] = load_pickle(eval_dir, 'evaluation_data')

        # Iterate through targets and create a directory in the evaluation directory
        # corresponding to each individual target.
        for target in targets:

            dir = os.path.join(system_dir, 'evaluation', target)

            if not os.path.isdir(dir):
                os.mkdir(dir)
                evaluate_regions(system, system.simulation['evaluation'], dir)

def evaluate(settings, systems):
    from th_e_sim.iotools import print_boxplot, write_excel

    def standard_kpi(data, data_target):
        #ToDo: Change to _p_err
        data_column = data_target + '_predict_err'

        mbe = data[data_column].mean()
        mae = data[data_column].abs().mean()
        rmse = (data[data_column] ** 2).mean() ** 0.5
        nrmse = rmse / 10000

        return mbe, mae, rmse, nrmse

    def doubt_kpi(data, data_target):
        # ToDo: Change to _p_err
        region_names = data.index.names
        data_column = data_target + '_predict_err'

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

