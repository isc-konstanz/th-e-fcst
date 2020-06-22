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
import inspect
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import multiprocessing as process
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser, RawTextHelpFormatter
from configparser import ConfigParser


def main(args):
    from th_e_fcst import System
    
    logger.info('Starting TH-E Simulation')
    
    settings_file = os.path.join(args.config_dir, 'settings.cfg')
    if not os.path.isfile(settings_file):
        raise ValueError('Unable to open simulation settings: {}'.format(settings_file))
    
    settings = ConfigParser()
    settings.read(settings_file)
    
    kwargs = vars(args)
    kwargs.update(dict(settings.items('General')))
    
    start = _get_time(settings['General']['start'])
    end = _get_time(settings['General']['end'])
    
    systems = System.read(**kwargs)
    for system in systems:
        _prepare_weather(system)
        _prepare_system(system)
        
        if not system.forecast._model.exists():
            system.forecast._model.train(system.forecast._get_data(_get_time(settings['Training']['start']), 
                                                                   _get_time(settings['Training']['end'])))
        
        data = system._database.get(start, end)
        weather = system.forecast._weather.get(start, end)
        features = system.forecast._model._parse_features(pd.concat([data, weather], axis=1))
        
        if settings.getboolean('General', 'threading', fallback=True):
            split = os.cpu_count() or 1
            
            results = list()
            futures = list()
            with ProcessPoolExecutor() as executor:
                manager = process.Manager()
                lock = manager.Lock()
                
                features_days = pd.Series(features.index.date, index=features.index)
                for features_range in np.array_split(sorted(set(features.index.date)), split):
                    features_split = features.loc[features_days.isin(features_range)]
                    futures.append(executor.submit(_simulate, 
                                                   settings, system, features_split, lock=lock))
                
                for future in as_completed(futures):
                    results.append(future.result())
            
            results = pd.concat(results, axis=0)
        else:
            results = _simulate(settings, system, features)
        
        if 'tolerance' in settings['General']:
            tolerance = float(settings['General']['tolerance'])
            results = results[(results['error'] < results['error'].quantile(1-tolerance/100)) & \
                              (results['error'] > results['error'].quantile(tolerance/100))]
        
        results_horizon1 = results[results['horizon'] == 1].assign(horizon=1)
        results_horizon3 = results[results['horizon'] == 3].assign(horizon=3)
        results_horizon24 = results[results['horizon'] == 24].assign(horizon=24)
        results_horizons = pd.concat([results_horizon1, results_horizon3, results_horizon24])
        
        _result_describe(system, results, results.index.hour, name='hours')
        _result_boxplot(system, results, results.index.hour, name='hours', label='Hours')
        _result_describe(system, results_horizon1, results_horizon1.index.hour, 'h1')
        _result_describe(system, results_horizon3, results_horizon3.index.hour, 'h3')
        _result_describe(system, results_horizon24, results_horizon24.index.hour, 'h24')
        _result_boxplot(system, results_horizons, results_horizons.index.hour, label='Hours', name='horizons', hue='horizon', colors=5)
        
        interval = settings.getint('General', 'interval')/60
        results = results[results['horizon'] <= interval].sort_index()
        del results['horizon']
        _result_write(system, results)
        _result_boxplot(system, results, results.index.hour, label='Hours')
        
        logger.info('Finished TH-E Simulation')

def _simulate(settings, system, features, lock=None, **kwargs):
    results = pd.DataFrame()
    
    system_dir = system._configs['General']['data_dir']
    database = copy.deepcopy(system._database)
    database.dir = system_dir
    #database.format = '%Y%m%d'
    database.enabled = True
    
    global logger
    if process.current_process().name != 'MainProcess':
        logger = process.get_logger()
    
    verbose = settings.getboolean('General', 'verbose', fallback=False)
    interval = settings.getint('General', 'interval')
    time = features.index[0] + system.forecast._model.time_prior
    end = features.index[-1] - system.forecast._model.time_horizon
    
    training_recursive = settings.getboolean('Training', 'recursive', fallback=False)
    #training_interval = settings.getint('Training', 'interval')
    #training_last = time
    
    while time <= end:
        if database.exists(time, subdir='results'):
            result = database.get(time, subdir='results')
            results = pd.concat([results, result], axis=0)
            
            time += dt.timedelta(minutes=interval)
            continue
        
        try:
            step_result = list()
            step_features = copy.deepcopy(features[time-system.forecast._model.time_prior:
                                                   time+system.forecast._model.time_horizon])
            
            index = step_features[time:].index
            for i in index:
                step_inputs = system.forecast._model._extract_inputs(step_features, i)
                
                if verbose:
                    database.persist(step_inputs, 
                                     subdir='inputs', 
                                     file=i.strftime('%Y%m%d_%H%M%S')+'.csv')
                
                inputs = np.squeeze(step_inputs.fillna(0).values)
                result = system.forecast._model._run_step(inputs)
                
                # Add predicted output to features of next iteration
                step_features.loc[i, system.forecast._model.features['target']] = result
                step_result.append(result)
            
            if training_recursive:
                training_features = features[time-system.forecast._model.time_prior:
                                             time+system.forecast._model.time_horizon]
                if lock is not None:
                    with lock:
                        system.forecast._model._train(training_features)
                else:
                    system.forecast._model._train(training_features)
                
                system.forecast._model._save_model()
            
            result = pd.DataFrame(step_result, index, columns=['prediction'])
            result.index.name = 'time'
            result['reference'] = features.loc[index, system.forecast._model.features['target']]
            result['error'] = result['prediction'] - result['reference']
            result['horizon'] = pd.Series(range(1, len(index)+1), index)
            result = result[['horizon', 'reference', 'prediction', 'error']]
            database.persist(result, subdir='results')
            
            results = pd.concat([results, result], axis=0)
            
        except ValueError as e:
            logger.debug("Skipping %s: %s", time, str(e))
        
        time += dt.timedelta(minutes=interval)
    
    return results

def _result_describe(system, results, index, name=None):
    median = results['error'].groupby([index]).median()
    median.name = 'median'
    desc = pd.concat([median, results.loc[:,'error'].groupby([index]).describe()], axis=1)
    _result_write(system, desc, name=name)

def _result_write(system, results, name=None):
    system_dir = system._configs['General']['data_dir']
    database = copy.deepcopy(system._database)
    database.dir = system_dir
    #database.format = '%Y%m%d'
    database.enabled = True
    
    csv_name = 'results'
    if name is not None:
        csv_name += '_{}'.format(name)
    csv_file = os.path.join(database.dir, csv_name+'.csv')
    
    results.to_csv(csv_file, 
                   sep=database.separator, 
                   decimal=database.decimal, 
                   encoding='utf-8')

def _result_boxplot(system, results, index, label='', name=None, colors=None, **kwargs):
    try:
        import seaborn as sns
        
        plot_name = 'results'
        if name is not None:
            plot_name += '_{}'.format(name)
        plot_file = os.path.join(system._configs['General']['data_dir'], plot_name+'.png')
        
        plt.figure()
        plot_fliers = dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='lightgrey')
        plot_colors = colors if colors is not None else index.nunique()
        plot_palette = sns.light_palette('#0069B4', n_colors=plot_colors, reverse=True)
        plot = sns.boxplot(x=index, y='error', data=results, palette=plot_palette, flierprops=plot_fliers, **kwargs) #, showfliers=False)
        plot.set(xlabel=label, ylabel='Error [W]')
        plot.figure.savefig(plot_file)
        plt.show(block=False)
    
    except ImportError:
        pass

def _prepare_weather(system):
    weather = system.forecast._weather._configs
    meteoblue = os.path.join(weather.get('General', 'lib_dir'), 'meteoblue')
    if not os.path.isdir(weather.get('Database', 'dir')) and os.path.isdir(meteoblue):
        os.makedirs(weather.get('Database', 'dir'), exist_ok=True)
        
        infos = []
        files = []
        for entry in os.scandir(meteoblue):
            if entry.is_file() and entry.path.endswith('.csv'):
                info = pd.read_csv(entry.path, skipinitialspace=True, low_memory=False, sep=';', header=None, index_col=[0]).iloc[:18,:]
                info.columns = info.iloc[3]
                
                infos.append(info.loc[:,~info.columns.duplicated()].dropna(axis=1, how='all'))
                files.append(pd.read_csv(entry.path, skipinitialspace=True, sep=';', header=[18], index_col=[0, 1, 2, 3, 4]))
        
        points = pd.concat(infos, axis=0).drop_duplicates()
        histories = pd.concat(files, axis=1)
        for point in points.columns.values:
            lat = float(points.loc['LAT', point])
            lon = float(points.loc['LON', point])
            loc = os.path.join(weather.get('Database', 'dir'), '{0:06.2f}'.format(lat).replace('.', '') + '_' + '{0:06.2f}'.format(lon).replace('.', ''))
            if not os.path.exists(loc):
                os.mkdir(loc)
            
            columns = [column for column in histories.columns.values if column.startswith(point + ' ')]
            history = histories[columns]
            history.columns = [c.replace(c.split(' ')[0], '').replace(c.split('[')[1], '').replace('  [', '') for c in columns]
            history['time'] = [dt.datetime(y, m, d, h, n) for y, m, d, h, n in history.index]
            history.set_index('time',inplace=True)
            history.index.tz_localize(tz.utc)
#             history.index.tz_localize('Europe/Berlin', ambiguous='infer')
#             history = histories.tz_convert(tz.utc)
            history.rename(columns={' Temperature': 'temp_air', 
                                    ' Wind Speed': 'wind_speed', 
                                    ' Wind Direction': 'wind_direction', 
                                    ' Wind Gust': 'wind_gust', 
                                    ' Relative Humidity': 'humidity_rel', 
                                    ' Mean Sea Level Pressure': 'pressure_sea', 
                                    ' Shortwave Radiation': 'ghi', 
                                    ' DNI - backwards': 'dni', 
                                    ' DIF - backwards': 'dhi', 
                                    ' Total Cloud Cover': 'total_clouds', 
                                    ' Low Cloud Cover': 'low_clouds', 
                                    ' Medium Cloud Cover': 'mid_clouds', 
                                    ' High Cloud Cover': 'high_clouds', 
                                    ' Total Precipitation': 'precipitation_total', 
                                    ' Snow Fraction': 'snow_fraction'}, inplace=True)
            
            time = history.index[0]
            while time <= history.index[-1]:
                file = os.path.join(loc, time.strftime('%Y%m%d_%H%M%S') + '.csv');
                history[time:time+dt.timedelta(hours=23)].to_csv(file, sep=',', encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
                
                time = time + dt.timedelta(hours=24)

def _prepare_system(system):
    opsd = os.path.join(system._configs.get('General', 'lib_dir'), 'opsd')
    if system._database is not None and not os.path.isdir(system._database.dir) and os.path.isdir(opsd):
        os.makedirs(system._database.dir, exist_ok=True)
        
        index = 'utc_timestamp'
        data = pd.read_csv(os.path.join(opsd, 'household_data_60min_singleindex.csv'), 
                           skipinitialspace=True, low_memory=False, sep=',',
                           index_col=[index], parse_dates=[index])
        
        data = data.filter(regex=(system.id)).dropna()
        columns = []
        for column in data.columns:
            columns.append(column.split(system.id+'_', 1)[1] + '_energy')
        data.columns = columns
        data.index.rename('time', inplace=True)
        
        # FIXME: investigate strange behaviour if this shift makes sense
        data = data.shift()
        data['grid_export_power'] = data['grid_export_energy'].diff()*1000
        data['grid_import_power'] = data['grid_import_energy'].diff()*1000
        data['pv_power'] = data['pv_energy'].diff()*1000
        
        data = data[['grid_export_power', 'grid_export_energy', 
                     'grid_import_power', 'grid_import_energy', 
                     'pv_power', 'pv_energy']]
        
        time = data.index[0]
        time = time.replace(hour=0, minute=0) + dt.timedelta(hours=24)
        while time <= data.index[-1]:
            file = os.path.join(system._database.dir, time.strftime('%Y%m%d_%H%M%S') + '.csv');
            csv = data[time:time+dt.timedelta(hours=23)]
            if not csv.empty and len(csv.index) == 24:
                csv.to_csv(file, sep=',', encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
            
            time = time + dt.timedelta(hours=24)

def _get_time(time_str):
    return tz.utc.localize(dt.datetime.strptime(time_str, '%d.%m.%Y'))

def _get_parser(root_dir):
    from th_e_fcst import __version__
    
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

