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
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser, RawTextHelpFormatter
from tensorboard import program

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

    settings = configs.read('settings.cfg', **vars(args))

    error = False
    kwargs = vars(args)
    kwargs.update(settings.items('General'))

    systems = System.read(**kwargs)

    verbose = settings.getboolean('General', 'verbose', fallback=False)

    tensorboard = _launch_tensorboard(systems, **kwargs)

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

        training_start = _get_time(settings['Training']['start'], timezone)
        training_end = _get_time(settings['Training']['end'], timezone)
        training_end = ceil_date(training_end, system.location.pytz)

        bldargs = dict(kwargs)
        bldargs['start'] = training_start
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
                        features = hdf.get('features').loc[training_start:training_end]
                else:
                    data = system.forecast._get_data_history(training_start, training_end)
                    features = system.forecast._model.features.extract(data)
                    features = system.forecast._model.features._add_meta(features)
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

                features = system.forecast._model.features.extract(data)
                features = system.forecast._model.features._add_meta(features)
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
                'durations': durations,
                'evaluation': None
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
def simulate(settings, system, features):
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
    end = ceil_date(features.index[-1] - resolution_max.time_horizon, timezone=system.location.tz)
    start = floor_date(features.index[0] + resolution_max.time_prior, timezone=system.location.tz)
    date = start

    training_recursive = settings.getboolean('Training', 'recursive', fallback=True)
    if training_recursive:
        training_interval = settings.getint('Training', 'interval', fallback=24)*60
        training_date = _next_date(date, training_interval*2)
        training_last = date
        date = _next_date(date, training_interval)

    results = pd.DataFrame()
    while date <= end:
        # Check if this step was simulated already and load the results, if so
        date_str = date.strftime('%Y%m%d_%H%M%S')
        date_dir = os.path.join(system_dir, 'results', date.strftime('%Y%m%d'))
        date_path = '/{0}.'.format(date)
        if date_path in datastore:
            result = datastore.get(date_path+'/outputs')
            input = datastore.get(date_path+'/input')
            target = datastore.get(date_path+'/target')

            # results[date] = (input, target, prediction)
            results = pd.concat([results, result], axis=0)

            date = _next_date(date, interval)
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

    database.close()
    datastore.close()

    return results

def evaluate(settings, systems):
    from th_e_data.io import write_excel
    from evaluation import Evaluations

    def concat_evaluation(sys_name, name, header, data):
        if data is None:
            return

        cols = [(header, sys_name, col) for col in data.columns]
        data.columns = pd.MultiIndex.from_tuples(cols, names=['target', 'system', 'metrics'])
        if name not in evaluations.keys():
            evaluations[name] = data
        else:
            evaluations[name] = pd.concat([evaluations[name], data], axis=1)

    def add_evaluation(sys_name, name, header, kpi, data=None):
        concat_evaluation(sys_name, name, header, data)
        if kpi is not None:
            summary_tbl.loc[sys_name, (header, name)] = kpi

    def _print_boxplot(system, labels, data, file, **kwargs):
        from th_e_data.io import print_boxplot
        try:
            colors = len(set(labels))
            print_boxplot(system, None, labels, data, file, colors=colors, **kwargs)

        except ImportError as e:
            logger.debug("Unable to plot boxplot for {} of system {}: {}".format(os.path.abspath(file), system.name,
                                                                                 str(e)))

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

        # Instantiate class Evaluations
    evals = Evaluations.read('conf')
    evals.run()

    for eval_id, eval in evals.items():
        for sys in eval.systems:
            for target in eval.targets:

                target_id = target.replace('_power', '')
                target_name = target_id if target_id not in TARGETS else TARGETS[target_id]

                metric_data = eval.evaluation.loc[:, (target, slice(None), sys)]
                metric_data.columns = metric_data.columns.get_level_values('metrics')
                add_evaluation(sys, eval.name, target_name, None, metric_data)

                for summary in eval.summaries:
                    summary_data = float(eval.kpi[target, summary, sys])
                    add_evaluation(sys, eval.name, target_name, kpi=summary_data, data=None)

    write_excel(settings, summary_tbl, evaluations)


def _launch_tensorboard(systems, **kwargs):
    # noinspection PyProtectedMember
    launch = any([system.forecast._model._tensorboard for system in systems])
    if launch:
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
    return (date + relativedelta(minutes=interval)).round('{minutes}s'.format(minutes=interval))


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
    logging.getLogger('tensorflow').setLevel(logging.WARN)

    logger = logging.getLogger('th-e-simulation')

    main(_get_parser(run_dir).parse_args())
