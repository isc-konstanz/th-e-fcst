#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    th-e-forecast
    ~~~~~
    
"""
import logging.config

import sys
import os
from boto import config
from pygments.lexers import configs
from theforecast.database import CsvDatabase
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))

from argparse import ArgumentParser
from configparser import ConfigParser
import json
import shutil
import inspect
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from keras.models import load_model
import numpy as np
import os
from theforecast import processing
import pandas

    
def main(rundir, args=None):
    from theforecast import Forecast, ForecastException
    from theforecast.neuralnetwork import NeuralNetwork
    from theforecast.io_control import IO_control
    
    if args.configs != None:
        configs = args.configs
        logger.debug('System configurations will be read in from "%s"', configs)
    else:
        configs = os.path.join(rundir, 'conf')
    logging = os.path.join(rundir, 'log')
        
    # Parameter and variables:
    IO_hist = np.zeros(1440)
    pred_start = 50 * 1440
    f_prediction = 20
    f_retrain = 200  # 6 * 60

    system = Forecast(configs)
    control = IO_control()
    data = system.databases['CSV'].data
    
    fig, axs = plt.subplots(2, 1)
    mng = plt.get_current_fig_manager()
    plt.pause(.1)
    mng.window.showMaximized()
    
    data_training = [data[0][:pred_start],  # BI
                     data[1][:pred_start]]  # timestamp
         
    X, Y = system.neuralnetwork.getInputVector(data_training,
                                               training=True)
    
    system.neuralnetwork.load(logging, 'myModel')
#     system.neuralnetwork.train(X, Y[:, 0, :], system.neuralnetwork.epochs_init)
#     system.neuralnetwork.train(X, Y[:, 0, :], 1)
#     system.neuralnetwork.model.save(logging + '\myModel')

    k = 0
    horizon = 18
    charge = 10
    t_request1 = 0
    t_request2 = 0
    status_predict = True
    stat_run = True
    
    while stat_run:
                   
        print('Iteration: ' + str(k))
        # GENERATE REQUEST
        if k % 1440 == 0:
            t_request1 = 6 * 60 + np.random.uniform() * 2 * 60   
            t_request2 = 12 * 60 + np.random.uniform() * 120  
        # daily request 1: 6-8h, horizon 3-4h      
        if (k % 1440) >= t_request1:  
            horizon = int(180 + np.random.uniform() * 60)
            charge = 120  # horizon - 60 - int(np.random.uniform() * 60)
            t_request1 = 1441
        # daily request 2: 12-14h, horizon 8-12h
        if (k % 1440) >= t_request2:    
            horizon = int(8 * 60 + np.random.uniform() * 4.5 * 60)
            charge = horizon - 100 - int(np.random.uniform() * 120)
            t_request2 = 1441
        
        # LOGGING
        if system.forecast.__len__() != 0:
            if len(control.IO_control) <= f_prediction:
                control.IO_control = np.append(control.IO_control, np.zeros(f_prediction - len(control.IO_control)))
            df = pandas.DataFrame({'unixtimestamp': system.databases['CSV'].data[1][pred_start + k : pred_start + k + f_prediction],
                                   'bi': system.databases['CSV'].data[0][pred_start + k : pred_start + k + f_prediction],
                                   'forecast': system.forecast[:f_prediction],
                                   'IO': control.IO_control[:f_prediction]})
            df = df.set_index('unixtimestamp')
            system.databases['CSV'].persist(df)
        
        # GET NEW DATA FROM DATABASE
        # TODO: get new data from DB
        
        # FORECAST   
        if charge < 5:
            status_predict = False
        elif charge >= 5:
            status_predict = True 
        try:
            system.execute(pred_start, k, f_retrain, status_predict, logging)
        except (ForecastException) as e:
            logger.error('Fatal forecast error: %s', str(e))
            sys.exit(1)  # abnormal termination
        
        # MPC
        if status_predict and horizon > 0:
            control.pred_horizon = horizon 
            control.charge_energy_amount = charge 
            control.u_init = IO_hist[-1]
        
            control.execute(system.forecast)
                 
            processing.plot_prediction(axs, k, pred_start, system)
            processing.plot_IO_control(axs, k, pred_start, system, control, IO_hist)
            plt.savefig(logging + '\\plots\\fig_' + str(int(k / 1440)) + '_' + str(k % 1440))
            
        elif charge < 5:
            print('no charge request - idle')
            control.IO_control = np.zeros(f_prediction)
            
        IO_hist = np.roll(IO_hist, -f_prediction)
        IO_hist[-f_prediction:] = np.concatenate((control.IO_control[:f_prediction],
                                             np.zeros(f_prediction - len(control.IO_control[:f_prediction]))))
        horizon = horizon - f_prediction
        charge = charge - sum(control.IO_control[:f_prediction])
        k = k + f_prediction
        
        # ENDING CONDITION
        if k >= 90 * 1440:
            stat_run = False
    
    print(' --- END of SIMULATION --- ')
    system.neuralnetwork.model.save('myModel_FinalLTS')
    os.system('shutdown -s')
    sys.exit(0)  # successful termination


if __name__ == '__main__':

    rundir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(rundir) == 'bin':
        rundir = os.path.dirname(rundir)
    
    # Load the logging configuration
    logging.config.fileConfig(os.path.join(rundir, 'conf', 'logging.cfg'))
    logger = logging.getLogger('th-e-forecast')
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--configs',
                        dest='configs',
                        help='Directory of system configuration files',
                        metavar='DIR')

    main(rundir, args=parser.parse_args())
  
