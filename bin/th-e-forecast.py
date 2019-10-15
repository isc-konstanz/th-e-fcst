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
import datetime as dt

    
def main(rundir, args=None):
    from theforecast import Forecast, ForecastException
    from theforecast.neuralnetwork import NeuralNetwork
    from theforecast.control import Control
    
    if args.configs != None:
        configs = args.configs
        logger.debug('System configurations will be read in from "%s"', configs)
    else:
        configs = os.path.join(rundir, 'conf')
    logging = os.path.join(rundir, 'log')
        
    # Parameter and variables:
    pred_start = 50 * 1440
    f_prediction = 20
    f_retrain = 200  # 6 * 60

    system = Forecast(configs)
    control = Control()

    system.neuralnetwork.load(logging, 'myModelInit')
#     system.neuralnetwork.train_initial(system.databases['CSV'].data)
#     system.neuralnetwork.model.save(logging + '\myModel')

#     t_start = dt.datetime.now()
#     t_now = dt.datetime.now()
#     while (t_now - t_start).seconds <= 10 * 60 * 60:  # trainingsintervall in seconds
#         system.neuralnetwork.train_initial(system.databases['CSV'].data)
#         system.neuralnetwork.model.save(logging + '\myModel')
#         system.neuralnetwork.model.save(logging + '\myModelInit')
#         t_now = dt.datetime.now()

    fig, axs = plt.subplots(2, 1)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.pause(.1)
    
    k = 0
    horizon = 150
    charge = 50
    t_request1 = 0
    t_request2 = 0
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
            df = pandas.DataFrame({'unixtimestamp': system.databases['CSV'].data.index[-f_prediction:],
                                   'bi': system.databases['CSV'].data.loc[:]['bi'][-f_prediction:],
                                   'forecast': (system.forecast[:f_prediction] * 2) - 1,
                                   'IO': control.history[-f_prediction:]})
            df = df.set_index('unixtimestamp')
            system.databases['CSV'].persist(df)
        
        # GET NEW DATA FROM DATABASE
        system.databases['CSV'].read_file(system.databases['CSV'].datafile, k, pred_start)
        
        # FORECAST   
        try:
            system.execute()
        except (ForecastException) as e:
            logger.error('Fatal forecast error: %s', str(e))
            sys.exit(1)  # abnormal termination
        
        # MPC
        if charge > 5 and horizon > 0:
            control.horizon = horizon 
            control.charge_energy_amount = charge             
            if charge <= horizon:
                control.execute(system.forecast)
            else:
                # TODO: what to do if this happens? why does it happen
                print('error: charge > horizon')
                control.control = np.roll(control.control, -f_prediction)
                
            # GRAPHICS
            processing.plot_prediction(axs, system)
            processing.plot_control(axs, system, control)
            plt.savefig(logging + '\\plots\\fig_' + str(int(k / 1440)) + '_' + str(k % 1440))
            
        else:
            print('no charge request - idle')
            control.control = np.zeros(f_prediction)
        
        # UPDATE MODEL
        if k % f_retrain == f_retrain - f_prediction:
            system.neuralnetwork.train(system.databases['CSV'].data)
            
        control.history = np.roll(control.history, -f_prediction)
        control.history[-f_prediction:] = np.concatenate((control.control[:f_prediction],
                                             np.zeros(f_prediction - len(control.control[:f_prediction]))))
        horizon = horizon - f_prediction
        charge = charge - sum(control.control[:f_prediction])
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
  
