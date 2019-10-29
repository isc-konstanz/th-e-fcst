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
import pandas as pd
from datetime import datetime, timedelta
import random

    
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
    f_prediction = 20
    f_retrain = 3 * 60  # 6 * 60

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
    
    time = datetime.strptime('2018-08-01 00:00:00', '%Y-%m-%d %H:%M:%S')  # change also in CSV-database -> __init__()-method
    horizon = 0
    charge = 0
    hp_req = False  # request from heatpump
    hp_req_gen = True
    
    while True:
        print('Iteration: ' + str(time))
        # TODO: make horizon to datetime object
        # TODO: plot req_time-stack
        # GENERATE REQUEST
        if hp_req_gen:
            req_time = time  # + timedelta(minutes=random.randrange(30, 3 * 60))
            hp_req_gen = False            
        
        if time >= req_time and not hp_req:
            horizon = int(60 + np.random.uniform() * 7 * 60)
            charge = int((.85 - 4 * np.random.uniform() / 10) * horizon)  # hier 
            hp_req = True
        
        # GET NEW DATA FROM DATABASE
        system.databases['CSV'].read_file(system.databases['CSV'].datafile, time)
        
       # FORECAST   
        try:
            system.execute()
        except (ForecastException) as e:
            logger.error('Fatal forecast error: %s', str(e))
            sys.exit(1)  # abnormal termination
        
        # MPC
        if hp_req and horizon > 0:           
            control.horizon = horizon 
            control.charge_energy_amount = charge             
            if charge <= horizon:
                control.execute(system.forecast)
            else:
                # TODO: what to do if this happens? why does it happen
                print('error: charge > horizon')
                control.control = np.roll(control.control, -f_prediction)
                horizon = charge
                
            # GRAPHICS
            processing.plot_prediction(axs[0], system)
            processing.plot_control(axs[1], system, control)
            plt.savefig(logging + '\\plots\\' + time.strftime("%Y-%m-%d %H-%M"))
            
        else:
            print('no charge request - idle')
            control.control = np.zeros(f_prediction)

        # LOGGING
        if system.forecast.__len__() != 0:
            df = pd.DataFrame({'unixtimestamp': system.databases['CSV'].data.index[-f_prediction:],
                                   'bi': system.databases['CSV'].data.loc[:]['bi'][-f_prediction:],
                                   'forecast': (system.forecast[:f_prediction] * 2) - 1,
                                   'IO': control.history[-f_prediction:]})
            df = df.set_index('unixtimestamp')
            system.databases['CSV'].persist(df)
   
        # UPDATE MODEL
        if int(time.timestamp() / 60) % f_retrain == 0:  # retrain every 3 hours
            system.neuralnetwork.train(system.databases['CSV'].data)

        if  horizon <= 0 and time >= req_time:
            hp_req = False
            hp_req_gen = True

        # ENDING CONDITION
        if time >= datetime.strptime('2018-08-28 23:00:00', '%Y-%m-%d %H:%M:%S'):
            break
        
        # NEXT TIMESTEP
        control.history = np.roll(control.history, -f_prediction)
        control.history[-f_prediction:] = np.concatenate((control.control[:f_prediction],
                                             np.zeros(f_prediction - len(control.control[:f_prediction]))))
        horizon = horizon - f_prediction
        charge = charge - sum(control.control[:f_prediction])
        time = time + timedelta(minutes=f_prediction)  
          
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
  
