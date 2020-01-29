"""
    th-e-forecast
    ~~~~~
    
"""
import logging.config

import sys
import os
# from boto import config
from pygments.lexers import configs
# from theforecast.database import CsvDatabase
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))

from argparse import ArgumentParser
# from configparser import ConfigParser
# import json
# import shutil
import inspect
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from theforecast import processing
from keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
# import pytz as tz

    
def main(rundir, args=None):
    from theforecast import Forecast, ForecastException
    from theforecast.neuralnetwork import NeuralNetwork
    from theforecast.mpc import MPC, MPCException
    
    if args.configs != None:
        configs = args.configs
        logger.debug('System configurations will be read in from "%s"', configs)
    else:
        configs = os.path.join(rundir, 'conf')
    log = os.path.join(rundir, 'log')
        
    # INITIALIZE VARIABLES AND CONSTANTS
    system = Forecast(configs)
    mpc = MPC()
    f_prediction = system.interval
    f_retrain = 4 * 60
    
    fig, axs = plt.subplots(2, 1)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.pause(.1)

    time = datetime.strptime(system.start, '%Y-%m-%d %H:%M:%S')  
    end = datetime.strptime('2018-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')
    req = time
    hp_threshold = 1
    horizon = 0
    charge = 0
    kpi_req = False
    kpi = []
    
    system.neuralnetwork.load(rundir)
    
# ---------- INITIAL MODEL TRAINING ----------
#     system.neuralnetwork.initialize(system.databases['CSV'].data)
#     system.neuralnetwork.model.save(log + '\myModel')
#     system.neuralnetwork.model.save(rundir + '\lib\my_model_init')
#     os.system('shutdown -s')
    
    while time < end:
        print('Iteration: ' + str(time))
        # TODO: plot req_time-stack
        # TODO: convert to utc time
        # GENERATE REQUESTS
        if time >= req:
            horizon = 45  # int(120 + np.random.uniform() * 7 * 60)
            charge = int((.80 - 4 * np.random.uniform() / 10) * horizon)
            
            req_horizon = horizon
            req_charge = charge
            req_start = time
            req_end = time + timedelta(minutes=horizon)        
            req = time + timedelta(minutes=horizon) + timedelta(minutes=random.randrange(15, 105))
        
        # GET NEW DATA FROM DATABASE
        system.databases['CSV'].read_file(system.databases['CSV'].datafile, time)
        
       # FORECAST   
        try:
            system.execute()
        except (ForecastException) as e:
            logger.error('Fatal forecast error %s', time)
        
        # MPC
        if horizon > hp_threshold:           
            mpc.horizon = horizon 
            mpc.charge_energy_amount = charge  
                       
            try:
                mpc.execute(system.forecast)
            except (MPCException) as e:
                logger.error('Fatal optimization error %s', time)                
                mpc.control = np.roll(mpc.control, -f_prediction)
                horizon = charge
            kpi_req = True
                
        # GRAPHICS
            try:
                processing.plot_prediction(axs[0], system)
                processing.plot_control(axs[1], system, mpc)
                plt.savefig(log + '\\plots\\' + time.strftime("%Y-%m-%d %H-%M"))
            except:
                logger.warning('plotting error at %s', time)
        else:
            print('no charge request - idle')
            mpc.control = np.zeros(f_prediction)

        # EVALUATION (KPI) 
        if time >= req_end:
            system.get_kpi_forecast(time)
            system.get_kpi_mpc(mpc, req_start, req_end)
            kpi.append(0)  # dataframe dt_start, dt_stop, charge, kpi
        
        # LOGGING
        if system.forecast.__len__() != 0:
            system.databases['CSV'].persist(system, mpc)
   
        # UPDATE MODEL
        if int(time.timestamp() / 60) % f_retrain == 0:
            system.neuralnetwork.train(system.databases['CSV'].data)

        # NEXT TIMESTEP
        mpc.history = np.roll(mpc.history, -f_prediction)
        mpc.history[-f_prediction:] = np.concatenate((mpc.control[:f_prediction],
                                             np.zeros(f_prediction - len(mpc.control[:f_prediction]))))
        horizon = horizon - f_prediction
        charge = charge - sum(mpc.control[:f_prediction])
        time = time + timedelta(minutes=f_prediction) 
    
    # --------------------------------------------      
    # ---------- SUCCESSFUL TERMINATION ----------
    # --------------------------------------------
    logger.info('Simulation terminated successfully at %s \n --- END ---', time)
    system.neuralnetwork.model.save('myModel_FinalLTS')
    os.system('shutdown -s')
    sys.exit(0)


if __name__ == '__main__':

    rundir = os.path.dirname(os.path.abspath(inspect.getsourcefile(main)))
    if os.path.basename(rundir) == 'bin':
        rundir = os.path.dirname(rundir)
    
    # Load the logging configuration
    logging.config.fileConfig(os.path.join(rundir, 'conf', 'logging.cfg'))
    logger = logging.getLogger('th-e-forecast')
    logger.info('\n ------------ starting th-e-forecast ------------')
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--configs',
                        dest='configs',
                        help='Directory of system configuration files',
                        metavar='DIR')

    main(rundir, args=parser.parse_args())

