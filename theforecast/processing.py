'''
Created on 09.08.2019

@author: sf
'''
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas


def get_daytime(data):
    """ returns the normalized Daytime [0,1]"""
    seconds = np.zeros([len(data)])
    for z in range(len(data)):
        seconds[z] = data[z].hour * 3600 + data[z].minute * 60 + data[z].second
    return seconds / (24 * 60 * 60)


def create_input_vector(data, theNN, l):
    '''Description: returns the input vector for the neural network'''
    dataX = np.zeros([l, theNN.look_back]) 
    i1 = 24 * 2 * 60
    i2 = 46 * 60  # 15 min interval      ~ 46h
    i3 = 2 * 60  # fMin min interval     ~ 2h
        
    for z in range(l):
        section1 = data[z:z + i1][int(60 / 2)::60]
        section2 = data[z + i1: z + i1 + i2][int(15 / 2)::15]
        section3 = data[z + i1 + i2: z + i1 + i2 + i3][int(theNN.fMin / 2)::theNN.fMin]
        dataX[z, :] = np.concatenate([section1, section2, section3])

    return dataX


def create_output_vector(data, theNN, l):
    '''Description: creates the output vector for the neural network. 
    Only used for training purposes.'''
    dataY = np.zeros([l, theNN.look_ahead])  
    i_start = 4 * 24 * 60  
    for z in range(l):
        section1 = data[z + i_start:z + i_start + theNN.look_ahead]
        dataY[z, :] = section1

    return dataY
    
    
def plot_prediction(axs, system):
    data = system.databases['CSV'].data
    data = [data.loc[:]['bi'].get_values(),
            pandas.Series.tolist(data.index)]
    bi = (data[0][-1440 * 5:] + 1) / 2
    data_input_pred = [data[0][-4 * 1440:],
                       data[1][-4 * 1440:]]
    input_vector = system.neuralnetwork.get_data_vector(data_input_pred, training=False)
                
    dt = data[1][-1440 * 5:]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base , base + timedelta(minutes=1440), timedelta(minutes=1)).astype(datetime)
    x_input = np.concatenate((np.arange(dt[-4 * 1440], dt[-2 * 1440], timedelta(minutes=60)),
                              np.arange(dt[-2 * 1440], dt[-2 * 1440 + 46 * 60], timedelta(minutes=15)),
                              np.arange(dt[-120], dt[-1], timedelta(minutes=system.neuralnetwork.fMin))))
    
    axs.clear()    
    axs.plot(dt, bi, 'c', linewidth=0.5, label='bi (raw)')
    axs.plot(x_input, input_vector[0, 0, :], 'r', label='input Data')
    axs.plot(x_pred, system.forecast, 'b', label='prediction') 
    
    axs.set_ylabel('prediction', color='tab:blue')
    axs.set_title('Results - prediction & control')
    axs.legend(loc='lower left')
    axs.grid(True)
    axs.set_xlim([base - timedelta(days=1.1), base + timedelta(days=1)])
    plt.pause(0.1)

    
def plot_control(axs, system, control):
    data = system.databases['CSV'].data
    data = [data['bi'].get_values(),
            pandas.Series.tolist(data.index)]
    bi = (data[0][-1440 * 5:] + 1) / 2
                
    dt = data[1][-1440 * 5:]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base + timedelta(minutes=1),
                       base + timedelta(minutes=1441),
                       timedelta(minutes=1)).astype(datetime)
    x_hist = np.arange(base - timedelta(minutes=1440),
                       base,
                       timedelta(minutes=1)).astype(datetime)
    ctrl = np.append(control.control, np.zeros(1440 - len(control.control)))  
    ts_horizon = dt[-1] + timedelta(minutes=len(control.control))
    
    axs.clear()    
    axs.plot(dt, bi, 'c', linewidth=0.5, label='bi (raw)')
    axs.plot(x_hist, control.history, 'r.', markersize=3, linewidth=0.4, label='MPC (history)')
    axs.plot(x_pred, ctrl, 'k', label='MPC')    
    axs.plot(x_pred, system.forecast, 'b', label='prediction')
    axs.plot([ts_horizon, ts_horizon], [-1.5, 1.5], 'g--', linewidth=0.7, label='horizon')
    
    axs.set_title('MPC')
    axs.legend(loc='lower left')
    axs.grid(True)
    axs.set_xlim([base - timedelta(days=1.1), base + timedelta(days=1)])
    axs.set_ylim([-.1, 1.1])
    plt.pause(.1)

