'''
Created on 09.08.2019

@author: sf
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def getDaytime(data):
    """ returns the normalized Daytime [0,1]"""
    seconds = np.zeros([len(data), 1])
    for z in range(0, len(data)):
        seconds[z, 0] = data[z].hour * 3600 + data[z].minute * 60 + data[z].second
    return seconds / (24 * 60 * 60)


def create_input_vector(data, look_back, look_ahead, fMin, training):
    if training == True:
        l = int(len(data) - 4 * 24 * 60 - look_ahead)
    elif training == False:
        l = 1 
    dataX = np.zeros([l, look_back]) 

    # i1 = 24 * 5 * 60  # 60 min interval      ~ 5d
    i1 = 24 * 2 * 60
    i2 = 46 * 60  # 15 min interval      ~ 46h
    i3 = 2 * 60  # fMin min interval     ~ 2h
        
    for z in range(l):
        section1 = data[z:z + i1, 0][int(60 / 2)::60]
        section2 = data[z + i1: z + i1 + i2, 0][int(15 / 2)::15]
        section3 = data[z + i1 + i2: z + i1 + i2 + i3, 0][int(fMin / 2)::fMin]
        dataX[z, :] = np.concatenate([section1, section2, section3])

    return dataX


def create_output_vector(data, look_ahead, fMin, training):
    if training == True:
        l = int(len(data) - 4 * 24 * 60 - look_ahead)
    elif training == False:
        l = 1 
    dataY = np.zeros([l, look_ahead])  
        
    i_start = 4 * 24 * 60  
    for z in range(l):
        section1 = data[z + i_start:z + i_start + look_ahead, 0]
        dataY[z, :] = section1
#         section1 = data[z + i_start:z + i_start + i1, 0]
#         section2 = data[z + i_start + i1: z + i_start + i1 + i2, 0][int(5 / 2)::f]
#         section3 = data[z + i_start + i1 + i2: z + i_start + i1 + i2 + i3, 0][int(15 / 2)::15]
#         dataY[z, :] = np.concatenate([section1, section2, section3])

    return dataY
    
    
def plot_prediction(axs, k, pred_start, system):
    data = system.databases['CSV'].data
    bi = (data[0][pred_start - 4 * 1440 + k:pred_start + 1440 + k] + 1) / 2
    b, a = signal.butter(8, 0.02)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
    bi_filtered = signal.filtfilt(b, a, bi, method='pad', padtype='even', padlen=150)         
    
    data_input_pred = [data[0][pred_start - 4 * 1440 + k : pred_start + k],
                       data[1][pred_start - 4 * 1440 + k : pred_start + k]]
    input_vector = system.neuralnetwork.getInputVector(data_input_pred,
                                                       system.neuralnetwork.lookBack,
                                                       system.neuralnetwork.lookAhead,
                                                       system.neuralnetwork.fMin,
                                                       training=False)
                
    dt = data[1][pred_start - 4 * 1440 + k:pred_start + 1440 + k]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base - timedelta(minutes=1440), base , timedelta(minutes=1)).astype(datetime)
    a1 = np.arange(dt[-5 * 1440], dt[-3 * 1440], timedelta(minutes=60))
    a2 = np.arange(dt[-3 * 1440], dt[-3 * 1440 + 46 * 60], timedelta(minutes=15))
    a3 = np.arange(dt[-1440 - 120], dt[-1440], timedelta(minutes=system.neuralnetwork.fMin))
    x_input = np.concatenate((a1, a2, a3))
    
    axs[0].clear()
    axs[0].plot(dt, bi, 'g--', linewidth=0.5, label='bi (raw)')
    axs[0].plot(dt, bi_filtered, 'k--', label='bi (filtered)')
    axs[0].plot(x_input, input_vector[0, 0, :], 'r', label='input Data')
    axs[0].plot(x_pred, system.forecast, 'b', label='prediction') 
    axs[0].plot(x_pred, system.forecast_unfiltered, 'b--', linewidth=0.5, label='prediction')
    
    axs[0].set_title('prediction')
    axs[0].legend(loc='lower left')
    axs[0].grid(True)
    axs[0].set_xlim([base - timedelta(days=2.1), base])
    plt.pause(0.1)

    
def plot_IO_control(axs, k, pred_start, system, control, IO_hist):
    dt = system.databases['CSV'].data[1][pred_start - 4 * 1440 + k:pred_start + 1440 + k]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base - timedelta(minutes=1440),
                       base ,
                       timedelta(minutes=1)).astype(datetime)
    x_hist = np.arange(base - timedelta(minutes=2880),
                       base - timedelta(minutes=1440),
                       timedelta(minutes=1)).astype(datetime)
    IO_control = np.append(control.IO_control, np.zeros(1440 - len(control.IO_control)))  
    scaler = MinMaxScaler(feature_range=(0, 1))
    pred_scaled = scaler.fit_transform(system.forecast.reshape(-1, 1))
    
    axs[1].clear()

    axs[1].plot(x_hist, IO_hist, 'r.', markersize=3, linewidth=0.4, label='MPC (history)')
    axs[1].plot(x_pred, IO_control, 'k', label='MPC')    
    axs[1].plot(x_pred, pred_scaled, 'b', label='prediction')
    axs[1].plot([x_pred[control.pred_horizon],
                 x_pred[control.pred_horizon]], [-1.5, 1.5], 'g--', linewidth=0.5, label='horizon')
    
    axs[1].set_title('MPC')
    axs[1].legend(loc='lower left')
    axs[1].grid(True)
    axs[1].set_xlim([base - timedelta(days=2.1), base])
    axs[1].set_ylim([-.1, 1.1])
    plt.pause(.1)
