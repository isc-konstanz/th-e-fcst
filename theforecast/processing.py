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
    """ returns the Daytime in seconds"""
    seconds = np.zeros([len(data), 1])
    for z in range(0, len(data)):
        seconds[z, 0] = data[z].hour * 3600 + data[z].minute * 60 + data[z].second
    return seconds / (24 * 60 * 60)


def create_dataset_mean(data, look_back, pred_horizon, fMin, training):
    look_ahead_total = 1440  # [min]
    if training == True:
        l = int(len(data) - 7 * 24 * 60 - look_ahead_total)
    elif training == False:
        l = 1
        
    dataX = np.zeros([l, look_back])
    dataY = np.zeros([l, int(pred_horizon / fMin)])  
    i1 = 120 * 60  # 60 min interval      = 5d
    i2 = 184 * 15  # 15 min interval      = 46h 46*60
    i3 = 120  # fMin interval     = 2h
    
    for z in range(l):
        section1Raw = data[z: z + i1, 0]
        section2Raw = data[z + i1: z + i1 + i2, 0]
        section3Raw = data[z + i1 + i2: z + i1 + i2 + i3, 0]
        
        section1Raw = section1Raw.reshape(120, 60)
        section2Raw = section2Raw.reshape(184, 15)
        section3Raw = section3Raw.reshape(int(120 / fMin), fMin)
        
        sec1Mean = np.mean(section1Raw, axis=1)
        sec2Mean = np.mean(section2Raw, axis=1)
        sec3Mean = np.mean(section3Raw, axis=1)
        
        dataX[z, :] = np.concatenate([sec1Mean, sec2Mean, sec3Mean])
        if training == True:
            dataY[z, :] = data[z + i1 + i2 + i3:z + i1 + i2 + i3 + pred_horizon][::fMin, 0]
    return dataX, dataY


def create_dataset(data, look_back, pred_horizon, fMin, training):
    look_ahead_total = 1440  # [min]
    if training == True:
        l = int(len(data) - 7 * 24 * 60 - look_ahead_total)
    elif training == False:
        l = 1 
    dataX = np.zeros([l, look_back]) 
    dataY = np.zeros([l, int(pred_horizon / fMin)])  
    i1 = 120 * 60  # 60 min interval      ~ 5d
    i2 = 184 * 15  # 15 min interval      ~ 46h
    i3 = 120  # fMin min interval     ~ 2h
    
    for z in range(l):
        section1 = data[z:z + i1, 0][int(60 / 2)::60]
        section2 = data[z + i1: z + i1 + i2, 0][int(15 / 2)::15]
        section3 = data[z + i1 + i2: z + i1 + i2 + i3, 0][int(fMin / 2)::fMin]
        dataX[z, :] = np.concatenate([section1, section2, section3])
        if training == True:
            dataY[z, :] = data[z + i1 + i2 + i3:z + i1 + i2 + i3 + pred_horizon][::fMin, 0]
    return dataX, dataY


def create_input_vector(data, look_back, look_ahead, fMin, training):
    if training == True:
        l = int(len(data) - 7 * 24 * 60 - look_ahead)
    elif training == False:
        l = 1 
    dataX = np.zeros([l, look_back]) 

    i1 = 120 * 60  # 60 min interval      ~ 5d
    i2 = 184 * 15  # 15 min interval      ~ 46h
    i3 = 120  # fMin min interval     ~ 2h
        
    for z in range(l):
        section1 = data[z:z + i1, 0][int(60 / 2)::60]
        section2 = data[z + i1: z + i1 + i2, 0][int(15 / 2)::15]
        section3 = data[z + i1 + i2: z + i1 + i2 + i3, 0][int(fMin / 2)::fMin]
        dataX[z, :] = np.concatenate([section1, section2, section3])

    return dataX


def create_output_vector(data, look_ahead, fMin, training):
    if training == True:
        l = int(len(data) - 7 * 24 * 60 - look_ahead)
    elif training == False:
        l = 1 
    dataY = np.zeros([l, look_ahead])  
#     i1 = 60  # 1 min interval      ~ 1h
#     i2 = 36 * 5  # 5 min interval      ~ 3h
#     i3 = 80 * 15  # 15 min interval     ~ 20h
        
    i_start = 7 * 24 * 60  
    for z in range(l):
        section1 = data[z + i_start:z + i_start + look_ahead, 0]
        dataY[z, :] = section1
#         section1 = data[z + i_start:z + i_start + i1, 0]
#         section2 = data[z + i_start + i1: z + i_start + i1 + i2, 0][int(5 / 2)::5]
#         section3 = data[z + i_start + i1 + i2: z + i_start + i1 + i2 + i3, 0][int(15 / 2)::15]
#         dataY[z, :] = np.concatenate([section1, section2, section3])

    return dataY


def getdBI(data, fMin):
    dBI = (data[1:] - data[:-1]) / fMin
    return dBI
    
    
def plot_prediction(axs, system, input_vector, prediction, k, pred_start):
    bi = (system.databases['CSV'].data[0][pred_start - 7 * 1440 + k:pred_start + 1440 + k] + 1) / 2
    b, a = signal.butter(8, 0.02)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
    bi_filtered = signal.filtfilt(b, a, bi, method='pad', padtype='even', padlen=150)

    dt = system.databases['CSV'].data[1][pred_start - 7 * 1440 + k:pred_start + 1440 + k]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base - timedelta(minutes=1440), base , timedelta(minutes=1)).astype(datetime)
    a1 = np.arange(dt[-8 * 1440], dt[-3 * 1440], timedelta(minutes=60))
    a2 = np.arange(dt[-3 * 1440], dt[-3 * 1440 + 46 * 60], timedelta(minutes=15))
    a3 = np.arange(dt[-1440 - 120], dt[-1440], timedelta(minutes=system.neuralnetwork.fMin))
    x_input = np.concatenate((a1, a2, a3))
    
    axs[0].clear()
    axs[0].plot(dt, bi, 'g--', linewidth=0.5, label='bi (raw)')
    axs[0].plot(dt, bi_filtered, 'k--', label='bi (filtered)')
    axs[0].plot(x_input, input_vector[0, 0, :], 'r', label='input Data')
    axs[0].plot(x_pred, prediction, 'b', label='prediction') 
    
    axs[0].set_title('prediction')
    axs[0].legend(loc='lower left')
    axs[0].grid(True)
    axs[0].set_xlim([base - timedelta(days=2.1), base])
    plt.pause(0.1)

    
def plot_IO_control(axs, system, IO_hist, control, k, pred_start):
    dt = system.databases['CSV'].data[1][pred_start - 7 * 1440 + k:pred_start + 1440 + k]
    base = dt[-1] + timedelta(minutes=1)
    x_pred = np.arange(base - timedelta(minutes=1440),
                       base ,
                       timedelta(minutes=1)).astype(datetime)
    x_hist = np.arange(base - timedelta(minutes=2880),
                       base - timedelta(minutes=1440),
                       timedelta(minutes=1)).astype(datetime)
    IO_control = np.append(control.IO_control, np.zeros(1440 - len(control.IO_control)))  
    scaler = MinMaxScaler(feature_range=(0, 1))
    pred_scaled = scaler.fit_transform(control.prediction.reshape(-1, 1))
    
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
