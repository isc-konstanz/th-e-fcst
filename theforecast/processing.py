'''
Created on 09.08.2019

@author: sf
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def getDaytime(data):
    """ returns the Daytime in seconds"""
    seconds = np.zeros([len(data), 1])
    for z in range(0, len(data)):
        seconds[z, 0] = data[z].hour * 3600 + data[z].minute * 60 + data[z].second
    return seconds


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
    i1 = 60  # 1 min interval      ~ 1h
    i2 = 36 * 5  # 5 min interval      ~ 3h
    i3 = 80 * 15  # 15 min interval     ~ 20h
        
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
    b, a = signal.butter(8, 0.015)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
    bi = signal.filtfilt(b, a, bi, padlen=150)
    
    axs[0].clear()
    
    axs[0].plot(np.linspace(50 + k / 1440, 58 + k / 1440, 8 * 1440), bi, 'k--')

    axs[0].plot(np.linspace(50 + k / 1440, 55 + k / 1440, 120), input_vector[0, 0, 0:120], 'r')
    axs[0].plot(np.linspace(55 + k / 1440, 56.9166 + k / 1440, 184), input_vector[0, 0, 120:304], 'r')
    axs[0].plot(np.linspace(56.9166 + k / 1440 + 7.5 / 1440, 57 + k / 1440, 24), input_vector[0, 0, 304:], 'r')
     
    axs[0].plot(np.linspace(57 + (k + 1) / 1440, 58 + (k + 1) / 1440, 1440), prediction, 'b') 
#     axs[0].plot(np.linspace(57 + k / 1440 + 1 / 1440, 57 + 1 / 24 + k / 1440 + 5 / 1440, 60), prediction[:60], 'b')
#     axs[0].plot(np.linspace(57 + 1 / 24 + k / 1440 + 10 / 1440, 57 + 4 / 24 + k / 1440 + 5 / 1440, 36), prediction[60:96], 'b')
#     axs[0].plot(np.linspace(57 + 4 / 24 + k / 1440 + 15 / 1440, 58 + k / 1440 + 5 / 1440, 80), prediction[96:], 'b')
#     
    axs[0].grid(True)
    axs[0].set_xlim([56 + k / 1440, 58.0 + k / 1440])
    plt.pause(0.1)

    
def plot_IO_control(axs, IO_hist, control, k, f_pred):
    axs[1].clear()
    IO_stack = control.IO_stack
    IO_control = control.IO_control
    
    axs[1].plot(np.linspace(56 + (k + f_pred) / 1440, 57 + (k + f_pred) / 1440, 1440), IO_hist, 'r.', markersize=3, linewidth=0.4)
    
    x1 = np.linspace(57 + k / 1440 + 1 / 1440, 57 + 1 / 24 + k / 1440 + f_pred / 1440, 60)
    x2 = np.linspace(57 + 1 / 24 + k / 1440 + 10 / 1440, 57 + 4 / 24 + k / 1440 + f_pred / 1440, 36)
    x3 = np.linspace(57 + 4 / 24 + k / 1440 + 15 / 1440, 58 + k / 1440 + f_pred / 1440, 80)
    
    for i in range(IO_stack.shape[0]):
        IO_actual_tmp = np.append(IO_stack[i, :], np.zeros(176 - IO_stack.shape[1]))
        axs[1].plot(x1, IO_actual_tmp[0:60], '--', linewidth=0.5)
        axs[1].plot(x2, IO_actual_tmp[60:96], '--', linewidth=0.5)
        axs[1].plot(x3, IO_actual_tmp[96:], '--', linewidth=0.5)
        
    IO_actual_tmp = np.append(IO_control, np.zeros(176 - len(IO_control)))
    axs[1].plot(x1, IO_actual_tmp[0:60], 'b')
    axs[1].plot(x2, IO_actual_tmp[60:96], 'b')
    axs[1].plot(x3, IO_actual_tmp[96:], 'b')  
      
    axs[1].grid(True)
    axs[1].set_xlim([56 + k / 1440, 58.0 + k / 1440])
    axs[1].set_ylim([-.1, 1.1])
    plt.pause(.1)
