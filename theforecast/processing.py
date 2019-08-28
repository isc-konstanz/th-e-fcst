'''
Created on 09.08.2019

@author: sf
'''
import numpy as np


def getDaytime(data):
    """ returns the Daytime in seconds"""
    seconds = np.zeros([len(data), 1])
    for z in range(0, len(data)):
        seconds[z, 0] = data[z].hour * 3600 + data[z].minute * 60 + data[z].second
    return seconds


def create_dataset_mean(data, look_back, pred_horizon, fMin, training):
    look_ahead_total = 1440  # [min]
    if training == True:
        # l = int((len(data) - 7 * 24 * 60 - (int(look_ahead_total / 60) - 0.5) * fMin)) 
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
        # l = int((len(data) - 7 * 24 * 60 - (int(look_ahead_total / 60) - 0.5) * fMin)) 
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

