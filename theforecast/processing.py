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


def create_datasetBI(data, look_back, look_ahead, fMin, training):
    nMinMean = 15
    if training == True:
        l = int((len(data) - 7 * 24 * 60 - (look_ahead - 0.5) * fMin)) 
    elif training == False:
        l = 1
        
    dataX = np.zeros([l, look_back])
    dataY = np.zeros([l, look_ahead])  
    i1 = 120 * 60  # 60 min intervall      = 5d
    i2 = 184 * nMinMean  # nMinMean interval      = 46h 46*60
    i3 = 120  # fMin intervall     = 2h
    
    for z in range(l):
        section1Raw = data[z: z + i1, 0]
        section2Raw = data[z + i1: z + i1 + i2, 0]
        section3Raw = data[z + i1 + i2: z + i1 + i2 + i3, 0]
        
        section1Raw = section1Raw.reshape(120, 60)
        section2Raw = section2Raw.reshape(184, nMinMean)
        section3Raw = section3Raw.reshape(int(120 / fMin), fMin)
        
        sec1Mean = np.mean(section1Raw, axis=1)
        sec2Mean = np.mean(section2Raw, axis=1)
        sec3Mean = np.mean(section3Raw, axis=1)
        
        dataX[z, :] = np.concatenate([sec1Mean, sec2Mean, sec3Mean])
        if training == True:
            dataY[z, :] = data[z + i1 + i2 + i3:z + i1 + i2 + i3 + 1440][::fMin, 0]
    return dataX, dataY


def create_datasetDT(data, look_back, look_ahead, fMin, training):
    nMinMean = 15
    if training == True:
        l = int((len(data) - 7 * 24 * 60 - (look_ahead - 0.5) * fMin)) 
    elif training == False:
        l = 1 
    dataX = np.zeros([l, look_back]) 
    dataY = np.zeros([l, look_ahead])  
    i1 = 120 * 60  # 60 min intervall      ~ 5d
    i2 = 184 * nMinMean  # 15 min interval      ~ 46h
    i3 = 120  # fMin min intervall     ~ 2h
    
    for z in range(l):
        section1 = data[z:z + i1, 0][30::60]
        section2 = data[z + i1: z + i1 + i2, 0][int(nMinMean / 2)::nMinMean]
        section3 = data[z + i1 + i2: z + i1 + i2 + i3, 0][int(fMin / 2)::fMin]
        dataX[z, :] = np.concatenate([section1, section2, section3])
        if training == True:
            dataY[z, :] = data[z + i1 + i2 + i3:z + i1 + i2 + i3 + 1440][::fMin, 0]
    return dataX, dataY

