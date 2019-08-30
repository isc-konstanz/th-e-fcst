'''
Created on 12.07.2019

@author: sf
'''
import os
from configparser import ConfigParser
import keras
import logging
import sys
import numpy as np
import pandas as pd
import theforecast.processing as processing
from datetime import datetime
from datetime import timedelta 
import matplotlib.pyplot as plt
from scipy import signal

logger = logging.getLogger(__name__)


class NeuralNetwork:
    
    def __init__(self, configs):
        # get NN configurations:
        neuralnetworkfile = os.path.join(configs, 'neuralnetwork.cfg')
        settings = ConfigParser()
        settings.read(neuralnetworkfile)
        self.fMin = settings.getint('Input vector', 'fMin')
        self.lookBack = int(settings.getint('Input vector', 'interval 1') / 60) + \
                        int(settings.getint('Input vector', 'interval 2') / 15) + \
                        int(settings.getint('Input vector', 'interval 3') / self.fMin)
        
        self.dropout = settings.getfloat('General', 'dropout')
        self.nLayers = settings.getint('General', 'layers')
        self.nNeurons = settings.getint('General', 'neurons')
        self.lookAhead = int(settings.getint('General', 'lookAhead'))
        if self.lookAhead < self.fMin:
            self.lookAhead = self.fMin
        self.dimension = settings.getint('General', 'dimension')
        
        # self.model = self.create()
        self.model = self.create_model_dropout()
    
    def create(self):    
        model = keras.Sequential()
        model.add(keras.layers.LSTM(self.nNeurons, input_shape=(self.dimension, self.lookBack), return_sequences=True))
        model.add(keras.layers.Dropout(self.dropout))
        for z in range(self.nLayers - 1):
            model.add(keras.layers.Dense(self.nNeurons, activation='sigmoid')) 
            model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(int(self.lookAhead / self.fMin), activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def create_model_dropout(self):
        inputs = keras.layers.Input(shape=(self.dimension, self.lookBack))
        x = keras.layers.LSTM(self.nNeurons, recurrent_dropout=self.dropout, return_sequences=True)(inputs, training=True)
        x = keras.layers.Dropout(self.dropout)(x, training=True)
        x = keras.layers.LSTM(self.nNeurons)(x, training=True)
        x = keras.layers.Dropout(self.dropout)(x, training=True)
        outputs = keras.layers.Dense(int(self.lookAhead / self.fMin))(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse']) 
        return model
    
    def train(self, trainingData):
        try: 
            self.model.fit(trainingData)
        except(ImportError) as e:
            logger.error('Trainig error : %s', str(e)) 
    
    def load(self, path):
        try:
            model = keras.models.load_model(path + '\\NNModel')
        except (OSError) as e:
            logger.error('Fatal forecast error: %s', str(e))
            sys.exit(1)  # abnormal termination
        return model   
    
    def getInputVector(self, data, lookBack, lookAhead, fMin, training=False):
        """ Description: input data will be normalized and shaped into specified form 
        :param data: 
            data which is loaded from the database
        :param lookBack:
            number of timesteps (in minutes) the NN looks back for prediction
        :param lookAhead:
            number of timesteps (in minutes) the NN predicts
        :param fMin:
            smallest interval in the created input vector. All other intervals are fixed 
        """
        
        dataBiNorm = (data[0] + 1) / 2
        b, a = signal.butter(8, 0.01)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
        dataBiNorm = signal.filtfilt(b, a, dataBiNorm, padlen=150)
        
        data[1] = data[1].reset_index()['unixtimestamp']
        dataDT = processing.getDaytime(data[1])
        dataDTNorm = dataDT / (24 * 60 * 60)
        
        hourOfYear = np.zeros([len(dataDT)])
        for i in range (len(data[1])): 
            hourOfYear[i] = data[1][i].timetuple().tm_yday * 24 + int(data[1][i].minute / 60)
                
        dataSeasonNorm = -0.5 * np.cos((hourOfYear - 360) / 365 / 24 * 2 * np.pi) + 0.5
        
        data_dBI = processing.getdBI(dataBiNorm, self.fMin) 
        data_dBI = np.append(data_dBI[0], data_dBI)
        # dataSeasonNorm = data[2]
        
        dataBiNorm = dataBiNorm.reshape(dataBiNorm.shape[0], 1)
        data_dBI = data_dBI.reshape(data_dBI.shape[0], 1)
        dataDTNorm = dataDTNorm.reshape(dataDTNorm.shape[0], 1)
        dataSeasonNorm = dataSeasonNorm.reshape(dataSeasonNorm.shape[0], 1)
        
        # reshape into X=t and Y=t+1 ( data needs to be normalized
        X_bi, Y_bi = processing.create_dataset(dataBiNorm, lookBack, lookAhead, fMin, training)
        # X_dbi, Y_dbi = processing.create_dataset(data_dBI, lookBack, lookAhead, fMin, training)
        X_dt, Y_dt = processing.create_dataset(dataDTNorm, lookBack, lookAhead, fMin, training)
        # X_season, Y_season = processing.create_dataset(dataSeasonNorm, lookBack, lookAhead, fMin, training)
        
        # reshape input to be [samples, time steps, features]
        X_bi = np.reshape(X_bi, (X_bi.shape[0], 1, X_bi.shape[1]))
        # X_dbi = np.reshape(X_dbi, (X_dbi.shape[0], 1, X_dbi.shape[1]))
        X_dt = np.reshape(X_dt, (X_dt.shape[0], 1, X_dt.shape[1]))
        # X_season = np.reshape(X_season, (X_season.shape[0], 1, X_season.shape[1]))
        # Xconcat = np.concatenate((X_bi , X_dbi, X_dt, X_season), axis=1)   
        Xconcat = np.concatenate((X_bi, X_dt), axis=1)    
        
        # create input vector for LSTM Model
        if training == True:
            Y_bi = np.reshape(Y_bi, (Y_bi.shape[0], 1, Y_bi.shape[1]))
            # Y_dbi = np.reshape(Y_dbi, (Y_dbi.shape[0], 1, Y_dbi.shape[1]))
            Y_dt = np.reshape(Y_dt, (Y_dt.shape[0], 1, Y_dt.shape[1]))
            # Y_season = np.reshape(Y_season, (Y_season.shape[0], 1, Y_season.shape[1]))
            # Yconcat = np.concatenate((Y_bi, Y_dbi, Y_dt, Y_season), axis=1)
            Yconcat = np.concatenate((Y_bi, Y_dt), axis=1)      
            return Xconcat, Yconcat
        elif training == False:     
            return Xconcat
        
    def predict_recursive(self, data):
        n_predictions = int(1440 / self.lookAhead)
        _predStack = np.zeros([self.dimension, 1440])
        v1 = np.zeros([self.dimension, self.lookAhead])

        plt.plot(data[0][-2880:], linewidth=0.5)
        
        for z in range(n_predictions):
            inputVectorTemp = self.getInputVector(data, self.lookBack, self.lookAhead, self.fMin, training=False)
            _predRecursive = self.model.predict(inputVectorTemp)
            for k in range(0, self.lookAhead, self.fMin):
                v1[0, k:k + self.fMin] = _predRecursive[0, 0, int(k / self.fMin)]
                v1[1, k:k + self.fMin] = _predRecursive[0, 1, int(k / self.fMin)]
            
            I = data[1].shape[0]
            ts = data[1][I - 1] 
            data = [np.roll(data[0], -self.lookAhead, axis=0),
                    data[1].shift(periods=-self.lookAhead)]
            data[0][-self.lookAhead:] = v1[0, :] * 2 - 1
            for k in range(self.lookAhead):
                data[1][I - self.lookAhead + k] = ts + np.timedelta64(k + 1, 'm')
                
            _predStack[:, z * _predRecursive.shape[2] : (z + 1) * _predRecursive.shape[2]] = _predRecursive[0, :, :]
            plt.plot(np.linspace(z * self.lookAhead, z * self.lookAhead + 2880, 2880), data[0][-2880:], linewidth=0.5)
            
        return _predStack
        
