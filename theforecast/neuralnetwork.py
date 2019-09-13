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
        model.add(keras.layers.Dense(int(self.lookAhead), activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def create_model_dropout(self):
        mode_training = False
        inputs = keras.layers.Input(shape=(self.dimension, self.lookBack))
        x = keras.layers.LSTM(self.nNeurons, recurrent_dropout=self.dropout, return_sequences=True)(inputs, training=mode_training)
        x = keras.layers.Dropout(self.dropout)(x, training=mode_training)
        x = keras.layers.LSTM(self.nNeurons)(x, training=mode_training)
        x = keras.layers.Dropout(self.dropout)(x, training=mode_training)
        outputs = keras.layers.Dense(int(self.lookAhead))(x)
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
        b, a = signal.butter(8, 0.02)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
         
        dataBiNorm = signal.filtfilt(b, a, dataBiNorm, padlen=150)
        dataDTNorm = processing.getDaytime(data[1]) / (24 * 60 * 60)
        
        hourOfYear = np.zeros([len(dataDTNorm)])
        for i in range (len(data[1])): 
            hourOfYear[i] = data[1][i].timetuple().tm_yday * 24 + int(data[1][i].minute / 60)
        dataSeasonNorm = -0.5 * np.cos((hourOfYear - 360) / 365 / 24 * 2 * np.pi) + 0.5
        
        dataBiNorm = dataBiNorm.reshape(dataBiNorm.shape[0], 1)
        dataDTNorm = dataDTNorm.reshape(dataDTNorm.shape[0], 1)
        dataSeasonNorm = dataSeasonNorm.reshape(dataSeasonNorm.shape[0], 1)
        
        # reshape into X=t and Y=t+1 ( data needs to be normalized
        X_bi = processing.create_input_vector(dataBiNorm, lookBack, lookAhead, fMin, training)
        X_dt = processing.create_input_vector(dataDTNorm, lookBack, lookAhead, fMin, training)
        if training == True:
            Y_bi = processing.create_output_vector(dataBiNorm, lookAhead, fMin, training)
            Y_dt = processing.create_output_vector(dataDTNorm, lookAhead, fMin, training)
#             Y_bi = processing.create_output_vector(dataBiNorm, lookAhead, fMin, training)
#             Y_dt = processing.create_output_vector(dataDTNorm, lookAhead, fMin, training)

        # reshape input to be [samples, time steps, features]
        X_bi = np.reshape(X_bi, (X_bi.shape[0], 1, X_bi.shape[1]))
        X_dt = np.reshape(X_dt, (X_dt.shape[0], 1, X_dt.shape[1]))
        # X_season = np.reshape(X_season, (X_season.shape[0], 1, X_season.shape[1]))        
        # X_dbi = np.reshape(X_dbi, (X_dbi.shape[0], 1, X_dbi.shape[1]))
        Xconcat = np.concatenate((X_bi, X_dt), axis=1)    
        
        # create input vector for model
        if training == True:
            Y_bi = np.reshape(Y_bi, (Y_bi.shape[0], 1, Y_bi.shape[1]))
            # Y_dbi = np.reshape(Y_dbi, (Y_dbi.shape[0], 1, Y_dbi.shape[1]))
            Y_dt = np.reshape(Y_dt, (Y_dt.shape[0], 1, Y_dt.shape[1]))
            # Y_season = np.reshape(Y_season, (Y_season.shape[0], 1, Y_season.shape[1]))

            Yconcat = np.concatenate((Y_bi, Y_dt), axis=1)      
            return Xconcat, Yconcat
        elif training == False:     
            return Xconcat
        
    def predict_recursive(self, data):
        n_predictions = int(1440 / self.lookAhead)
        predStack = np.zeros(1440)  # predStack = np.zeros([self.dimension, 1440])
        
        for z in range(n_predictions):
            inputVectorTemp = self.getInputVector(data, self.lookBack, self.lookAhead, self.fMin, training=False)
            pred = self.model.predict(inputVectorTemp)
            
            data = [np.roll(data[0], -self.lookAhead, axis=0),
                    np.roll(data[1], -self.lookAhead, axis=0)]
            
            data[0][-self.lookAhead:] = pred * 2 - 1
            
            I = data[0].shape[0]
            ts = data[1][I - 1] 
            for k in range(self.lookAhead):
                data[1][I - self.lookAhead + k] = ts + np.timedelta64(k + 1, 'm')
                
            predStack[z * self.lookAhead : (z + 1) * self.lookAhead] = pred
            
        return predStack
        
