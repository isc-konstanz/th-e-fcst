'''
Created on 12.07.2019

@author: sf
'''
import os
from configparser import ConfigParser
import keras
import logging
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
        self.look_back = int(settings.getint('Input vector', 'interval 1') / 60) + \
                        int(settings.getint('Input vector', 'interval 2') / 15) + \
                        int(settings.getint('Input vector', 'interval 3') / self.fMin)
        self.dropout = settings.getfloat('General', 'dropout')
        self.layers = settings.getint('General', 'layers')
        self.neurons = settings.getint('General', 'neurons')
        self.look_ahead = int(settings.getint('General', 'lookAhead'))
        self.dimension = settings.getint('General', 'dimension')
        self.epochs_retrain = settings.getint('General', 'epochs_retrain')
        self.epochs_init = settings.getint('General', 'epochs_init')
        self.model = self.create_model()
    
    def create_model(self):
        mode_training = False
        inputs = keras.layers.Input(shape=(self.dimension, self.look_back))
        x = keras.layers.LSTM(self.neurons, recurrent_dropout=self.dropout, return_sequences=True)(inputs, training=mode_training)
        x = keras.layers.Dropout(self.dropout)(x, training=mode_training)
        x = keras.layers.LSTM(self.neurons)(x, training=mode_training)
        x = keras.layers.Dropout(self.dropout)(x, training=mode_training)
        outputs = keras.layers.Dense(int(self.look_ahead))(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse']) 
        return model
    
    def train(self, X, Y, epochs=1):
        try: 
            self.model.fit(X, Y, epochs=epochs, batch_size=64, verbose=2)
        except(ImportError) as e:
            logger.error('Trainig error : %s', str(e)) 
    
    def load(self, path, name):
        modelfile = os.path.join(path, name)
        if os.path.isfile(modelfile):
            self.model = keras.models.load_model(modelfile)
        else: 
            # TODO: logger info: no modelfile found in directory
            print('logger info: no model-file found in directory.')
    
    def getInputVector(self, data, training=False):
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
        look_back = self.look_back
        look_ahead = self.look_ahead
        fMin = self.fMin
        
        dataBiNorm = (data[0] + 1) / 2
        b, a = signal.butter(8, 0.022)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
         
        dataBiNorm = signal.filtfilt(b, a, dataBiNorm, method='pad', padtype='even', padlen=150)
        dataDTNorm = processing.getDaytime(data[1]) 
#         c='b'
#         plt.clf()
#         plt.plot(dataBiNorm)
#         plt.plot(signal.filtfilt(b, a, dataBiNorm, method='gust'), label='gust')
#         plt.plot(signal.filtfilt(b, a, dataBiNorm, method='pad', padtype='odd', padlen=150),c, label='odd')
#         plt.plot(signal.filtfilt(b, a, dataBiNorm, method='pad', padtype='even', padlen=150),c, label='even')
#         plt.plot(signal.filtfilt(b, a, dataBiNorm, method='pad', padtype='constant', padlen=150),c, label='constant')
#         plt.legend()
        # even, constant
        hourOfYear = np.zeros([len(dataDTNorm)])
        for i in range (len(data[1])): 
            hourOfYear[i] = data[1][i].timetuple().tm_yday * 24 + int(data[1][i].minute / 60)
        dataSeasonNorm = -0.5 * np.cos((hourOfYear - 360) / 365 / 24 * 2 * np.pi) + 0.5
        
        dataBiNorm = dataBiNorm.reshape(dataBiNorm.shape[0], 1)
        dataDTNorm = dataDTNorm.reshape(dataDTNorm.shape[0], 1)
        dataSeasonNorm = dataSeasonNorm.reshape(dataSeasonNorm.shape[0], 1)
        
        # reshape into X=t and Y=t+1 ( data needs to be normalized
        X_bi = processing.create_input_vector(dataBiNorm, look_back, look_ahead, fMin, training)
        X_dt = processing.create_input_vector(dataDTNorm, look_back, look_ahead, fMin, training)
        if training == True:
            Y_bi = processing.create_output_vector(dataBiNorm, look_ahead, fMin, training)
            Y_dt = processing.create_output_vector(dataDTNorm, look_ahead, fMin, training)
#             Y_dt = processing.create_output_vector(dataDTNorm, lookAhead, fMin, training)

        # reshape input to be [samples, time steps, features]
        X_bi = np.reshape(X_bi, (X_bi.shape[0], 1, X_bi.shape[1]))
        X_dt = np.reshape(X_dt, (X_dt.shape[0], 1, X_dt.shape[1]))
#         X_season = np.reshape(X_season, (X_season.shape[0], 1, X_season.shape[1]))        
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
        '''Description: recursiveley predicts the BI over 1 Day.
        :param data: raw Data '''
        n_predictions = int(1440 / self.look_ahead)
        predStack = np.zeros(1440)  # predStack = np.zeros([self.dimension, 1440])
        
        for z in range(n_predictions):
            inputVectorTemp = self.getInputVector(data, training=False)
            pred = self.model.predict(inputVectorTemp)
            
#             plt.figure(2)
#             plt.clf()
#             plt.plot((data[0] + 1) / 2)
#             a1 = np.linspace(0, 2 * 1440, 48)
#             a2 = np.linspace(2 * 1440, 2 * 1440 + 46 * 60, 46 * 4)
#             a3 = np.linspace(2 * 1440 + 46 * 60, 4 * 1440, 24)
#             plt.plot(np.concatenate((a1, a2, a3)), inputVectorTemp[0, 0, :])
            
            data = [np.roll(data[0], -self.look_ahead, axis=0),
                    np.roll(data[1], -self.look_ahead, axis=0)]
            
            data[0][-self.look_ahead:] = pred * 2 - 1
            
            I = data[0].shape[0]
            ts = data[1][I - 1] 
            for k in range(self.look_ahead):
                data[1][I - self.look_ahead + k] = ts + np.timedelta64(k + 1, 'm')
                
            predStack[z * self.look_ahead : (z + 1) * self.look_ahead] = pred
            
        return predStack
        
