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
import theforecast.processing as processing
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
        self.lookAhead = int(settings.getint('General', 'lookAhead') / self.fMin)
        self.dimension = settings.getint('General', 'dimension')
        
        self.model = self.create()
    
    def create(self):    
        model = keras.Sequential()
        model.add(keras.layers.LSTM(self.nNeurons, input_shape=(3, self.lookBack), return_sequences=True))
        model.add(keras.layers.Dropout(self.dropout))
        for z in range(self.nLayers - 1):
            model.add(keras.layers.Dense(self.nNeurons, activation='sigmoid')) 
            model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(self.lookAhead, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
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
    
    def save(self, model, path):
        # safe the neural network
        model.save(path + '\\NNModel')
    
    def predict(self, model, inputData):
        return 
    
    def predictRecursive(self):
        return
    
    def getInputVector(self, data, lookBack, lookAhead, fMin, training=False):
        """ Description: input data will be normalized and shaped into specified form 
        :param data: 
            data which is loaded from the database
        :param lookBack:
            number of timesteps (in minutes) the NN looks back for prediction
        :param lookAhead:
            number of timesteps (in fMin) the NN predicts
        :param fMin:
            smallest interval in the created input vector. All other intervals are fixed 
        """
        
        dataBiNorm = (data[0] + 1) / 2
        # TODO: data[1] is a series of datetimes and hast index starting at 33121. need to reset this index
        # how can i handle this???
        dataDT = processing.getDaytime(data[1])
        dataDTNorm = dataDT / (24 * 60 * 60)
        dataSeasonNorm = data[2]
        
        dataBiNorm = dataBiNorm.reshape(dataBiNorm.shape[0], 1)
        dataDTNorm = dataDTNorm.reshape(dataDTNorm.shape[0], 1)
        dataSeasonNorm = dataSeasonNorm.reshape(dataSeasonNorm.shape[0], 1)
        
        # reshape into X=t and Y=t+1 ( data needs to be normalized
        X_bi, Y_bi = processing.create_datasetBI(dataBiNorm, lookBack, lookAhead, fMin, training)
        X_dt, Y_dt = processing.create_datasetDT(dataDTNorm, lookBack, lookAhead, fMin, training)
        X_season, Y_season = processing.create_datasetDT(dataSeasonNorm, lookBack, lookAhead, fMin, training)
        
        # reshape input to be [samples, time steps, features]
        X_bi = np.reshape(X_bi, (X_bi.shape[0], 1, X_bi.shape[1]))
        X_dt = np.reshape(X_dt, (X_dt.shape[0], 1, X_dt.shape[1]))
        X_season = np.reshape(X_season, (X_season.shape[0], 1, X_season.shape[1]))
        Xconcat = np.concatenate((X_bi , X_dt, X_season), axis=1)    
        
        # create input vector for LSTM Model
        if training == True:
            Y_bi = np.reshape(Y_bi, (Y_bi.shape[0], 1, Y_bi.shape[1]))
            Y_dt = np.reshape(Y_dt, (Y_dt.shape[0], 1, Y_dt.shape[1]))
            Y_season = np.reshape(Y_season, (Y_season.shape[0], 1, Y_season.shape[1]))
            Yconcat = np.concatenate((Y_bi , Y_dt, Y_season), axis=1)      
            return Xconcat, Yconcat
        elif training == False:     
            return Xconcat
