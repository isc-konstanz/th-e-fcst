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

logger = logging.getLogger(__name__)


class NeuralNetwork:
    
    def __init__(self, configs):
        # get NN configurations:
        neuralnetworkfile = os.path.join(configs, 'neuralnetwork.cfg')
        settings = ConfigParser()
        settings.read(neuralnetworkfile)
        self.dropout = settings.getfloat('General', 'dropout')
        self.nLayers = settings.getint('General', 'layers')
        self.nNeurons = settings.getint('General', 'neurons')
        self.lookBack = settings.getint('General', 'lookBack')
        self.lookAhead = settings.getint('General', 'lookAhead')
        self.dimension = settings.getint('General', 'dimension')
    
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
    
    def train(self, model, trainingData):
        try: 
            model.fit(trainingData)
        except(ImportError) as e:
            logger.error('Trainig error : %s', str(e))
        return model  
    
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
    
