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
        # get CNN configurations:
        # TODO: define CNN config-file
        neuralnetworkfile = os.path.join(configs, 'neuralnetwork.cfg')
        settings = ConfigParser()
        settings.read(neuralnetworkfile)
        self.dropout = settings.getint('General', 'dropout')
        self.nLayers = settings.getint('General', 'layers')
        self.nNeurons = settings.getint('General', 'neurons')
        self.lookBack = settings.getint('General', 'lookBack')
        self.dimension = settings.getint('General', 'dimension')
        
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
    
    def train(self, model, inputData):
        
        return model
    
    def predict(self, model, inputData):
        return 
    
    def predictRecursive(self):
        return
    
