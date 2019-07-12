'''
Created on 12.07.2019

@author: sf
'''
import os
from configparser import ConfigParser
import keras


class NeuralNetwork:
    
    def __init__(self, configs):
        # get CNN configurations:
        # TODO: define CNN config-file
        neuralnetworkfile = os.path.join(configs, 'neuralnetwork.cfg')
        settings = ConfigParser()
        settings.read(neuralnetworkfile)
    
    def get(self, configs):
        # TODO: load the trained neural network
        return
    
    def set(self, model, configs):
        # TODO: safe the neural network
        # model.save()
        return 
    
    def train(self, model, inputData):
        return
    
    def predict(self, model, inputData):
        return 
    
    def predictRecursive(self):
        return
    
