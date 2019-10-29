'''
Created on 12.07.2019

@author: sf
'''
import os
from configparser import ConfigParser
import keras
import logging
import numpy as np
import theforecast.processing as processing
from datetime import timedelta 
from scipy import signal
import pandas as pd

logger = logging.getLogger(__name__)


class NeuralNetwork:
    
    def __init__(self, configs):
        ''' get configuration from configs-path. '''
        
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
        self.n_samples_retrain = settings.getint('Prediction', 'training samples')
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
    
    def train(self, data):
        '''Description: trains the neural network
        
        :param data: takes all the data from the database. The function automatically selects the last 
        values (number of values specified in the config-file).
        :dtype DataFrame:
        '''
        data = [data.loc[:]['bi'].get_values()[-self.n_samples_retrain:],
                data.index[-self.n_samples_retrain:]]
        X, Y = self.get_data_vector(data, training=True)
        try: 
            # self.model.fit(X, Y[:,0,:], epochs=self.epochs_retrain, batch_size=64, verbose=2)
            self.model.fit(X, Y[:, 0, :], epochs=1, batch_size=64, verbose=2)
        except(ImportError) as e:
            logger.error('Trainig error : %s', str(e)) 
            
    def train_initial(self, data):
        '''Description: initially trains the neural network with all data available
        :param data: 
        input data to train the neural network with
        :dtype DataFrame:
        '''
        data = [data.loc[:]['bi'].get_values(),
                data.index]
        X, Y = self.get_data_vector(data, training=True)
        try: 
            # self.model.fit(X, Y[:, 0, :], epochs=self.epochs_init, batch_size=64, verbose=2)
            self.model.fit(X, Y[:, 0, :], epochs=1, batch_size=64, verbose=2)
        except(ImportError) as e:
            logger.error('Initial trainig error : %s', str(e)) 
    
    def load(self, path, name):
        modelfile = os.path.join(path, name)
        if os.path.isfile(modelfile):
            self.model = keras.models.load_model(modelfile)
        else: 
            # TODO: logger info: no modelfile found in directory
            print('logger info: no model-file found in directory.')
    
    def get_data_vector(self, data, training=False):
        """ Description: input data will be normalized and shaped into specified form 
        :param data: 
            data which is loaded from the database
        :param training:
            defines if the function additionally returns an output vector or just an input vector
        """
        data_bi_norm = (data[0] + 1) / 2
        b, a = signal.butter(8, 0.022)  # lowpass filter of order = 8 and critical frequency = 0.01 (-3dB)
        data_bi_norm = signal.filtfilt(b, a, data_bi_norm, method='pad', padtype='even', padlen=150)
        data_daytime_norm = processing.get_daytime(data[1]) 

        hour_of_year = np.zeros([len(data_daytime_norm)])
        for i in range (len(data[1])): 
            hour_of_year[i] = data[1][i].timetuple().tm_yday * 24 + int(data[1][i].minute / 60)
        data_season_norm = -0.5 * np.cos((hour_of_year - 360) / 365 / 24 * 2 * np.pi) + 0.5
        
        data_array = np.zeros([self.dimension, data_bi_norm.__len__()])
        data_array[0, :] = data_bi_norm
        data_array[1, :] = data_daytime_norm
        data_array[2, :] = data_season_norm

        if training == True:
            length = int(len(data_array[0]) - 4 * 24 * 60 - self.look_ahead)
            Y = np.zeros([length, 1, self.look_ahead])
            Y[:, 0, :] = processing.create_output_vector(data_array[0, :], self, length)
        else:
            length = 1
            
        X = np.zeros([length, self.dimension, self.look_back])
        for i in range(self.dimension):
            X[:, i, :] = processing.create_input_vector(data_array[i, :], self, length)
        
        if training == True:
            return X, Y
        else:
            return X
        
    def predict_recursive(self, data):
        '''Description: recursively predicts the BI over next 24 hours.
        :param data: raw Data
        :dtype list with dimension = conf - file'''
        data = [data.loc[:]['bi'].get_values()[-4 * 1440:],
                pd.Series.tolist(data.index[-4 * 1440:])]
        
        n_predictions = int(1440 / self.look_ahead)
        predStack = np.zeros(1440)  # predStack = np.zeros([self.dimension, 1440])
        
        for z in range(n_predictions):
            inputVectorTemp = self.get_data_vector(data, training=False)
            pred = self.model.predict(inputVectorTemp)
            
            data[0] = np.roll(data[0], -self.look_ahead, axis=0)
            data[1][0] = data[1][self.look_ahead]
            for i in range(1440 - 1):
                data[1][i + 1] = data[1][i] + timedelta(minutes=1)
                
            predStack[z * self.look_ahead : (z + 1) * self.look_ahead] = pred
            
        return predStack
        
