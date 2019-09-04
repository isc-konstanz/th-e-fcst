'''
Created on 02.09.2019

@author: sf
'''
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class IO_control:

    def __init__(self, fMin, f_prediction):
        self.pred_horizon_max = int(1440 / fMin)
        self.pred_horizon = 0
        self.prediction = []
        self.u_init = []
        self.charge_energy_amount = []
        self.IO_control = np.zeros(10)
        self.f_prediction = f_prediction
        self.fMin = fMin

        self.const = ({'type': 'eq', 'fun': self.fConst0},
                      {'type': 'eq', 'fun': self.fConstEnd})

#                     {'type': 'eq', 'fun': self.fConstInit})
    def get_IO(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        initial_guess = scaler.fit_transform(self.prediction.reshape(-1, 1))
        n_iter = 0
        
        bnds = ((0, 1),)
        for i in range(self.pred_horizon - 1):
            bnds = bnds + ((0, 1),)
            
        while n_iter <= 50:
            res = optimize.minimize(self.function,
                                    initial_guess[:self.pred_horizon],
                                    bounds=bnds,
                                    constraints=self.const,
                                    options={'disp': True})
            initial_guess = res.x
            n_iter = n_iter + res.nit
            
        self.IO_control = res.x
        return res.x
    
    def function(self, u):
        sumDiff = sum(-(self.prediction[:self.pred_horizon] * u)) 
        u = np.append(self.u_init, u)
        sumDeltaDerivation = sum(((2 * u[1:-1] - u[:-2] - u[2:])) ** 2)
        return sumDiff + 1 * sumDeltaDerivation
    
    # sum of all inputs must equal the desired power
    def fConst0(self, u):
        return sum(u) - self.charge_energy_amount
    
    # last value/input must be 0
    def fConstEnd(self, u):
        return u[-1]
    
    # initial conditions
    def fConstInit(self, u):
        return u[0] - self.IO_control[self.fMin]
    
    # constraining the absolute differntiation 
    def fConst2(self, u): 
        return np.absolute((u[1:] - u[:-1])) - 0.002

