'''
Created on 02.09.2019

@author: sf
'''
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt


class IO_control:

    def __init__(self, fMin, f_prediction):
#         self.pred_horizon_max = int(1440 / fMin)
#         self.f_prediction = f_prediction
        self.pred_horizon = 0
        self.prediction = []
        self.u_init = []
        self.charge_energy_amount = []
        self.IO_control = np.zeros(176)
        self.fMin = fMin
        self.IO_stack = []
        
        self.const = ({'type': 'eq', 'fun': self.fConstPower},
                      {'type': 'eq', 'fun': self.fConstEnd})

    def get_IO(self):
        self.IO_stack = np.zeros([10, self.pred_horizon])
        function_values = np.zeros(10)
        initial_guess = self.IO_control[:self.pred_horizon]
        if self.pred_horizon > len(self.IO_control):
            initial_guess = np.append(initial_guess, np.zeros(self.pred_horizon - len(self.IO_control)))
        bnds = [(0, 1)] * self.pred_horizon
   
        for i in range(10):
            res = optimize.minimize(self.function,
                                    initial_guess,
                                    bounds=bnds,
                                    constraints=self.const)
            function_values[i] = res.fun
            # initial_guess = res.x + np.random.normal(0, .1, self.pred_horizon)
            initial_guess[:10] = res.x[:10] + np.random.normal(0, .1, 10)
            self.IO_stack[i, :] = res.x
            
        self.IO_control = self.IO_stack[function_values.argmin(), :]

    def function(self, u):
        sumDiff = sum(-(self.prediction[:self.pred_horizon] * u)) 
        u = np.concatenate((self.u_init, np.append(u, 0)))
        sumDeltaDerivation = sum(((2 * u[1:-1] - u[:-2] - u[2:])) ** 2)
        return sumDiff + .4 * sumDeltaDerivation
    
    # sum of all inputs must equal the desired power
    def fConstPower(self, u):
        return sum(u) - self.charge_energy_amount
    
    # last value/input must be 0
    def fConstEnd(self, u):
        return u[-1]
    
#     # initial conditions
#     def fConstInit(self, u):
#         return u[0] - self.IO_control[self.fMin]
#     
#     # constraining the absolute differntiation 
#     def fConst2(self, u): 
#         return np.absolute((u[1:] - u[:-1])) - 0.002
#     
#     def getData(self, nSamples):
#         data = np.zeros(nSamples)
#         x = np.linspace(-5, 20, nSamples)
#         for i in range(nSamples):
#             data[i] = np.sin(x[i]) / x[i] - 0.1
#         return data
