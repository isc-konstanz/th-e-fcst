'''
Created on 02.09.2019

@author: sf
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import warnings


class IO_control:

    def __init__(self):
        self.pred_horizon = 0
        self.charge_energy_amount = 0
        self.IO_control = []
        self.IO_history = np.zeros(1440)

    def execute(self, prediction):
        lb = 0
        ub = 2  # number of controlling steps; depends on device
        self.IO_control = np.zeros(self.pred_horizon)
        solver = pywraplp.Solver('simple_lp_program',
                                      pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        x = {}
        
        # Create the variables
        for i in range(self.pred_horizon):
            x[i] = solver.IntVar(lb, ub, 'x_%i' % i)
        for j in range(1, self.pred_horizon + 1):    
            x[i + j] = solver.IntVar(lb, ub, 'T_%i' % j)
        # add constraints    
        solver.Add(solver.Sum([x[i] for i in range(self.pred_horizon)]) == self.charge_energy_amount * ub)
        solver.Add(x[self.pred_horizon - 1] == 0)
        
        solver.Add(-self.IO_history[-1] - x[0] + x[self.pred_horizon] <= 0)
        solver.Add(self.IO_history[-1] - x[0] - x[self.pred_horizon] <= 0)
        solver.Add(-self.IO_history[-1] + x[0] - x[self.pred_horizon] <= 0)
        solver.Add(self.IO_history[-1] + x[0] + x[self.pred_horizon] <= 2 * ub)
        for i in range(self.pred_horizon - 1):
            solver.Add(-x[i] - x[i + 1] + x[i + 1 + self.pred_horizon] <= 0)
            solver.Add(x[i] - x[i + 1] - x[i + 1 + self.pred_horizon] <= 0)
            solver.Add(-x[i] + x[i + 1] - x[i + 1 + self.pred_horizon] <= 0)
            solver.Add(x[i] + x[i + 1] + x[i + 1 + self.pred_horizon] <= 2 * ub)
            
        # define cost function    
        solver.Maximize(solver.Sum([prediction[i] * x[i] for i in range(self.pred_horizon)]) - 
                        0.2 * solver.Sum([x[i + self.pred_horizon] for i in range(self.pred_horizon - 1)]))
        
        solver.Solve()
        for i in range(self.pred_horizon):
            self.IO_control[i] = x[i].solution_value() / ub

