'''
Created on 02.09.2019

@author: sf
'''

from __future__ import print_function
import numpy as np
from ortools.linear_solver import pywraplp


class Control:

    def __init__(self):
        self.horizon = 0
        self.charge_energy_amount = 0
        self.control = []
        self.history = np.zeros(1440)

    def execute(self, prediction):
        '''Description: executes the MPC optimization
        :input prediction:
        BI prediction of the next 1440 minutes 
        :dtype ndarray:
        '''
        lb = 0
        ub = 1  # number of controlling steps; depends on device
        self.control = np.zeros(self.horizon)
        solver = pywraplp.Solver('simple_lp_program',
                                      pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        delta = 0 
        for i in range(120):
            if self.history[-120 + i] - self.history[-121 + i] != 0:
                delta += 1
        weight_penalty_function = 0.5 + delta * 2
        # Create the variables
        x = {}
        for i in range(self.horizon):
            x[i] = solver.IntVar(lb, ub, 'x_%i' % i)
        for j in range(1, self.horizon + 1):    
            x[i + j] = solver.IntVar(lb, ub, 'T_%i' % j)
            
        # add constraints    
        solver.Add(solver.Sum([x[i] for i in range(self.horizon)]) == self.charge_energy_amount * ub)
        solver.Add(x[self.horizon - 1] == 0)
        
        solver.Add(-self.history[-1] - x[0] + x[self.horizon] <= 0)
        solver.Add(self.history[-1] - x[0] - x[self.horizon] <= 0)
        solver.Add(-self.history[-1] + x[0] - x[self.horizon] <= 0)
        solver.Add(self.history[-1] + x[0] + x[self.horizon] <= 2 * ub)
        for i in range(self.horizon - 1):
            solver.Add(-x[i] - x[i + 1] + x[i + 1 + self.horizon] <= 0)
            solver.Add(x[i] - x[i + 1] - x[i + 1 + self.horizon] <= 0)
            solver.Add(-x[i] + x[i + 1] - x[i + 1 + self.horizon] <= 0)
            solver.Add(x[i] + x[i + 1] + x[i + 1 + self.horizon] <= 2 * ub)
            
        # define cost function    
        solver.Maximize(solver.Sum([prediction[i] * x[i] for i in range(self.horizon)]) - 
                        weight_penalty_function * solver.Sum([x[i + self.horizon] for i in range(self.horizon - 1)]))
        
        # execute optimization
        if solver.Solve() == 0:
            for i in range(self.horizon):
                self.control[i] = x[i].solution_value() / ub
        else: 
            # TODO: logger info: optimization error
            print('Optimization error. Solution not found.')
