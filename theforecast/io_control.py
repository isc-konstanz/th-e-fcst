'''
Created on 02.09.2019

@author: sf
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp


class IO_control:

    def __init__(self):
        self.pred_horizon = 0
        self.prediction = []
        self.u_init = [0, 0]
        self.charge_energy_amount = 0
        self.IO_control = []

    def get_IO(self):
        lb = 0
        ub = 10  # number of controlling steps; depends on device

        solver = pywraplp.Solver('simple_lp_program',
                                      pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        
        # Create the variables
        x = {}
        for i in range(self.pred_horizon):
            x[i] = solver.IntVar(lb, ub, 'x_%i' % i)
        # add constraints    
        solver.Add(solver.Sum([x[i] for i in range(self.pred_horizon)]) == self.charge_energy_amount * ub)
        solver.Add(x[self.pred_horizon - 1] == 0)
        solver.Add(-3 <= x[0] - self.u_init[-1] * ub <= 3)
        for i in range(self.pred_horizon - 1):
            solver.Add(-3 <= x[i + 1] - x[i] <= 3)
        # define cost function    
        solver.Maximize(solver.Sum([self.prediction[i] * x[i] for i in range(self.pred_horizon)]))
        
        solver.Solve()
        solver.Objective().Value()
    
        self.IO_control = np.zeros(self.pred_horizon)
        for i in range(self.pred_horizon):
            self.IO_control[i] = x[i].solution_value() / ub

