# -*- coding: utf-8 -*-
"""
Execute optimization calculation
"""
##############################################################################
#%% Python Imports
##############################################################################
import sys
import os
import pygmo as pg   # pygmo optimization
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from src.mathematical_problem.optimization_calculation import  optimizeGMO
from src.mathematical_problem.helper_functions import func_plot

sys.path.append(os.path.join(sys.path[0], '..'))

##############################################################################
#%% start optimization

print('Starting optimization')
startTime = timer()

solution = optimizeGMO(verbosity=1)
print('solution = ', solution)

##############################################################################
#%% Finished
endTime = timer()
print(endTime - startTime, 's')

##########################################################################
#%% Plots
func_plot(solution)
##########################################################################

##############################################################################
sys.exit('end of program')
