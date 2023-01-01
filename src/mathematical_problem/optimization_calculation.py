# A module to define the optimization algorithm using pygmo liberary

import math
import pygmo as pg
import pandas as pd
import numpy as np

# from numba import jit
from typing import List, int
from src.mathematical_problem.model import model

# extract setpoints
df = pd.read_csv('src/mathematical_problem/setpoints.csv')
setpoint1 = df['setpoint1']
setpoint2 = df['setpoint2']
setpoint3 = df['setpoint3']
setpoint4 = df['setpoint4']

# class used to create pygmo problem class
class GMO:
    """
    A custom class to define pygmo optimization problem
    """

    def __init__(self, step: int):
        self.dim = 6
        self.c_tol = 1*E-5
        self.step = step

    ###########################################################################
    #%% pygmo related methods
    ###########################################################################

    # @jit(nopython=True)
    def fitness(self, x: List):
        """
        this is the objective function for the pygmo liberary based optimization
        """
        # x is the decision vector
        # 6 inputs or  6 optimization variables in the objective func, so we got 6 (min,max) boundaries for these 6

        model_state0, model_state1  = model(x)
        # objective function 1
        obj1 = ((setpoint1[self.step] - (model_state0 - model_state1) + (model_state0 * 1.20))**2) * 0.40

        # objective function 2
        obj2 = ((setpoint2[self.step] - (model_state0 + model_state1 / model_state0) + (model_state1 * 1.50))**2) * 0.20

        # objective function 3
        obj3 = ((setpoint3[self.step] - (x[3] * model_state1 /2.5) + (model_state0 * 2.50))**2) * 0.10

        # objective function 4
        obj4 = ((setpoint4[self.step] - (model_state0 /1.5) - (math.log(model_state1*x[0]*x[5])))**2) * 0.30

        # weighted decomposition for decomposing multi objectives into a single objective function
        obj =  0.2 * obj1 + 0.2 * obj2 + 0.2 * obj3 + 0.2 * obj4

        # constraints
        c1, c2, c3, c4 = self.const_eq(x)
        c5, c6 = self.const_ieq(x)

        obj_plus_all_constrains_list = [obj, c1, c2, c3, c4, c5, c6]

        return obj_plus_all_constrains_list


    def get_bounds(self):
        "put boundaries  here..........."
        # 6 inputs or  6 optimization variables in the objective func, so we got 6 (min,max) boundaries for these 6


        min_boundaries = [-100]*6
        max_boundaries = [100]*6

        return (min_boundaries, max_boundaries)

    # Return number of objectives
    def get_nobj(self):
        return 1

    def get_name(self):
        return "pygmo Main optimization func"

    # number of Inequality Constraints
    def get_nic(self):
        return 2

    # number of Equality Constraints
    def get_nec(self):
        return 4

    def gradient(self, x: List):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def has_gradient(self):
        return True

    def hessians(self, x: List):
       return pg.estimate_gradient_h(lambda x: self.gradient(x), x)

    def has_hessians(self):
        return False

    # here f is the input arg which is the fitness vector(objects + constraints)

    def feasibility_f(self, f: List):
        """
        This method will check the feasibility of a fitness vector f against the tolerances returned by c_tol.
        """
        cons = f[1:]
        val = self.c_tol
        feasb = all(k <= val[0] for k in cons)  # True or False
        # feasb = all(k <= 0 for k in cons)
        return feasb

    def feasibility_x(self, x):
        """
        This method will check the feasibility of the fitness corresponding to a decision vector x against
         the tolerances returned by c_tol.
         decision vector  = x or (optimization variables) = [x1, x2, x3]
        """
        fitness_vector = self.fitness(x)
        feasb = self.feasibility_f(fitness_vector)
        return feasb


    @staticmethod
    def const_eq(x: List):
        """
        equality constraint functions:
        """

        ce1 = 4*(x[0]-x[1])**2+x[1]-x[2]**2+x[2]-x[3]**2
        ce2 = 8*x[1]*(x[1]**2-x[0])-2*(1-x[1])+4*(x[1]-x[2])**2+x[0]**2+x[2]-x[3]**2+x[3]-x[4]**2
        ce3 = 8*x[2]*(x[2]**2-x[1])-2*(1-x[2])+4*(x[2]-x[3])**2+x[1]**2-x[0]+x[3]-x[4]**2+x[0]**2+x[4]-x[5]**2
        ce4 = 8*x[3]*(x[3]**2-x[2])-2*(1-x[3])+4*(x[3]-x[4])**2+x[2]**2-x[1]+x[4]-x[5]**2+x[1]**2+x[5]-x[0]

        return np.array([ce1, ce2, ce3, ce4])


    @staticmethod
    def const_ieq(x):
        """
        inequality constraint functions:
        """

        ci1 = 8*x[4]*(x[4]**2-x[3])-2*(1-x[4])+4*(x[4]-x[5])**2+x[3]**2-x[2]+x[5]+x[2]**2-x[1]
        ci2 = -(8*x[5] * (x[5]**2-x[4])-2*(1-x[5]) +x[4]**2-x[3]+x[3]**2 - x[4])

        return np.array([ci1, ci2])


# main optimization algorithm which runs using the pygmo liberary
# @jit(parallel=True, nopython=True, fastmath=True)
def optimizeGMO(verbosity: int):
    """
    implementation of the pygmo optimization algorithm

    """
    result = []
    for step in range(24):

        prob = pg.problem(GMO(step))
        print("step = ", step)
        print("check problem ->\n", prob)

        # define algorithms
        al = pg.gaco(gen=20, ker=10, q=0.1, impstop=30, acc=0.01, oracle=0.05, focus=1E-8, memory=False, seed=32*5000)
        # al = pg.cstrs_self_adaptive(iters=5, algo=pg.gwo(gen=10), seed=32*5000)
        # al = pg.cstrs_self_adaptive(iters=3, algo=pg.simulated_annealing(Ts=10.0, Tf=0.1, n_T_adj=10, n_range_adj=10, bin_size=10, start_range=1.0), seed=32*5000)
        # al = pg.cstrs_self_adaptive(iters=5, algo=pg.bee_colony(gen=3, limit=1, seed=32*5000), seed=32*5000)
        # al = pg.cstrs_self_adaptive(iters=3, algo=pg.pso(gen=3), seed=32*5000)
        # al = pg.pso(gen=5, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, variant=5, neighb_type=2, neighb_param=4, memory=False, seed=32*5000)
        # al = pg.cstrs_self_adaptive(iters=5, algo=pg.cmaes(gen=3, ftol=1e-08, xtol=1e-08), seed=32*5000)
        # al = pg.mbh(algo=pg.nlopt('auglag_eq'), stop=5, perturb=0.01, seed=32*5000)
        # al = pg.mbh(algo=pg.compass_search(max_fevals=1, start_range=0.1, stop_range=0.01, reduction_coeff=0.5), stop=5, perturb=0.01, seed=32*5000)
        # al = pg.ihs(gen=60, seed=32*5000)
        algo = pg.algorithm(uda=al)
        algo.set_verbosity(verbosity)


        pop = pg.population(prob=prob, size=10)
        pop = algo.evolve(pop)

        #res = pop.get_x()[
        #        pop.best_idx()
        #]

        res = pop.champion_x
        result.append(res)

    result = np.array(result)
    solution_df = pd.DataFrame(result, columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])

    # additional check
    print('fevals: ', pop.problem.get_fevals())
    print('gevals: ', pop.problem.get_gevals())
    return solution_df
