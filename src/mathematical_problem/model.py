# The module that represents the mathematical equations of the model

import math

def model(x):
    """
    the mathematical model of the system
    """
    # input x :- an array of 6 elments [x0,x1,...x5]

    # model has two states
    state0 = 0
    state1 = 0

    # first state
    for i in range(1, 4):
        state0 += (x[2*i-2]-3)**2 / 1000. - (x[2*i-2]-x[2*i-1]) + math.exp(20.*(x[2*i - 2]-x[2*i-1]))

    # second state
    for i in range(1, 4):
        state1 += (x[2*i-2]-x[2*i-1])**3 / 500. - (x[2*i-2]-3) + (x[2*i - 2]-x[2*i-1])


    return (state0, state1)
