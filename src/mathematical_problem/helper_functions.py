import pandas as pd
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from src.mathematical_problem.model import model


def func_plot(solution: List):

    df = pd.read_csv('src/mathematical_problem/setpoints.csv')
    steps = df['step']
    setpoint1 = df['setpoint1']
    setpoint2 = df['setpoint2']
    setpoint3 = df['setpoint3']
    setpoint4 = df['setpoint4']

    quantity1 = []
    quantity2 = []
    quantity3 = []
    quantity4 = []

    for index, row in df.iterrows():
        x = [row['x1'], row['x2'], row['x3'], row['x4'], row['x5'], row['x6']]
        model_state0, model_state1  = model(x)

        quantity1.append([(model_state0 - model_state1) + (model_state0 * 1.20)])
        quantity2.append([(model_state0 + model_state1 / model_state0) + (model_state1 * 1.50)])
        quantity3.append([(row['x4'] * model_state1 /2.5) + (model_state0 * 2.50)])
        quantity4.append([(model_state0 /1.5) - (math.log(model_state1*x[0]*row['x6']))])



    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax1.plot(steps, setpoint1, label = 'setpoint')
    ax1.plot(steps, quantity1, label = 'optimized quantity')
    ax1.set_title('1st Objective')
    ax1.set_xlabel('steps')
    ax1.legend(['setpoint', 'optimized quantity'])
    plt.savefig('src/mathematical_problem/plots/fig1.png')

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(steps, setpoint2, label = 'setpoint')
    ax2.plot(steps, quantity2, label = 'optimized quantity')
    ax2.set_title('2nd Objective')
    ax2.set_xlabel('steps')
    ax2.legend(['setpoint', 'optimized quantity'])
    plt.savefig('src/mathematical_problem/plots/fig2.png')

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(steps, setpoint3, label = 'setpoint')
    ax3.plot(steps, quantity3, label = 'optimized quantity')
    ax3.set_title('3rd Objective')
    ax3.set_xlabel('steps')
    ax3.legend(['setpoint', 'optimized quantity'])
    plt.savefig('src/mathematical_problem/plots/fig3.png')

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.plot(steps, setpoint4, label = 'setpoint')
    ax4.plot(steps, quantity4, label = 'optimized quantity')
    ax4.set_title('4th Objective')
    ax4.set_xlabel('steps')
    ax4.legend(['setpoint', 'optimized quantity'])
    plt.savefig('src/mathematical_problem/plots/fig4.png')

    plt.show(block=False)
