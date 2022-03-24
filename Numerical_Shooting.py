import numpy as np
import matplotlib.pyplot as plt
from ODE_Solver import *
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from function_examples import *


def isolate_orbit(ode_data, time_data):
    """
    :param ode_data: Data created by an ODE solver (currently set as RK4)
    :param time_data: A linspace of time which the function will search over to find an orbit
    :return: If an orbit is found returns [initial x value, initial y value, time period] for the orbit
            else returns None
    """
    x_data = ode_data[:, 0]
    y_data = ode_data[:, 1]
    maximums = argrelextrema(x_data, np.greater)[0]
    previous_value = False
    previous_time = 0
    for i in maximums:
        if previous_value:
            if math.isclose(x_data[i], previous_value, abs_tol=1e-4):
                period = time_data[i] - previous_time
                return x_data[i], y_data[i], period
        previous_value = x_data[i]
        previous_time = time_data[i]
    return


def shooting_conditions(ode, u0, args):
    """
    :param ode: ODE to find orbit for
    :param u0: initial conditions
    :param args: Arguments for the ODE
    :return: The augmented equation suitable for fsolve()
    """
    x0 = u0[:-1]
    t0 = u0[-1]
    sol = solve_ivp(ode, (0, t0), x0, max_step=1e-2, args=args)
    x_condition = x0 - sol.y[:, -1]
    t_condition = np.asarray(ode(t0, x0, *args)[0])
    g_condition = np.concatenate((x_condition, t_condition), axis=None)
    return g_condition


def shooting(ode, u0, args):
    """"
    A function that uses numerical shooting to find limit cycles of a specified ODE.

    Parameters
    ode : function
        The ODE to apply shooting to. The ode function should take arguments for the independent variable, dependant
        variable and constants, and return the right-hand side of the ODE as a numpy.array.
    u0 : numpy.array
        An initial guess at the initial values for the limit cycle.

    Returns
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """
    final = fsolve(lambda x: shooting_conditions(ode, x, args=args), u0)
    return final


def plot_function(u0, step_size, solver):
    """
    :param u0: Initial values
    :param step_size: maximum step-size for the solver
    :param solver: chosen solver (not yet implemented, currently set RK4 as default)
    :return: Plot of the isolated orbit found by the numerical shooter
    """
    x0 = u0[:-1]
    t0 = u0[-1]
    times = np.linspace(0, t0, num=1000)
    data_values = np.asarray(solve_ode(times, x0, step_size, RK4, pred_prey, args))
    plt.plot(times, data_values[:, 0])
    plt.plot(times, data_values[:, 1])
    plt.xlabel('time')
    plt.ylabel('u')
    plt.show()


'''
# args = np.array([1, -1])
args = np.array([1, 0.2, 0.1])
#print(shooting(pred_prey, np.array([1, 1, 20]), args))


# Plots the solved ODE
times1 = np.linspace(0, 400, num=1000)
# RK4_values = np.asarray(solve_ode(times1, np.array([0.9, 0]), 0.1, RK4, hopf_bif, args))
RK4_values = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, pred_prey, args))
init_vals = isolate_orbit(RK4_values, times1)
plot_function(init_vals, 0.1, 0)
'''

'''
# Plots the solved ODE with varying b values

fig, axs = plt.subplots(2, 2)
times1 = np.linspace(0, 200, num=1000)
RK4_values_1 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.15, 0.1])))
RK4_values_2 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.25, 0.1])))
RK4_values_3 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.27, 0.1])))
RK4_values_4 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, RK4, ode_num, np.array([1, 0.5, 0.1])))


axs[0, 0].plot(RK4_values_1[:, 0])
axs[0, 0].plot(RK4_values_1[:, 1])
axs[0, 0].set_title('b = 0.15')
axs[0, 1].plot(RK4_values_2[:, 0])
axs[0, 1].plot(RK4_values_2[:, 1])
axs[0, 1].set_title('b = 0.25')
axs[1, 0].plot(RK4_values_3[:, 0])
axs[1, 0].plot(RK4_values_3[:, 1])
axs[1, 0].set_title('b = 0.27')
axs[1, 1].plot(RK4_values_4[:, 0])
axs[1, 1].plot(RK4_values_4[:, 1])
axs[1, 1].set_title('b = 0.5')
for ax in axs.flat:
    ax.set(xlabel='t', ylabel='u')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()
'''



