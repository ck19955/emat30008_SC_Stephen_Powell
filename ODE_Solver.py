# Solves ODEs
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from function_examples import *


def euler_step(t_n, x_n, step_size, ode, args):
    """
    Executes a single step of the forward euler method for given value, t_n

    Parameters:
    ----------
        t_n : float
            The value of the independent variable
        x_n : numpy array
            The value of the dependant variable
        step_size : integer
            The step-size of the euler step to be executed
        ode : function
            The ODE for which the euler step predicts
        args : list
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable after an euler step
    """

    x = x_n + step_size * ode(t_n, x_n, *args)
    return x


def RK4(t_n, x_n, step_size,ode,args):
    """
    The RK4() function executes a single step of the 4th Order Runge Kutta method for given value, t_n

    Parameters:
        t_n         - The value of the independent variable
        x_n         - The value of the dependant variable
        step_size   - The step-size of the RK4 to be executed

    Returns:
        x           - The new value of the dependant variable after a RK4 step
    """
    k1 = ode(t_n, x_n, *args)
    k2 = ode(t_n + step_size/2, x_n + k1*(step_size/2), *args)
    k3 = ode(t_n + step_size/2, x_n + k1*(step_size/2), *args)
    k4 = ode(t_n + step_size, x_n + k3*step_size, *args)
    x = x_n + ((k1 + 2*k2 + 2*k3 + k4)/6)*step_size
    return x


def solve_to(t_0, t_end, x_0, deltaT_max, method, ode, args):
    """
    The solve_to() function solves an ODE for an array of values of the independent variable.

    Parameters:
        t_0         - The initial value of the independent variable
        t_end       - The final value of the independent variable
        x_0         - The initial value of the dependant variable
        deltaT_max  - The maximum value the step-size can take when performing a step of a numerical method
        method      - The numerical method to be used to solve the ODE

    Returns:
        x           - The final value of the dependant variable
    """

    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = method(t, x, deltaT_max, ode, args)
            t += deltaT_max
        else:
            deltaT_max = t_end - t  # Reduces deltaT_max to fit next iteration perfectly
    return x


def solve_ode(t_values, x_0, deltaT_max, method, ode, args):
    """
    The solve_ode() function solves an ODE from an initial value, t_0 to a final value t_end. However, the difference
    between the two values must be smaller than deltaT_max.

    Parameters:
        t_values    - An array of values of the independent variable, for which each t_value should be a assigned an
                       x value
        x_0         - The initial value of the dependant variable
        deltaT_max  - The maximum value the step-size can take when performing a step of a numerical method
        method      - The numerical method to be used to solve the ODE

    Returns:
        x           - The final value of the dependant variable
    """

    x_values = [0] * len(t_values)
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(t_values[i], t_values[i + 1], x_values[i], deltaT_max, method, ode, args)
    return x_values


def error_plot(t_values, x_0, ode, exact, args):
    """
    The error_plot() function examines the change in error when varying the step-size for various numerical methods. In
    addition, it compares the time taken for both the euler method and fourth order runge kutta (RK4 to produce a
    solution to the ODE with the same magnitude of error.

    Parameters:
        t_values         - The range of values of the independent variable
        x_0              - The initial value of the dependant variable

    Returns:
        error_eul           - The array of errors for the euler method at each step-size value
        error_RK4           - The array of errors for the RK4 method at each step-size value
        time_eul            - The time take for the euler method to reach an error specified in the function
        time_RK4            - The time take for the RK4 method to reach an error specified in the function
    """

    x_value = exact(t_values[1], 0)
    step_sizes = np.logspace(-6, 0, 10)
    error_eul = np.zeros(len(step_sizes))
    error_RK4 = np.zeros(len(step_sizes))
    error_match = 1e-6
    time_eul = 0
    time_RK4 = 0
    for i in range(len(step_sizes)):
        init_time = time.perf_counter()
        predict_eul = solve_ode(t_values, x_0, step_sizes[i], euler_step, ode, args)
        time_1 = time.perf_counter() - init_time
        init_time = time.perf_counter()
        predict_RK4 = solve_ode(t_values, x_0, step_sizes[i], RK4, ode, args)
        time_2 = time.perf_counter() - init_time
        error_eul[i] = abs(predict_eul[-1] - x_value)
        error_RK4[i] = abs(predict_RK4[-1] - x_value)
        if math.isclose(error_match, error_eul[i], abs_tol=1e-5):
            time_eul = time_1
        if math.isclose(error_match, error_RK4[i], abs_tol=1e-5):
            time_RK4 = time_2

    # Plot the errors of euler and RK4
    plt.loglog(step_sizes, error_eul, label='Euler Method')
    plt.loglog(step_sizes, error_RK4, label='RK4 Method')
    plt.legend()
    plt.ylabel("Error of approximation")
    plt.xlabel("Size of timestep")
    plt.show()
    return error_eul, error_RK4, time_eul, time_RK4


def plot_approx(t_values, x_values, step_size, ode, exact, args):
    """
    The plot_approx() function plots the numerical solutions from the numerical methods.

    Parameters:
        t_values        - The range of values of the independent variable
        x_values        - The initial values of the dependant variables
        step_size       - The step-size of the numerical methods to be executed
    """

    RK4_values = np.asarray(solve_ode(t_values, x_values, step_size, RK4, ode, args))
    euler_values = np.asarray(solve_ode(t_values, x_values, step_size, euler_step, ode, args))
    exact_values = [0]*len(RK4_values)
    for i in range(len(t_values)):
        exact_values[i] = exact(t_values[i], x_values)

    # Plot x against dx/dt
    plt.plot([item[0] for item in exact_values], [item[1] for item in exact_values])
    plt.plot(RK4_values[:, 0], RK4_values[:, 1])
    plt.plot(euler_values[:, 0], euler_values[:, 1])
    plt.legend()
    plt.show()
    return


# times1 = np.linspace(0, 20, num=50)
# times = [0, 1]
# error_1, error_2, time_euler, time_RungeKutta = error_plot(times, 1, ode_first_order, exponential, [])
# plot_approx(times1, np.array([3, 4]), 0.1, ode_second_order, exact_second_order, [])
