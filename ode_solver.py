# Solves ODEs
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from function_examples import ode_first_order, ode_second_order, exact_second_order, exponential


def euler_step(t_n, x_n, step_size, ode, args):
    """
    Executes a single step of the forward euler method for a given value, t_n

    Parameters:
    ----------
        t_n : float
            The value of the independent variable
        x_n : numpy array
            The value of the dependant variable(s)
        step_size : float
            The step-size of the euler step to be executed
        ode : function
            The ODE for which the euler step predicts
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable(s) after an euler step
    """

    x = x_n + step_size * ode(t_n, x_n, *args)
    return x


def rk4(t_n, x_n, step_size, ode, args):
    """
    Executes a single step of the 4th Order Runge Kutta method for given value, t_n

    Parameters:
    ----------
        t_n : float
            The value of the independent variable
        x_n : numpy array
            The value of the dependant variable(s)
        step_size : float
            The step-size of the rk4 step to be executed
        ode : function
            The ODE for which the rk4 step predicts
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable(s) after an rk4 step
"""
    k1 = ode(t_n, x_n, *args)
    k2 = ode(t_n + step_size/2, x_n + k1*(step_size/2), *args)
    k3 = ode(t_n + step_size/2, x_n + k2*(step_size/2), *args)
    k4 = ode(t_n + step_size, x_n + k3*step_size, *args)
    x = x_n + ((k1 + 2*k2 + 2*k3 + k4)/6)*step_size
    return x


def improved_euler_step(t_n, x_n, step_size, ode, args):
    """
    Executes a single step of the improved euler method for given value, t_n

    Parameters:
    ----------
        t_n : float
            The value of the independent variable
        x_n : numpy array
            The value of the dependant variable(s)
        step_size : float
            The step-size of the improved euler step to be executed
        ode : function
            The ODE for which the improved euler step predicts
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable(s) after an improved euler step
    """

    x = x_n + step_size/2 * (ode(t_n, x_n, *args) + ode(t_n + step_size, x_n + step_size*ode(t_n, x_n, *args), *args))
    return x


def heun_step(t_n, x_n, step_size, ode, args):
    """
    Executes a single step of Heun's method for given value, t_n

    Parameters:
    ----------
        t_n : float
            The value of the independent variable
        x_n : numpy array
            The value of the dependant variable(s)
        step_size : float
            The step-size of a single Heun's method step to be executed
        ode : function
            The ODE for which a single Heun's method step predicts
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The new value of the dependant variable(s) after a single Heun's step
    """

    k1 = ode(t_n, x_n, *args)
    k2 = ode(t_n + step_size/3, x_n + k1*(step_size/3), *args)
    k3 = ode(t_n + 2*step_size/3, x_n + k2*(2*step_size/3), *args)

    x = x_n + ((k1 + 3*k3)/4)*step_size
    return x


def solve_to(t_0, t_end, x_0, deltaT_max, method, ode, args):
    """
    Solves an ODE for an array of values of the independent variable, t.

    Parameters:
    ----------
        t_0 : float
            The initial value of the independent variable
        t_end : float
            The final value of the independent variable
        x_0 : numpy array
            The value of the dependant variable(s)
        deltaT_max : float
            The maximum value the step-size can take when performing a step of a numerical method
        ode : function
            The ODE to be integrated
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            The final value of the dependant variable(s)
    """

    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = method(t, x, deltaT_max, ode, args)
            t += deltaT_max
        else:
            deltaT_max = t_end - t  # Reduces deltaT_max to fit next iteration perfectly
            # This coding decision was chosen to ensure the final time value, t_end is accounted for
    return x


def solve_ode(t_values, x_0, deltaT_max, method, ode, args=None):
    """
    Solves an ODE from an initial value, t_0 to a final value t_end. However, the difference
    between the two values must be smaller than deltaT_max.

    Parameters:
    ----------
        t_values : array
            values of the independent variable
        x_0 : numpy array
            The initial value of the dependant variable(s)
        deltaT_max : float
            The maximum value the step-size can take when performing a step of a numerical method
        method : function
            The one-step integration method to be used on the ODE
        ode : function
            The ODE to be integrated
        args : numpy array
            The parameters of the ODE

    Returns:
    ----------
        x : numpy array
            An array of dependant variable values for the integrated ODE
    """
    if args is None:  # args cannot be passed through as None because this induces an error when trying to unpack
        args = []  # This allows for unpacking

    x_values = [0] * len(t_values)  # Create a list of solution x values
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(t_values[i], t_values[i + 1], x_values[i], deltaT_max, method, ode, args)
    return x_values


def error_plot(t_values, x_0, ode, exact, error_match, args=None):
    """
    Examines the change in error when varying the step-size for various numerical methods. In addition, it compares the
    time taken and step size required for both the methods to reach same magnitude of error. Also plots a log-log plot
    the different methods as the step-size varies.

    Parameters:
    ----------
        t_values : array
            values of the independent variable
        x_0 : numpy array
            The initial value of the dependant variable(s)
        ode : function
            The ODE to be integrated
        exact : function
            The exact function which returns the true solution
        error_match : float
            The error used to find how long each method takes and the step-size required to reach a specific error
        args : numpy array
            The parameters of the ODE
    """

    if args is None:  # args cannot be passed through as None because this induces an error when trying to unpack
        args = []  # This allows for unpacking

    exact_value = exact(t_values[1], 0)
    step_sizes = np.logspace(-6, 0, 10)

    # Set up an array for the errors of the different one-step integration methods
    error_eul = np.zeros(len(step_sizes))
    error_rk4 = np.zeros(len(step_sizes))
    error_improv = np.zeros(len(step_sizes))
    error_heun = np.zeros(len(step_sizes))

    # Initialise the values for the step_size for each method to reach a specific error
    step_size_eul = 0
    step_size_rk4 = 0
    step_size_improv = 0
    step_size_heun = 0

    # Initialise the values for the time taken for each method to reach a specific error
    final_time_eul = 0
    final_time_rk4 = 0
    final_time_improv = 0
    final_time_heun = 0

    for i in range(len(step_sizes)):
        # Here time.perf_counter() is used instead of time.time() because perf_counter is more accurate and does not
        # appear to be significantly more computationally expensive

        init_time = time.perf_counter()  # Set time
        predict_eul = solve_ode(t_values, x_0, step_sizes[i], euler_step, ode, args)
        time_eul = time.perf_counter() - init_time  # Find difference in time since the previously set time

        init_time = time.perf_counter()  # Set time
        predict_rk4 = solve_ode(t_values, x_0, step_sizes[i], rk4, ode, args)
        time_rk4 = time.perf_counter() - init_time  # Find difference in time since the previously set time

        init_time = time.perf_counter()  # Set time
        predict_improv = solve_ode(t_values, x_0, step_sizes[i], improved_euler_step, ode, args)
        time_improv = time.perf_counter() - init_time  # Find difference in time since the previously set time

        init_time = time.perf_counter()  # Set time
        predict_heun = solve_ode(t_values, x_0, step_sizes[i], heun_step, ode, args)
        time_heun = time.perf_counter() - init_time  # Find difference in time since the previously set time

        # Find errors by comparing to the exact value
        error_eul[i] = abs(predict_eul[-1] - exact_value)
        error_rk4[i] = abs(predict_rk4[-1] - exact_value)
        error_improv[i] = abs(predict_improv[-1] - exact_value)
        error_heun[i] = abs(predict_heun[-1] - exact_value)

        # Check if the error from each method is close to the specific error chosen (error_match)
        if math.isclose(error_match, error_eul[i], abs_tol=error_match):
            # Using math.isclose() because it is unlikely for the error to be equal to the error specified
            final_time_eul = time_eul
            step_size_eul = step_sizes[i]
        if math.isclose(error_match, error_rk4[i], abs_tol=error_match):
            final_time_rk4 = time_rk4
            step_size_rk4 = step_sizes[i]
        if math.isclose(error_match, error_improv[i], abs_tol=error_match):
            final_time_improv = time_improv
            step_size_improv = step_sizes[i]
        if math.isclose(error_match, error_heun[i], abs_tol=error_match):
            final_time_heun = time_heun
            step_size_heun = step_sizes[i]
    print('Time taken by Euler Method: ', final_time_eul)
    print('Time taken by 4th Order Runge Kutta Method: ', final_time_rk4)
    print('Time taken by Improved Euler Method: ', final_time_improv)
    print('Time taken by Heun Method: ', final_time_heun)

    print('Step-size for Euler Method: ', step_size_eul)
    print('Step-size for 4th Order Runge Kutta Method: ', step_size_rk4)
    print('Step-size for Improved Euler Method: ', step_size_improv)
    print('Step-size for Heun Method: ', step_size_heun)

    # Plot the errors of euler and RK4
    plt.loglog(step_sizes, error_eul, label='Euler Method')
    plt.loglog(step_sizes, error_rk4, label='rk4 Method')
    plt.loglog(step_sizes, error_improv, label='Improved Euler Method')
    plt.loglog(step_sizes, error_heun, label='Heun Method')
    plt.legend()
    plt.ylabel("Error of approximation")
    plt.xlabel("Size of timestep")
    plt.show()
    return


def plot_approx(t_values, x_values, step_size, ode, exact, args=None):
    """
    Plots the numerical solutions from the different one-step integration methods. Also plots x against dx/dt to
    illustrate that the Forward Euler method is poor at solving second-order ODEs.

    Parameters:
    ----------
        t_values : array
            values of the independent variable
        x_values : numpy array
            values of the dependant variable(s)
        step_size : float
            The step-size of a single Heun's method step to be executed
        ode : function
            The ODE to be integrated
        exact : function
            The exact function which returns the true solution
        args : numpy array
            The parameters of the ODE
    """

    if args is None:  # args cannot be passed through as None because this induces an error when trying to unpack
        args = []  # This allows for unpacking

    # Initialise the solutions of all the one-step inegration methods
    rk4_values = np.asarray(solve_ode(t_values, x_values, step_size, rk4, ode, args))
    euler_values = np.asarray(solve_ode(t_values, x_values, step_size, euler_step, ode, args))
    improved_euler_values = np.asarray(solve_ode(t_values, x_values, step_size, improved_euler_step, ode, args))
    heun_values = np.asarray(solve_ode(t_values, x_values, step_size, heun_step, ode, args))

    # Find the exact solutions
    exact_values = [0]*len(rk4_values)
    for i in range(len(t_values)):
        exact_values[i] = exact(t_values[i], x_values, *args)

    # Plot solution of ode
    plt.plot(rk4_values, label='rk4 Method')
    plt.plot(exact_values, label='Exact Values')
    plt.plot(euler_values, label='Euler Method')
    plt.plot(improved_euler_values, label='improved')
    plt.plot(heun_values, label='heun')
    plt.ylabel("x")
    plt.xlabel("Time")
    plt.legend(prop={'size': 9})
    plt.show()

    # Plot x against dx/dt
    plt.plot([item[0] for item in exact_values], [item[1] for item in exact_values], label='Exact Values')
    plt.plot(rk4_values[:, 0], rk4_values[:, 1], label='rk4 Method')
    plt.plot(euler_values[:, 0], euler_values[:, 1], label='Euler Method')
    plt.ylabel("x")
    plt.xlabel("dx/dt")
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    # Plots the errors of each one-step integration method and prints the time taken for each method and the step-size
    # required to reach an accuracy of 1e-6.
    error_plot([0, 1], 1, ode_first_order, exponential, 1e-6)

    # Plots the solutions of the ode from the different integration methods and plots the limitation in the use of
    # Euler's method.
    times1 = np.linspace(0, 20, num=100)
    plot_approx(times1, np.array([3, 4]), 0.1, ode_second_order, exact_second_order)
