import numpy as np
import math
import matplotlib.pyplot as plt
from ode_solver import rk4, solve_ode
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from function_examples import pred_prey


def isolate_orbit(ode_data, time_data):
    """
    Finds the initial conditions and time period of an orbit using data from a solved ODE

    Parameters:
    ----------
        ode_data : array
            Data created by an ODE solver (currently set as RK4 for accuracy)
        time_data : array
             Time data which the function will search over to find an orbit

    Returns:
    ----------
        solution_array : list
            The initial conditions and time period of the orbit found
    """
    x_data = ode_data[:, 0]
    y_data = ode_data[:, 1]
    maximums = find_peaks(x_data)[0]
    previous_value = False
    previous_time = 0
    for i in maximums:
        if previous_value:
            if math.isclose(x_data[i], previous_value, abs_tol=1e-4):
                period = time_data[i] - previous_time
                solution_array = [x_data[i], y_data[i], period]
                return solution_array
        previous_value = x_data[i]
        previous_time = time_data[i]
    raise RuntimeError("No orbit found")


def shooting_conditions(ode, u0, pseudo, orbit, args):
    """
    Creates the augmented equation containing the conditions necessary to find an orbit of the ODE.

    Parameters:
    ----------
        ode : function
            ODE to find an orbit for
        u0 : array
            Current guess for the initial conditions of the orbit
        pseudo : list
            A list comprising of the parameters for numerical shooting when using pseudo-arclength
        orbit : Boolean
            Decides whether an orbit is being found or not
        args : numpy array
            The parameters of the ODE


    Returns:
    ----------
        g_condition : numpy array
            Contains the augmented equation to solve for. It is suitable for fsolve()
    """

    x0 = u0[:-1]  # Define independent variables
    t0 = u0[-1]
    sol = solve_ivp(ode, (0, t0), x0, max_step=1e-2, args=args)  # Solve ODE
    x_condition = x0 - sol.y[:, -1]
    t_condition = np.asarray(ode(t0, x0, *args)[0])  # Phase Condition
    g_condition = np.concatenate((x_condition, t_condition), axis=None)  # Group conditions together
    return g_condition


def pseudo_arclength_conditions(ode, u0, pseudo_info, orbit, args):
    """
    Creates the augmented equation containing the conditions necessary to find an orbit of the ODE when using
    pseudo-arclength continuation.

    Parameters:
    ----------
        ode : function
            ODE to find an orbit for
        u0 : array
            Current guess for the initial conditions of the orbit
        pseudo_info : list
            A list comprising of the parameters for numerical shooting when using pseudo-arclength
        orbit : Boolean
            Decides whether an orbit is being found or not
        args : numpy array
            The parameters of the ODE


    Returns:
    ----------
        g_condition : numpy array
            Contains the augmented equation to solve for. It is suitable for fsolve()
    """

    # Unpack pseudo argument
    state_prediction, state_secant, param_prediction, param_secant = pseudo_info[1:]
    if orbit:
        x0 = u0[:-2]  # Define independent variables
        t0 = u0[-2]
        vary_par = u0[-1]
        args[pseudo_info[0]] = vary_par
        sol = solve_ivp(ode, (0, t0), x0, max_step=1e-2, args=args)  # Solves ODE

        x_condition = x0 - sol.y[:, -1]
        t_condition = np.asarray(ode(t0, x0, *args)[0])  # Phase condition
        # Pseudo condition
        pseudo_info = np.dot(u0[:-1] - state_prediction, state_secant) + np.dot(vary_par - param_prediction,
                                                                                param_secant)
        g_condition = np.concatenate((x_condition, t_condition, pseudo_info), axis=None)  # Group conditions together
    else:
        x0 = u0[:-1]  # Define independent variables
        vary_par = u0[-1]
        args[pseudo_info[0]] = vary_par
        x_condition = np.asarray(ode(0, x0, *args)[0])
        # Pseudo condition
        pseudo_info = np.dot(x0 - state_prediction, state_secant) + np.dot(vary_par - param_prediction, param_secant)
        g_condition = np.concatenate((x_condition, pseudo_info), axis=None)  # Group conditions together
    return g_condition


def shooting(ode, u0, conditions, pseudo_info, orbit, args):
    """"
    Uses numerical shooting to find a limit cycle of a specified ODE.

    Parameters:
    ----------
        ode : function
            ODE to find an orbit for
        u0 : array
            Current guess for the initial conditions of the orbit
        conditions : function
            The type of conditions required for the task (normal or pseudo-arclength)
        pseudo_info : list
            A list comprising of the parameters for numerical shooting when using pseudo-arclength
        orbit : Boolean
            Decides whether an orbit is being found or not
        args : numpy array
            The parameters of the ODE


    Returns:
    ----------
        final : numpy array
            Contains the initial values for the limit cycle
    """

    final = fsolve(lambda x: conditions(ode, x, pseudo_info, orbit, args=args), u0)
    return final


def plot_function(ode, u0, time_range, step_size, solver, args):
    """
    Plots the solution of the ODE over a desired range of time

    Parameters:
    ----------
        ode : function
            ODE to find an orbit for
        u0 : array
            Current guess for the initial conditions of the orbit
        time_range : tuple
            The range of time to plot over
        step_size : float
            Maximum step-size for the solver
        solver : function
            Method to solve the ODE
        args : numpy array
            The parameters of the ODE
    """

    x0 = u0[:-1]  # Remove the time period from the initial condition of the orbit
    times = np.linspace(0, time_range, num=1000)

    # Solve the ODE using the initial conditions
    data_values = np.asarray(solve_ode(times, x0, step_size, solver, ode, args))

    # Plot the values
    for i in range(len(data_values[0])):
        plt.plot(times, data_values[:, i])
    # Using for loop so that the function is general and works for different dimensions

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('u', fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()


if __name__ == '__main__':

    # Plots the solved ODE with varying b values
    fig, axs = plt.subplots(2, 2)
    times1 = np.linspace(0, 200, num=1000)
    rk4_values_1 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.15, 0.1])))
    rk4_values_2 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.25, 0.1])))
    rk4_values_3 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.27, 0.1])))
    rk4_values_4 = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, rk4, pred_prey, np.array([1, 0.5, 0.1])))
    
    axs[0, 0].plot(rk4_values_1[:, 0])
    axs[0, 0].plot(rk4_values_1[:, 1])
    axs[0, 0].set_title('b = 0.15')
    axs[0, 1].plot(rk4_values_2[:, 0])
    axs[0, 1].plot(rk4_values_2[:, 1])
    axs[0, 1].set_title('b = 0.25')
    axs[1, 0].plot(rk4_values_3[:, 0])
    axs[1, 0].plot(rk4_values_3[:, 1])
    axs[1, 0].set_title('b = 0.27')
    axs[1, 1].plot(rk4_values_4[:, 0])
    axs[1, 1].plot(rk4_values_4[:, 1])
    axs[1, 1].set_title('b = 0.5')
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='u')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

    # Plot solution of the predator-prey model
    args = np.array([1, 0.2, 0.1])
    times1 = np.linspace(0, 400, num=1000)
    RK4_values = np.asarray(solve_ode(times1, np.array([1, 1]), 0.1, rk4, pred_prey, args))
    plot_function(pred_prey, np.array([1, 1, 20]), 200, 0.1, rk4, args)

    # Find an orbit of the predator-prey model
    init_vals = isolate_orbit(RK4_values, times1)
    # State the initial values and period of the orbit
    print('Initial values for the orbit found: ', init_vals[:-1])
    print('Period of the orbit found: ', init_vals[-1])
    # Plot the orbit
    plot_function(pred_prey, init_vals, init_vals[-1], 0.1, rk4, args)

    # Check if the numerical shooting function can find the same orbit found previously
    shooting_values = shooting(pred_prey, init_vals, shooting_conditions, False, True, args)
    print('Initial values for the orbit found from numerical shooting: ', shooting_values[:-1])
    print('Period of the orbit found from numerical shooting: ', shooting_values[-1])

    plot_function(pred_prey, shooting_values, shooting_values[-1], 0.1, rk4, args)
