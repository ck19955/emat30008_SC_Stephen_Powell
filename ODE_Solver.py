# Solves ODEs
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def euler_step(t_n, x_n, step_size):
    x = x_n + step_size * ode(t_n, x_n)
    return x


def RK4(t_n, x_n, step_size):
    k1 = ode(t_n, x_n)
    k2 = ode(t_n + step_size/2, x_n + k1*(step_size/2))
    k3 = ode(t_n + step_size/2, x_n + k1*(step_size/2))
    k4 = ode(t_n + step_size, x_n + k3*step_size)
    x = x_n + ((k1 + 2*k2 + 2*k3 + k4)/6)*step_size
    return x


def solve_to(t_0, t_end, x_0, deltaT_max, method):
    x = x_0
    t = t_0
    while t < t_end:
        if t + deltaT_max <= t_end:
            x = method(t, x, deltaT_max)
            t = t + deltaT_max
        else:
            deltaT_max = t_end - t
    return x


def solve_ode(t_values, x_0, deltaT_max, method):
    x_values = [0] * len(t_values)
    x_values[0] = x_0
    for i in range(len(t_values)-1):
        x_values[i+1] = solve_to(t_values[i], t_values[i + 1], x_values[i], deltaT_max, method)
    return x_values


def ode(t, x):
    return x


def f(t, x):
    return math.exp(t)


def error_plot(t_values, x_0):
    x_value = f(t_values[1], 0)
    step_sizes = np.linspace(t_values[0], t_values[-1], num=1000)[1:]
    error_eul = np.zeros(len(step_sizes))
    error_RK4 = np.zeros(len(step_sizes))
    error_match = 1e-2
    time_eul = 0
    time_RK4 = 0
    for i in range(len(step_sizes)):
        init_time = time.perf_counter()
        predict_eul = solve_ode(t_values, x_0, step_sizes[i], euler_step)
        time_1 = time.perf_counter() - init_time
        predict_RK4 = solve_ode(t_values, x_0, step_sizes[i], RK4)
        time_2 = time.perf_counter() - init_time
        error_eul[i] = abs(predict_eul[-1] - x_value)
        error_RK4[i] = abs(predict_RK4[-1] - x_value)
        if math.isclose(error_match, error_eul[i], abs_tol=1e-3):
            time_eul = time_1
        if math.isclose(error_match, error_RK4[i], abs_tol=1e-3):
            time_RK4 = time_2
    plt.loglog(step_sizes, error_eul, label='Euler Method')
    plt.loglog(step_sizes, error_RK4, label='RK4 Method')
    plt.show()
    return error_eul, error_RK4, time_eul, time_RK4


times = [0, 1]
error_1, error_2, time_euler, time_RungeKutta = error_plot(times, 1)
print(time_euler)
print(time_RungeKutta)

